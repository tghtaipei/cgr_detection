from datetime import datetime
import time
import cv2
import numpy as np
from ov_inference import cgr_detect_with_onnx, cgr_detect_sahi
# import trt_inference_detr
# import trt_inference_yolo
import smoke_inference

class Colors:
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


# 绘制函数颜色库
colors = Colors()
kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# 保存间隔
count = [0]
# 吸烟置信度
cgr_conf=[0.4]
# 使用模型
model=[2]
# 人员信息列表，包括每个人的id与吸烟检测累计值，累计超过吸烟设定阈值就会被认为在吸烟
ids = {}

# ── 判定條件 1：10 秒內手靠近嘴部 2 次以上（高權重）──────────────────────
mouth_approach_log = {}   # {id: [timestamp, ...]}  各次靠近的時間戳
MOUTH_WINDOW        = 10.0  # 觀察窗口（秒）
MOUTH_MIN_COUNT     = 2     # 窗口內最少靠近次數
APPROACH_GAP        = 1.0   # 兩次靠近事件的最小間隔（秒），避免同一次靠近重複計數
HIGH_WEIGHT_BONUS   = 15    # 觸發時加分

# ── 判定條件 2：嫌疑人在同一位置超過 10 秒（一般權重）──────────────────────
position_log = {}    # {id: {'ref_center': (cx, cy), 'start_time': float}}
POSITION_WINDOW     = 10.0  # 停留時間門檻（秒）
POSITION_MOVE_RATIO = 0.30  # 位移超過人體框寬/高的此比例才視為移動
NORMAL_WEIGHT_BONUS = 8     # 觸發時加分

# ── 防止同一條件在短時間內重複加分 ───────────────────────────────────────
bonus_cooldown = {}   # {id: {'mouth': float, 'position': float}}
BONUS_COOLDOWN_SEC  = 5.0   # 同一條件再次觸發的最短間隔（秒）

# ── 曾確認持有香菸的人員集合 ──────────────────────────────────────────────
# 一旦某人被 status==2 確認，後續 status==1 改為累加而非遞減
confirmed_smokers = set()  # {id, ...}

# ── 條件 3：偵測到煙霧與人物重疊（中等權重 +10）────────────────────────
SMOKE_BONUS          = 10    # 偵測到煙霧時加分
smoke_cooldown       = {}    # {id: float}  上次觸發時間
_smoke_available     = None  # 延遲初始化：None=未檢查, True/False=檢查結果

# SAHI 採樣窗口：角度 < 55° 後，每隔 1 秒採樣一次，共採 3 秒
# {id: {'window_start': float, 'last_sample': float,
#        'smoke_boxes': list, 'hit': bool}}
SAHI_SAMPLE_INTERVAL = 1.0   # 每次採樣間隔（秒）
SAHI_WINDOW_SEC      = 3.0   # 採樣窗口總長（秒）
SAHI_PERSON_COOLDOWN = 10.0  # 同一人兩次窗口的最短間隔（秒）
_sahi_state          = {}    # 每人的採樣狀態
_sahi_last_close     = {}    # {id: float} 上次窗口關閉時間


def init_model(models):
    model[0]=models

def judge_smoke(pose_result, img, label):
    k = pose_result.keypoints
    left_angle, right_angle = cal_angle(k)
    left_hand_index = 9
    right_hand_index = 10
    # 如果角度小于55度或受手嘴距离小于0.8
    if int(left_angle) < 55 or cal_dis(k, left_hand_index) < 0.8:
        if cgr_detect(pose_result, img, left_hand_index, label):
            # 检测到香烟
            return 2
        else:
            return 1

    elif int(right_angle) < 55 or cal_dis(k, right_hand_index) < 0.8:
        if cgr_detect(pose_result, img, right_hand_index, label):
            return 2
        else:
            return 1

    return 0


def detect_and_draw(pose_result, img, opt):
    global _smoke_available
    smoking_threshold = opt.threshold
    cgr_conf[0] = opt.cgr_conf
    cgrlabel = []
    now = time.time()

    # ── 條件 3：每幀全畫面煙霧偵測（簡單全幀，非 SAHI）────────────────────
    if _smoke_available is None:
        _smoke_available = smoke_inference.is_model_available()
        if _smoke_available:
            print("[func] 煙霧偵測模型已啟用（條件 3）")
        else:
            print("[func] 煙霧偵測模型未找到，條件 3 停用（請執行 train_smoke.py 後複製模型）")
    smoke_boxes = smoke_inference.detect_smoke(img) if _smoke_available else []

    all_sahi_cig_boxes: list = []   # 收集所有人的 SAHI 香菸框（供畫圖用）

    for d in pose_result:
        conf, idd = float(d.conf), None if d.id is None else int(d.id)
        if idd not in ids.keys():
            ids[idd] = np.array([idd, 0])

        condition = ids[idd]
        status = judge_smoke(d, img, cgrlabel)

        # ── SAHI 香菸偵測：手角度 < 55° → 啟動採樣窗口，每秒採樣 1 次共 3 秒 ──
        person_sahi_cig_boxes: list = []
        if idd is not None:
            left_angle, right_angle = cal_angle(d.keypoints)
            pose_active = min(left_angle, right_angle) < 55

            if pose_active:
                state = _sahi_state.get(idd)
                if state is None:
                    # 10 秒冷卻內不重新開窗
                    last_close = _sahi_last_close.get(idd, 0.0)
                    if (now - last_close) < SAHI_PERSON_COOLDOWN:
                        pass  # 冷卻中，跳過
                    else:
                        _sahi_state[idd] = {
                            'window_start': now, 'last_sample': 0.0,
                            'cig_boxes': [], 'hit': False
                        }
                    state = _sahi_state.get(idd)

                if state is not None:
                    window_elapsed = now - state['window_start']
                    if window_elapsed <= SAHI_WINDOW_SEC:
                        # 窗口內：每隔 SAHI_SAMPLE_INTERVAL 秒採樣一次
                        if (now - state['last_sample']) >= SAHI_SAMPLE_INTERVAL:
                            boxes = cgr_detect_sahi(img, person_xyxy=d.xyxy)
                            state['last_sample'] = now
                            if boxes:
                                state['cig_boxes'] = boxes
                                state['hit'] = True
                    # 沿用窗口內最後一次有偵測到的結果（供畫圖）
                    person_sahi_cig_boxes = state.get('cig_boxes', [])
                    all_sahi_cig_boxes.extend(person_sahi_cig_boxes)
            else:
                # 手放下，若窗口仍開著則關閉並記錄冷卻起點
                if idd in _sahi_state:
                    _sahi_last_close[idd] = now
                    _sahi_state.pop(idd, None)

        # ── 吸菸判定計分邏輯 ─────────────────────────────────────────────
        if status == 2:
            # 香菸確認：標記此人，每幀 +20
            if idd is not None:
                confirmed_smokers.add(idd)
            if condition[1] < 100:
                condition[1] += 20
            if condition[1] < smoking_threshold:
                box_label(d.xyxy, img, 3, "Suspicious", (28, 172, 255))
        elif status == 1:
            # 手靠近嘴但沒香菸：
            #   曾被確認抽菸 → +5（持續可疑行為仍累加）
            #   未曾確認    → -1（原始行為，維持緩降）
            if idd is not None and idd in confirmed_smokers:
                if condition[1] < 100:
                    condition[1] += 5
            else:
                if condition[1] > 0:
                    condition[1] -= 1
            box_label(d.xyxy, img, 3, "Suspicious", (28, 172, 255))
        else:
            if condition[1] > 0:
                condition[1] -= 1

        # ── 條件 1：10 秒內手靠近嘴部 2 次以上（高權重 +15）────────────
        if idd is not None and status >= 1:
            log = mouth_approach_log.setdefault(idd, [])
            # 距離上次靠近事件超過 APPROACH_GAP 秒才記錄新事件
            if not log or (now - log[-1]) > APPROACH_GAP:
                log.append(now)
            # 清除窗口外的舊紀錄
            mouth_approach_log[idd] = [t for t in log if now - t <= MOUTH_WINDOW]

        if idd is not None and idd in mouth_approach_log:
            recent = len(mouth_approach_log[idd])
            last_bonus = bonus_cooldown.get(idd, {}).get('mouth', 0)
            if recent >= MOUTH_MIN_COUNT and (now - last_bonus) >= BONUS_COOLDOWN_SEC:
                condition[1] = min(100, int(condition[1]) + HIGH_WEIGHT_BONUS)
                bonus_cooldown.setdefault(idd, {})['mouth'] = now
                # 在畫面上標示觸發原因
                x1, y1 = int(d.xyxy[0]), int(d.xyxy[1])
                cv2.putText(img, f"[+{HIGH_WEIGHT_BONUS} Repeated]",
                            (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 200, 255), 2, cv2.LINE_AA)

        # ── 條件 3a：煙霧框與人物重疊 → 加分 ────────────────────────────
        if _smoke_available and smoke_boxes and idd is not None:
            if smoke_inference.smoke_overlaps_person(smoke_boxes, d.xyxy):
                last_smoke = smoke_cooldown.get(idd, 0)
                if (now - last_smoke) >= BONUS_COOLDOWN_SEC:
                    condition[1] = min(100, int(condition[1]) + SMOKE_BONUS)
                    smoke_cooldown[idd] = now
                    x1, y1 = int(d.xyxy[0]), int(d.xyxy[1])
                    cv2.putText(img, f"[+{SMOKE_BONUS} Smoke]",
                                (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 128, 255), 2, cv2.LINE_AA)

        # ── 條件 3b：SAHI 3 秒窗口內確認香菸 → 加分 ─────────────────────
        state = _sahi_state.get(idd) if idd is not None else None
        if (state is not None and state['hit']
                and (now - state['window_start']) > SAHI_WINDOW_SEC):
            last_smoke = smoke_cooldown.get(idd, 0)
            if (now - last_smoke) >= BONUS_COOLDOWN_SEC:
                condition[1] = min(100, int(condition[1]) + SMOKE_BONUS)
                smoke_cooldown[idd] = now
                x1, y1 = int(d.xyxy[0]), int(d.xyxy[1])
                cv2.putText(img, f"[+{SMOKE_BONUS} SAHI Cig]",
                            (x1, y1 - 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 100, 0), 2, cv2.LINE_AA)
            # 結算後記錄關閉時間，10 秒內不重新開窗
            _sahi_last_close[idd] = now
            _sahi_state.pop(idd, None)

        # ── 條件 2：在同一位置停留超過 10 秒（一般權重 +8）─────────────
        if idd is not None:
            box = d.xyxy
            cx = (float(box[0]) + float(box[2])) / 2
            cy = (float(box[1]) + float(box[3])) / 2
            bw = float(box[2]) - float(box[0])
            bh = float(box[3]) - float(box[1])

            if idd not in position_log:
                position_log[idd] = {'ref_center': (cx, cy), 'start_time': now}
            else:
                ref_cx, ref_cy = position_log[idd]['ref_center']
                moved_x = abs(cx - ref_cx) > POSITION_MOVE_RATIO * bw
                moved_y = abs(cy - ref_cy) > POSITION_MOVE_RATIO * bh
                if moved_x or moved_y:
                    # 人物移動，重置參考點
                    position_log[idd] = {'ref_center': (cx, cy), 'start_time': now}
                else:
                    # 人物未移動，檢查停留時間
                    duration = now - position_log[idd]['start_time']
                    last_bonus = bonus_cooldown.get(idd, {}).get('position', 0)
                    if duration >= POSITION_WINDOW and (now - last_bonus) >= BONUS_COOLDOWN_SEC:
                        condition[1] = min(100, int(condition[1]) + NORMAL_WEIGHT_BONUS)
                        bonus_cooldown.setdefault(idd, {})['position'] = now
                        x1, y1 = int(d.xyxy[0]), int(d.xyxy[1])
                        cv2.putText(img, f"[+{NORMAL_WEIGHT_BONUS} Staying]",
                                    (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 165, 0), 2, cv2.LINE_AA)

        # ── 超過閾值顯示吸菸警告 ─────────────────────────────────────────
        if condition[1] > smoking_threshold:
            box_label(d.xyxy, img, 3, "Target is Smoking", (0, 0, 255))

        ids[idd] = condition

        if opt.skeleton:
            key_label(d.keypoints, img, img.shape, kpt_line=True)

    cgr_box = np.array([t[:4] for t in cgrlabel])
    if opt.cig_box:
        for i in cgr_box:
            box_label(i, img, 3, label='Cig', color=(0, 0, 255), txt_color=(255, 255, 255))
        # 顯示煙霧偵測框（橘色）
        for s in smoke_boxes:
            box_label(s[:4], img, 2, label=f'Smoke {s[4]:.2f}', color=(0, 128, 255), txt_color=(255, 255, 255))
        # 顯示 SAHI 香菸偵測框（藍色，區別於口部裁切的一般香菸框）
        for s in all_sahi_cig_boxes:
            box_label(s[:4], img, 2, label=f'Cig-S {s[4]:.2f}', color=(255, 100, 0), txt_color=(255, 255, 255))

    return img


def cal_dis(kpt, direction):
    # 计算手嘴距离，以上身长度为参照
    nose, wrist, shoulder, hip = kpt[0], kpt[direction], kpt[5], kpt[11]
    difference = nose - wrist
    standard = shoulder - hip
    # 计算欧氏距离
    distance = np.linalg.norm(difference)
    standdis = np.linalg.norm(standard)
    return distance / standdis


def cal_angle(kpt):
    # 计算关节夹角，以向量夹角方式计算
    lshoulder, lelbow, lwrist = kpt[5], kpt[7], kpt[9]
    rshoulder, relbow, rwrist = kpt[6], kpt[8], kpt[10]
    left_shoulder_vector = lshoulder - lelbow
    left_wrist_vector = lwrist - lelbow
    right_shoulder_vector = rshoulder - relbow
    right_wrist_vector = rwrist - relbow
    # 计算向量的夹角（弧度）
    left_angle_radian = np.arccos(
        np.dot(left_shoulder_vector, left_wrist_vector) / (
                np.linalg.norm(left_shoulder_vector) * np.linalg.norm(left_wrist_vector)))
    right_angle_radian = np.arccos(
        np.dot(right_shoulder_vector, right_wrist_vector) / (
                np.linalg.norm(right_shoulder_vector) * np.linalg.norm(right_wrist_vector)))

    # 将弧度转换为角度
    right_angle_degree = np.degrees(right_angle_radian)
    left_angle_degree = np.degrees(left_angle_radian)

    return left_angle_degree, right_angle_degree


def box_label(box, im, lw, label='', color=(255, 255, 64), txt_color=(255, 255, 255)):
    # 画出检测框，box为xyxy格式，来源于ultralytics官方
    """Add one xyxy box to image with label."""
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)


def cgr_detect(k, img, direction, label):
    count[0] += 1
    box = k.xyxy
    right = k.keypoints[0]
    # 锁定嘴部位置
    length = int(0.4 * (box[2] - box[0]))
    lengths = int(0.3 * (box[3] - box[1]))
    box = box.astype(np.int32)
    box[1] = np.max([int(right[1]) - length, 0])
    box[3] = np.min([int(right[1]) + length, img.shape[0]])
    box[0] = np.max([int(right[0]) - lengths, 0])
    box[2] = np.min([int(right[0]) + lengths, img.shape[1]])
    # 挖出嘴部图片
    person = img[box[1]:box[3], box[0]:box[2]]
    # if count[0] % 5 == 0:
    #     cv2.imwrite(f"video/{k.id}.jpg", person)

    if person.shape[0] != 0 and person.shape[1] != 0:
        # 对挖出图片进行香烟目标检测
        if model[0]==0:
            boxes, scores = trt_inference_detr.cgr_detect_with_onnx(person)
        if model[0]==1:
            boxes, scores = trt_inference_yolo.cgr_detect_with_onnx(person)
        if model[0]==2:
            boxes, scores = cgr_detect_with_onnx(person)
        # boxes, scores = cgr_detect_alternative(person)
        for i, c in enumerate(scores):
            # 若存在，则添加至香烟队列（用于画图）
            if c > cgr_conf[0]:
                label.append([int(boxes[i][0]) + int(box[0]), int(boxes[i][1]) + int(box[1]),
                              int(boxes[i][2]) + int(box[0]), int(boxes[i][3]) + int(box[1]), c])
                return True
            else:
                return False


def key_label(kpts, im, shape=(640, 640), radius=5, kpt_line=True):
    # 骨架绘图函数，来源于ultralytics库
    """
    Args:
        kpts (ndarray): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
        shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
        radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                   for human pose. Default is True.
    """
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim == 3
    kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]]
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.4:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(im, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

