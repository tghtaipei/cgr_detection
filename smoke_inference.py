"""
煙霧偵測推論模組
使用 ONNX Runtime 載入訓練好的 YOLOv8 煙霧偵測模型

模型路徑：models/smoke_detector.onnx
  → 請先執行 scripts/download_smoke_dataset.py 下載資料集
  → 再執行 train_smoke.py 訓練，訓練完後將 best.onnx 複製到上述路徑
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple

# 延遲載入 ONNX Runtime，若模型不存在不影響主系統
_session = None
_MODEL_PATH = Path(__file__).parent / "models" / "smoke_detector.onnx"
_INPUT_SIZE = 640
_CONF_THRESHOLD = 0.15  # 低閾值提升 Recall，容許較多 FP 以避免漏偵測
_IOU_THRESHOLD = 0.45


def _load_session():
    global _session
    if _session is not None:
        return _session
    if not _MODEL_PATH.exists():
        return None
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _session = ort.InferenceSession(str(_MODEL_PATH), providers=providers)
        print(f"[smoke_inference] 載入煙霧偵測模型：{_MODEL_PATH}")
    except Exception as e:
        print(f"[smoke_inference] 無法載入模型：{e}")
        _session = None
    return _session


def is_model_available() -> bool:
    """檢查煙霧偵測模型是否已就緒"""
    return _MODEL_PATH.exists()


def _sahi_slices(img_w: int, img_h: int,
                 slice_size: int = _INPUT_SIZE, overlap: float = 0.2) -> list[tuple]:
    """產生覆蓋整張影像的重疊切片座標清單。"""
    step = int(slice_size * (1 - overlap))
    slices = []
    y = 0
    while True:
        y2 = min(y + slice_size, img_h)
        y1 = max(0, y2 - slice_size)
        x = 0
        while True:
            x2 = min(x + slice_size, img_w)
            x1 = max(0, x2 - slice_size)
            slices.append((x1, y1, x2, y2))
            if x2 == img_w:
                break
            x += step
        if y2 == img_h:
            break
        y += step
    return slices


def detect_smoke_sahi(img: np.ndarray, roi_xyxy=None,
                      conf_threshold: float = _CONF_THRESHOLD) -> list[list[float]]:
    """
    SAHI 切片煙霧偵測，在 roi_xyxy（人體框）周圍執行，
    比全幀偵測更能抓到小煙霧，且只在有吸菸姿勢時才呼叫以節省運算。

    Args:
        img:       原始 BGR 全幀影像
        roi_xyxy:  人體框 [x1, y1, x2, y2]；None 則對全圖執行 SAHI
        conf_threshold: 置信度閾值

    Returns:
        list of [x1, y1, x2, y2, score]（座標為原始影像像素）
    """
    sess = _load_session()
    if sess is None:
        return []

    orig_h, orig_w = img.shape[:2]

    # 以人體框為中心擴展 ROI（左右各 50%，往上 80% 捕捉頭頂煙霧）
    if roi_xyxy is not None:
        px1, py1, px2, py2 = [float(v) for v in roi_xyxy]
        mx = (px2 - px1) * 0.5
        my = (py2 - py1) * 0.8
        rx1 = max(0, int(px1 - mx))
        ry1 = max(0, int(py1 - my))
        rx2 = min(orig_w, int(px2 + mx))
        ry2 = min(orig_h, int(py2))
        region = img[ry1:ry2, rx1:rx2]
        off_x, off_y = rx1, ry1
    else:
        region = img
        off_x, off_y = 0, 0

    rh, rw = region.shape[:2]
    if rh == 0 or rw == 0:
        return []

    input_name = sess.get_inputs()[0].name
    all_boxes: list[list[float]] = []

    for (sx1, sy1, sx2, sy2) in _sahi_slices(rw, rh):
        patch = region[sy1:sy2, sx1:sx2]
        ph, pw = patch.shape[:2]
        if ph == 0 or pw == 0:
            continue
        tensor, pad_info = _preprocess(patch)
        output = sess.run(None, {input_name: tensor})[0]
        boxes = _postprocess(output, pw, ph, pad_info, conf_threshold)
        for b in boxes:
            all_boxes.append([b[0] + sx1 + off_x, b[1] + sy1 + off_y,
                               b[2] + sx1 + off_x, b[3] + sy1 + off_y, b[4]])

    if not all_boxes:
        return []

    xs1 = np.array([b[0] for b in all_boxes])
    ys1 = np.array([b[1] for b in all_boxes])
    xs2 = np.array([b[2] for b in all_boxes])
    ys2 = np.array([b[3] for b in all_boxes])
    sc  = np.array([b[4] for b in all_boxes])
    keep = _nms(xs1, ys1, xs2, ys2, sc, _IOU_THRESHOLD)
    return [[float(xs1[i]), float(ys1[i]), float(xs2[i]), float(ys2[i]), float(sc[i])] for i in keep]


def detect_smoke(img: np.ndarray, conf_threshold: float = _CONF_THRESHOLD) -> list[list[float]]:
    """
    對整張影像進行煙霧偵測

    Args:
        img: BGR 格式的 numpy 影像
        conf_threshold: 置信度閾值

    Returns:
        list of [x1, y1, x2, y2, score]  (座標為原始影像像素)
        若模型未載入則回傳空列表
    """
    sess = _load_session()
    if sess is None:
        return []

    orig_h, orig_w = img.shape[:2]

    # 前處理
    input_tensor, pad_info = _preprocess(img)

    # 推論
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: input_tensor})[0]

    # 後處理
    detections = _postprocess(output, orig_w, orig_h, pad_info, conf_threshold)
    return detections


def _preprocess(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Letterbox resize + normalize"""
    orig_h, orig_w = img.shape[:2]
    scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (_INPUT_SIZE - new_w) // 2
    pad_h = (_INPUT_SIZE - new_h) // 2
    canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, /255
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # HWC→NCHW

    pad_info = {"scale": scale, "pad_w": pad_w, "pad_h": pad_h}
    return tensor, pad_info


def _postprocess(output: np.ndarray, orig_w: int, orig_h: int, pad_info: dict, conf_thr: float) -> list:
    """YOLOv8 輸出解碼 [1, 5, N] 或 [1, N, 5]"""
    pred = np.squeeze(output)
    if pred.ndim == 1:
        return []

    # YOLOv8 輸出格式為 [5, N]，轉置為 [N, 5]
    if pred.shape[0] == 5:
        pred = pred.T

    boxes_cx = pred[:, 0]
    boxes_cy = pred[:, 1]
    boxes_w  = pred[:, 2]
    boxes_h  = pred[:, 3]
    scores   = pred[:, 4]

    mask = scores > conf_thr
    if not mask.any():
        return []

    boxes_cx, boxes_cy = boxes_cx[mask], boxes_cy[mask]
    boxes_w,  boxes_h  = boxes_w[mask],  boxes_h[mask]
    scores = scores[mask]

    scale = pad_info["scale"]
    pad_w = pad_info["pad_w"]
    pad_h = pad_info["pad_h"]

    x1 = np.clip((boxes_cx - boxes_w / 2 - pad_w) / scale, 0, orig_w)
    y1 = np.clip((boxes_cy - boxes_h / 2 - pad_h) / scale, 0, orig_h)
    x2 = np.clip((boxes_cx + boxes_w / 2 - pad_w) / scale, 0, orig_w)
    y2 = np.clip((boxes_cy + boxes_h / 2 - pad_h) / scale, 0, orig_h)

    # NMS
    keep = _nms(x1, y1, x2, y2, scores, _IOU_THRESHOLD)
    return [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(scores[i])] for i in keep]


def _nms(x1, y1, x2, y2, scores, iou_thr: float) -> list[int]:
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def smoke_overlaps_person(smoke_boxes: list, person_xyxy, iou_threshold: float = 0.01) -> bool:
    """
    判斷是否有煙霧框與人體框重疊

    Args:
        smoke_boxes: detect_smoke 的回傳結果
        person_xyxy: 人體框 [x1, y1, x2, y2]
        iou_threshold: 重疊比例閾值（相對於煙霧框面積）

    Returns:
        True 表示有煙霧與該人物重疊
    """
    if not smoke_boxes:
        return False

    px1, py1, px2, py2 = float(person_xyxy[0]), float(person_xyxy[1]), float(person_xyxy[2]), float(person_xyxy[3])
    # 擴大人體框偵測範圍（往上延伸 50%，捕捉頭頂上方的煙霧）
    extend_h = (py2 - py1) * 0.5
    py1_ext = max(0, py1 - extend_h)

    for s in smoke_boxes:
        sx1, sy1, sx2, sy2 = s[0], s[1], s[2], s[3]
        inter_x1 = max(sx1, px1)
        inter_y1 = max(sy1, py1_ext)
        inter_x2 = min(sx2, px2)
        inter_y2 = min(sy2, py2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        smoke_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
        if inter_area / smoke_area >= iou_threshold:
            return True
    return False
