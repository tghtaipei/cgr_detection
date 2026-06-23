# 吸菸行為偵測系統

YOLOv8-Pose 姿態估計 + ByteTrack 追蹤 + RT-DETR 香菸偵測 + SAHI 小目標增強 + 煙霧偵測  
推論框架支援：OpenVINO / ONNX Runtime / TensorRT

---

## 系統架構

```
輸入影像
  │
  ├─ YOLOv8-Pose      偵測人體位置、提取 17 個關節座標
  │     └─ ByteTrack  跨幀追蹤，維持每人唯一 ID
  │
  ├─ RT-DETR（口部裁切）  每幀對嘴部區域進行香菸偵測
  ├─ SAHI 香菸偵測        每 10 秒最多啟動一次，對上半身做切片推論
  └─ YOLOv8n 煙霧偵測    每幀全畫面偵測煙霧
```

---

## 偵測邏輯與計分系統

每個被追蹤的人都有一個 **吸菸分數（0–100）**，超過閾值（預設 50）即判定為正在吸菸。

### 基礎計分

| 狀態 | 條件 | 分數變化 |
|------|------|---------|
| 香菸確認（status 2） | RT-DETR 在口部偵測到香菸 | +20 / 幀 |
| 可疑姿勢（status 1） | 手靠近嘴但無香菸 | 曾確認吸菸者 +5；未確認者 -1 |
| 無動作（status 0） | 手未抬起 | -1 / 幀 |

### 加分條件

**條件 1 ── 10 秒內手靠近嘴部 2 次以上（+15）**
- 記錄每次靠近事件的時間戳
- 10 秒觀察窗口內累積 ≥ 2 次即觸發
- 同一條件 5 秒內不重複加分

**條件 2 ── 同一位置停留超過 10 秒（+8）**
- 追蹤人體框中心點，位移超過框寬/高的 30% 才算移動
- 停留超過 10 秒觸發
- 同一條件 5 秒內不重複加分

**條件 3a ── 偵測到煙霧與人物區域重疊（+10）**
- 每幀對全畫面執行 YOLOv8n 煙霧偵測
- 煙霧框與人體框重疊（含往上延伸 50% 範圍）即觸發
- 5 秒冷卻

**條件 3b ── SAHI 確認香菸（+10）**
- 手肘角度 < 55° 時啟動採樣窗口
- 每秒採樣 1 次，共 3 秒（最多 3 次 SAHI 推論）
- 窗口結束後有命中即觸發
- 10 秒冷卻，避免短時間重複觸發

---

## SAHI 香菸偵測

**SAHI（Slicing Aided Hyper Inference）** 透過將影像切成重疊的小區塊分別推論，再合併結果，能有效偵測遠距離或畫面中較小的香菸目標。

### 工作流程

```
手肘角度 < 55°（吸菸姿勢）
  │
  ├─ 10 秒冷卻中 → 跳過
  │
  └─ 開啟 3 秒採樣窗口
       ├─ t=0s → cgr_detect_sahi()  第 1 次
       ├─ t=1s → cgr_detect_sahi()  第 2 次
       ├─ t=2s → cgr_detect_sahi()  第 3 次
       └─ t=3s 結算
            ├─ 有命中 → 條件 3b +10，進入 10 秒冷卻
            └─ 無命中 → 清除窗口，進入 10 秒冷卻
```

### `cgr_detect_sahi()` 內部流程

```
傳入：全幀影像 + 人體框
  │
  ├─ 裁出上半身 ROI（人體框上 60%）
  ├─ 切成重疊的 320×320 切片（重疊率 20%）
  ├─ 每個切片 → RT-DETR 香菸偵測
  ├─ 座標換算回全幀座標
  └─ 全域 NMS 去除重複框
```

### 與原有口部裁切的差異

| | 口部裁切 `cgr_detect()` | SAHI `cgr_detect_sahi()` |
|--|------------------------|--------------------------|
| 執行頻率 | 每幀 | 最多 3 次 / 10 秒 |
| 偵測範圍 | 口部小區域 | 上半身整體 ROI |
| 適合場景 | 近距離、香菸在嘴邊 | 遠距離、香菸在手持位置 |
| 畫面標示 | 紅框 `Cig` | 藍框 `Cig-S` |

---

## 煙霧偵測

使用 **YOLOv8n** 訓練的煙霧偵測模型，透過 ONNX Runtime 推論。

- 模型路徑：`models/smoke_detector.onnx`
- 置信度閾值：0.25（低閾值提升召回率，煙霧模型資料量較少）
- 每幀對全畫面執行，偵測到的煙霧框顯示為橘色

### 訓練方式

使用 `train_smoke_colab.ipynb` 在 Google Colab（T4 GPU）訓練：
- 資料集：[Roboflow smoke-gxoy3](https://universe.roboflow.com/naruesuan-university/smoke-gxoy3)（438 張）
- 模型：YOLOv8n（無增強，直接訓練效果最佳）
- 基準效能：mAP50 = 0.6665，Precision = 0.674，Recall = 0.618

---

## 模型說明

### 香菸偵測模型（RT-DETR）
相較於 YOLO 系列，RT-DETR 以較短訓練時長（75–80 epoch）和較少資料增強策略，在同等測試條件下（640×640）展現更強效能。

訓練結果：Precision 0.951、Recall 0.929、mAP50 0.961、mAP50-95 0.620

### 姿態估計模型（YOLOv8-Pose）
提取人體 17 個關節點，計算手肘夾角與手嘴距離，作為吸菸姿勢的判斷依據。

---

## 檔案說明

```
cgr_detection/
│
├─ infer_main.py          主程式，啟動推論（桌面端）
├─ infer_colab.py         Colab 推論腳本（無 GUI，讀取影片檔）
├─ smoking_detection.ipynb Colab 完整推論流程筆記本
├─ func.py                核心邏輯：吸菸判斷、計分、SAHI 採樣窗口
├─ ov_inference.py        OpenVINO 推論模組（含 cgr_detect_sahi）
├─ ort_inference.py       ONNX Runtime 推論模組
├─ smoke_inference.py     煙霧偵測推論模組（ONNX Runtime）
├─ qt_main.py             GUI 主視窗
├─ bytetrack_init.py      ByteTrack 初始化與參數設定
├─ train_smoke_colab.ipynb 煙霧偵測模型訓練（Google Colab）
├─ train_smoke.py         煙霧偵測模型本機訓練腳本
├─ trt_infer.ipynb        從 ONNX 生成本機 TensorRT 模型
├─ requirements.txt       環境需求
│
└─ models/
     ├─ last.onnx              RT-DETR 香菸偵測 ONNX 模型
     ├─ yolov8n-pose.onnx      YOLOv8-Pose 姿態估計 ONNX 模型
     └─ smoke_detector.onnx    YOLOv8n 煙霧偵測 ONNX 模型
```

---

## 使用方式

### 桌面端（GUI）

```bash
python qt_main.py
```

啟動後：選擇偵測模型 → 點擊初始化模型 → 開始推論

GUI 可調整：香菸框顯示、骨架顯示、香菸置信度、偵測閾值、影片儲存

### Colab 推論

使用 `smoking_detection.ipynb`，依序執行：

1. 確認 GPU（T4）
2. Clone 專案（含模型）
3. 安裝套件
4. 上傳影片
5. 執行偵測
6. 預覽結果
7. 下載結果影片

### 本機推論（無 GUI）

```bash
python infer_main.py
```

---

## 更換推論框架

修改 `func.py` 與 `infer_main.py` 中的 import：

```python
# OpenVINO（預設）
from ov_inference import pose_estimate_with_onnx, cgr_detect_with_onnx

# ONNX Runtime
from ort_inference import pose_estimate_with_onnx, cgr_detect_with_onnx
```

---

## 注意事項

- TensorRT 模型需在目標機器上重新生成（使用 `trt_infer.ipynb`）
- 煙霧偵測模型（`smoke_detector.onnx`）已包含在 `models/` 資料夾，Clone 後即可使用
- ByteTrack 需替換 ultralytics 函式庫內的追蹤器：以專案內 `byte_tracker.py` 替換 `ultralytics/trackers/byte_tracker.py`
