"""
煙霧偵測模型訓練腳本
使用 YOLOv8n 在 Roboflow smoke-gxoy3 資料集上訓練

使用方式：
    # 標準訓練
    python train_smoke.py

    # 指定參數
    python train_smoke.py --data data_smoke.yaml --epochs 100 --batch 16

    # 從現有 checkpoint 繼續訓練
    python train_smoke.py --resume runs/train/smoke_detector/weights/last.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="訓練煙霧偵測 YOLOv8 模型")
    parser.add_argument("--data", default="data_smoke.yaml", help="資料集 YAML 路徑")
    parser.add_argument("--model", default="yolov8n.pt", help="預訓練模型（yolov8n/s/m.pt）")
    parser.add_argument("--epochs", type=int, default=100, help="訓練 epoch 數")
    parser.add_argument("--batch", type=int, default=16, help="Batch size（GPU 記憶體不足時調小）")
    parser.add_argument("--imgsz", type=int, default=640, help="訓練圖片大小")
    parser.add_argument("--device", default="0", help="GPU ID 或 'cpu'")
    parser.add_argument("--project", default="runs/train", help="輸出目錄")
    parser.add_argument("--name", default="smoke_detector", help="實驗名稱")
    parser.add_argument("--resume", default=None, help="從 checkpoint 繼續訓練")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  煙霧偵測模型訓練")
    print(f"  資料集：{args.data}")
    print(f"  模型：{args.model}  epochs={args.epochs}  batch={args.batch}")
    print("=" * 60)

    if args.resume:
        print(f"[INFO] 從 checkpoint 繼續：{args.resume}")
        model = YOLO(args.resume)
    else:
        if not Path(args.data).exists():
            print(f"[ERROR] 找不到資料集設定檔：{args.data}")
            print("        請先執行：python scripts/download_smoke_dataset.py --api-key YOUR_KEY")
            return

        model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        # 增強設定（適合煙霧這類形態多變的目標）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # 儲存設定
        save=True,
        save_period=10,
        exist_ok=True,
    )

    # 驗證最終模型
    print("\n[INFO] 驗證最終模型...")
    metrics = model.val()
    print(f"[OK] 訓練完成")
    print(f"     mAP50   : {metrics.box.map50:.4f}")
    print(f"     mAP50-95: {metrics.box.map:.4f}")
    print(f"     Precision: {metrics.box.mp:.4f}")
    print(f"     Recall   : {metrics.box.mr:.4f}")

    # 匯出為 ONNX 以供推論使用
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n[INFO] 匯出 ONNX 模型...")
        export_model = YOLO(str(best_pt))
        export_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        onnx_path = best_pt.with_suffix(".onnx")
        print(f"[OK] ONNX 匯出完成：{onnx_path}")
        print(f"\n[HINT] 將 {onnx_path} 複製到 models/smoke_detector.onnx 後重啟系統")
    else:
        print(f"[WARN] 找不到 {best_pt}，請手動匯出模型")


if __name__ == "__main__":
    main()
