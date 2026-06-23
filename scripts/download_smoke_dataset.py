"""
Download the smoke detection dataset from Roboflow Universe:
  https://universe.roboflow.com/naruesuan-university/smoke-gxoy3

Usage:
    # 方法一：使用 Roboflow API Key（推薦）
    python scripts/download_smoke_dataset.py --api-key YOUR_API_KEY

    # 方法二：不需要 API key（使用公開下載連結）
    python scripts/download_smoke_dataset.py --no-api

Roboflow API Key 取得方式：
    1. 前往 https://app.roboflow.com/
    2. 登入後點右上角頭像 → Settings → Roboflow API
    3. 複製 Private API Key
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "smoke_roboflow"
ROBOFLOW_WORKSPACE = "naruesuan-university"
ROBOFLOW_PROJECT = "smoke-gxoy3"
ROBOFLOW_VERSION = 3  # 使用第 3 版本（最新穩定）
EXPORT_FORMAT = "yolov8"


def download_with_roboflow_api(api_key: str):
    """使用 Roboflow Python SDK 下載資料集"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] 請先安裝 roboflow: pip install roboflow")
        sys.exit(1)

    print(f"[INFO] 連接 Roboflow API ...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 下載資料集到 {DATASET_DIR} ...")
    dataset = version.download(EXPORT_FORMAT, location=str(DATASET_DIR))
    print(f"[OK] 資料集下載完成：{dataset.location}")
    return dataset.location


def download_without_api():
    """使用公開下載連結（無需 API key）"""
    # Roboflow 公開資料集下載 URL 格式
    url = (
        f"https://universe.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}"
        f"/dataset/{ROBOFLOW_VERSION}/download/{EXPORT_FORMAT}"
    )
    zip_path = DATASET_DIR.parent / "smoke_roboflow.zip"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 從 Roboflow Universe 下載資料集...")
    print(f"[INFO] URL: {url}")
    print("[WARN] 如果下載失敗，請改用 --api-key 選項")

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print(f"\n[INFO] 解壓縮到 {DATASET_DIR} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATASET_DIR)
        zip_path.unlink()
        print(f"[OK] 完成，資料集位於 {DATASET_DIR}")
        return str(DATASET_DIR)
    except Exception as e:
        print(f"\n[ERROR] 公開下載失敗：{e}")
        print("[HINT] 請使用 --api-key 選項並提供 Roboflow API Key")
        sys.exit(1)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%  {downloaded//1024}KB/{total_size//1024}KB", end="", flush=True)


def verify_dataset(dataset_dir: str) -> bool:
    """驗證下載的資料集結構是否符合 YOLOv8 格式"""
    d = Path(dataset_dir)
    required = ["images/train", "images/val", "labels/train", "labels/val"]
    missing = [r for r in required if not (d / r).exists()]

    # 有些 Roboflow 資料集目錄結構稍有不同，也接受 train/images 形式
    if missing:
        alt = ["train/images", "valid/images", "train/labels", "valid/labels"]
        missing_alt = [r for r in alt if not (d / r).exists()]
        if not missing_alt:
            print("[INFO] 偵測到替代目錄結構 (train/valid)，將進行調整...")
            _normalize_structure(d)
            return True
        print(f"[WARN] 資料集目錄缺少：{missing}")
        return False

    train_imgs = list((d / "images/train").glob("*.jpg")) + list((d / "images/train").glob("*.png"))
    val_imgs = list((d / "images/val").glob("*.jpg")) + list((d / "images/val").glob("*.png"))
    print(f"[OK] 資料集驗證通過：train={len(train_imgs)} 張，val={len(val_imgs)} 張")
    return True


def _normalize_structure(dataset_dir: Path):
    """將 train/valid 目錄結構轉換為 images/train & labels/train 格式"""
    import shutil

    mappings = [
        ("train/images", "images/train"),
        ("valid/images", "images/val"),
        ("train/labels", "labels/train"),
        ("valid/labels", "labels/val"),
    ]
    for src_rel, dst_rel in mappings:
        src = dataset_dir / src_rel
        dst = dataset_dir / dst_rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
    print("[OK] 目錄結構調整完成")


def generate_data_yaml(dataset_dir: str):
    """生成 YOLOv8 訓練用的 data_smoke.yaml"""
    yaml_path = Path(__file__).parent.parent / "data_smoke.yaml"
    content = f"""# Smoke detection dataset (Roboflow: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION})
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

nc: 1
names:
  - smoke
"""
    yaml_path.write_text(content)
    print(f"[OK] 訓練設定檔已生成：{yaml_path}")
    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="下載 Roboflow 煙霧偵測資料集")
    parser.add_argument("--api-key", type=str, help="Roboflow API Key")
    parser.add_argument("--no-api", action="store_true", help="不使用 API key（公開下載）")
    parser.add_argument("--version", type=int, default=ROBOFLOW_VERSION, help=f"資料集版本 (預設: {ROBOFLOW_VERSION})")
    args = parser.parse_args()

    global ROBOFLOW_VERSION
    ROBOFLOW_VERSION = args.version

    print("=" * 60)
    print("  煙霧偵測資料集下載器")
    print(f"  資料集：{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}")
    print("=" * 60)

    if args.api_key:
        dataset_loc = download_with_roboflow_api(args.api_key)
    elif args.no_api:
        dataset_loc = download_without_api()
    else:
        print("[ERROR] 請提供 --api-key YOUR_KEY 或使用 --no-api")
        print("        取得 API Key：https://app.roboflow.com/ → Settings → API")
        sys.exit(1)

    verify_dataset(dataset_loc)
    yaml_path = generate_data_yaml(dataset_loc)

    print("\n" + "=" * 60)
    print("  下一步：執行訓練")
    print(f"  python train_smoke.py --data {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
