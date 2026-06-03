"""
infer_colab.py — Colab-compatible cigarette detection inference script.

Usage:
    python infer_colab.py --source /content/test.mp4 --output /content/output.mp4 \
        --cgr_conf 0.4 --cig_box

No OpenCV GUI required; reads from a video file and writes results to an output video.
"""

import argparse
import os
import sys
import time

import cv2

from ov_inference import pose_estimate_with_onnx
from func import detect_and_draw


class Result:
    def __init__(self, box, score, id, kpts):
        self.xyxy = box
        self.conf = score
        self.id = id
        self.keypoints = kpts


def cgr_detect(image, opt):
    """Run pose estimation + cigarette detection on a single frame.

    Returns (annotated_image, fps, elapsed_seconds).
    """
    start_time = time.time()

    box, score, ids, kpts = pose_estimate_with_onnx(image)
    if ids is not None and box is not None and len(ids) > 0:
        pose_result = [Result(i, j, k, m) for i, j, k, m in zip(box, score, ids, kpts)]
        image = detect_and_draw(pose_result, image, opt)

    elapsed_time = time.time() - start_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
    cv2.putText(image, f"FPS: {round(fps, 2)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image, fps, elapsed_time


def build_video_writer(output_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Create a VideoWriter using XVID codec inside AVI container (most universal)."""
    # Force .avi extension regardless of what was passed
    avi_path = os.path.splitext(output_path)[0] + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        # XVID unavailable; fall back to MJPG which ships with almost every OpenCV build
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                "VideoWriter could not be opened with XVID or MJPG codec.\n"
                "Try: pip install opencv-python-headless --upgrade"
            )
        print("[WARNING] XVID not available; using MJPG codec (larger file size)")
    if avi_path != output_path:
        print(f"[INFO] Output format fixed to AVI: {avi_path}")
    return writer


def main():
    parser = argparse.ArgumentParser(
        description="Cigarette detection inference — Colab-compatible, no GUI"
    )
    # I/O
    parser.add_argument('--source', type=str, required=True,
                        help='Path to input video file, e.g. /content/test.mp4')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Path to output video file (default: output.mp4)')
    # Detection parameters (kept identical to infer_main.py)
    parser.add_argument('--cgr_conf', type=float, default=0.4,
                        help='Cigarette detection confidence threshold (default: 0.4)')
    parser.add_argument('--skeleton', action='store_true',
                        help='Draw skeleton keypoints on detected persons')
    parser.add_argument('--cig_box', action='store_true',
                        help='Draw bounding boxes around detected cigarettes')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Consecutive-frame smoking threshold (default: 50, not recommended to change)')
    opt = parser.parse_args()

    # ── Validate source ─────────────────────────────────────────────────────
    if not os.path.isfile(opt.source):
        raise FileNotFoundError(
            f"Input video not found: {opt.source}\n"
            "Make sure you uploaded the file and provided the correct path."
        )

    cap = cv2.VideoCapture(opt.source)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open the video: {opt.source}\n"
            "Possible causes: corrupted file, unsupported codec, wrong path."
        )

    # ── Read video metadata ──────────────────────────────────────────────────
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Source  : {opt.source}")
    print(f"[INFO] Output  : {opt.output}")
    print(f"[INFO] Size    : {frame_width}x{frame_height}")
    print(f"[INFO] FPS     : {source_fps:.2f}")
    print(f"[INFO] Frames  : {total_frames if total_frames > 0 else 'unknown'}")
    print(f"[INFO] cgr_conf: {opt.cgr_conf}  skeleton: {opt.skeleton}  "
          f"cig_box: {opt.cig_box}  threshold: {opt.threshold}")
    print("-" * 60)

    # ── Set up output writer ─────────────────────────────────────────────────
    out_dir = os.path.dirname(opt.output)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    writer = build_video_writer(opt.output, frame_width, frame_height, source_fps)

    # ── Inference loop ───────────────────────────────────────────────────────
    frame_count = 0
    fps_history = []
    loop_start = time.time()

    try:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated, fps_frame, elapsed = cgr_detect(image, opt)
            fps_history.append(fps_frame)
            writer.write(annotated)

            # Progress report every 30 frames
            if frame_count % 30 == 0:
                avg_fps = sum(fps_history) / len(fps_history)
                wall_elapsed = time.time() - loop_start
                progress = (f"{frame_count}/{total_frames}" if total_frames > 0
                            else str(frame_count))
                print(
                    f"[PROGRESS] frame {progress} | "
                    f"instant FPS: {fps_frame:.2f} | "
                    f"avg FPS: {avg_fps:.2f} | "
                    f"frame time: {elapsed*1000:.1f} ms | "
                    f"wall time: {wall_elapsed:.1f}s"
                )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        writer.release()

    # ── Summary ──────────────────────────────────────────────────────────────
    total_wall = time.time() - loop_start
    avg_fps_final = (sum(fps_history) / len(fps_history)) if fps_history else 0.0
    actual_output = opt.output if os.path.isfile(opt.output) else opt.output.rsplit('.', 1)[0] + '.avi'

    print("-" * 60)
    print(f"[DONE] Processed {frame_count} frames in {total_wall:.2f}s "
          f"(avg {avg_fps_final:.2f} FPS)")
    print(f"[DONE] Output saved to: {os.path.abspath(actual_output)}")


if __name__ == '__main__':
    main()
