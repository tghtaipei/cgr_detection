"""
SAHI (Slicing Aided Hyper Inference) utilities for small-object cigarette detection.

Strategy:
  - Slice the mouth ROI into overlapping patches (default 320×320, 20% overlap).
  - Run the RT-DETR inference on each patch independently.
  - Restore each patch's box coordinates back to the ROI coordinate system.
  - Merge all detections with NMS to remove duplicates.
  - Fall back to single-inference when the ROI is smaller than one slice.
"""

import numpy as np
from yolov8onnx.utils import nms


def slice_image(image, slice_h: int, slice_w: int, overlap_ratio: float):
    """
    Divide *image* into overlapping patches.

    Returns
    -------
    list of (patch_img, x_offset, y_offset)
        x_offset / y_offset: top-left corner of each patch in the original image.
    """
    img_h, img_w = image.shape[:2]
    step_y = max(1, int(slice_h * (1 - overlap_ratio)))
    step_x = max(1, int(slice_w * (1 - overlap_ratio)))

    slices = []
    y = 0
    while y < img_h:
        x = 0
        y_end = min(y + slice_h, img_h)
        while x < img_w:
            x_end = min(x + slice_w, img_w)
            patch = image[y:y_end, x:x_end]
            slices.append((patch, x, y))
            if x_end == img_w:
                break
            x += step_x
        if y_end == img_h:
            break
        y += step_y

    return slices


def merge_detections(patch_results, iou_threshold: float = 0.5):
    """
    Merge (boxes, scores) tuples from multiple patches into a single list.

    Each element of *patch_results* is ((boxes_array, scores_array), x_off, y_off).
    Boxes are in xyxy format relative to their patch origin; they are shifted to
    the ROI coordinate system before merging.

    Returns
    -------
    (merged_boxes, merged_scores) as numpy arrays, or ([], []) when empty.
    """
    all_boxes = []
    all_scores = []

    for (boxes, scores), x_off, y_off in patch_results:
        if len(boxes) == 0:
            continue
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        # Shift coordinates from patch space → ROI space
        boxes[:, 0] += x_off
        boxes[:, 2] += x_off
        boxes[:, 1] += y_off
        boxes[:, 3] += y_off
        all_boxes.append(boxes)
        all_scores.append(scores)

    if not all_boxes:
        return [], []

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    keep = nms(all_boxes, all_scores, iou_threshold)
    return all_boxes[keep], all_scores[keep]


def sahi_infer(image, infer_fn, slice_size: int = 320, overlap: float = 0.2,
               iou_threshold: float = 0.5):
    """
    Run SAHI inference on *image* using *infer_fn* as the single-patch detector.

    Parameters
    ----------
    image      : np.ndarray  BGR image (the cropped mouth ROI)
    infer_fn   : callable    Signature: infer_fn(patch) -> (boxes, scores)
                             where boxes is xyxy in patch coordinates.
    slice_size : int         Height and width of each slice (pixels).
    overlap    : float       Fractional overlap between adjacent slices [0, 1).
    iou_threshold : float    IoU threshold for the final NMS merge step.

    Returns
    -------
    (boxes, scores) in ROI coordinates, as numpy arrays.
    """
    img_h, img_w = image.shape[:2]

    # If the ROI fits inside a single slice, skip slicing entirely.
    if img_h <= slice_size and img_w <= slice_size:
        return infer_fn(image)

    patches = slice_image(image, slice_size, slice_size, overlap)

    patch_results = []
    for patch, x_off, y_off in patches:
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            continue
        boxes, scores = infer_fn(patch)
        patch_results.append(((boxes, scores), x_off, y_off))

    return merge_detections(patch_results, iou_threshold)
