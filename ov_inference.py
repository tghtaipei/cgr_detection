import time
import cv2
import numpy
import numpy as np
from typing import Tuple
try:
    import openvino as ov
except ImportError:
    import openvino.runtime as ov

from bytetrack_init import bytetrack,make_parser
from yolov8onnx.utils import xywh2xyxy, nms
from trackers.byte_tracker import BYTETracker

core1 = ov.Core()
core2 = ov.Core()
cgr_model = core1.compile_model("models/last.onnx", "AUTO")
pose_model = core2.compile_model("models/yolov8n-pose.onnx", "AUTO")
output_node = pose_model.outputs[0]
infer_cgr = cgr_model.create_infer_request()
infer_pose = pose_model.create_infer_request()
args, _ = make_parser().parse_known_args()
tracker = BYTETracker(args, frame_rate=30)
tracker_cgr=BYTETracker(args, frame_rate=60)


def prepare_input(image):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize input image
    input_img,border = proportional_resize_with_padding(input_img, (640,640))

    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor,border


def process_output(output,conf_threshold,iou_threshold,img,inputimg,border):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes =extract_boxes(predictions,img,inputimg,border)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, iou_threshold)
    # indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)

    return boxes[indices], scores[indices], class_ids[indices]


def extract_boxes(predictions,ori,inputimg,border):
    # Extract boxes from predictions
    boxes = predictions[:, :4]
    # Scale boxes to original image dimensions
    boxes = rscale_box_with_padding((border[4],border[5]),boxes,(ori.shape[1],ori.shape[0]),border)
    # boxes = resize_box_with_padding(boxes,ori,inputimg)
    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    return boxes


def proportional_resize_with_padding(img,new_shape: Tuple[int, int])-> np.array:
    try:
        # 获取原始图片的宽高
        original_height, original_width, _ = img.shape

        # 计算宽高比例
        width_ratio = new_shape[1] / original_width
        height_ratio = new_shape[0] / original_height

        # 选择缩放比例较小的那个，以保持原始图像的纵横比
        resize_ratio = min(width_ratio, height_ratio)

        # 计算缩放后的尺寸
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)

        # 进行缩放
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 计算边框大小
        top = (new_shape[0] - new_height) // 2
        bottom = new_shape[0] - new_height - top
        left = (new_shape[1] - new_width) // 2
        right = new_shape[1] - new_width - left

        # 添加边框
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return padded_img,(top,bottom,left,right,new_width,new_height)
    except Exception as e:
        print(f"Error: {e}")
        return None


def xywh2xyxy_rescale(x,scale,is_scale):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    if is_scale:
        y[..., 0] = x[..., 0]*scale
        y[..., 1] = x[..., 1]*scale
        y[..., 2] = (x[..., 0] + x[..., 2])*scale
        y[..., 3] = (x[..., 1] + x[..., 3])*scale
    else:
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = (x[..., 0] + x[..., 2])
        y[..., 3] = (x[..., 1] + x[..., 3])
    return y

def rscale_box_with_padding(original_size,boxes, target_size,border):
    """
    Resize a detection box from original size to target size with padding.

    Parameters:
        original_size (tuple): A tuple (width, height) representing the original image size.
        original_box_xyxy (tuple): A tuple (x1, y1, x2, y2) representing the bounding box coordinates
                                   in the original image before padding.
        target_size (tuple): A tuple (width, height) representing the target image size after padding.

    Returns:
        tuple: A tuple (x1, y1, x2, y2) representing the bounding box coordinates in the target image.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size


    # Calculate scaling factors in x and y directions
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # Adjust the box coordinates to account for padding
    boxes-= np.array([border[2],border[0],0,0])
    boxes *= np.array([scale_x,scale_y,scale_x,scale_y])
    return boxes


def cgr_detect_with_onnx(image):
    img ,border= prepare_input(image)
    start_time = time.time()
    # Create tensor from external memory
    input_tensor = ov.Tensor(array=img)
    # Set input tensor for models with one input
    infer_cgr.set_input_tensor(input_tensor)
    infer_cgr.infer()
    # Get output tensor for models with one output
    output = infer_cgr.get_output_tensor()
    output_buffer = output.data
    boxes, scores, class_ids = process_output(output_buffer, 0.25, 0.7, image, img,border)
    boxes=np.array(boxes)
    scores=np.array(scores)
    class_ids=np.array(class_ids)
    # boxes, result = bytetrack(boxes, scores,class_ids, tracker_cgr)
    if isinstance(boxes, numpy.ndarray) and boxes.shape[0]!=0:
        # boxes = xywh2xyxy_rescale(boxes, 0, False)
        return boxes,scores
    else:
        return [],[]


def _nms_boxes(boxes_list: list, iou_thr: float = 0.45) -> list:
    if not boxes_list:
        return []
    xs1 = np.array([b[0] for b in boxes_list])
    ys1 = np.array([b[1] for b in boxes_list])
    xs2 = np.array([b[2] for b in boxes_list])
    ys2 = np.array([b[3] for b in boxes_list])
    sc  = np.array([b[4] for b in boxes_list])
    areas = (xs2 - xs1) * (ys2 - ys1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(xs1[i], xs1[order[1:]])
        yy1 = np.maximum(ys1[i], ys1[order[1:]])
        xx2 = np.minimum(xs2[i], xs2[order[1:]])
        yy2 = np.minimum(ys2[i], ys2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return [[float(xs1[i]), float(ys1[i]), float(xs2[i]), float(ys2[i]), float(sc[i])] for i in keep]


def cgr_detect_sahi(img_full: np.ndarray, person_xyxy,
                    conf_threshold: float = 0.25,
                    slice_size: int = 320, overlap: float = 0.2) -> list:
    """
    SAHI 香菸偵測：對人體上半身 ROI 做重疊切片推論，合併後回傳全圖座標。

    Args:
        img_full:        原始 BGR 全幀影像
        person_xyxy:     人體框 [x1, y1, x2, y2]
        conf_threshold:  香菸置信度閾值
        slice_size:      每個切片的邊長（像素）
        overlap:         切片重疊比例

    Returns:
        list of [x1, y1, x2, y2, score]（座標為全幀像素）
    """
    orig_h, orig_w = img_full.shape[:2]
    px1, py1, px2, py2 = [float(v) for v in person_xyxy]
    body_h = py2 - py1

    # 僅取上半身 60%（含頭部、手臂、嘴部），避免對下半身做無謂運算
    roi_x1 = max(0, int(px1))
    roi_y1 = max(0, int(py1))
    roi_x2 = min(orig_w, int(px2))
    roi_y2 = min(orig_h, int(py1 + body_h * 0.6))

    region = img_full[roi_y1:roi_y2, roi_x1:roi_x2]
    rh, rw = region.shape[:2]
    if rh == 0 or rw == 0:
        return []

    step = max(1, int(slice_size * (1 - overlap)))
    all_boxes: list = []

    y = 0
    while True:
        y2s = min(y + slice_size, rh)
        y1s = max(0, y2s - slice_size)
        x = 0
        while True:
            x2s = min(x + slice_size, rw)
            x1s = max(0, x2s - slice_size)
            patch = region[y1s:y2s, x1s:x2s]
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                boxes, scores = cgr_detect_with_onnx(patch)
                for i, s in enumerate(scores):
                    if float(s) > conf_threshold:
                        b = boxes[i]
                        all_boxes.append([
                            float(b[0]) + x1s + roi_x1,
                            float(b[1]) + y1s + roi_y1,
                            float(b[2]) + x1s + roi_x1,
                            float(b[3]) + y1s + roi_y1,
                            float(s)
                        ])
            if x2s == rw:
                break
            x += step
        if y2s == rh:
            break
        y += step

    return _nms_boxes(all_boxes)

def pose_estimate_with_onnx(frame):
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    # 基于OpenVINO实现推理计算
    outputs = infer_pose.infer(blob)[output_node]
    outputs = np.array([cv2.transpose(outputs[0])])

    classes_scores = outputs[:, :, 4]
    key_points = outputs[:, :, 5:]

    # Create a mask to filter rows based on classes_scores >= 0.5
    mask = classes_scores >= 0.5

    # Use the mask to get the filtered_outputs array
    filtered_outputs = outputs[mask]

    # Calculate boxes using vectorized operations
    boxes = filtered_outputs[:, 0:4] - np.column_stack(
        [(0.5 * filtered_outputs[:, 2]), (0.5 * filtered_outputs[:, 3]), np.zeros_like(filtered_outputs[:, 2]),
         np.zeros_like(filtered_outputs[:, 3])])

    # Extract scores and key points
    scores = filtered_outputs[:, 4]
    preds_kpts = key_points[mask]

    # Optionally, convert the boxes list to a NumPy array

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.5, 0.5)

    box = np.array(boxes)[result_boxes].reshape(-1, 4)
    box = xywh2xyxy_rescale(box, scale,True)
    scores = np.array(scores)[result_boxes]
    cls=numpy.zeros(box.shape[0])
    box,result = bytetrack(box, scores,cls,tracker)
    if isinstance(box,numpy.ndarray) and box.shape[0]!=0:
        box= xywh2xyxy_rescale(box,scale,False)
    kpts = np.array(preds_kpts)[result_boxes].reshape(-1,17,3)*scale
    return box,scores,result,kpts


def cgr_detect_alternative(frame):
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    infer_cgr.infer(blob)
    # Get output tensor for models with one output
    output = infer_cgr.get_output_tensor()
    outputs = output.data

    classes_scores = outputs[:, :, 4]
    # Create a mask to filter rows based on classes_scores >= 0.5
    mask = classes_scores >= 0.5

    # Use the mask to get the filtered_outputs array
    filtered_outputs = outputs[mask]

    # Calculate boxes using vectorized operations
    boxes = filtered_outputs[:, 0:4] - np.column_stack(
        [(0.5 * filtered_outputs[:, 2]), (0.5 * filtered_outputs[:, 3]), np.zeros_like(filtered_outputs[:, 2]),
         np.zeros_like(filtered_outputs[:, 3])])

    # Extract scores and key points
    scores = filtered_outputs[:, 4]
    # Optionally, convert the boxes list to a NumPy array
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.5, 0.5)
    box = np.array(boxes)[result_boxes].reshape(-1, 4)
    box = xywh2xyxy_rescale(box, scale)
    scores = np.array(scores)[result_boxes]
    return box,scores

# if __name__ == '__main__':
#     # cap= cv2.VideoCapture(0)
#     # cap.set(4, 1080)
#     # cap.set(3, 720)
#     # cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("Detected Objects", 1920,1080)
#     # while cap.isOpened():
#     #
#     #     # Read frame from the video
#     #     ret, frame = cap.read()
#     #
#     #     if not ret:
#     #         break
#     frame=cv2.imread("hands/10.jpg")
#     # image = cv2.imread("hands/220.jpg")
#     img,border=prepare_input(frame)
#     start_time = time.time()
#     # Create tensor from external memory
#     input_tensor = ov.Tensor(array=img)
#     # Set input tensor for models with one input
#     infer_request.set_input_tensor(input_tensor)
#     infer_request.infer()
#     # infer_request.start_async()
#     # infer_request.wait()
#     # Get output tensor for models with one output
#     output = infer_request.get_output_tensor()
#     output_buffer = output.data
#     boxes, scores, class_ids=process_output(output_buffer,0.4,0.8,frame,img,border)
#     current_time = time.time()
#     elapsed_time = current_time - start_time
#     print(boxes)
#     combined_img = draw_detections(frame, boxes, scores, class_ids, mask_alpha=0.4)
#     cv2.imwrite("compare1.jpg",combined_img)
#     # cv2.imshow("Detected Objects", combined_img)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break