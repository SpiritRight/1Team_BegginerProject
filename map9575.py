import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ------------------------------
# IoU 계산
# ------------------------------
def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter
    return inter / union if union != 0 else 0


# ------------------------------
# AP 계산
# ------------------------------
def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))

    for i in range(len(precisions)-1, 0, -1):
        precisions[i-1] = max(precisions[i-1], precisions[i])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[idx+1] - recalls[idx]) * precisions[idx+1])
    return ap


# ------------------------------
# AP @ IoU threshold
# ------------------------------
def ap_at_iou(pred_boxes, pred_scores, gt_boxes, iou_threshold):

    sorted_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched = set()

    for i, pbox in enumerate(pred_boxes):
        best_iou = 0
        best_gt = -1

        for j, gt in enumerate(gt_boxes):
            iou = compute_iou(pbox, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold and best_gt not in matched:
            tp[i] = 1
            matched.add(best_gt)
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / len(gt_boxes) if len(gt_boxes) > 0 else np.array([0.0])
    precisions = tp / (tp + fp + 1e-6)

    return compute_ap(recalls, precisions)


# ------------------------------
# mAP75-95 계산
# ------------------------------
def map75_95(pred_boxes, pred_scores, gt_boxes):
    if len(gt_boxes) == 0:
        # gt box 존재하지 않을시 0 리턴
        return 0.0, []

    iou_thresholds = np.arange(0.75, 1.00, 0.05)
    aps = []

    for t in iou_thresholds:
        ap = ap_at_iou(pred_boxes, pred_scores, gt_boxes, t)
        aps.append(ap)

    return np.mean(aps), aps

def run_inference_and_evaluate(model_path, img_path, gt_json_dir):
    model = YOLO(model_path)

    # 사이즈, conf 값 조정가능
    results = model.predict(source=img_path, conf=0.25, imgsz=768, verbose=False) # verbose=False to reduce output

    # 잘 작동하는지 시각화
    res_plotted = results[0].plot()
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Prediction for: {os.path.basename(img_path)}")
    plt.show()

    # bbox, score 추출
    if results[0].boxes:
        pred_boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        pred_scores = results[0].boxes.conf.cpu().numpy()
    else:
        pred_boxes_xyxy = np.array([])
        pred_scores = np.array([])

    # gt 값들 추출
    img_name = os.path.basename(img_path)
    json_name = os.path.splitext(img_name)[0] + '.json'
    gt_json_path = os.path.join(gt_json_dir, json_name)

    gt_boxes = []
    if os.path.exists(gt_json_path):
        with open(gt_json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        for ann in gt_data.get('annotations', []):
            bbox = ann.get('bbox', []) # format: [x, y, w, h]
            # bbox 좌표가 4개인지 검증
            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, w, h = bbox
                gt_boxes.append([x, y, x + w, y + h]) # convert to [x1, y1, x2, y2]
            else:
                print(f"Warning: Malformed bbox found in {json_name}: {bbox}")
    gt_boxes = np.array(gt_boxes)

    
    mean_ap, _ = map75_95(pred_boxes_xyxy, pred_scores, gt_boxes)
    print(f"\nmAP@.75-.95 for {os.path.basename(img_path)}: {mean_ap:.4f}")

    return mean_ap



model_path = '모델 경로 입력'

# gt annotation 입력
SRC_JSON_DIR = "JSON 파일 경로"

# 실험으로 validation 이미지 사용
val_images_dir = "이미지 경로"
val_image_files = os.listdir(val_images_dir)

if val_image_files:
    num_images_to_evaluate = 10
    print(f"Evaluating on {num_images_to_evaluate} validation images...")
    for i, example_img_name in enumerate(val_image_files[:num_images_to_evaluate]):
        print(f"\n--- Evaluating Image {i+1}/{num_images_to_evaluate} ---")
        img_path = os.path.join(val_images_dir, example_img_name)
        print(f"Using validation image: {img_path}")
        run_inference_and_evaluate(model_path, img_path, SRC_JSON_DIR)
else:
    print(f"No validation images found in {val_images_dir} to perform evaluation.")
