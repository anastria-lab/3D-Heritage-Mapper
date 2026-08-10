import json
import os

# Paths and Settings
json_path = r"projects\bdd\data\metadata\bdd100k_labels_images_val.json"
predictions_base = r"projects\bdd\yolo_predictions_tel"
conditions = ["daytime", "night", "fog"]

IMG_W, IMG_H = 1280, 720
IOU_THRESHOLD = 0.2  
MIN_BOX_AREA = (
    100  
)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


print(f"Reading Ground Truth from: {json_path}...")
with open(json_path, "r") as f:
    data = json.load(f)

gt_dict = {}
for img in data:
    img_name = img["name"]
    boxes = []
    if "labels" in img:
        for label in img["labels"]:
            if label["category"] == "traffic light" and "box2d" in label:
                b = label["box2d"]
                w = b["x2"] - b["x1"]
                h = b["y2"] - b["y1"]

                if (w * h) >= MIN_BOX_AREA:
                    boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
    if boxes:
        gt_dict[img_name] = boxes

print("\nStarting evaluation with optimized thresholds...")

summary_results = []

for condition in conditions:
    labels_dir = os.path.join(predictions_base, condition, "labels")
    if not os.path.exists(labels_dir):
        continue

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    total_gt_lights = 0
    total_pred_lights = 0
    true_positives = 0
    false_positives = 0
    confidences = []

    for txt_file in txt_files:
        img_name = txt_file.replace(".txt", ".jpg")

        if img_name not in gt_dict:
            continue

        gt_boxes = gt_dict[img_name]
        total_gt_lights += len(gt_boxes)

        pred_boxes = []
        with open(os.path.join(labels_dir, txt_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = float(parts[0])
                    if int(cls) != 9:
                        continue

                    x, y, w, h = map(float, parts[1:5])

                    if len(parts) >= 6:
                        confidences.append(float(parts[5]))

                    x_center, y_center = x * IMG_W, y * IMG_H
                    width, height = w * IMG_W, h * IMG_H

                    x1 = x_center - (width / 2)
                    y1 = y_center - (height / 2)
                    x2 = x_center + (width / 2)
                    y2 = y_center + (height / 2)

                    pred_boxes.append([x1, y1, x2, y2])

        total_pred_lights += len(pred_boxes)

        matched_gt = set()
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = calculate_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= IOU_THRESHOLD:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1

    recall = (
        (true_positives / total_gt_lights) * 100 if total_gt_lights > 0 else 0
    )
    precision = (
        (true_positives / total_pred_lights) * 100
        if total_pred_lights > 0
        else 0
    )
    mean_conf = (
        (sum(confidences) / len(confidences)) * 100 if confidences else 0
    )

    summary_results.append({
        "condition": condition.capitalize(),
        "gt": total_gt_lights,
        "pred": total_pred_lights,
        "tp": true_positives,
        "fp": false_positives,
        "precision": f"{precision:.1f}%",
        "recall": f"{recall:.1f}%",
        "mean_conf": f"{mean_conf:.1f}%" if confidences else "N/A",
    })

# --- ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑC ΦΑΣΗΣ 2 ---
print("\n" + "=" * 90)
print(
    f"{'Condition':<10} | {'GT Lights':<10} | {'YOLO Preds':<10} | {'TP (True)':<10} | {'FP (False)':<10} | {'Precision':<10} | {'Recall':<10} | {'Mean Conf':<10}"
)
print("=" * 90)

for res in summary_results:
    print(
        f"{res['condition']:<10} | "
        f"{res['gt']:<10} | "
        f"{res['pred']:<10} | "
        f"{res['tp']:<10} | "
        f"{res['fp']:<10} | "
        f"{res['precision']:<10} | "
        f"{res['recall']:<10} | "
        f"{res['mean_conf']:<10}"
    )

print("=" * 90)