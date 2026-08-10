import os
import json

# Paths
json_path = r"projects\bdd\data\metadata\bdd100k_labels_images_val.json"
predictions_base = r"projects\bdd\yolo_predictions_tel"
conditions = ["daytime", "night", "fog"]

with open(json_path, "r") as f:
    data = json.load(f)


ground_truth_dict = {}
for img in data:
    has_light = False
    if "labels" in img:
        for label in img["labels"]:
            if label["category"] == "traffic light":
                has_light = True
                break
    ground_truth_dict[img["name"]] = has_light

summary_results = []

for condition_name in conditions:
    images_dir = os.path.join(predictions_base, condition_name)
    labels_dir = os.path.join(predictions_base, condition_name, "labels")

    if not os.path.exists(images_dir):
        continue

    test_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    total_images = len(test_images)

    if total_images == 0:
        continue

    images_with_gt_light = 0
    yolo_found = 0
    yolo_missed = 0

    for img_name in test_images:
        actual_has_light = ground_truth_dict.get(img_name, False)

        txt_filename = img_name.replace(".jpg", ".txt")
        txt_path = os.path.join(labels_dir, txt_filename)
        yolo_has_light = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0

        if actual_has_light:
            images_with_gt_light += 1
            if yolo_has_light:
                yolo_found += 1
            else:
                yolo_missed += 1

    success_rate = (
        (yolo_found / images_with_gt_light) * 100
        if images_with_gt_light > 0
        else 0
    )

    rate_str = f"{success_rate:.1f}%".replace(".0%", "%")

    summary_results.append({
        "condition": condition_name.capitalize(),
        "total": total_images,
        "true": yolo_found,
        "false": yolo_missed,
        "rate": rate_str,
    })


border = "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*8 + "+" + "-"*9 + "+" + "-"*11 + "+"
header_border = "+" + "="*12 + "+" + "="*10 + "+" + "="*8 + "+" + "="*9 + "+" + "="*11 + "+"

print(border)
print(f"| {'Συνθήκη':<10} | {'Εικόνες':<8} | {'TRUE':<6} | {'FALSE':<7} | {'Ποσοστό':<9} |")
print(header_border)

for r in summary_results:
    print(f"| {r['condition']:<10} | {r['total']:<8} | {r['true']:<6} | {r['false']:<7} | {r['rate']:<9} |")
    print(border)