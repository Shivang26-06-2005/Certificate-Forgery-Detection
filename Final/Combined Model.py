from ultralytics import YOLO
import os
import csv
import cv2
import numpy as np

# ----------------------------
# Config
# ----------------------------
detection_model_path = r"D:\Certificate-Detection\Second Dataset\outputs\certificate_retrain2\weights\best.pt"
forgery_model_path   = r"D:\Certificate-Detection\First Dataset\certificate_forgery\weights\best.pt"

input_folders = [
    r"D:\Certificate-Detection\Second Dataset\valid\images",
    r"D:\Certificate-Detection\First Dataset\valid\images"
]

output_folder = r"D:\Certificate-Detection\Outputs\Annotated"
os.makedirs(output_folder, exist_ok=True)
csv_file = r"D:\Certificate-Detection\Outputs\certificate_results.csv"

iou_threshold = 0.3  # minimum overlap to count as fake
forg_conf_threshold = 0.5  # minimum forgery confidence to consider

# Load models
detection_model = YOLO(detection_model_path)
forgery_model   = YOLO(forgery_model_path)

# ----------------------------
# Helper functions
# ----------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ----------------------------
# Process one certificate
# ----------------------------
def check_certificate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "ERROR: Cannot read image"

    annotated = img.copy()
    certificate_fake = False

    # Step 1: Run forgery model on whole image
    forg_results = forgery_model(img)
    forg_boxes = forg_results[0].boxes
    forg_labels = forg_results[0].names
    forg_scores = getattr(forg_results[0], 'probs', [1]*len(forg_boxes))  # confidence scores

    fake_regions = []
    if forg_boxes is not None and len(forg_boxes) > 0:
        for i, f_box in enumerate(forg_boxes):
            f_cls = int(f_box.cls)
            f_label = forg_labels.get(f_cls, "")
            conf = float(f_box.conf) if hasattr(f_box, "conf") else 1.0
            if f_label == "fake" and conf >= forg_conf_threshold:
                x1, y1, x2, y2 = f_box.xyxy[0].tolist()
                fake_regions.append([x1, y1, x2, y2])

    # Step 2: Run detection model on whole image
    det_results = detection_model(img)
    det_boxes = det_results[0].boxes
    det_labels = det_results[0].names

    if det_boxes is not None and len(det_boxes) > 0:
        for i, det_box in enumerate(det_boxes):
            x1, y1, x2, y2 = det_box.xyxy[0].tolist()
            label_id = int(det_box.cls)
            label_name = det_labels.get(label_id, "unknown")

            element_fake = False
            # Step 3: Check overlap with any fake region
            for f_box in fake_regions:
                iou = compute_iou([x1, y1, x2, y2], f_box)
                if iou >= iou_threshold:
                    element_fake = True
                    certificate_fake = True

            # Draw detection box
            color = (0,0,255) if element_fake else (0,255,0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, f"{label_name}: {'FAKE' if element_fake else 'GENUINE'}",
                        (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Step 4: Optionally overlay all fake regions for visualization
    for f_box in fake_regions:
        fx1, fy1, fx2, fy2 = map(int, f_box)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0,0,255), -1)
        cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)

    # Save annotated image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_folder, f"{base_name}_annotated.jpg")
    cv2.imwrite(out_path, annotated)

    return "FAKE" if certificate_fake else "GENUINE"

# ----------------------------
# Collect all images recursively
# ----------------------------
all_images = []
for folder in input_folders:
    if not os.path.isdir(folder):
        continue
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                all_images.append(os.path.join(root, f))

# ----------------------------
# Run pipeline & write CSV
# ----------------------------
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
with open(csv_file, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["Filename", "Result"])
    for img_path in all_images:
        try:
            result = check_certificate(img_path)
        except Exception as e:
            result = f"ERROR: {e}"
        writer.writerow([img_path, result])
        print(f"{img_path} → {result}")

print(f"\n✅ Done. Annotated images saved in: {output_folder}")
print(f"✅ CSV results saved at: {csv_file}")
