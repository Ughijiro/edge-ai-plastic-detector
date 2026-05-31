from ultralytics import YOLO
import cv2
import os
from decision.decision_logic import decide_action

model = YOLO("yolov8n.pt")

test_folders = ["test_images", "test_videos"]
plastic_detected = 0
total_plastic = 0
warnings = 0

for folder in test_folders:
    if not os.path.exists(folder):
        continue
    print(f"\n=== Testez folderul: {folder} ===")
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            frame = cv2.imread(path)
        elif file.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
        else:
            continue

        if frame is None:
            continue

        results = model(frame, conf=0.5, verbose=False)
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in ["bottle", "cup", "bowl"]:
                    total_plastic += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area_ratio = ((x2 - x1) * (y2 - y1)) / (frame.shape[1] * frame.shape[0])
                    detection = {"label": label, "confidence": float(box.conf[0]), "area_ratio": round(area_ratio, 3)}
                    action = decide_action(detection)
                    print(f"  {file} → {label} size {area_ratio:.1%} → {action}")
                    if action in ["COLLECT", "WARNING"]:
                        plastic_detected += 1
                    if action == "WARNING":
                        warnings += 1

print(f"\n=== REZULTATE FINALE ===")
print(f"Plastice detectate corect: {plastic_detected}/{total_plastic} ({(plastic_detected/total_plastic*100 if total_plastic > 0 else 0):.1f}%)")
print(f"Warning-uri emise (gunoi mare): {warnings}")
print("Dacă sub 70-75% → mergem la fine-tuning.")