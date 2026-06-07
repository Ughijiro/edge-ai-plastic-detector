#from ultralytics import YOLO
#
#model = YOLO("yolov8n.pt")
#
#ALLOWED_CLASSES = ["bottle", "person", "bird", "boat"]
#
#def detect_objects(frame):
#    results = model(frame, verbose=False)
#    detections = []
#
#    for result in results:
#        for box in result.boxes:
#            class_id = int(box.cls[0])
#            label = model.names[class_id]
#            confidence = round(float(box.conf[0]), 2)
#
#            if label in ALLOWED_CLASSES and confidence >= 0.5:
#                detections.append({
#                    "label": label,
#                    "confidence": confidence
#               })
#
#    return detections

# VERSIUNE SOFTWARE ONLY 
'''from ultralytics import YOLO

model = YOLO('runs/detect/plastic_detector_finetuned-4/weights/best.pt')
# clase de gunoi/plastic pe care le detectăm acum (din modelul pre-trained)
WASTE_CLASSES = ["bottle", "cup", "bowl"]  
# pentru "plastic bag", "foam", "can" etc. va trebui fine-tuning mai târziu (cum zice și ideea #14 din proiect)

def detect_objects(frame):
    # prag mai jos ca să vedem tot ce detectează modelul
    results = model(frame, conf=0.25, verbose=False)
    detections = []

    print("   [DEBUG] Detectii brute gasite:")   # asta o sa vezi tot

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = round(float(box.conf[0]), 2)

            # calculam aria oricum
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area_ratio = (w * h) / (frame.shape[1] * frame.shape[0])

            print(f"     → {label} | conf={confidence} | aria={area_ratio:.1%}")

            # păstrăm doar gunoiul, dar acum vedem si celelalte
            if label in WASTE_CLASSES and confidence >= 0.35:
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "area_ratio": round(area_ratio, 3)
                })

    return detections '''

#TEST PENTRU VIDEO 
from ultralytics import YOLO

# Modelul tău fine-tuned (ultimul antrenat)
model = YOLO('runs/detect/plastic_detector_finetuned-4/weights/yolov8n.pt')

WASTE_CLASSES = ["bottle", "plastic", "can", "carton", "paper"]

def detect_objects(frame):
    # tracking cu persist=True ca să țină minte ID-urile de la frame la frame
    results = model.track(frame, persist=True, conf=0.45, verbose=False)   # crescut de la 0.35
    detections = []

    print("   [DEBUG] Detectii cu tracking:")

    for result in results:
        if not result.boxes or not result.boxes.is_track:
            continue

        for box in result.boxes:
            track_id = int(box.id[0]) if box.id is not None else -1
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = round(float(box.conf[0]), 2)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area_ratio = (w * h) / (frame.shape[1] * frame.shape[0])

            print(f"     → ID{track_id} | {label} | conf={confidence} | aria={area_ratio:.1%}")

            if label in WASTE_CLASSES and confidence >= 0.50:
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "area_ratio": round(area_ratio, 3),
                    "track_id": track_id
                })

    return detections