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
'''from ultralytics import YOLO

# Modelul tău fine-tuned (ultimul antrenat)
model = YOLO('runs/detect/plastic_detector_finetuned-4/weights/best.pt')

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

    return detections'''



#VARIANTA PE POZE DUPA ANTRENARE 100 EPOCI
'''from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

CONF_THRESHOLD = 0.25
IMGSZ = 960

LARGE_WASTE_THRESHOLD = 0.15

WASTE_CLASSES = {
    "plastic-bottle",
    "plastic-bag",
    "plastic-cup",
    "plastic-wrapper",
    "plastic-other",
    "can",
    "carton",
    "foam",
    "other-waste",
}

NON_WASTE_CLASSES = {
    "natural-debris",
}

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Nu am găsit modelul la: {MODEL_PATH}\n"
        f"Pune best.pt în folderul: {PROJECT_ROOT / 'models'}"
    )

model = YOLO(str(MODEL_PATH))


def decide_action(label, area_ratio):
    """
    Decide ce ar trebui să facă sistemul pe baza clasei detectate.

    - dacă NU e deșeu -> IGNORE
    - dacă e deșeu mic/normal -> COLLECT_SERVO
    - dacă e deșeu mare -> WARNING_LED_BUZZER
    """

    if label not in WASTE_CLASSES:
        return "IGNORE"

    if area_ratio >= LARGE_WASTE_THRESHOLD:
        return "WARNING_LED_BUZZER"

    return "COLLECT_SERVO"


def detect_objects(frame, conf_threshold=CONF_THRESHOLD, imgsz=IMGSZ):
    """
    Primește o imagine OpenCV și returnează lista de detecții.

    Fiecare detecție conține:
    - label
    - confidence
    - bbox
    - area_ratio
    - is_waste
    - action
    """

    results = model.predict(
        source=frame,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        image_height, image_width = result.orig_shape
        image_area = image_width * image_height

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            area_ratio = box_area / image_area

            is_waste = label in WASTE_CLASSES
            action = decide_action(label, area_ratio)

            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1)
                ],
                "area_ratio": round(area_ratio, 4),
                "area_percent": round(area_ratio * 100, 2),
                "is_waste": is_waste,
                "action": action
            })

    return detections


def get_model_classes():
    return model.names
'''



from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

model = YOLO(str(MODEL_PATH))

WASTE_CLASSES = {
    "plastic-bottle", "plastic-bag", "plastic-cup",
    "plastic-wrapper", "plastic-other",
    "can", "carton", "foam", "other-waste",
}
# natural-debris e detectat și afișat, dar NU intră în decizie
CONF_THRESHOLD = 0.45

def detect_objects(frame):
    results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    h, w = frame.shape[:2]

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            confidence = round(float(box.conf[0]), 2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
            track_id = int(box.id[0]) if box.id is not None else -1

            if label in WASTE_CLASSES:
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "area_ratio": round(area_ratio, 3),
                    "track_id": track_id,
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                })
    return detections