from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

CONF_THRESHOLD = 0.45
LARGE_WASTE_THRESHOLD = 0.35

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
        f"Model not found at: {MODEL_PATH}\n"
        f"Put best.pt inside: {PROJECT_ROOT / 'models'}"
    )


model = YOLO(str(MODEL_PATH))


def get_model_classes():
    return model.names


def get_recommended_action(label, area_ratio):
    if label not in WASTE_CLASSES:
        return "IGNORE"

    if area_ratio >= LARGE_WASTE_THRESHOLD:
        return "WARNING_LED_BUZZER"

    return "COLLECT_SERVO"


def detect_objects(frame):
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    detections = []
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = round(float(box.conf[0]), 2)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1
            area_ratio = (box_width * box_height) / frame_area

            track_id = int(box.id[0]) if box.id is not None else -1
            is_waste = label in WASTE_CLASSES
            action = get_recommended_action(label, area_ratio)

            if is_waste:
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "area_ratio": round(area_ratio, 3),
                    "area_percent": round(area_ratio * 100, 2),
                    "track_id": track_id,
                    "bbox": [
                        round(x1),
                        round(y1),
                        round(x2),
                        round(y2)
                    ],
                    "is_waste": is_waste,
                    "action": action
                })

    return detections