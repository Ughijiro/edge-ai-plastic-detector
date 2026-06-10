from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

CONF_THRESHOLD = 0.40
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.25
OVERLAP_WIDTH_RATIO = 0.25

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


detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=str(MODEL_PATH),
    confidence_threshold=CONF_THRESHOLD,
    device="cpu"
)


def get_model_classes():
    return detection_model.category_mapping


def get_recommended_action(label, area_ratio):
    if label not in WASTE_CLASSES:
        return "IGNORE"

    if area_ratio >= LARGE_WASTE_THRESHOLD:
        return "WARNING_LED_BUZZER"

    return "COLLECT_SERVO"


def detect_objects(frame):
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        verbose=0
    )

    detections = []

    for prediction in result.object_prediction_list:
        label = prediction.category.name
        confidence = round(float(prediction.score.value), 2)

        x1 = prediction.bbox.minx
        y1 = prediction.bbox.miny
        x2 = prediction.bbox.maxx
        y2 = prediction.bbox.maxy

        box_width = x2 - x1
        box_height = y2 - y1
        area_ratio = (box_width * box_height) / frame_area

        is_waste = label in WASTE_CLASSES
        action = get_recommended_action(label, area_ratio)

        if is_waste:
            detections.append({
                "label": label,
                "confidence": confidence,
                "area_ratio": round(area_ratio, 3),
                "area_percent": round(area_ratio * 100, 2),
                "track_id": -1,
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