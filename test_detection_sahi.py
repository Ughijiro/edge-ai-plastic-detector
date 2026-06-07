from pathlib import Path
import cv2
import shutil
from collections import Counter

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
TEST_IMAGES_FOLDER = PROJECT_ROOT / "test_images"
OUTPUT_FOLDER = PROJECT_ROOT / "runs" / "annotated_test_images_SAHI"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CONF_THRESHOLD = 0.40

SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.25
OVERLAP_WIDTH_RATIO = 0.25

LARGE_WASTE_THRESHOLD = 0.15

LABEL_MODE = "id_only"
# poți schimba în:
# LABEL_MODE = "id_label_conf"


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


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass

    return "cpu"


DEVICE = get_device()


def decide_action(label, area_ratio):
    if label not in WASTE_CLASSES:
        return "IGNORE"

    if area_ratio >= LARGE_WASTE_THRESHOLD:
        return "WARNING_LED_BUZZER"

    return "COLLECT_SERVO"


def get_box_color(action):
    if action == "COLLECT_SERVO":
        return (0, 120, 0)       # verde închis
    elif action == "WARNING_LED_BUZZER":
        return (0, 0, 170)       # roșu închis
    else:
        return (80, 80, 80)      # gri închis


def rectangles_overlap(rect1, rect2):
    ax1, ay1, ax2, ay2 = rect1
    bx1, by1, bx2, by2 = rect2

    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def find_label_position(x1, y1, x2, y2, text_width, text_height,
                        image_width, image_height, used_rects):
    padding = 4

    candidates = [
        # deasupra boxului
        (
            x1,
            max(0, y1 - text_height - 12),
            x1 + text_width + 2 * padding,
            max(0, y1 - 12) + text_height + 2 * padding
        ),

        # sub box
        (
            x1,
            min(image_height - text_height - 2 * padding, y2 + 5),
            x1 + text_width + 2 * padding,
            min(image_height, y2 + 5 + text_height + 2 * padding)
        ),

        # dreapta
        (
            min(image_width - text_width - 2 * padding, x2 + 5),
            y1,
            min(image_width, x2 + 5 + text_width + 2 * padding),
            min(image_height, y1 + text_height + 2 * padding)
        ),

        # stânga
        (
            max(0, x1 - text_width - 2 * padding - 5),
            y1,
            max(0, x1 - 5),
            min(image_height, y1 + text_height + 2 * padding)
        ),
    ]

    for rect in candidates:
        overlaps = any(rectangles_overlap(rect, used_rect) for used_rect in used_rects)
        if not overlaps:
            return rect

    return candidates[0]


def prepare_detections_for_display(detections):
    detections_sorted = sorted(
        detections,
        key=lambda d: (d["bbox"][1], d["bbox"][0])
    )

    prepared = []

    for idx, detection in enumerate(detections_sorted, start=1):
        new_detection = detection.copy()
        new_detection["display_id"] = idx
        prepared.append(new_detection)

    return prepared


def get_label_text(detection):
    display_id = detection["display_id"]
    label = detection["label"]
    confidence = detection["confidence"]

    if LABEL_MODE == "id_label_conf":
        return f"#{display_id} {label} {confidence:.2f}"

    return f"#{display_id}"


def draw_detections(image, detections):
    annotated = image.copy()
    image_height, image_width = annotated.shape[:2]

    used_label_rects = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    padding = 4

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        action = detection["action"]
        color = get_box_color(action)

        text = get_label_text(detection)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        lx1, ly1, lx2, ly2 = find_label_position(
            x1, y1, x2, y2,
            text_width, text_height,
            image_width, image_height,
            used_label_rects
        )

        used_label_rects.append((lx1, ly1, lx2, ly2))

        cv2.rectangle(
            annotated,
            (lx1, ly1),
            (lx2, ly2),
            color,
            -1
        )

        cv2.putText(
            annotated,
            text,
            (lx1 + padding, ly1 + text_height + padding),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return annotated


def load_sahi_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu am găsit modelul la: {MODEL_PATH}\n"
            f"Pune best.pt în folderul: {PROJECT_ROOT / 'models'}"
        )

    print(f"Încarc modelul SAHI + YOLO de la: {MODEL_PATH}")
    print(f"Device folosit: {DEVICE}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(MODEL_PATH),
        confidence_threshold=CONF_THRESHOLD,
        device=DEVICE,
    )

    return detection_model


def detect_objects_sahi(image_path, detection_model):
    frame = cv2.imread(str(image_path))

    if frame is None:
        return None, []

    image_height, image_width = frame.shape[:2]
    image_area = image_width * image_height

    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        verbose=0,
    )

    detections = []

    for prediction in result.object_prediction_list:
        label = str(prediction.category.name)
        confidence = float(prediction.score.value)

        bbox = prediction.bbox

        x1 = float(bbox.minx)
        y1 = float(bbox.miny)
        x2 = float(bbox.maxx)
        y2 = float(bbox.maxy)

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

    return frame, detections


def main():
    print("\n==============================")
    print(" TESTARE MODEL CU SAHI")
    print("==============================\n")

    print(f"CONF_THRESHOLD: {CONF_THRESHOLD}")
    print(f"SLICE_HEIGHT:   {SLICE_HEIGHT}")
    print(f"SLICE_WIDTH:    {SLICE_WIDTH}")
    print(f"OVERLAP:        {OVERLAP_HEIGHT_RATIO}, {OVERLAP_WIDTH_RATIO}")

    if not TEST_IMAGES_FOLDER.exists():
        print(f"\nFolderul nu există: {TEST_IMAGES_FOLDER}")
        print("Creează folderul test_images și pune pozele acolo.")
        return

    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"\nFolder output creat: {OUTPUT_FOLDER}")

    image_paths = [
        path for path in TEST_IMAGES_FOLDER.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"\nNu am găsit poze în folderul: {TEST_IMAGES_FOLDER}")
        return

    print(f"\nAm găsit {len(image_paths)} imagini pentru test.\n")

    detection_model = load_sahi_model()

    total_detections = 0
    class_counter = Counter()
    action_counter = Counter()

    for image_path in image_paths:
        print("\n--------------------------------")
        print(f"Imagine: {image_path.name}")
        print("--------------------------------")

        frame, detections = detect_objects_sahi(image_path, detection_model)

        if frame is None:
            print("Nu pot citi imaginea.")
            continue

        detections = prepare_detections_for_display(detections)

        if not detections:
            print("Nicio detecție.")

            output_path = OUTPUT_FOLDER / f"{image_path.stem}_sahi_annotated.jpg"
            cv2.imwrite(str(output_path), frame)

            print(f"Imagine salvată fără detecții la: {output_path}")
            continue

        for detection in detections:
            display_id = detection["display_id"]
            label = detection["label"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            area_percent = detection["area_percent"]
            is_waste = detection["is_waste"]
            action = detection["action"]

            total_detections += 1
            class_counter[label] += 1
            action_counter[action] += 1

            print(f"\nDetecția #{display_id}")
            print(f"  Clasă:      {label}")
            print(f"  Confidence: {confidence}")
            print(f"  BBox:       {bbox}")
            print(f"  Arie:       {area_percent}% din imagine")
            print(f"  Este deșeu: {is_waste}")
            print(f"  Acțiune:    {action}")

        annotated_image = draw_detections(frame, detections)

        output_path = OUTPUT_FOLDER / f"{image_path.stem}_sahi_annotated.jpg"
        success = cv2.imwrite(str(output_path), annotated_image)

        if success:
            print(f"\nImagine SAHI anotată salvată la: {output_path}")
        else:
            print(f"\nNu am putut salva imaginea la: {output_path}")

    print("\n==============================")
    print(" REZUMAT FINAL SAHI")
    print("==============================")

    print(f"\nTotal imagini testate: {len(image_paths)}")
    print(f"Total detecții:        {total_detections}")

    print("\nDetecții pe clase:")
    if class_counter:
        for label, count in class_counter.items():
            print(f"  {label}: {count}")
    else:
        print("  Nicio clasă detectată.")

    print("\nAcțiuni generate:")
    if action_counter:
        for action, count in action_counter.items():
            print(f"  {action}: {count}")
    else:
        print("  Nicio acțiune generată.")

    print(f"\nImaginile anotate cu SAHI se găsesc în folderul:")
    print(f"{OUTPUT_FOLDER}")

    print("\nTest SAHI terminat.")


if __name__ == "__main__":
    main() 
