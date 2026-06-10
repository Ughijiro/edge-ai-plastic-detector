from pathlib import Path
import cv2
import shutil
import json
import csv
from collections import Counter

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
TEST_IMAGES_FOLDER = PROJECT_ROOT / "test_images"
OUTPUT_FOLDER = PROJECT_ROOT / "runs" / "annotated_test_images_hybrid"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}



# 1) YOLO NORMAL - pentru obiecte medii / mari + contextul întreg
NORMAL_CONF = 0.40
NORMAL_IOU = 0.60
NORMAL_IMGSZ = 1280


# 2) SAHI - pentru obiecte mici / îndepărtate
SAHI_CONF = 0.40
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.20
OVERLAP_WIDTH_RATIO = 0.20

# Păstrăm din SAHI doar obiectele mici / mici-medii
# ca să nu concureze inutil cu detecția normală pentru obiecte mari
SAHI_MAX_AREA_RATIO = 0.08

# SAHI + YOLO duplicate merge
MERGE_IOU_THRESHOLD = 0.50



# 3) Threshold-uri pe clase

DEFAULT_CLASS_THRESHOLD = 0.30

CLASS_CONF_THRESHOLDS = {
    "plastic-bottle": 0.30,
    "plastic-bag": 0.32,
    "plastic-cup": 0.30,
    "plastic-wrapper": 0.40,
    "plastic-other": 0.45,
    "can": 0.30,
    "carton": 0.32,
    "foam": 0.42,
    "other-waste": 0.45,
    "natural-debris": 0.35,
}



# 4) Logica proiectului

LARGE_WASTE_THRESHOLD = 0.15

LABEL_MODE = "id_only"
# dacă vrei mai mult text pe imagine:
# LABEL_MODE = "id_label_conf_source"

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


def get_effective_threshold(label):
    return CLASS_CONF_THRESHOLDS.get(label, DEFAULT_CLASS_THRESHOLD)


def decide_action(label, area_ratio, confidence):
    """
    Decizie de proiect:
    - non-waste -> IGNORE
    - clase vagi cu confidence mic -> REVIEW_ONLY
    - deșeu mare -> WARNING_LED_BUZZER
    - deșeu mic/mediu -> COLLECT_SERVO
    """
    if label not in WASTE_CLASSES:
        return "IGNORE"

    if label in {"plastic-other", "other-waste"} and confidence < 0.55:
        return "REVIEW_ONLY"

    if area_ratio >= LARGE_WASTE_THRESHOLD:
        return "WARNING_LED_BUZZER"

    return "COLLECT_SERVO"


def get_box_color(action):
    """
    OpenCV folosește BGR.
    """
    if action == "COLLECT_SERVO":
        return (0, 120, 0)        # verde închis
    elif action == "WARNING_LED_BUZZER":
        return (0, 0, 170)        # roșu închis
    elif action == "REVIEW_ONLY":
        return (0, 140, 180)      # portocaliu
    else:
        return (80, 80, 80)       # gri închis


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


def compute_iou(box1, box2):
    """
    box = [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def build_detection(label, confidence, x1, y1, x2, y2, image_width, image_height, source):
    image_area = image_width * image_height
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = max(0.0, box_width) * max(0.0, box_height)
    area_ratio = box_area / image_area if image_area > 0 else 0.0

    is_waste = label in WASTE_CLASSES
    action = decide_action(label, area_ratio, confidence)

    return {
        "label": label,
        "confidence": round(float(confidence), 3),
        "bbox": [
            round(float(x1), 1),
            round(float(y1), 1),
            round(float(x2), 1),
            round(float(y2), 1)
        ],
        "area_ratio": round(area_ratio, 4),
        "area_percent": round(area_ratio * 100, 2),
        "is_waste": is_waste,
        "action": action,
        "source": source
    }


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
    source = detection["source"]

    if LABEL_MODE == "id_label_conf_source":
        return f"#{display_id} {label} {confidence:.2f} {source}"

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

        # chenare
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


def load_normal_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu am găsit modelul la: {MODEL_PATH}\n"
            f"Pune best.pt în folderul: {PROJECT_ROOT / 'models'}"
        )

    print(f"Încarc modelul YOLO normal de la: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    return model


def load_sahi_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu am găsit modelul la: {MODEL_PATH}\n"
            f"Pune best.pt în folderul: {PROJECT_ROOT / 'models'}"
        )

    print(f"Încarc modelul SAHI + YOLO de la: {MODEL_PATH}")
    print(f"Device folosit: {DEVICE}")

    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(MODEL_PATH),
            confidence_threshold=SAHI_CONF,
            device=DEVICE,
        )
    except Exception:
        print("Nu a mers model_type='ultralytics'. Încerc model_type='yolov8'...")

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(MODEL_PATH),
            confidence_threshold=SAHI_CONF,
            device=DEVICE,
        )

    return detection_model


def detect_objects_normal(frame, normal_model):
    detections = []

    results = normal_model.predict(
        source=frame,
        conf=NORMAL_CONF,
        iou=NORMAL_IOU,
        imgsz=NORMAL_IMGSZ,
        verbose=False
    )

    for result in results:
        if result.boxes is None:
            continue

        image_height, image_width = result.orig_shape

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = normal_model.names[class_id]
            confidence = float(box.conf[0])

            min_conf = get_effective_threshold(label)
            if confidence < min_conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = build_detection(
                label=label,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_width=image_width,
                image_height=image_height,
                source="normal"
            )

            detections.append(detection)

    return detections


def detect_objects_sahi(image_path, frame, sahi_model):
    detections = []

    image_height, image_width = frame.shape[:2]

    result = get_sliced_prediction(
        str(image_path),
        sahi_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        perform_standard_pred=False,
        verbose=0,
    )

    for prediction in result.object_prediction_list:
        label = str(prediction.category.name)
        confidence = float(prediction.score.value)

        min_conf = get_effective_threshold(label)
        if confidence < min_conf:
            continue

        bbox = prediction.bbox
        x1 = float(bbox.minx)
        y1 = float(bbox.miny)
        x2 = float(bbox.maxx)
        y2 = float(bbox.maxy)

        # Păstrăm din SAHI doar obiectele mici / mici-medii
        box_width = x2 - x1
        box_height = y2 - y1
        image_area = image_width * image_height
        box_area = max(0.0, box_width) * max(0.0, box_height)
        area_ratio = box_area / image_area if image_area > 0 else 0.0

        if area_ratio > SAHI_MAX_AREA_RATIO:
            continue

        detection = build_detection(
            label=label,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=image_width,
            image_height=image_height,
            source="sahi"
        )

        detections.append(detection)

    return detections


def merge_detections(normal_detections, sahi_detections, iou_threshold=MERGE_IOU_THRESHOLD):
    """
    Combină:
    - detecțiile normale (baza)
    - detecțiile SAHI (mai ales pentru small objects)

    Dacă o detecție SAHI se suprapune puternic cu una YOLO de aceeași clasă,
    păstrăm una singură și marcăm source='both'.
    """
    final_detections = [d.copy() for d in normal_detections]

    for sahi_det in sahi_detections:
        matched_index = None
        best_iou = 0.0

        for idx, existing_det in enumerate(final_detections):
            if existing_det["label"] != sahi_det["label"]:
                continue

            iou = compute_iou(existing_det["bbox"], sahi_det["bbox"])

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                matched_index = idx

        if matched_index is None:
            final_detections.append(sahi_det.copy())
        else:
            existing_det = final_detections[matched_index]

            if sahi_det["confidence"] > existing_det["confidence"]:
                merged = sahi_det.copy()
            else:
                merged = existing_det.copy()

            merged["source"] = "both"
            merged["matched_iou"] = round(best_iou, 3)

            final_detections[matched_index] = merged

    return final_detections


def save_detections_json_and_csv(all_detections):
    json_path = OUTPUT_FOLDER / "detections_hybrid.json"
    csv_path = OUTPUT_FOLDER / "detections_hybrid.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=4, ensure_ascii=False)

    fieldnames = [
        "image",
        "display_id",
        "label",
        "confidence",
        "bbox",
        "area_ratio",
        "area_percent",
        "is_waste",
        "action",
        "source"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for detection in all_detections:
            writer.writerow(detection)

    print(f"\nJSON salvat la: {json_path}")
    print(f"CSV salvat la:  {csv_path}")


def main():
    print("\n======================================")
    print(" TESTARE HYBRID: YOLO NORMAL + SAHI")
    print("======================================\n")

    print(f"PROJECT_ROOT:        {PROJECT_ROOT}")
    print(f"MODEL_PATH:          {MODEL_PATH}")
    print(f"TEST_IMAGES_FOLDER:  {TEST_IMAGES_FOLDER}")
    print(f"OUTPUT_FOLDER:       {OUTPUT_FOLDER}")

    print(f"\nNORMAL_CONF:         {NORMAL_CONF}")
    print(f"NORMAL_IOU:          {NORMAL_IOU}")
    print(f"NORMAL_IMGSZ:        {NORMAL_IMGSZ}")

    print(f"\nSAHI_CONF:           {SAHI_CONF}")
    print(f"SLICE_HEIGHT:        {SLICE_HEIGHT}")
    print(f"SLICE_WIDTH:         {SLICE_WIDTH}")
    print(f"OVERLAP:             {OVERLAP_HEIGHT_RATIO}, {OVERLAP_WIDTH_RATIO}")
    print(f"SAHI_MAX_AREA_RATIO: {SAHI_MAX_AREA_RATIO}")
    print(f"MERGE_IOU_THRESHOLD: {MERGE_IOU_THRESHOLD}")

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

    normal_model = load_normal_model()
    sahi_model = load_sahi_model()

    total_final_detections = 0
    class_counter = Counter()
    action_counter = Counter()
    source_counter = Counter()
    all_detections = []

    for image_path in image_paths:
        print("\n--------------------------------")
        print(f"Imagine: {image_path.name}")
        print("--------------------------------")

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Nu pot citi imaginea.")
            continue

        normal_detections = detect_objects_normal(frame, normal_model)
        sahi_detections = detect_objects_sahi(image_path, frame, sahi_model)
        final_detections = merge_detections(normal_detections, sahi_detections)
        final_detections = prepare_detections_for_display(final_detections)

        print(f"Detecții YOLO normal: {len(normal_detections)}")
        print(f"Detecții SAHI:        {len(sahi_detections)}")
        print(f"Detecții finale:      {len(final_detections)}")

        if not final_detections:
            print("Nicio detecție finală.")

            output_path = OUTPUT_FOLDER / f"{image_path.stem}_hybrid_annotated.jpg"
            cv2.imwrite(str(output_path), frame)

            print(f"Imagine salvată fără detecții la: {output_path}")
            continue

        for detection in final_detections:
            display_id = detection["display_id"]
            label = detection["label"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            area_percent = detection["area_percent"]
            is_waste = detection["is_waste"]
            action = detection["action"]
            source = detection["source"]

            total_final_detections += 1
            class_counter[label] += 1
            action_counter[action] += 1
            source_counter[source] += 1

            detection_for_file = detection.copy()
            detection_for_file["image"] = image_path.name
            all_detections.append(detection_for_file)

            print(f"\nDetecția #{display_id}")
            print(f"  Clasă:      {label}")
            print(f"  Confidence: {confidence}")
            print(f"  BBox:       {bbox}")
            print(f"  Arie:       {area_percent}% din imagine")
            print(f"  Este deșeu: {is_waste}")
            print(f"  Acțiune:    {action}")
            print(f"  Sursă:      {source}")

        annotated_image = draw_detections(frame, final_detections)

        output_path = OUTPUT_FOLDER / f"{image_path.stem}_hybrid_annotated.jpg"
        success = cv2.imwrite(str(output_path), annotated_image)

        if success:
            print(f"\nImagine hybrid anotată salvată la: {output_path}")
        else:
            print(f"\nNu am putut salva imaginea la: {output_path}")

    save_detections_json_and_csv(all_detections)

    print("\n======================================")
    print(" REZUMAT FINAL HYBRID")
    print("======================================")

    print(f"\nTotal imagini testate: {len(image_paths)}")
    print(f"Total detecții finale: {total_final_detections}")

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

    print("\nSurse detecții:")
    if source_counter:
        for source, count in source_counter.items():
            print(f"  {source}: {count}")
    else:
        print("  Nicio sursă înregistrată.")

    print(f"\nImaginile anotate hybrid se găsesc în folderul:")
    print(f"{OUTPUT_FOLDER}")

    print("\nTest hybrid terminat.")


if __name__ == "__main__":
    main()