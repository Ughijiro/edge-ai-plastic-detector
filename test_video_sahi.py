from pathlib import Path
import cv2
import shutil
import json
import csv
from collections import Counter

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# PATHS


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
TEST_VIDEOS_FOLDER = PROJECT_ROOT / "test_videos"

# Folder separat pentru rezultatele:
# SAHI, confidence 0.40, maximum 15 secunde
OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "runs"
    / "annotated_test_videos_sahi_0_40_15s"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}



# PARAMETRI SAHI


CONF_THRESHOLD = 0.40

SLICE_HEIGHT = 640
SLICE_WIDTH = 640

OVERLAP_HEIGHT_RATIO = 0.25
OVERLAP_WIDTH_RATIO = 0.25



# PARAMETRI VIDEO


# Procesează maximum primele 15 secunde din fiecare videoclip
MAX_VIDEO_SECONDS = 15

# Rulează SAHI pe frame-urile:
# 0, 5, 10, 15, 20...
PROCESS_EVERY_N_FRAMES = 5

# False = afișează doar rezumatul fiecărui frame procesat
# True = afișează toate detaliile fiecărei detecții
PRINT_DETAILED_DETECTIONS = False



# LOGICA PROIECTULUI


LARGE_WASTE_THRESHOLD = 0.15

# Pe video apare doar #1, #2, #3...
LABEL_MODE = "id_only"

# Pentru clasă și confidence pe video:
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



# DEVICE


def get_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"

    except Exception:
        pass

    return "cpu"


DEVICE = get_device()



# DECIZIA PENTRU HARDWARE


def decide_action(label, area_ratio):
    """
    IGNORE:
        obiectul nu este deșeu

    COLLECT_SERVO:
        deșeu suficient de mic pentru colectare

    WARNING_LED_BUZZER:
        deșeu prea mare pentru mecanismul de colectare
    """

    if label not in WASTE_CLASSES:
        return "IGNORE"

    if area_ratio >= LARGE_WASTE_THRESHOLD:
        return "WARNING_LED_BUZZER"

    return "COLLECT_SERVO"



# DESENAREA BOUNDING BOX-URILOR

def get_box_color(action):
    """
    OpenCV folosește BGR, nu RGB.
    """

    if action == "COLLECT_SERVO":
        return 0, 120, 0       # verde închis

    if action == "WARNING_LED_BUZZER":
        return 0, 0, 170       # roșu închis

    return 80, 80, 80          # gri închis pentru IGNORE


def rectangles_overlap(rect1, rect2):
    ax1, ay1, ax2, ay2 = rect1
    bx1, by1, bx2, by2 = rect2

    return not (
        ax2 < bx1
        or ax1 > bx2
        or ay2 < by1
        or ay1 > by2
    )


def clamp_label_rectangle(
    x,
    y,
    width,
    height,
    image_width,
    image_height
):
    """
    Păstrează eticheta complet în interiorul imaginii.
    """

    x = max(0, min(x, image_width - width))
    y = max(0, min(y, image_height - height))

    return (
        x,
        y,
        x + width,
        y + height
    )


def find_label_position(
    x1,
    y1,
    x2,
    y2,
    text_width,
    text_height,
    image_width,
    image_height,
    used_rects
):
    """
    Încearcă să plaseze eticheta fără suprapunere.
    """

    padding = 4

    label_width = text_width + 2 * padding
    label_height = text_height + 2 * padding + 2

    candidates = [
        # Deasupra bounding box-ului
        clamp_label_rectangle(
            x1,
            y1 - label_height - 4,
            label_width,
            label_height,
            image_width,
            image_height
        ),

        # Sub bounding box
        clamp_label_rectangle(
            x1,
            y2 + 4,
            label_width,
            label_height,
            image_width,
            image_height
        ),

        # În dreapta
        clamp_label_rectangle(
            x2 + 4,
            y1,
            label_width,
            label_height,
            image_width,
            image_height
        ),

        # În stânga
        clamp_label_rectangle(
            x1 - label_width - 4,
            y1,
            label_width,
            label_height,
            image_width,
            image_height
        ),
    ]

    for candidate in candidates:
        overlaps = any(
            rectangles_overlap(candidate, used_rect)
            for used_rect in used_rects
        )

        if not overlaps:
            return candidate

    return candidates[0]


def prepare_detections_for_display(detections):
    """
    Sortează detecțiile de sus în jos și de la stânga la dreapta.
    Adaugă display_id: #1, #2, #3...
    """

    detections_sorted = sorted(
        detections,
        key=lambda detection: (
            detection["bbox"][1],
            detection["bbox"][0]
        )
    )

    prepared = []

    for index, detection in enumerate(
        detections_sorted,
        start=1
    ):
        new_detection = detection.copy()
        new_detection["display_id"] = index
        prepared.append(new_detection)

    return prepared


def get_label_text(detection):
    display_id = detection["display_id"]
    label = detection["label"]
    confidence = detection["confidence"]

    if LABEL_MODE == "id_label_conf":
        return f"#{display_id} {label} {confidence:.2f}"

    return f"#{display_id}"


def draw_detections(frame, detections):
    """
    Desenează bounding box-urile pe frame.
    """

    annotated = frame.copy()
    image_height, image_width = annotated.shape[:2]

    used_label_rects = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 2
    box_thickness = 2
    padding = 4

    for detection in detections:
        bbox = detection.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), image_width - 1))
        y1 = max(0, min(int(y1), image_height - 1))
        x2 = max(0, min(int(x2), image_width - 1))
        y2 = max(0, min(int(y2), image_height - 1))

        action = detection.get("action", "IGNORE")
        color = get_box_color(action)

        text = get_label_text(detection)

        # Bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness
        )

        (text_width, text_height), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness
        )

        label_rect = find_label_position(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            text_width=text_width,
            text_height=text_height,
            image_width=image_width,
            image_height=image_height,
            used_rects=used_label_rects
        )

        lx1, ly1, lx2, ly2 = label_rect
        used_label_rects.append(label_rect)

        # Fundalul etichetei
        cv2.rectangle(
            annotated,
            (lx1, ly1),
            (lx2, ly2),
            color,
            -1
        )

        # Text alb
        cv2.putText(
            annotated,
            text,
            (lx1 + padding, ly1 + text_height + padding),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

    return annotated


# ÎNCĂRCAREA MODELULUI
def load_sahi_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu am găsit modelul la: {MODEL_PATH}\n"
            f"Pune best.pt în folderul: "
            f"{PROJECT_ROOT / 'models'}"
        )

    print(f"Încarc modelul SAHI + YOLO de la: {MODEL_PATH}")
    print(f"Device folosit: {DEVICE}")

    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(MODEL_PATH),
            confidence_threshold=CONF_THRESHOLD,
            device=DEVICE,
        )

    except Exception as first_error:
        print(
            "Nu a mers model_type='ultralytics'. "
            "Încerc model_type='yolov8'..."
        )
        print(f"Prima eroare: {first_error}")

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(MODEL_PATH),
            confidence_threshold=CONF_THRESHOLD,
            device=DEVICE,
        )

    return detection_model



# DETECȚIE SAHI PE UN FRAME
def detect_frame_sahi(frame, detection_model):
    image_height, image_width = frame.shape[:2]
    image_area = image_width * image_height

    # OpenCV citește BGR; pentru SAHI îl transformăm în RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = get_sliced_prediction(
        frame_rgb,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
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

        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        box_area = box_width * box_height

        area_ratio = (
            box_area / image_area
            if image_area > 0
            else 0.0
        )

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



# JSON ȘI CSV
def save_detections_json_and_csv(all_detections):
    json_path = (
        OUTPUT_FOLDER
        / "detections_video_sahi_0_40_15s.json"
    )

    csv_path = (
        OUTPUT_FOLDER
        / "detections_video_sahi_0_40_15s.csv"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as json_file:
        json.dump(
            all_detections,
            json_file,
            indent=4,
            ensure_ascii=False
        )

    fieldnames = [
        "video",
        "frame_index",
        "time_seconds",
        "display_id",
        "label",
        "confidence",
        "bbox",
        "area_ratio",
        "area_percent",
        "is_waste",
        "action"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for detection in all_detections:
            writer.writerow(detection)

    print(f"\nJSON salvat la: {json_path}")
    print(f"CSV salvat la:  {csv_path}")


# PROCESAREA UNUI VIDEOCLIP
def process_video(video_path, detection_model):
    print("\n================================")
    print(f"Video: {video_path.name}")
    print("================================")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Nu pot deschide videoclipul.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("FPS invalid. Folosesc 25 FPS.")
        fps = 25.0

    if width <= 0 or height <= 0:
        print("Rezoluția videoclipului este invalidă.")
        cap.release()
        return []

    # Numărul maxim de frame-uri corespunzător celor 15 secunde
    max_frames_for_15_seconds = int(
        fps * MAX_VIDEO_SECONDS
    )

    if total_frames > 0:
        frames_to_write = min(
            total_frames,
            max_frames_for_15_seconds
        )
    else:
        frames_to_write = max_frames_for_15_seconds

    actual_duration = frames_to_write / fps

    print(f"Rezoluție:              {width}x{height}")
    print(f"FPS:                    {fps:.2f}")
    print(f"Frame-uri totale:       {total_frames}")
    print(f"Limită de timp:         {MAX_VIDEO_SECONDS}s")
    print(f"Frame-uri de procesat:  {frames_to_write}")
    print(f"Durată rezultat:        {actual_duration:.2f}s")
    print(
        f"SAHI rulează la fiecare "
        f"{PROCESS_EVERY_N_FRAMES} frame-uri."
    )

    output_video_path = (
        OUTPUT_FOLDER
        / f"{video_path.stem}_sahi_0_40_15s_annotated.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Nu pot crea videoclipul output.")
        cap.release()
        return []

    video_detections = []

    frame_index = 0
    processed_frame_count = 0
    total_detections_for_video = 0

    # Detecțiile ultimului frame analizat cu SAHI
    last_detections = []

    # Afișăm progresul aproximativ o dată pe secundă
    progress_interval = max(1, int(fps))

    while frame_index < frames_to_write:
        ret, frame = cap.read()

        if not ret:
            break

        should_process = (
            frame_index % PROCESS_EVERY_N_FRAMES == 0
        )

        if should_process:
            detections = detect_frame_sahi(
                frame,
                detection_model
            )

            detections = prepare_detections_for_display(
                detections
            )

            last_detections = detections

            processed_frame_count += 1
            total_detections_for_video += len(detections)

            time_seconds = frame_index / fps

            for detection in detections:
                detection_for_file = detection.copy()

                detection_for_file["video"] = video_path.name
                detection_for_file["frame_index"] = frame_index
                detection_for_file["time_seconds"] = round(
                    time_seconds,
                    2
                )

                video_detections.append(
                    detection_for_file
                )

                if PRINT_DETAILED_DETECTIONS:
                    print(
                        f"\n  Detecția "
                        f"#{detection['display_id']}"
                    )
                    print(
                        f"    Clasă:      "
                        f"{detection['label']}"
                    )
                    print(
                        f"    Confidence: "
                        f"{detection['confidence']}"
                    )
                    print(
                        f"    BBox:       "
                        f"{detection['bbox']}"
                    )
                    print(
                        f"    Arie:       "
                        f"{detection['area_percent']}%"
                    )
                    print(
                        f"    Este deșeu: "
                        f"{detection['is_waste']}"
                    )
                    print(
                        f"    Acțiune:    "
                        f"{detection['action']}"
                    )

        # Pe frame-urile neprocesate desenăm ultimele detecții cunoscute
        annotated_frame = draw_detections(
            frame,
            last_detections
        )

        # Scrie frame-ul CU chenare în videoclipul rezultat
        writer.write(annotated_frame)

        if frame_index % progress_interval == 0:
            current_time = frame_index / fps

            print(
                f"Progres: {current_time:.1f}s / "
                f"{actual_duration:.1f}s | "
                f"ultimele detecții: "
                f"{len(last_detections)}"
            )

        frame_index += 1

    cap.release()
    writer.release()

    print(f"\nVideo anotat salvat la:")
    print(output_video_path)

    print(
        f"Frame-uri analizate efectiv cu SAHI: "
        f"{processed_frame_count}"
    )

    print(
        f"Total detecții în frame-urile analizate: "
        f"{total_detections_for_video}"
    )

    return video_detections


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========================================")
    print(" TESTARE VIDEO CU SAHI")
    print(" CONFIDENCE 0.40 — LIMITĂ 15 SECUNDE")
    print("========================================\n")

    print(f"MODEL_PATH:         {MODEL_PATH}")
    print(f"TEST_VIDEOS_FOLDER: {TEST_VIDEOS_FOLDER}")
    print(f"OUTPUT_FOLDER:      {OUTPUT_FOLDER}")

    print(f"\nCONF_THRESHOLD:     {CONF_THRESHOLD}")
    print(f"SLICE_HEIGHT:       {SLICE_HEIGHT}")
    print(f"SLICE_WIDTH:        {SLICE_WIDTH}")

    print(
        f"OVERLAP:            "
        f"{OVERLAP_HEIGHT_RATIO}, "
        f"{OVERLAP_WIDTH_RATIO}"
    )

    print(
        f"PROCESS_EVERY_N:    "
        f"{PROCESS_EVERY_N_FRAMES}"
    )

    print(
        f"MAX_VIDEO_SECONDS:  "
        f"{MAX_VIDEO_SECONDS}"
    )

    if not TEST_VIDEOS_FOLDER.exists():
        print(
            f"\nFolderul nu există: "
            f"{TEST_VIDEOS_FOLDER}"
        )
        print(
            "Creează folderul test_videos "
            "și pune videoclipurile acolo."
        )
        return

    video_paths = sorted(
        path
        for path in TEST_VIDEOS_FOLDER.iterdir()
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_paths:
        print(
            f"\nNu am găsit videoclipuri în folderul: "
            f"{TEST_VIDEOS_FOLDER}"
        )
        return

    # Șterge numai rezultatele vechi ale acestei configurații
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)

    # Creează automat folderul nou
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nFolder output creat:")
    print(OUTPUT_FOLDER)

    print(
        f"\nAm găsit {len(video_paths)} "
        f"videoclipuri pentru test.\n"
    )

    detection_model = load_sahi_model()

    all_detections = []
    class_counter = Counter()
    action_counter = Counter()

    for video_path in video_paths:
        detections = process_video(
            video_path,
            detection_model
        )

        all_detections.extend(detections)

        for detection in detections:
            class_counter[detection["label"]] += 1
            action_counter[detection["action"]] += 1

    save_detections_json_and_csv(all_detections)

    print("\n========================================")
    print(" REZUMAT FINAL VIDEO SAHI")
    print("========================================")

    print(
        f"\nTotal videoclipuri testate: "
        f"{len(video_paths)}"
    )

    print(
        f"Total detecții în frame-urile analizate: "
        f"{len(all_detections)}"
    )

    print("\nDetecții pe clase:")

    if class_counter:
        for label, count in class_counter.most_common():
            print(f"  {label}: {count}")
    else:
        print("  Nicio clasă detectată.")

    print("\nAcțiuni generate:")

    if action_counter:
        for action, count in action_counter.most_common():
            print(f"  {action}: {count}")
    else:
        print("  Nicio acțiune generată.")

    print("\nVideoclipurile anotate se găsesc în:")
    print(OUTPUT_FOLDER)

    print("\nTest video SAHI terminat.")


if __name__ == "__main__":
    main()