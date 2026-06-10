from pathlib import Path
from collections import Counter
import cv2
import shutil

from detection.detector import (
    detect_objects,
    get_model_classes,
    MODEL_PATH
)



# PATH
PROJECT_ROOT = Path(__file__).resolve().parent

TEST_VIDEOS_FOLDER = PROJECT_ROOT / "test_videos"

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "runs"
    / "annotated_test_videos_no_sahi"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}



# CONFIGURAȚIE VIDEO
# Se procesează maximum primele 15 secunde din fiecare videoclip
MAX_VIDEO_SECONDS = 15

# 1 = detecție pe fiecare frame
# 2 = detecție o dată la două frame-uri
# 5 = detecție o dată la cinci frame-uri
PROCESS_EVERY_N_FRAMES = 5

# True = deschide și fereastra cu videoclipul în timpul procesării
# False = doar salvează videoclipul rezultat
SHOW_PREVIEW = False

# "id_only" -> #1, #2, #3
# "id_label_conf" -> #1 plastic-bottle 0.74
LABEL_MODE = "id_label_conf"



# CULORI ȘI DESENARE

def get_box_color(action):
    """
    OpenCV folosește formatul BGR.

    COLLECT_SERVO:
        verde închis

    WARNING_LED_BUZZER:
        roșu închis

    REVIEW_ONLY:
        portocaliu

    IGNORE:
        gri închis
    """

    if action == "COLLECT_SERVO":
        return 0, 120, 0

    if action == "WARNING_LED_BUZZER":
        return 0, 0, 170

    if action == "REVIEW_ONLY":
        return 0, 140, 180

    return 80, 80, 80


def rectangles_overlap(rect1, rect2):
    """
    Verifică dacă două dreptunghiuri se suprapun.

    Dreptunghiurile au forma:
    (x1, y1, x2, y2)
    """

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
    Păstrează dreptunghiul etichetei în interiorul imaginii.
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
    Încearcă să plaseze eticheta fără să se suprapună
    cu etichetele deja desenate.
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

        # În dreapta bounding box-ului
        clamp_label_rectangle(
            x2 + 4,
            y1,
            label_width,
            label_height,
            image_width,
            image_height
        ),

        # În stânga bounding box-ului
        clamp_label_rectangle(
            x1 - label_width - 4,
            y1,
            label_width,
            label_height,
            image_width,
            image_height
        )
    ]

    for candidate in candidates:
        overlaps = any(
            rectangles_overlap(candidate, used_rect)
            for used_rect in used_rects
        )

        if not overlaps:
            return candidate

    # Dacă toate pozițiile se suprapun, folosim prima variantă
    return candidates[0]


def prepare_detections_for_display(detections):
    """
    Sortează detecțiile de sus în jos și de la stânga la dreapta,
    apoi adaugă un ID vizual pentru frame-ul curent.
    """

    sorted_detections = sorted(
        detections,
        key=lambda detection: (
            detection["bbox"][1],
            detection["bbox"][0]
        )
    )

    prepared_detections = []

    for display_id, detection in enumerate(
        sorted_detections,
        start=1
    ):
        prepared_detection = detection.copy()
        prepared_detection["display_id"] = display_id
        prepared_detections.append(prepared_detection)

    return prepared_detections


def get_label_text(detection):
    display_id = detection["display_id"]
    label = detection.get("label", "unknown")
    confidence = float(detection.get("confidence", 0))

    if LABEL_MODE == "id_label_conf":
        return f"#{display_id} {label} {confidence:.2f}"

    return f"#{display_id}"


def draw_detections(frame, detections):
    """
    Desenează bounding box-urile și etichetele pe frame.
    """

    annotated_frame = frame.copy()

    image_height, image_width = annotated_frame.shape[:2]

    used_label_rects = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    text_thickness = 1
    box_thickness = 2
    padding = 4

    for detection in detections:
        bbox = detection.get("bbox")

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Păstrăm coordonatele în interiorul imaginii
        x1 = max(0, min(int(x1), image_width - 1))
        y1 = max(0, min(int(y1), image_height - 1))
        x2 = max(0, min(int(x2), image_width - 1))
        y2 = max(0, min(int(y2), image_height - 1))

        action = detection.get("action", "IGNORE")
        color = get_box_color(action)

        text = get_label_text(detection)

        # Desenează bounding box
        cv2.rectangle(
            annotated_frame,
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

        # Fundal închis pentru etichetă
        cv2.rectangle(
            annotated_frame,
            (lx1, ly1),
            (lx2, ly2),
            color,
            -1
        )

        text_x = lx1 + padding
        text_y = ly1 + text_height + padding

        # Text alb
        cv2.putText(
            annotated_frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

    return annotated_frame



# PROCESAREA UNUI VIDEOCLIP
def process_video(video_path):
    print("\n========================================")
    print(f"Video: {video_path.name}")
    print("========================================")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print("Nu pot deschide videoclipul.")
        return Counter(), Counter(), 0

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("FPS invalid. Folosesc valoarea implicită 25 FPS.")
        fps = 25.0

    if width <= 0 or height <= 0:
        print("Rezoluția videoclipului nu poate fi citită.")
        capture.release()
        return Counter(), Counter(), 0

    maximum_frames = int(MAX_VIDEO_SECONDS * fps)

    if total_frames > 0:
        frames_to_process = min(total_frames, maximum_frames)
    else:
        frames_to_process = maximum_frames

    duration_to_process = frames_to_process / fps

    print(f"Rezoluție:            {width}x{height}")
    print(f"FPS:                  {fps:.2f}")
    print(f"Frame-uri originale:  {total_frames}")
    print(f"Frame-uri procesate:  maximum {frames_to_process}")
    print(f"Durată procesată:     maximum {duration_to_process:.2f}s")
    print(
        f"Detecție la fiecare:  "
        f"{PROCESS_EVERY_N_FRAMES} frame-uri"
    )

    output_path = (
        OUTPUT_FOLDER
        / f"{video_path.stem}_no_sahi_annotated.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print(f"Nu pot crea videoclipul rezultat: {output_path}")
        capture.release()
        return Counter(), Counter(), 0

    frame_index = 0
    processed_detection_frames = 0
    total_detections = 0

    class_counter = Counter()
    action_counter = Counter()

    # Folosit doar dacă PROCESS_EVERY_N_FRAMES este mai mare decât 1
    last_detections = []

    # Afișăm progresul aproximativ o dată pe secundă
    progress_interval = max(1, int(fps))

    while frame_index < frames_to_process:
        success, frame = capture.read()

        if not success:
            break

        should_detect = (
            frame_index % PROCESS_EVERY_N_FRAMES == 0
        )

        if should_detect:
            detections = detect_objects(frame)

            detections = prepare_detections_for_display(
                detections
            )

            last_detections = detections
            processed_detection_frames += 1
            total_detections += len(detections)

            for detection in detections:
                label = detection.get("label", "unknown")
                action = detection.get("action", "UNKNOWN")

                class_counter[label] += 1
                action_counter[action] += 1

        annotated_frame = draw_detections(
            frame,
            last_detections
        )

        writer.write(annotated_frame)

        if frame_index % progress_interval == 0:
            current_second = frame_index / fps

            print(
                f"Progres: {current_second:.1f}s / "
                f"{duration_to_process:.1f}s | "
                f"Detecții pe ultimul frame analizat: "
                f"{len(last_detections)}"
            )

        if SHOW_PREVIEW:
            cv2.imshow(
                f"Detectare fără SAHI - {video_path.name}",
                annotated_frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Q sau ESC oprește videoclipul curent
            if key == ord("q") or key == 27:
                print("Procesarea videoclipului a fost oprită.")
                break

        frame_index += 1

    capture.release()
    writer.release()

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print(f"\nVideoclip salvat la:")
    print(output_path)

    print(
        f"Frame-uri pe care s-a rulat modelul: "
        f"{processed_detection_frames}"
    )

    print(
        f"Total detecții în frame-urile analizate: "
        f"{total_detections}"
    )

    return class_counter, action_counter, total_detections



# MAIN
def main():
    print("\n========================================")
    print(" TESTARE VIDEO FĂRĂ SAHI")
    print("========================================\n")

    print(f"Model folosit: {MODEL_PATH}")

    print("\nClasele modelului:")
    print(get_model_classes())

    print(f"\nFolder videoclipuri:")
    print(TEST_VIDEOS_FOLDER)

    print(f"\nFolder rezultate:")
    print(OUTPUT_FOLDER)

    print(
        f"\nLimită pentru fiecare videoclip: "
        f"{MAX_VIDEO_SECONDS} secunde"
    )

    if not TEST_VIDEOS_FOLDER.exists():
        print("\nFolderul test_videos nu există.")
        print("Creează-l și pune videoclipurile în el.")
        return

    video_paths = sorted(
        path
        for path in TEST_VIDEOS_FOLDER.iterdir()
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_paths:
        print("\nNu am găsit videoclipuri în test_videos.")
        return

    # Șterge numai rezultatele vechi ale acestui test
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nFolder output creat:")
    print(OUTPUT_FOLDER)

    print(
        f"\nAm găsit {len(video_paths)} "
        f"videoclipuri pentru test."
    )

    total_class_counter = Counter()
    total_action_counter = Counter()
    total_detections = 0

    for video_path in video_paths:
        class_counter, action_counter, video_detections = (
            process_video(video_path)
        )

        total_class_counter.update(class_counter)
        total_action_counter.update(action_counter)
        total_detections += video_detections

    print("\n========================================")
    print(" REZUMAT FINAL — VIDEO FĂRĂ SAHI")
    print("========================================")

    print(f"\nVideoclipuri testate: {len(video_paths)}")
    print(f"Total detecții:       {total_detections}")

    print("\nDetecții pe clase:")

    if total_class_counter:
        for label, count in total_class_counter.most_common():
            print(f"  {label}: {count}")
    else:
        print("  Nicio detecție.")

    print("\nAcțiuni generate:")

    if total_action_counter:
        for action, count in total_action_counter.most_common():
            print(f"  {action}: {count}")
    else:
        print("  Nicio acțiune.")

    print("\nVideoclipurile rezultate sunt în:")
    print(OUTPUT_FOLDER)

    print("\nTest terminat.")


if __name__ == "__main__":
    main()