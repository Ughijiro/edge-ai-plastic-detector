'''import cv2
from detection.detector import detect_objects

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    frame_count += 1

    # Run AI every 20 frames
    if frame_count % 20 == 0:

        detections = detect_objects(frame)

        for detection in detections:
            print(detection)

    cv2.imshow("AI Detection Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()'''


#TEST FOR IMAGES
'''import cv2
import os
from detection.detector import detect_objects
from decision.decision_logic import decide_action

# folderul cu pozele tale
image_folder = "test_images"

# toate pozele (jpg, jpeg, png, webp)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

print(f"Am găsit {len(image_files)} poze pentru test.\n")

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    
    if frame is None:
        print(f"Nu pot citi imaginea: {img_name}")
        continue
    
    print(f"\n=== Test pe: {img_name} ===")
    
    detections = detect_objects(frame)
    
    if not detections:
        print("Nicio detectie de gunoi (bottle/cup/bowl)")
    else:
        for det in detections:
            print(f"Detectat: {det}")
            action = decide_action(det)
            print(f"Acțiune: {action}")
    
    # arată poza (apasă orice tastă pentru următoarea, ESC ca să oprești)
    cv2.imshow("Test Image - apasa tasta", frame)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
print("Test terminat pe toate pozele.")
'''

#TEST FOR VIDEOS
'''import cv2
import os
from detection.detector import detect_objects
from decision.decision_logic import decide_action

video_folder = "test_videos"

video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

print(f"Am găsit {len(video_files)} videoclipuri pentru test.\n")

for video_name in video_files:
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    
    print(f"\n=== Test pe video: {video_name} ===")
    
    seen_tracks = set()        # aici ținem minte obiectele deja procesate
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 5 != 0:      # procesăm fiecare al 5-lea frame
            continue
            
        detections = detect_objects(frame)
        
        for det in detections:
            track_id = det.get("track_id", -1)
            if track_id != -1 and track_id not in seen_tracks:
                seen_tracks.add(track_id)                     # marchez ca procesat O SINGURĂ DATĂ
                print(f"Frame {frame_count} | NOU gunoi detectat (ID {track_id}): {det}")
                action = decide_action(det)
                print(f"Acțiune: {action}")
        
        cv2.imshow(f"Video: {video_name} (apasa q sa opresti)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

print("\nTest pe toate videoclipurile terminat.")'''


#test pe imagini fine tuning 
'''from pathlib import Path
import cv2
from collections import Counter

from detection.detector import detect_objects, get_model_classes, MODEL_PATH


TEST_IMAGES_FOLDER = Path("test_images")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def main():
    print("\n==============================")
    print(" TESTARE MODEL PE IMAGINI")
    print("==============================\n")

    print(f"Model folosit: {MODEL_PATH}")
    print("\nClasele modelului:")
    print(get_model_classes())

    if not TEST_IMAGES_FOLDER.exists():
        print(f"\nFolderul '{TEST_IMAGES_FOLDER}' nu există.")
        print("Creează folderul test_images și pune pozele acolo.")
        return

    image_paths = [
        path for path in TEST_IMAGES_FOLDER.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"\nNu am găsit poze în folderul '{TEST_IMAGES_FOLDER}'.")
        return

    print(f"\nAm găsit {len(image_paths)} imagini pentru test.\n")

    total_detections = 0
    class_counter = Counter()
    action_counter = Counter()

    for image_path in image_paths:
        print("\n--------------------------------")
        print(f"Imagine: {image_path.name}")
        print("--------------------------------")

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Nu pot citi imaginea.")
            continue

        detections = detect_objects(frame)

        if not detections:
            print("Nicio detecție.")
            continue

        for index, detection in enumerate(detections, start=1):
            label = detection["label"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            area_percent = detection["area_percent"]
            is_waste = detection["is_waste"]
            action = detection["action"]

            total_detections += 1
            class_counter[label] += 1
            action_counter[action] += 1

            print(f"\nDetecția #{index}")
            print(f"  Clasă:      {label}")
            print(f"  Confidence: {confidence}")
            print(f"  BBox:       {bbox}")
            print(f"  Arie:       {area_percent}% din imagine")
            print(f"  Este deșeu: {is_waste}")
            print(f"  Acțiune:    {action}")

    print("\n==============================")
    print(" REZUMAT FINAL")
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

    print("\nTest terminat.")


if __name__ == "__main__":
    main()'''

#test fine tuning (deseneaza chenare unde vede obiectele)

'''from pathlib import Path
import cv2
from collections import Counter

from detection.detector import detect_objects, get_model_classes, MODEL_PATH


TEST_IMAGES_FOLDER = Path("test_images")
OUTPUT_FOLDER = Path("runs/annotated_test_images")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_box_color(action):
    """
    Alege culoarea chenarului în funcție de acțiune:
    - verde = COLLECT_SERVO
    - roșu = WARNING_LED_BUZZER
    - gri = IGNORE
    """
    if action == "COLLECT_SERVO":
        return (0, 255, 0)       # verde
    elif action == "WARNING_LED_BUZZER":
        return (0, 0, 255)       # roșu
    else:
        return (180, 180, 180)   # gri


def draw_detections(image, detections):
    """
    Desenează chenare și etichete pe imagine.
    """
    annotated = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        label = detection["label"]
        confidence = detection["confidence"]
        action = detection["action"]
        area_percent = detection["area_percent"]

        color = get_box_color(action)

        # Chenar
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Text pentru etichetă
        text = f"{label} | {confidence:.2f} | {action} | {area_percent:.1f}%"

        # Dimensiune text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        # Fundal text
        text_y = max(y1 - 10, text_height + 10)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_height - 8),
            (x1 + text_width + 6, text_y + baseline - 4),
            color,
            -1
        )

        # Text
        cv2.putText(
            annotated,
            text,
            (x1 + 3, text_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated


def main():
    print("\n==============================")
    print(" TESTARE MODEL PE IMAGINI")
    print("==============================\n")

    print(f"Model folosit: {MODEL_PATH}")
    print("\nClasele modelului:")
    print(get_model_classes())

    if not TEST_IMAGES_FOLDER.exists():
        print(f"\nFolderul '{TEST_IMAGES_FOLDER}' nu există.")
        print("Creează folderul test_images și pune pozele acolo.")
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in TEST_IMAGES_FOLDER.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"\nNu am găsit poze în folderul '{TEST_IMAGES_FOLDER}'.")
        return

    print(f"\nAm găsit {len(image_paths)} imagini pentru test.\n")

    total_detections = 0
    class_counter = Counter()
    action_counter = Counter()

    for image_path in image_paths:
        print("\n--------------------------------")
        print(f"Imagine: {image_path.name}")
        print("--------------------------------")

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Nu pot citi imaginea.")
            continue

        detections = detect_objects(frame)

        if not detections:
            print("Nicio detecție.")

            # salvează și imaginea fără detecții, dacă vrei
            output_path = OUTPUT_FOLDER / image_path.name
            cv2.imwrite(str(output_path), frame)
            print(f"Imagine salvată fără detecții la: {output_path}")
            continue

        for index, detection in enumerate(detections, start=1):
            label = detection["label"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            area_percent = detection["area_percent"]
            is_waste = detection["is_waste"]
            action = detection["action"]

            total_detections += 1
            class_counter[label] += 1
            action_counter[action] += 1

            print(f"\nDetecția #{index}")
            print(f"  Clasă:      {label}")
            print(f"  Confidence: {confidence}")
            print(f"  BBox:       {bbox}")
            print(f"  Arie:       {area_percent}% din imagine")
            print(f"  Este deșeu: {is_waste}")
            print(f"  Acțiune:    {action}")

        # Desenează chenarul pe imagine
        annotated_image = draw_detections(frame, detections)

        # Salvează imaginea rezultată
        output_path = OUTPUT_FOLDER / image_path.name
        cv2.imwrite(str(output_path), annotated_image)

        print(f"\nImagine anotată salvată la: {output_path}")

    print("\n==============================")
    print(" REZUMAT FINAL")
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

    print(f"\nImaginile anotate se găsesc în folderul: {OUTPUT_FOLDER}")
    print("\nTest terminat.")


if __name__ == "__main__":
    main()'''






from pathlib import Path
import cv2
import shutil
from collections import Counter

from detection.detector import detect_objects, get_model_classes, MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parent

TEST_IMAGES_FOLDER = PROJECT_ROOT / "test_images"
OUTPUT_FOLDER = PROJECT_ROOT / "runs" / "annotated_test_images(before SAHI)"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Recomandat pentru imagini aglomerate:
# "id_only"       -> pe imagine apare doar #1, #2, #3 etc.
# "id_label_conf" -> pe imagine apare #1 plastic-bottle 0.34
LABEL_MODE = "id_only"


def get_box_color(action):
    """
    OpenCV folosește culori BGR, nu RGB.

    verde închis = COLLECT_SERVO
    roșu închis  = WARNING_LED_BUZZER
    gri închis   = IGNORE
    """
    if action == "COLLECT_SERVO":
        return (0, 120, 0)
    elif action == "WARNING_LED_BUZZER":
        return (0, 0, 170)
    else:
        return (80, 80, 80)


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
    """
    Sortează detecțiile și le pune id-uri: #1, #2, #3...
    Așa ce vezi pe imagine corespunde cu ce vezi în terminal.
    """
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

        # Chenar obiect
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Dimensiune text
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        # Poziție label
        lx1, ly1, lx2, ly2 = find_label_position(
            x1, y1, x2, y2,
            text_width, text_height,
            image_width, image_height,
            used_label_rects
        )

        used_label_rects.append((lx1, ly1, lx2, ly2))

        # Fundal mai închis pentru text
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
            thickness,
            cv2.LINE_AA
        )

    return annotated


def main():
    print("\n==============================")
    print(" TESTARE MODEL PE IMAGINI")
    print("==============================\n")

    print(f"Model folosit: {MODEL_PATH}")

    print("\nClasele modelului:")
    print(get_model_classes())

    if not TEST_IMAGES_FOLDER.exists():
        print(f"\nFolderul nu există: {TEST_IMAGES_FOLDER}")
        print("Creează folderul test_images și pune pozele acolo.")
        return

    # Șterge rezultatele vechi, ca să nu te încurci
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

    total_detections = 0
    class_counter = Counter()
    action_counter = Counter()

    for image_path in image_paths:
        print("\n--------------------------------")
        print(f"Imagine: {image_path.name}")
        print("--------------------------------")

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Nu pot citi imaginea.")
            continue

        detections = detect_objects(frame)
        detections = prepare_detections_for_display(detections)

        if not detections:
            print("Nicio detecție.")

            output_path = OUTPUT_FOLDER / f"{image_path.stem}_annotated.jpg"
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

        output_path = OUTPUT_FOLDER / f"{image_path.stem}_annotated.jpg"
        success = cv2.imwrite(str(output_path), annotated_image)

        if success:
            print(f"\nImagine anotată salvată la: {output_path}")
        else:
            print(f"\nNu am putut salva imaginea la: {output_path}")

    print("\n==============================")
    print(" REZUMAT FINAL")
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

    print(f"\nImaginile anotate se găsesc în folderul:")
    print(f"{OUTPUT_FOLDER}")

    print("\nTest terminat.")


if __name__ == "__main__":
    main()