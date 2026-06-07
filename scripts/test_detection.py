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
print("Test terminat pe toate pozele.")'''


#TEST FOR VIDEOS
import cv2
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

print("\nTest pe toate videoclipurile terminat.")