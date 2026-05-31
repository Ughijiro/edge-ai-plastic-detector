import cv2
from datetime import datetime
from detection.detector import detect_objects
from decision.decision_logic import decide_action
from communication.client import send_command


cap = cv2.VideoCapture(1)

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

    if frame_count % 20 == 0:
        detections = detect_objects(frame)

        for detection in detections:
            action = decide_action(detection)

            print("Detection:", detection)
            print("Action:", action)

            if action != "IGNORE":
                event = {
                    "action": action,
                    "label": detection["label"],
                    "confidence": detection["confidence"],
                    "timestamp": datetime.now().isoformat()
                }

                send_command(event)

    cv2.imshow("AI Plastic Detector", frame)

    if cv2.waitKey(1) == 27:
        break
    
send_command({
    "action": "STOP",
    "timestamp": datetime.now().isoformat()
})

cap.release()
cv2.destroyAllWindows()