import cv2
import time
from datetime import datetime

from detection.detector import detect_objects
from decision.decision_logic import decide_action
from communication.client import send_command

from cloud.aws_publisher import publish_event


DEVICE_ID = "plastic-detector-01"
SOURCE = "laptop-ai"

CAMERA_INDEX = 0
PROCESS_EVERY_N_FRAMES = 20

COLLECT_COOLDOWN_SECONDS = 5
ALARM_COOLDOWN_SECONDS = 10

collected_count = 0
alarm_count = 0

last_collect_time = 0
last_alarm_time = 0


def build_event(decision):
    selected = decision.get("selected_detection")

    event = {
        "device_id": DEVICE_ID,
        "source": SOURCE,
        "action": decision["action"],
        "reason": decision["reason"],
        "detections_count": decision["detections_count"],
        "collected_count": collected_count,
        "alarm_count": alarm_count,
        "timestamp": datetime.now().isoformat()
    }

    if selected is not None:
        event["selected_label"] = selected.get("label")
        event["selected_confidence"] = selected.get("confidence")
        event["selected_area_ratio"] = selected.get("area_ratio")
        event["selected_track_id"] = selected.get("track_id")

    return event


def should_send_action(action):
    global last_collect_time, last_alarm_time

    now = time.time()

    if action == "COLLECT":
        if now - last_collect_time >= COLLECT_COOLDOWN_SECONDS:
            last_collect_time = now
            return True
        print("[COOLDOWN] COLLECT ignored")
        return False

    if action == "ALARM":
        if now - last_alarm_time >= ALARM_COOLDOWN_SECONDS:
            last_alarm_time = now
            return True
        print("[COOLDOWN] ALARM ignored")
        return False

    return False


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

frame_count = 0
last_event = None

print("[INFO] AI Plastic Detector started")
print("[INFO] Press ESC to stop")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        detections = detect_objects(frame)
        decision = decide_action(detections)

        action = decision["action"]

        print("\n[DETECTIONS]", detections)
        print("[DECISION]", decision)

        if should_send_action(action):
            if action == "COLLECT":
                collected_count += 1
            elif action == "ALARM":
                alarm_count += 1

            event = build_event(decision)
            last_event = event

            print("[EVENT]", event)
            send_command(event)
            publish_event(event)

    if last_event is not None:
        cv2.putText(
            frame,
            f"Last action: {last_event['action']} | Detected: {last_event['detections_count']} | Collected: {collected_count} | Alarms: {alarm_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("AI Plastic Detector", frame)

    if cv2.waitKey(1) == 27:
        break

stop_event = {
    "device_id": DEVICE_ID,
    "source": SOURCE,
    "action": "STOP",
    "reason": "application_stopped",
    "timestamp": datetime.now().isoformat()
}

send_command(stop_event)

cap.release()
cv2.destroyAllWindows()