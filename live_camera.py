"""
live_camera.py — Monitor LIVE de la camera, conectat la dashboard
=================================================================
Deschide camera, ruleaza detectia (acelasi model si desenare ca test_video_sahi.py),
ia o decizie pe frame (STOP / COLLECT / ALARM, exact ca main.py + decision_logic),
afiseaza starea SERVO / LED / BUZZER si SCRIE fiecare eveniment ca sa apara in dashboard.

Se pune in RADACINA proiectului (langa test_video_sahi.py si folderul models/).

Rulare:
    python live_camera.py

Taste in fereastra:
    q = iesire    s = snapshot    m = comuta fast/sahi

LEGAREA CU DASHBOARD-UL
-----------------------
1) LOG_TO_DASHBOARD = True  -> scrie evenimentele in dashboard/live_events.jsonl.
   In dashboard alegi sursa "Live (local)". Merge pe un singur laptop, fara AWS.
2) PUBLISH_TO_AWS = True    -> trimite si in AWS IoT (prin cloud/aws_publisher.py).
   ATENTIE: aws_publisher.py are caile catre certificate hardcodate pe alt PC;
   trebuie sa pui certificatele tale si sa schimbi acele cai ca sa mearga.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

import test_video_sahi as tv   # model, desenare, decizie per-obiect, clase


# --------------------------------------------------------------------------- #
# Setari
# --------------------------------------------------------------------------- #
CAMERA_INDEX = 1
DETECTION_MODE = "fast"          # "fast" (fluid) sau "sahi" (mai exact, mai lent)
PROCESS_EVERY_N_FRAMES = 3
SERVO_HOLD_SECONDS = 1.5
WINDOW_NAME = "Water waste detection — LIVE"

DEVICE_ID = "plastic-detector-01"
SOURCE = "laptop-ai-live"

# legarea cu dashboard-ul
LOG_TO_DASHBOARD = True
LIVE_EVENTS_PATH = tv.PROJECT_ROOT / "dashboard" / "live_events.jsonl"
PUBLISH_TO_AWS = False           # pune True doar daca ai certificatele AWS configurate

# logica de decizie pe frame (identica cu decision/decision_logic.py de pe cloud)
LARGE_THRESHOLD = 0.15
MAX_OBJECTS_BEFORE_ALARM = 15
COLLECT_COOLDOWN_SECONDS = 5
ALARM_COOLDOWN_SECONDS = 10

SNAPSHOT_DIR = tv.PROJECT_ROOT / "runs" / "live_snapshots"


# --------------------------------------------------------------------------- #
# Detector "fast" (fara slicing) — flux live fluid
# --------------------------------------------------------------------------- #
def load_fast_model():
    return YOLO(str(tv.MODEL_PATH))


def detect_frame_fast(frame, model):
    h, w = frame.shape[:2]
    image_area = w * h
    results = model.predict(source=frame, conf=tv.CONF_THRESHOLD, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_ratio = ((x2 - x1) * (y2 - y1)) / image_area if image_area else 0.0
            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "area_ratio": round(area_ratio, 4),
                "area_percent": round(area_ratio * 100, 2),
                "is_waste": label in tv.WASTE_CLASSES,
                "action": tv.decide_action(label, area_ratio),
            })
    return detections


# --------------------------------------------------------------------------- #
# Decizia pe frame (ca main.py) — doar pe deseuri, ca in pipeline-ul cloud
# --------------------------------------------------------------------------- #
def decide_frame(waste_dets):
    n = len(waste_dets)
    if n == 0:
        return {"action": "STOP", "reason": "no_garbage_detected", "selected": None, "count": 0}
    if n >= MAX_OBJECTS_BEFORE_ALARM:
        return {"action": "ALARM", "reason": "high_pollution_level", "selected": None, "count": n}
    sel = max(waste_dets, key=lambda d: d.get("area_ratio", 0.0))
    if sel.get("area_ratio", 0.0) > LARGE_THRESHOLD:
        return {"action": "ALARM", "reason": "object_too_large", "selected": sel, "count": n}
    return {"action": "COLLECT", "reason": "collectable_garbage_detected", "selected": sel, "count": n}


def build_event(decision, collected_count, alarm_count):
    sel = decision["selected"]
    event = {
        "device_id": DEVICE_ID, "source": SOURCE,
        "action": decision["action"], "reason": decision["reason"],
        "detections_count": decision["count"],
        "collected_count": collected_count, "alarm_count": alarm_count,
        "timestamp": datetime.now().isoformat(),
    }
    if sel is not None:
        event["selected_label"] = sel.get("label")
        event["selected_confidence"] = sel.get("confidence")
        event["selected_area_ratio"] = sel.get("area_ratio")
        event["selected_track_id"] = sel.get("track_id", -1)
    return event


def log_event_local(event):
    LIVE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# AWS optional (se importa doar daca PUBLISH_TO_AWS)
_publish_aws = None
if PUBLISH_TO_AWS:
    try:
        from cloud.aws_publisher import publish_event as _publish_aws
    except Exception as exc:  # noqa: BLE001
        print(f"[AWS] nu pot importa aws_publisher: {exc}")
        _publish_aws = None


# --------------------------------------------------------------------------- #
# Panou stare actuatoare
# --------------------------------------------------------------------------- #
def draw_status_panel(frame, action, servo_active, fps, n_det, mode):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 152), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    alarm = (action == "ALARM")

    def lamp(y, label, on, on_color):
        color = on_color if on else (90, 90, 90)
        cv2.circle(frame, (28, y), 8, color, -1)
        cv2.putText(frame, f"{label}: {'ON' if on else 'off'}", (46, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    lamp(40, "SERVO", servo_active, (0, 180, 0))
    lamp(70, "LED", alarm, (0, 0, 210))
    lamp(100, "BUZZER", alarm, (0, 0, 210))
    cv2.putText(frame, f"{action}  |  obiecte: {n_det}  |  {fps:.0f} FPS  [{mode}]",
                (20, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# Bucla principala
# --------------------------------------------------------------------------- #
def main():
    print("Pornesc camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Nu pot deschide camera (index {CAMERA_INDEX}). Incearca alt index.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mode = DETECTION_MODE
    print(f"Mod detectie: {mode}. Incarc modelul...")
    fast_model = load_fast_model()
    sahi_model = None
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_TO_DASHBOARD:
        print(f"Scriu evenimente pentru dashboard in: {LIVE_EVENTS_PATH}")

    last_detections = []
    last_action = "STOP"
    servo_until = 0.0
    collected_count = 0
    alarm_count = 0
    last_collect_time = 0.0
    last_alarm_time = 0.0
    frame_index = 0
    t_prev = time.time()
    fps = 0.0

    print("Camera pornita. q = iesire, s = snapshot, m = schimba modul.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu mai primesc frame-uri.")
            break

        if frame_index % PROCESS_EVERY_N_FRAMES == 0:
            if mode == "sahi":
                if sahi_model is None:
                    print("Incarc modelul SAHI...")
                    sahi_model = tv.load_sahi_model()
                dets = tv.detect_frame_sahi(frame, sahi_model)
            else:
                dets = detect_frame_fast(frame, fast_model)
            last_detections = tv.prepare_detections_for_display(dets)

            # decizia pe frame se ia DOAR pe deseuri (ca in pipeline-ul cloud)
            waste = [d for d in last_detections if d.get("is_waste")]
            decision = decide_frame(waste)
            last_action = decision["action"]

            now = time.time()
            send = False
            if last_action == "COLLECT" and now - last_collect_time >= COLLECT_COOLDOWN_SECONDS:
                last_collect_time = now
                collected_count += 1
                servo_until = now + SERVO_HOLD_SECONDS
                send = True
            elif last_action == "ALARM" and now - last_alarm_time >= ALARM_COOLDOWN_SECONDS:
                last_alarm_time = now
                alarm_count += 1
                send = True

            if send:
                event = build_event(decision, collected_count, alarm_count)
                if LOG_TO_DASHBOARD:
                    log_event_local(event)
                if PUBLISH_TO_AWS and _publish_aws is not None:
                    _publish_aws(event)
                print("[EVENT]", event["action"], event["reason"],
                      event.get("selected_label", ""))

        servo_active = time.time() < servo_until
        annotated = tv.draw_detections(frame, last_detections)
        annotated = draw_status_panel(
            annotated, last_action, servo_active, fps, len(last_detections), mode)
        cv2.imshow(WINDOW_NAME, annotated)

        t_now = time.time()
        dt = t_now - t_prev
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t_prev = t_now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = SNAPSHOT_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), annotated)
            print(f"Snapshot salvat: {path}")
        elif key == ord("m"):
            mode = "sahi" if mode == "fast" else "fast"
            print(f"Mod detectie: {mode}")
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Inchis. Colectate: {collected_count} // Alarme: {alarm_count}")


if __name__ == "__main__":
    main()