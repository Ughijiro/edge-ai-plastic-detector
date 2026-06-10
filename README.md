# Edge AI Plastic Waste Detector

A real-time computer-vision system that detects floating plastic waste, decides what to do, drives actuators on a Raspberry Pi (servo / LED / buzzer), and logs every event to AWS for a live dashboard. The YOLOv8 model runs on a laptop (the "brain"); the Raspberry Pi performs the physical action (the "hands").

## What It Does

A camera watches the water. The AI classifies each object as **waste** or **not waste** — only waste drives the hardware. Per frame, the system picks one action:

- **COLLECT** → manageable waste → activate the **servo** (scoops it out).
- **ALARM** → waste too large, or too many objects (heavy pollution) → **LED + buzzer**, servo stays put.
- **STOP** → no waste → do nothing.

Non-waste objects (e.g. `natural-debris`) are recognized only so they can be ignored. Every COLLECT/ALARM event is saved as JSON for the dashboard.

## Pipeline

**Step 1 — See:** the camera captures frames of the water.

**Step 2 — Detect:** `detection/` runs YOLOv8 on each frame, classifying every object as waste or not and measuring its size.

**Step 3 — Decide:** `decision/` looks at the waste and picks ONE action for the frame:

```
   COLLECT  →  small, manageable waste
   ALARM    →  waste too large, or too many objects
   STOP     →  no waste
```

**Step 4 — Act:** the action is sent down one (or both) of these paths:

```
                        ┌─────────────────────────────┐
                        │   Step 3: decide the action  │
                        │   COLLECT / ALARM / STOP      │
                        └───────────────┬──────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                │                                                 │
                ▼                                                 ▼
     ━━━ HARDWARE PATH ━━━                            ━━━ DASHBOARD PATH ━━━
     (act in the real world)                          (record & visualize)

   communication/client.py                            live_camera.py
   sends the action by Wi-Fi                           saves the event
            │                                                 │
            ▼                                          ┌───────┴────────┐
   Raspberry Pi: server.py                             ▼                ▼
            │                                    local .jsonl      AWS IoT
            ▼                                    file               │
   COLLECT → servo moves                              │             ▼
   ALARM  → LED + buzzer                               │          DynamoDB
                                                       │             │
                                                       └──────┬──────┘
                                                              ▼
                                                    dashboard/ (Streamlit)
                                                    charts, KPIs, event log
```

In short: the laptop is the **brain** (Steps 1–3). The **hardware path** lets the Raspberry Pi physically react, and the **dashboard path** records every event so you can see what happened.

Two runnable paths share the same detection + decision core:
1. **Hardware path** — `main.py` → `communication/client.py` → Pi `communication/server.py` drives the servo / LED / buzzer.
2. **Dashboard path** — `live_camera.py` logs events to a local `.jsonl` file and/or publishes to AWS IoT; the Streamlit `dashboard/` visualizes them (from Demo, the local file, or DynamoDB).

## Components

| Folder / file | Role |
|---|---|
| `camera/` | Test the webcam and print resolution / FPS. |
| `detection/detector.py` | Fast YOLOv8 detection + object tracking. |
| `detection/detector_sahi.py` | SAHI sliced inference — better on small/distant objects. |
| `decision/decision_logic.py` | Picks one action per frame (largest object, object count). |
| `communication/client.py` | Laptop → Pi over TCP, sends JSON. |
| `communication/server.py` | Pi side: maps `COLLECT`/`ALARM`/`STOP` to servo / LED / buzzer. |
| `cloud/aws_publisher.py` | Publishes events to AWS IoT Core over MQTT (topic `plastic-detector/events`). |
| `cloud/dynamodb_reader.py` | Reads stored events back from the DynamoDB table `PlasticDetectorEvents`. |
| `main.py` | Entry point for the hardware path. |
| `live_camera.py` | Entry point for the dashboard path (status panel, event logging). |
| `dashboard/` | Streamlit app + data layer (KPIs, charts, event log, CSV export). |
| `train.py` | Trains the YOLOv8 model on a Roboflow floating-waste dataset. |

**Waste classes:** `plastic-bottle`, `plastic-bag`, `plastic-cup`, `plastic-wrapper`, `plastic-other`, `can`, `carton`, `foam`, `other-waste`. Size/pollution thresholds are configurable constants at the top of the relevant files.

## Event JSON

```json
{
  "action": "COLLECT",
  "reason": "collectable_garbage_detected",
  "detections_count": 3,
  "selected_label": "plastic-bottle",
  "selected_confidence": 0.91,
  "selected_area_ratio": 0.08,
  "timestamp": "2026-06-10T15:30:12"
}
```

## Getting Started

```bash
pip install -r requirements.txt    # opencv-python, ultralytics, torch, torchvision
# extras: sahi (SAHI mode); streamlit / plotly / pandas (dashboard);
#         awsiotsdk + boto3 (AWS publishing + DynamoDB)
```

1. Put trained weights at `models/best.pt` (git-ignored — train with `train.py` or add your own).
2. Camera index defaults to `1`; change to `0` if the webcam won't open. Verify with `python camera/camera_test.py`.

**Hardware path:**
```bash
python communication/server.py   # on the Raspberry Pi
python main.py                    # on the laptop (set RASPBERRY_PI_IP in client.py)
```

**Dashboard path (no hardware needed):**
```bash
python live_camera.py             # produce live events
streamlit run dashboard/App_.py   # view them (pick "Live (local)", or "Demo" for sample data)
```

## Cloud (AWS)

The cloud layer uses **AWS IoT Core** (MQTT) for ingest and **DynamoDB** (`PlasticDetectorEvents`, region `eu-north-1`) for storage. To enable it: set `PUBLISH_TO_AWS = True` in `live_camera.py`, and in `cloud/aws_publisher.py` update the certificate paths (`CERT_PATH`, `PRIVATE_KEY_PATH`, `ROOT_CA_PATH`) to your own AWS IoT credentials — the current paths are hardcoded to one machine. Without this, the local `.jsonl` log and the Demo source work fine.

## Notes

- This is a proof-of-concept combining computer vision, embedded systems, real-time detection, and cloud logging.
