# Edge AI Plastic Waste Detector

A real-time computer-vision system that detects floating plastic waste, decides what to do, drives actuators on a Raspberry Pi (servo / LED / buzzer), and logs every event for a live dashboard. The YOLOv8 model runs on a laptop (the "brain"); the Raspberry Pi performs the physical action (the "hands").

## What It Does

A camera watches the water. The AI classifies each object as **waste** or **not waste** — only waste drives the hardware. Per frame, the system picks one action:

- **COLLECT** → manageable waste → activate the **servo** (scoops it out).
- **ALARM** → waste too large, or too many objects (heavy pollution) → **LED + buzzer**, servo stays put.
- **STOP** → no waste → do nothing.

Non-waste objects (e.g. `natural-debris`) are recognized only so they can be ignored. Every COLLECT/ALARM event is saved as JSON for the dashboard.

## Pipeline

```
Camera → detection/ (YOLOv8: classify + size) → decision/ (COLLECT/ALARM/STOP)
                                                         │
                ┌────────────────────────────────────────┴───────────────┐
                ▼                                                          ▼
   communication/client.py ──TCP──> Raspberry Pi server.py        live_camera.py → dashboard/
        (sends JSON command)        (servo / LED / buzzer)        (logs events, charts them)
```

Two runnable paths share the same detection + decision core:
1. **Hardware path** — `main.py` → `communication/client.py` → Pi `communication/server.py` drives the servo / LED / buzzer.
2. **Dashboard path** — `live_camera.py` logs events to a `.jsonl` file (or AWS), and the Streamlit `dashboard/` visualizes them.

## Components

| Folder / file | Role |
|---|---|
| `camera/` | Test the webcam and print resolution / FPS. |
| `detection/detector.py` | Fast YOLOv8 detection + object tracking. |
| `detection/detector_sahi.py` | SAHI sliced inference — better on small/distant objects. |
| `decision/decision_logic.py` | Picks one action per frame (largest object, object count). |
| `communication/client.py` | Laptop → Pi over TCP, sends JSON. |
| `communication/server.py` | Pi side: maps `COLLECT`/`ALARM`/`STOP` to servo / LED / buzzer. |
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
# + sahi (SAHI mode), streamlit / plotly / pandas (dashboard)
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

## Notes

- The cloud layer (AWS IoT + DynamoDB, referenced in `cloud/`) is not committed, so AWS publishing is off by default — the local `.jsonl` log and Demo source work without it.
- Dashboard UI text is currently in Romanian.
- This is a proof-of-concept combining computer vision, embedded systems, real-time detection, and cloud logging.
