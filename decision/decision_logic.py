LARGE_THRESHOLD = 0.15
TOO_MANY_OBJECTS_THRESHOLD = 15


def decide_action(detections):
    detections_count = len(detections)

    if detections_count == 0:
        return {
            "action": "STOP",
            "reason": "no_garbage_detected",
            "selected_detection": None,
            "detections_count": 0
        }

    # Selectăm cea mai mare detecție din frame
    selected_detection = max(
        detections,
        key=lambda detection: detection.get("area_ratio", 0)
    )

    # Dacă sunt prea multe obiecte detectate, ridicăm alarmă
    if detections_count >= TOO_MANY_OBJECTS_THRESHOLD:
        return {
            "action": "ALARM",
            "reason": "too_many_objects_detected",
            "selected_detection": selected_detection,
            "detections_count": detections_count
        }

    # Dacă obiectul selectat ocupă prea mult din imagine, ridicăm alarmă
    if selected_detection.get("area_ratio", 0) >= LARGE_THRESHOLD:
        return {
            "action": "ALARM",
            "reason": "object_too_large",
            "selected_detection": selected_detection,
            "detections_count": detections_count
        }

    # Altfel, obiectul este considerat colectabil
    return {
        "action": "COLLECT",
        "reason": "collectable_garbage_detected",
        "selected_detection": selected_detection,
        "detections_count": detections_count
    }