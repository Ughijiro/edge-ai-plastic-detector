LARGE_THRESHOLD = 0.15
MAX_OBJECTS_BEFORE_ALARM = 15


def select_main_detection(detections):
    """
    Selects the most relevant detection from the current processed frame.
    For now, the largest object is selected using area_ratio.
    """
    if not detections:
        return None

    return max(detections, key=lambda detection: detection.get("area_ratio", 0.0))


def decide_action(detections):
    """
    Decides one action for all detections from the current processed frame.

    Rules:
    - no detections -> STOP
    - too many detections -> ALARM
    - selected object too large -> ALARM
    - otherwise -> COLLECT
    """
    detections_count = len(detections)

    if detections_count == 0:
        return {
            "action": "STOP",
            "reason": "no_garbage_detected",
            "selected_detection": None,
            "detections_count": detections_count
        }

    if detections_count >= MAX_OBJECTS_BEFORE_ALARM:
        return {
            "action": "ALARM",
            "reason": "high_pollution_level",
            "selected_detection": None,
            "detections_count": detections_count
        }

    selected_detection = select_main_detection(detections)

    if selected_detection.get("area_ratio", 0.0) > LARGE_THRESHOLD:
        return {
            "action": "ALARM",
            "reason": "object_too_large",
            "selected_detection": selected_detection,
            "detections_count": detections_count
        }

    return {
        "action": "COLLECT",
        "reason": "collectable_garbage_detected",
        "selected_detection": selected_detection,
        "detections_count": detections_count
    }