from datetime import datetime
from cloud.aws_publisher import publish_event, disconnect_from_aws

event = {
    "device_id": "plastic-detector-01",
    "source": "python-test",
    "action": "COLLECT",
    "reason": "python_mqtt_test",
    "detections_count": 1,
    "collected_count": 1,
    "alarm_count": 0,
    "selected_label": "bottle",
    "selected_confidence": 0.87,
    "selected_area_ratio": 0.09,
    "selected_track_id": 1,
    "timestamp": datetime.now().isoformat()
}

publish_event(event)
disconnect_from_aws()