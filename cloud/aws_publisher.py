import json
from awscrt import mqtt
from awsiot import mqtt_connection_builder

AWS_IOT_ENDPOINT = "a4zx6wl5nemnb-ats.iot.eu-north-1.amazonaws.com"
CLIENT_ID = "plastic-detector-laptop-client"
TOPIC = "plastic-detector/events"

CERT_PATH = r"C:/Users/tamas/Desktop/aws_cert/335d0cba3ed33e014e44f544d4570611210040db227428b408c2a21cad130da3-certificate.pem.crt"
PRIVATE_KEY_PATH = r"C:/Users/tamas/Desktop/aws_cert/335d0cba3ed33e014e44f544d4570611210040db227428b408c2a21cad130da3-private.pem.key"
ROOT_CA_PATH = r"C:/Users/tamas/Desktop/aws_cert/AmazonRootCA1.pem"

mqtt_connection = None


def connect_to_aws():
    global mqtt_connection

    if mqtt_connection is not None:
        return mqtt_connection

    print("[AWS] Connecting to AWS IoT Core...")

    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=AWS_IOT_ENDPOINT,
        cert_filepath=CERT_PATH,
        pri_key_filepath=PRIVATE_KEY_PATH,
        ca_filepath=ROOT_CA_PATH,
        client_id=CLIENT_ID,
        clean_session=False,
        keep_alive_secs=30
    )

    connect_future = mqtt_connection.connect()
    connect_future.result()

    print("[AWS] Connected to AWS IoT Core")
    return mqtt_connection


def publish_event(event):
    try:
        connection = connect_to_aws()

        payload = json.dumps(event)

        publish_future, _ = connection.publish(
            topic=TOPIC,
            payload=payload,
            qos=mqtt.QoS.AT_LEAST_ONCE
        )

        publish_future.result()

        print(
            f"[AWS] Published: action={event.get('action')}, "
            f"label={event.get('selected_label')}, "
            f"confidence={event.get('selected_confidence')}"
        )

        return True

    except Exception as error:
        print(f"[AWS] Failed to publish event: {error}")
        return False


def disconnect_from_aws():
    global mqtt_connection

    if mqtt_connection is not None:
        print("[AWS] Disconnecting...")
        disconnect_future = mqtt_connection.disconnect()
        disconnect_future.result()
        mqtt_connection = None
        print("[AWS] Disconnected")