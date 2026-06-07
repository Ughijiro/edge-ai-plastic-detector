#nume_rasp = "raspberrypi20261@192.168.1.106"
#pass = "raspberry20261"

import socket
import json

RASPBERRY_PI_IP = "192.168.1.106"
PORT = 65432
TIMEOUT_SECONDS = 3


def send_command(data):
    """
    Sends a JSON event to the Raspberry Pi server.
    If the Raspberry Pi is offline, the laptop AI loop continues running.
    """
    try:
        message = json.dumps(data)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(TIMEOUT_SECONDS)
            client.connect((RASPBERRY_PI_IP, PORT))
            client.sendall(message.encode("utf-8"))

        print(f"[SOCKET] Sent: {message}")
        return True

    except ConnectionRefusedError:
        print("[SOCKET] Raspberry Pi server refused the connection. Is server.py running?")
        return False

    except TimeoutError:
        print("[SOCKET] Connection timed out. Check Raspberry Pi IP/network.")
        return False

    except OSError as error:
        print(f"[SOCKET] Could not send event: {error}")
        return False