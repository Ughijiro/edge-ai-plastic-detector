#nume_rasp = "raspberrypi20261@192.168.1.106"
#pass = "raspberry20261"

import socket
import json

RASPBERRY_PI_IP = "192.168.1.106"  # pune IP-ul real
PORT = 65432

def send_command(data):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((RASPBERRY_PI_IP, PORT))

    message = json.dumps(data)
    client.sendall(message.encode())

    print(f"Sent: {message}")

    client.close()