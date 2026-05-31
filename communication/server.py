import socket
import json

HOST = "0.0.0.0"
PORT = 65432

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Raspberry Pi server running...")
print("Waiting for commands...")

while True:
    conn, addr = server.accept()
    print(f"Connected by {addr}")

    data = conn.recv(1024)

    if data:
        message = data.decode()
        event = json.loads(message)

        action = event.get("action")

        print("Received event:", event)

        if action == "COLLECT":
            print("Servo activated")

        elif action == "ALARM":
            print("Alarm activated")

        elif action == "STOP":
            print("Stopping server...")
            conn.close()
            break

    conn.close()

server.close()