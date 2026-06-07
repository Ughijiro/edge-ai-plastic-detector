import socket
import json
import time
from time import sleep
from gpiozero import LED, Servo, TonalBuzzer
from gpiozero.tones import Tone

HOST = "0.0.0.0"
PORT = 65432

SERVO_GPIO = 17
LED_GPIO = 27
BUZZER_GPIO = 22

# Timing settings
GLOBAL_COOLDOWN = 5.0
COLLECT_COOLDOWN = 5.0
ALARM_COOLDOWN = 6.0
ALARM_DURATION = 3.0

led = LED(LED_GPIO)
buzzer = TonalBuzzer(BUZZER_GPIO)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Raspberry Pi hardware server running...")
print("Waiting for commands...")

last_collect_time = 0
last_alarm_time = 0
last_any_action_time = 0

collected_count = 0


def can_run_action(action):
    global last_collect_time, last_alarm_time, last_any_action_time

    current_time = time.time()

    if current_time - last_any_action_time < GLOBAL_COOLDOWN:
        print(f"{action} ignored: global hardware cooldown active")
        return False

    if action == "COLLECT":
        if current_time - last_collect_time < COLLECT_COOLDOWN:
            print("COLLECT ignored: collect cooldown active")
            return False

        last_collect_time = current_time

    elif action == "ALARM":
        if current_time - last_alarm_time < ALARM_COOLDOWN:
            print("ALARM ignored: alarm cooldown active")
            return False

        last_alarm_time = current_time

    last_any_action_time = current_time
    return True


def move_servo():
    global collected_count

    print("Servo activated")

    servo = Servo(SERVO_GPIO)

    servo.min()
    sleep(0.7)

    servo.max()
    sleep(0.7)

    servo.mid()
    sleep(0.5)

    servo.detach()
    servo.close()

    collected_count += 1

    print("Servo action finished")
    print(f"Collected garbage count: {collected_count}")


def activate_alarm():
    print("Alarm activated")

    led.on()
    buzzer.play(Tone("A4"))

    sleep(ALARM_DURATION)

    buzzer.stop()
    led.off()

    print("Alarm finished")


try:
    while True:
        conn, addr = server.accept()
        print(f"Connected by {addr}")

        data = conn.recv(1024)

        if data:
            event = json.loads(data.decode())
            action = event.get("action")

            print("Received event:", event)

            if action == "COLLECT":
                if can_run_action("COLLECT"):
                    move_servo()

            elif action == "ALARM":
                if can_run_action("ALARM"):
                    activate_alarm()

            elif action == "STATUS":
                print(f"Collected garbage count: {collected_count}")

            elif action == "STOP":
                print("Stopping server...")
                break

        conn.close()

finally:
    led.off()
    buzzer.stop()

    led.close()
    buzzer.close()

    server.close()

    print(f"Final collected garbage count: {collected_count}")
    print("Hardware server closed")