import cv2

camera_index = 1

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Webcam started successfully")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    cv2.imshow("Edge AI Plastic Detector", frame)

    # ESC key to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()