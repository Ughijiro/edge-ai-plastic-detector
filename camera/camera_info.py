import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print("Width:", width)
print("Height:", height)
print("FPS:", fps)

cap.release()