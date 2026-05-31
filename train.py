import os
from ultralytics import YOLO

# debug ca să vedem exact
print("Director curent:", os.getcwd())
print("Folder dataset există?", os.path.exists('floating_waste'))

model = YOLO('yolov8n.pt')

data_path = 'floating_waste/data.yaml'

print("Caut data.yaml la:", data_path)
print("Există data.yaml?", os.path.exists(data_path))

model.train(
    data=data_path,
    epochs=20,
    imgsz=640,
    batch=8,
    name='plastic_detector_finetuned',
    device='cpu'
)

print("Antrenarea s-a terminat! Modelul e în runs/detect/plastic_detector_finetuned/weights/best.pt")