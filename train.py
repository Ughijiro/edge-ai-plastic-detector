from ultralytics import YOLO

# Incarcă modelul de bază (poți schimba cu yolov8m.pt sau yolov8l.pt pentru mai multă precizie)
model = YOLO("yolov8n.pt")  

# Calea către data.yaml-ul exportat din Roboflow (schimbă cu locația ta exactă)
data_yaml_path = "floating-waste-v4/data.yaml"  # sau numele folderului tău

# Antrenare
results = model.train(
    data=data_yaml_path,
    epochs=100,           # crește la 150-200 dacă ai timp/GPU puternic
    imgsz=640,            # 1280 dacă vrei mai bine pe obiecte mici/departe (necesită mai mult VRAM)
    batch=16,             # ajustează după memoria GPU (8 sau 32)
    name="floating_waste_v4",
    patience=50,          # early stopping
    augment=True,         # activează augmentările Roboflow/YOLO
    degrees=15,           # rotații utile pentru apă
    translate=0.1,
    scale=0.7,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0
)

print("Antrenarea s-a terminat. Cel mai bun model este salvat în runs/detect/floating_waste_v4/weights/best.pt")