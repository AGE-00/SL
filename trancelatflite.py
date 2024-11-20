from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("C:/Users/students/Documents/GitHub/SL/train15/weights/best.pt")

results = model(0,show=True,conf=0.5) 