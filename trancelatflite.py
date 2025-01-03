from ultralytics import YOLO

model = YOLO("C:/Users/students/Documents/GitHub/SL/train15/weights/best.pt")
model.export(format="tflite")
