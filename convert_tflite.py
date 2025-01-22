from ultralytics import YOLO
model = YOLO("C:/myprograms/SL/runs/detect/train25/weights/best.pt")
model.export(format="tflite")
tflite_model = YOLO("yolo11n_float32.tflite")