from ultralytics import YOLO
path="your_model_path"
model = YOLO(path)
model.export(format="tflite")
tflite_model = YOLO("yoloxx.tflite")