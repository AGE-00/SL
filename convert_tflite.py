from ultralytics import YOLO
import tensorflow

# Load the YOLO11 model
model = YOLO("C:/Myprograms/SL/runs/detect/train25/weights/best.pt")

# Export the model to TFLite format
model.export(format="tflite",dynamic=True,int8=True)  # creates 'yolo11n_float32.tflite'