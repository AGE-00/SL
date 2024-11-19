from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("C:/myprograms/SL/yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="C:/Myprograms/SL/mysearch-1/data.yaml", epochs=100, imgsz=640)