from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("C:/myprograms/SL/yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="C:/Users/nagas/Downloads/mysearch.v1i.yolov11/data.yaml", epochs=40, imgsz=640 ,batch=0.5)   