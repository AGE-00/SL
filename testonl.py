import ultralytics

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("mysearch-1/train/images/-2024-08-19-105836_png.rf.27f2393f0eda22d996b15d05d7e2a945.jpg")