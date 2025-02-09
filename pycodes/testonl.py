from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("C:/myprograms/SL/yolo11n.pt")
    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="C:/Users/nagas/Downloads/mysearch.v3-mysearch.yolov11/data.yaml", epochs=100, imgsz=640 ,device=0)

if __name__ == '__main__':
    freeze_support()  # フリーズ時に必要
    main()