from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == '__main__':
    freeze_support()  # Windowsでマルチプロセスを使用するために必要
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model.train(data=R'C:\Users\students\Documents\GitHub\SL\mysearch-1\mysearch-1\valid\images', epochs=100, imgsz=640 ) #device=0 GPUの指定 CPUの場合は削除する