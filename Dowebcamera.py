import cv2
from ultralytics import YOLO

# YOLOv8モデルのロード (最適なモデルを選択)
model = YOLO("C:/myprograms/SL/runs/detect/train25/weights/best.pt")

# カメラキャプチャの初期化
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read(0)  # カメラからフレームを読み込む
    if not ret:
        break

    # YOLOv8モデルを使って物体検出
    results = model(frame)
    
    # 検出結果をフレームに描画
    annotated_frame = results[0].plot()
    
    # フレームを表示
    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)
    
    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
