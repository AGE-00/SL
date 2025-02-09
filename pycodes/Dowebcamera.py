import cv2
from ultralytics import YOLO
path="your_model_path"
model = YOLO(path)
video_path =0
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read(0)  # カメラからフレームを読み込む
    if not ret:
        break

    results = model(frame)

    # 検出結果をフレームに描画
    annotated_frame = results[0].plot()

    # フレームを表示
    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
