import torch
import cv2

# YOLOv5モデルの読み込み（PyTorch Hubを使用）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 動画の読み込み
video_path = '/Users/nagas/Downloads/traffic.mp4'  # 入力動画ファイルのパス
cap = cv2.VideoCapture(video_path)

# 動画の出力設定
output_path = R'"C:\Users\students\Downloads\IMG_4493.mp4"'  # 出力動画ファイルのパス
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力ファイル形式
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5モデルを使って信号機を検出
    results = model(frame)

    # 検出結果をPandas DataFrameとして取得
    detections = results.pandas().xyxy[0]

    # 検出された信号機の領域を矩形で囲む
    for index, row in detections.iterrows():
        if row['name'] == 'traffic light':  # YOLOv5のラベルが 'traffic light' の場合
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # 信号機領域を囲む矩形を描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 信号機のラベルを描画
            label = "Traffic Light"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 結果のフレームを動画に書き込み
    out.write(frame)

    # 動画をリアルタイムで表示（必要に応じて）
    cv2.imshow('Traffic Light Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 動画オブジェクトの解放
cap.release()
out.release()
cv2.destroyAllWindows()
