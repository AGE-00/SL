import torch
import cv2
import numpy as np

# YOLOv5モデルの読み込み（PyTorch Hubを使用）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 画像を読み込む
image = cv2.imread('/Users/nagas/Downloads/traffiic.jpg')

# YOLOv5モデルを使って信号機を検出
results = model(image)

# 検出結果をPandas DataFrameとして取得
detections = results.pandas().xyxy[0]

# 検出された信号機の領域を抽出
for index, row in detections.iterrows():
    if row['name'] == 'traffic light':  # YOLOv5のラベルが 'traffic light' の場合
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # 信号機領域の切り抜き
        roi = image[y1:y2, x1:x2]
        
        # 切り抜いた領域で色フィルタリング（例として赤色フィルタを適用）
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_lower = np.array([0, 70, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv_roi, red_lower, red_upper)
        
        # フィルタされた結果をマージ
        filtered = cv2.bitwise_and(roi, roi, mask=red_mask)
        
        # 結果を元の画像に反映
        image[y1:y2, x1:x2] = filtered
        
        # 信号機領域を囲む矩形を描画 (太い線に変更)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # 線の太さを4に設定

        # 信号機のラベルを描画
        label = "Traffic Light"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 結果を表示
cv2.imshow('Detected Traffic Lights with Color Filtering', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
