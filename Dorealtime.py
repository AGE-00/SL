import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path =1 # 本体に付属のカメラを指定
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
	success, frame = cap.read()
	if success:
		# 推論を実行
		results = model(frame)

		# キャプチャした画像サイズを取得
		imageWidth = results[0].orig_shape[0]
		imageHeight = results[0].orig_shape[1]

		# 後のオブジェクト名出力などのため
		names = results[0].names
		classes = results[0].boxes.cls
		boxes = results[0].boxes
		annotatedFrame = results[0].plot()

		# 検出したオブジェクトのバウンディングボックス座標とオブジェクト名を取得し、ターミナルに出力
		for box, cls in zip(boxes, classes):
			name = names[int(cls)]
			x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
			print(f"Object: {name} Coordinates: StartX={x1}, StartY={y1}, EndX={x2}, EndY={y2}")
			# バウンディングBOXの座標情報を書き込む
			cv2.putText(annotatedFrame, f"{x1} {y1} {x2} {y2}", (x1, y1 - 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2, cv2.LINE_AA)

		# プレビューウィンドウに画像出力
		cv2.imshow("YOLOv8 Inference", annotatedFrame)
	
		# アプリケーション終了
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break	
	else: 
		# キャプチャに失敗したら終了
		break
        
# 終了処理
cap.release()
cv2.destroyAllWindows()

##credit https://zenn.dev/collabostyle/articles/3d0f210c59679e
