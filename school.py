import cv2
import torch
from yolov5 import YOLOv5

# Load the YOLOv5 model
model = YOLOv5("yolov5s.pt", device="cuda" if torch.cuda.is_available() else "cpu")

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()