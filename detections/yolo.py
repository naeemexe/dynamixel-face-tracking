import cv2
from ultralytics import YOLO

model_path = 'models/yolov8n.pt'
model = YOLO(model_path)
TARGET_CLASS_ID = 0  # The class ID for 'person' in the standard COCO dataset
CONFIDENCE_THRESHOLD = 0.5 
cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0].item())
            if class_id == TARGET_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    cv2.imshow("YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("YOLOv8", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()