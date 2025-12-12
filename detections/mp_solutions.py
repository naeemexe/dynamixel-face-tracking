import cv2
import mediapipe as mp
import time

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 120)
pTime = 0

while True:
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    success, frame = cap.read()
    if not success:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    results = face_detection.process(img)
    if results.detections:
        bbox_norm = results.detections[0].location_data.relative_bounding_box
        x1 = int(bbox_norm.xmin * w)
        y1 = int(bbox_norm.ymin * h)
        x2 = int((bbox_norm.xmin + bbox_norm.width) * w)
        y2 = int((bbox_norm.ymin + bbox_norm.height) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Mediapipe", frame)
    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Mediapipe", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()