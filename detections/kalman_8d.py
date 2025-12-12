import cv2
import mediapipe as mp
import numpy as np
from filterpy.kalman import KalmanFilter 

# Setup 8D Kalman Filter
def initialize_kalman_filter(initial_box):

    kf = KalmanFilter(dim_x=8, dim_z=4) 
    x1, y1, x2, y2 = initial_box
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    dt = 1.0 # Fixed dt
    kf.F = np.array([
        [1, 0, 0, 0, dt, 0, 0, 0],
        [0, 1, 0, 0, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0, dt, 0],
        [0, 0, 0, 1, 0, 0, 0, dt],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ])
    q_scale = 1.0
    kf.Q[0:4, 0:4] *= q_scale
    kf.Q[4:8, 4:8] *= 0.1
    kf.R *= 25.0 
    kf.x = np.array([x, y, w, h, 0, 0, 0, 0])
    kf.P *= 1000.0
    
    return kf


mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 30)
kf_tracker = None 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Predicts every frame except for the first
    if kf_tracker is not None:
        kf_tracker.predict()
    
    results = face_detection.process(img)
    if results.detections:
        det = max(results.detections, key=lambda d: d.score[0])
        bbox_norm = det.location_data.relative_bounding_box
        x1_raw = int(bbox_norm.xmin * w)
        y1_raw = int(bbox_norm.ymin * h)
        x2_raw = int((bbox_norm.xmin + bbox_norm.width) * w)
        y2_raw = int((bbox_norm.ymin + bbox_norm.height) * h)
        cv2.rectangle(frame, (x1_raw, y1_raw), (x2_raw, y2_raw), (0, 0, 255), 2)

        x_m = (x1_raw + x2_raw) / 2.0
        y_m = (y1_raw + y2_raw) / 2.0
        w_m = x2_raw - x1_raw
        h_m = y2_raw - y1_raw
        measurement = np.array([[x_m], [y_m], [w_m], [h_m]])

        if kf_tracker is None: # Only runs the first time
            kf_tracker = initialize_kalman_filter([x1_raw, y1_raw, x2_raw, y2_raw])
            x_smooth, y_smooth, w_smooth, h_smooth = x_m, y_m, w_m, h_m
        else: 
            kf_tracker.update(measurement)
            x_smooth, y_smooth, w_smooth, h_smooth = kf_tracker.x[:4].flatten()
        
        x1_smooth = int(x_smooth - w_smooth / 2)
        y1_smooth = int(y_smooth - h_smooth / 2)
        x2_smooth = int(x_smooth + w_smooth / 2)
        y2_smooth = int(y_smooth + h_smooth / 2)
        cv2.rectangle(frame, (x1_smooth, y1_smooth), (x2_smooth, y2_smooth), (255, 0, 0), 2)
        cv2.putText(frame, "Smoothed", (x1_smooth, y1_smooth - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # if detection found this runs
    elif kf_tracker is not None:
        x_smooth, y_smooth, w_smooth, h_smooth = kf_tracker.x[:4].flatten()    
        x1_smooth = int(x_smooth - w_smooth / 2)
        y1_smooth = int(y_smooth - h_smooth / 2)
        x2_smooth = int(x_smooth + w_smooth / 2)
        y2_smooth = int(y_smooth + h_smooth / 2)
        cv2.rectangle(frame, (x1_smooth, y1_smooth), (x2_smooth, y2_smooth), (0, 255, 0), 2)
        cv2.putText(frame, "Predicted", (x1_smooth, y1_smooth - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Kalman Smoothed", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.getWindowProperty("Kalman Smoothed", cv2.WND_PROP_VISIBLE) < 1: 
        break

cap.release()
cv2.destroyAllWindows()