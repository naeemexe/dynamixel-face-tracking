import os, warnings, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
from filterpy.kalman import KalmanFilter

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Setup Kalman 4D
kf = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.0 / 30.0
kf.F = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1,  0],
                 [0, 0, 0,  1]], dtype=np.float32)
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]], dtype=np.float32)

kf.P *= 500.0           
kf.R = np.eye(2) * 5.0 
kf.Q = np.eye(4) * 0.01 
initialized = False
prev_time = time.time()

def update_F_with_dt(kf_obj, dt_val: float):
    kf_obj.F[0, 2] = dt_val
    kf_obj.F[1, 3] = dt_val

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not grab frame. Try a different camera index.")
        break

    now = time.time()
    dt = max(1e-3, now - prev_time)  # avoid zero dt
    prev_time = now
    update_F_with_dt(kf, dt)

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        det = max(results.detections, key=lambda d: float(d.score[0]))
        bbox = det.location_data.relative_bounding_box

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        # clamp to frame just in case
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if not initialized:  # initialize state at first detection (x, y, vx, vy)
            kf.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
            initialized = True
        # Kalman update with each detection
        kf.update(np.array([float(cx), float(cy)], dtype=np.float32))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
    else:
        # no measurement this frame -> predict only
        pass
    # always predict next state (constant velocity model)
    kf.predict()

    # read predicted position (cast via float() to avoid NumPy deprecation warnings)
    pred_x = int(float(kf.x[0]))
    pred_y = int(float(kf.x[1]))

    # draw Kalman-smoothed target point
    cv2.circle(frame, (pred_x, pred_y), 7, (0, 80, 255), -1)
    cv2.putText(frame, "Kalman smoothed", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2)

    cv2.imshow("Face detection (Kalman only)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
