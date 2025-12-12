import cv2
import numpy as np
import onnxruntime as ort
from filterpy.kalman import KalmanFilter
import time

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession("models/ultraface.onnx", providers=providers)
print("Model Inputs:", [i.name for i in sess.get_inputs()])
print("Model Outputs:", [o.name for o in sess.get_outputs()])
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def preprocess(frame):
    img = cv2.resize(frame, (320, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]
    inp = preprocess(frame)
    start = time.time()

    outs = sess.run(output_names, {"input": inp})
    scores, boxes = outs
    scores = scores[0][:, 1]
    boxes = boxes[0]
    infer_time = (time.time() - start) * 1000

    mask = scores > 0.6
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) > 0:
        sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best = np.argmax(sizes)
        x1, y1, x2, y2 = boxes[best]
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"FPS: {round(1000 / infer_time, 1) if infer_time > 0 else '...'}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("UltraFace Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
