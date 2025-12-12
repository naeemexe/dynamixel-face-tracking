import cv2

FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
CY = int(FRAME_HEIGHT/2)
CX = int(FRAME_WIDTH/2)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, -64)
cap.set(cv2.CAP_PROP_CONTRAST, 64)
cap.set(cv2.CAP_PROP_HUE, 3)
cap.set(cv2.CAP_PROP_GAMMA, 420)
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)

while True:
    success, image = cap.read()
    if not success:
        break

    cv2.line(image, (0, CY), (FRAME_WIDTH, CY), (255, 0, 0), 2)
    cv2.line(image, (CX, 0), (CX, FRAME_HEIGHT), (255, 0, 0), 2)
    cv2.imshow('MediaPipe', image)
    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('MediaPipe', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
