import cv2
import mediapipe as mp
from dynamixel_sdk import *
import time

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_POSITION_P_GAIN = 84
ADDR_POSITION_D_GAIN = 80
P_GAIN_VALUE = 3000
D_GAIN_VALUE = 16383
BAUDRATE = 4000000
TORQUE_ENABLE = 1
FRAME_WIDTH = 800
FRAME_HEIGHT = 600

portHandler = PortHandler('COM12')
packetHandler = PacketHandler(2.0)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)
packetHandler.write1ByteTxRx(portHandler, 1, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
packetHandler.write1ByteTxRx(portHandler, 2, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, 2048)
packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, 1100)
packetHandler.write2ByteTxRx(portHandler, 1, ADDR_POSITION_P_GAIN, P_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 1, ADDR_POSITION_D_GAIN, D_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 2, ADDR_POSITION_P_GAIN, P_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 2, ADDR_POSITION_D_GAIN, D_GAIN_VALUE)


face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)

CY = int(FRAME_HEIGHT/2)
CX = int(FRAME_WIDTH/2)

pTime = 0

while True:
    success, image = cap.read()
    if not success:
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.detections:
        box = results.detections[0].location_data.relative_bounding_box
        xmid = (box.xmin + box.width * 0.5) * FRAME_WIDTH
        ymid = (box.ymin + box.height * 0.5) * FRAME_HEIGHT
        cv2.circle(image, (int(xmid), int(ymid)), 5, (0, 0, 255), -1)
        pix_xerr = CX - xmid
        pix_yerr = CY - ymid

        if abs(pix_xerr) > 2 or abs(pix_yerr) > 2:
          x_err = (pix_xerr/FRAME_WIDTH) * 17
          y_err = (pix_yerr/FRAME_HEIGHT) * 14
          err_xpos = int(x_err * 11.378)
          err_ypos = int(y_err * 11.378)

          present_xpos, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 1, ADDR_PRESENT_POSITION)
          present_ypos, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 2, ADDR_PRESENT_POSITION)
          new_xpos = present_xpos + err_xpos
          new_ypos = present_ypos + err_ypos

          if 1024 < new_xpos < 3072:
              packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, new_xpos)
          if 1024 < new_ypos < 2048:
              packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, new_ypos)

    cv2.putText(image, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.line(image, (0, CY), (FRAME_WIDTH, CY), (255, 0, 0), 2)
    cv2.line(image, (CX, 0), (CX, FRAME_HEIGHT), (255, 0, 0), 2)
    cv2.imshow('MediaPipe', image)
    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('MediaPipe', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
portHandler.closePort()
