import cv2
import mediapipe as mp
from dynamixel_sdk import *
import time

ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

BAUDRATE = 4000000
P_GAIN_VALUE = 2800
D_GAIN_VALUE = 16383
I_GAIN_VALUE = 500
TORQUE_ENABLE = 1

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
CY = int(FRAME_HEIGHT/2)
CX = int(FRAME_WIDTH/2)

portHandler = PortHandler('COM12')
packetHandler = PacketHandler(2.0)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)
packetHandler.write1ByteTxRx(portHandler, 1, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
packetHandler.write1ByteTxRx(portHandler, 2, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, 2048)
packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, 1300)
packetHandler.write2ByteTxRx(portHandler, 1, ADDR_POSITION_P_GAIN, P_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 1, ADDR_POSITION_I_GAIN, I_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 1, ADDR_POSITION_D_GAIN, D_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 2, ADDR_POSITION_P_GAIN, P_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 2, ADDR_POSITION_I_GAIN, I_GAIN_VALUE)
packetHandler.write2ByteTxRx(portHandler, 2, ADDR_POSITION_D_GAIN, D_GAIN_VALUE)

groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, 4)
groupSyncRead.addParam(1)
groupSyncRead.addParam(2)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4)

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BRIGHTNESS, -64)
cap.set(cv2.CAP_PROP_CONTRAST, 64)
cap.set(cv2.CAP_PROP_HUE, 3)
cap.set(cv2.CAP_PROP_GAMMA, 420)
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)

FOV_X = 15
FOV_Y = 10
SCALE_X = FOV_X / 800
SCALE_Y = FOV_Y / 600
UNIT = 1024 / 90
pTime = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    result = face_detection.process(rgb)
    frame.flags.writeable = True

    if result.detections:
        face = result.detections[0].location_data.relative_bounding_box
        xmid = int((face.xmin + face.width  * 0.5) * 800)
        ymid = int((face.ymin + face.height * 0.5) * 600)
        cv2.circle(frame, (int(xmid), int(ymid)), 5, (0, 0, 255), -1)
        dx = CX - xmid
        dy = CY - ymid

        groupSyncRead.txRxPacket()
        present_x = groupSyncRead.getData(1,  ADDR_PRESENT_POSITION, 4)
        present_y = groupSyncRead.getData(2, ADDR_PRESENT_POSITION, 4)

        if dx > 2 or dx < -2:
            deg_x = dx * SCALE_X
            new_x = present_x + int(deg_x * UNIT)
            if new_x < 1024:
                new_x = 1024
            elif new_x > 3072:
                new_x = 3072
        else:
            new_x = present_x

        if dy > 2 or dy < -2:
            deg_y = dy * SCALE_Y
            new_y = present_y + int(deg_y * UNIT)
            if new_y < 1024:
                new_y = 1024
            elif new_y > 2048:
                new_y = 2048
        else:
            new_y = present_y

        groupSyncWrite.clearParam()
        groupSyncWrite.addParam(1,  [new_x & 0xFF, (new_x>>8)&0xFF, (new_x>>16)&0xFF, (new_x>>24)&0xFF])
        groupSyncWrite.addParam(2, [new_y & 0xFF, (new_y>>8)&0xFF, (new_y>>16)&0xFF, (new_y>>24)&0xFF])
        groupSyncWrite.txPacket()
   
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.line(frame, (CX,0), (CX,600), (255,0,0), 2)
    cv2.line(frame, (0,CY), (800,CY), (255,0,0), 2)
    cv2.imshow("FaceTrack", frame)   
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
portHandler.closePort()
