import cv2
import mediapipe as mp
import math
import wmi
import time
import numpy as np
import hand_detector as hd
import os
import pyautogui
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
#from brightness import set_brightness
#from volume import set_volume1

from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 노트북의 볼륨 조절
def set_volume(level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevel(level, None)

    volume_range = volume.GetVolumeRange()
    min_volume = volume_range[0]
    max_volume = volume_range[1]
    print("유효한 볼륨 레벨 범위:", min_volume+65.25, "-", max_volume+65.25)
    master_volume = volume.GetMasterVolumeLevel()
    print("현재 마스터 볼륨 레벨:", master_volume+65.25)

    # 노트북의 밝기 조절
def set_brightness(level):
    wmi_interface = wmi.WMI(namespace='wmi')     # WMI 인터페이스 초기화
    brightness_instance = wmi_interface.WmiMonitorBrightnessMethods()[0]   # 모니터 설정 변경을 위한 인스턴스 가져오기
    brightness_instance.WmiSetBrightness(level, 0)     # 밝기 레벨 설정

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
my_hands = mp_hands.Hands()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#def dist(x1, y1, x2, y2):
#    return math.sqrt(math.pow(x1 - x2,2)) + math.sqrt(math.pow(y1 - y2,2))
def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
 
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
 
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
 
cap.set(3, wCam)
cap.set(4, hCam) 
detector = hd.handDetector(detectionCon=0.7)
wScr, hScr = pyautogui.size()
print(wScr, hScr)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5) as hands:  # 손가락 detection 모듈 초기화 후, # 한 사람의 두손의 모션 이용  # mp.solutions.hands모듈을 hands로 명명
 
    while cap.isOpened():
        success, img = cap.read()  # 변수 값 1프레임씩 무한루프 돌며 읽어내기
        h,w,c = img.shape

        if not success:            
            continue

        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)  # 좌우반전 / 영상형식 opencv : BGR, mediapipe : RGB
        results = my_hands.process(image)    # image를 전처리 및 모델 추론
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

        left_hand_landmarks = None
        right_hand_landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
               handedness = hand_handedness.classification[0].label
               if handedness == "Left":
                   left_hand_landmarks = hand_landmarks
               elif handedness == "Right":
                   right_hand_landmarks = hand_landmarks

              
            # 왼손 볼륨 조절
            if left_hand_landmarks:
                choice1 = dist(left_hand_landmarks.landmark[0].x, left_hand_landmarks.landmark[0].y,left_hand_landmarks.landmark[14].x,left_hand_landmarks.landmark[14].y) < dist(left_hand_landmarks.landmark[0].x, left_hand_landmarks.landmark[0].y,left_hand_landmarks.landmark[16].x,left_hand_landmarks.landmark[16].y)
                choice2 = dist(left_hand_landmarks.landmark[0].x, left_hand_landmarks.landmark[0].y,left_hand_landmarks.landmark[10].x,left_hand_landmarks.landmark[10].y) < dist(left_hand_landmarks.landmark[0].x, left_hand_landmarks.landmark[0].y,left_hand_landmarks.landmark[12].x,left_hand_landmarks.landmark[12].y)
               
                if choice1 == True:
                   ldist = -dist(left_hand_landmarks.landmark[4].x, left_hand_landmarks.landmark[4].y,left_hand_landmarks.landmark[8].x, left_hand_landmarks.landmark[8].y) / (dist(left_hand_landmarks.landmark[2].x, left_hand_landmarks.landmark[2].y,left_hand_landmarks.landmark[5].x, left_hand_landmarks.landmark[5].y) * 2)
                   ldist = ldist * 50
                   ldist = -65 - ldist
                   ldist = min(0,ldist)
            
                   set_volume(ldist)
                mp_drawing.draw_landmarks(image, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)   

                if choice2 == True:
                   rdist = int(dist(left_hand_landmarks.landmark[4].x, left_hand_landmarks.landmark[4].y,left_hand_landmarks.landmark[8].x, left_hand_landmarks.landmark[8].y) / (dist(left_hand_landmarks.landmark[2].x, left_hand_landmarks.landmark[2].y,left_hand_landmarks.landmark[5].x, left_hand_landmarks.landmark[5].y) * 2))
                   rdist = rdist*255
                       
                   set_brightness(int(rdist))
                mp_drawing.draw_landmarks(image, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if not choice1 & choice2 == True: continue

            if right_hand_landmarks:
                           
                while True:
                    success, img = cap.read()
                    img = detector.findHands(img)
                    lmList, bbox = detector.findPosition(img)
                    output = img.copy()
                
                    if len(lmList) != 0:
                        # print(lmList[4], lmList[8])
                        x1, y1 = lmList[8][1:]
                        x2, y2 = lmList[12][1:]
                
                        fingers = detector.fingersUp()
                        # print(fingers)
                        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (205, 250, 255), -1)
                        img = cv2.addWeighted(img, 0.5, output, 1 - .5, 0, output)
                
                        # Only Index Finger : Moving Mode
                        if fingers[1] == 1 and fingers[2] == 0 and right_hand_landmarks is not None:
                            # Convert Coordinates
                            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                
                            # Smoothen Values
                            clocX = plocX + (x3 - plocX) / smoothening
                            clocY = plocY + (y3 - plocY) / smoothening
                
                            # Move Mouse
                            pyautogui.moveTo(wScr - clocX, clocY)
                            cv2.circle(img, (x1, y1), 6, (255, 28, 0), cv2.FILLED)
                            plocX, plocY = clocX, clocY
                            # cv2.putText(img, 'Moving Mode', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                
                        # Both Index and middle fingers are up : Clicking Mode
                        if fingers[1] == 1 and fingers[2] == 1 and right_hand_landmarks is not None: 
                            # Find distance between fingers
                            length, img, lineInfo = detector.findDistance(8, 12, img)
                
                            # Click mouse if distance short
                            if length < 40:
                                cv2.circle(img, (lineInfo[4], lineInfo[5]), 6, (0, 255, 0), cv2.FILLED)
                                # cv2.putText(img, 'Click!!', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                                pyautogui.click()
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break


            # cv2.putText(
            #    image, text='brightness=%d volume=%.2f' % (int(rdist), ldist+66), org=(10, 30),
            #    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            #    color=255, thickness=2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.imshow("Vitual mouse monitor", img)
        cv2.setWindowProperty("Vitual mouse monitor", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(1) == ord('q'):
            break
 
cap.release()