import mediapipe as mp
import cv2
import math
import wmi
import pyautogui
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume = cast(interface,POINTER(IAudioEndpointVolume))

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# space_pressed = False
# hand_flipped = False

def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2,2)) + math.sqrt(math.pow(y1 - y2,2))

def set_brightness(level):
    wmi_interface = wmi.WMI(namespace='wmi')     # WMI 인터페이스 초기화
    brightness_instance = wmi_interface.WmiMonitorBrightnessMethods()[0]   # 모니터 설정 변경을 위한 인스턴스 가져오기
    brightness_instance.WmiSetBrightness(level, 0)     # 밝기 레벨 설정

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


while True:
    success,img = cap.read()
    h,w,c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = my_hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            
            open = dist(
                handlms.landmark[0].x,
                handlms.landmark[0].y,
                handlms.landmark[14].x,
                handlms.landmark[14].y
            ) < dist(
                handlms.landmark[0].x,
                handlms.landmark[0].y,
                handlms.landmark[16].x,
                handlms.landmark[16].y)
            
            thumb_folded = dist(
                handlms.landmark[4].x,
                handlms.landmark[4].y,
                handlms.landmark[10].x,
                handlms.landmark[10].y
            ) < dist(
                handlms.landmark[3].x,
                handlms.landmark[3].y,
                handlms.landmark[10].x,
                handlms.landmark[10].y
            )
            
            index_folded = dist(
                handlms.landmark[8].x,
                handlms.landmark[8].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            ) < dist(
                handlms.landmark[5].x,
                handlms.landmark[5].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            )
            
            middle_folded = dist(
                handlms.landmark[12].x,
                handlms.landmark[12].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            ) < dist(
                handlms.landmark[9].x,
                handlms.landmark[9].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            )
            
            ring_folded = dist(
                handlms.landmark[16].x,
                handlms.landmark[16].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            ) < dist(
                handlms.landmark[13].x,
                handlms.landmark[13].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            )
            
            pinky_folded = dist(
                handlms.landmark[20].x,
                handlms.landmark[20].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            ) < dist(
                handlms.landmark[17].x,
                handlms.landmark[17].y,
                handlms.landmark[0].x,
                handlms.landmark[0].y
            )

            if (open == False) and not thumb_folded:
                cv2.putText(
                    img, text="Adjusting Sound",
                    org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=255, thickness=2)
                ldist = -dist(handlms.landmark[4].x, handlms.landmark[4].y,
                              handlms.landmark[8].x, handlms.landmark[8].y) / (dist(handlms.landmark[2].x, handlms.landmark[2].y
                                                                                    ,handlms.landmark[5].x, handlms.landmark[5].y) * 2)
                ldist = ldist * 50
                ldist = -60 - ldist
                ldist = min(0,ldist)
                
                set_volume(ldist)
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)    

            if ring_folded & pinky_folded & thumb_folded:
                cv2.putText(
                    img, text="Virtual Mouse",
                    org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=255, thickness=2)
                # finger1 = int(handlms.landmark[8].x * 600)
                # finger2 = int(handlms.landmark[12].x ^ 600)
                # dist2 = int(abs(finger1 - finger2))
            
            if ring_folded and thumb_folded and middle_folded:
                cv2.putText(
                    img, text="Adjusting Brightness",
                    org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=255, thickness=2)
                rdist = int(dist(handlms.landmark[8].x, handlms.landmark[8].y,
                                 handlms.landmark[20].x, handlms.landmark[20].y) / (dist(handlms.landmark[2].x, handlms.landmark[2].y,
                                                                                         handlms.landmark[5].x, handlms.landmark[5].y) * 2))
                rdist = rdist*255
                
                set_brightness(int(rdist))
            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
        

    cv2.imshow("HandTracking", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
