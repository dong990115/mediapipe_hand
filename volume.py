import cv2
import mediapipe as mp
import math
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 노트북의 볼륨 조절
def set_volume1(level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevel(level,None)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
my_hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2,2)) + math.sqrt(math.pow(y1 - y2,2)) 

with mp_hands.Hands(
    max_num_hands=1,
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

        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                open = dist(hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y,hand_landmarks.landmark[14].x,hand_landmarks.landmark[14].y) < dist(hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y,hand_landmarks.landmark[16].x,hand_landmarks.landmark[16].y)
            
                if open == False:
                    curdist = -dist(hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y,hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y) / (dist(hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y,hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y) * 2)
                    curdist = curdist * 50
                    curdist = -60 - curdist
                    curdist = min(0,curdist)
                # 노트북의 볼륨 조절
                    set_volume1(curdist)

                #cv2.putText(
                #    image, text='volume=%d' % (dist+100), org=(10, 30),
                #    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                #    color=255, thickness=2)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        # cv2.imshow('volume', image)
        # if cv2.waitKey(1) == ord('q'):
        #     break
 
#cap.release()