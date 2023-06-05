import cv2
import mediapipe as mp

from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
 
 
mute_status = volume.GetMute()
if mute_status:
    print("볼륨이 음소거되었습니다.")
else:
    print("볼륨이 음소거되지 않았습니다.")


master_volume = volume.GetMasterVolumeLevel()
print("현재 마스터 볼륨 레벨:", master_volume)


volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
print("유효한 볼륨 레벨 범위:", min_volume, "-", max_volume)


volume.SetMasterVolumeLevel(-20.0, None)
print("마스터 볼륨 레벨이 -20.0으로 설정되었습니다.")


cap = cv2.VideoCapture(0)
 
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5) as hands:  # 손가락 detection 모듈 초기화 후, # 한 사람의 두손의 모션 이용  # mp.solutions.hands모듈을 hands로 명명
 
    while cap.isOpened():
        success, image = cap.read()  # 변수 값 1프레임씩 무한루프 돌며 읽어내기
        if not success:            
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # 좌우반전 / 영상형식 opencv : BGR, mediapipe : RGB
 
        results = hands.process(image)    # image를 전처리 및 모델 추론
 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger1 = int(hand_landmarks.landmark[4].y * 100 )   # 손가락 좌표의 y값 변화를 감지
                finger2 = int(hand_landmarks.landmark[8].y * 100 )
                
                dist = int(2*abs(finger1 - finger2))

                cv2.putText(
                    image, text='f1=%d f2=%d volume=%d ' % (finger1,finger2,dist), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)
 
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        cv2.imshow('gesture detection', image)
        if cv2.waitKey(1) == ord('q'):
            break
 
cap.release()