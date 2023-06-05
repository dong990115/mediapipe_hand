import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
my_hands = mp_hands.Hands()

# 볼륨 범위 및 음소거 상태 확인
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
mute_status = volume.GetMute()
if mute_status:
    print("볼륨이 음소거되었습니다.")
else:
    print("볼륨이 음소거되지 않았습니다.")

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, img = cap.read()
        h, w, c = img.shape

        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        results = my_hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger1 = int(hand_landmarks.landmark[4].y * 100)  # 엄지 손가락 좌표의 y값 변화를 감지
                finger2 = int(hand_landmarks.landmark[8].y * 100)

                dist = int(abs(finger1 - finger2))
                dist = -65 + dist
                dist = min(0, dist)

                # 실제 음량 범위로 변환하여 볼륨 조절
                volume_level = min_volume + (max_volume - min_volume) * (dist / -6.5)
                volume.SetMasterVolumeLevel(volume_level, None)

                cv2.putText(
                    image, text='volume=%d' % (dist + 100), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=2)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('gesture detection', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
