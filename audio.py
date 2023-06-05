from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#volume.GetMute()  # 음소거 상태
#volume.GetMasterVolumeLevel()   # 마스터 볼륨 레벨 가져올 때
#volume.GetVolumeRange()         # 유효한 볼륨 레벨의 범위 가져올 때
#volume.SetMasterVolumeLevel(-20.0, None)   # 마스터 볼륨 레벨을 특정 값으로 설정

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
