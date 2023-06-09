def sound_dist(dist, lms):

    sdist = -dist(
            lms.landmark[4].x, 
            lms.landmark[4].y,
            lms.landmark[8].x,
            lms.landmark[8].y
            ) / (dist(
            lms.landmark[2].x,
            lms.landmark[2].y,
            lms.landmark[5].x,
            lms.landmark[5].y) * 2)
    #referenced by https://developeralice.tistory.com/11
    #엄지 손가락 끝과 검지 손가락 끝의 거리를 엄지,검지 마디 끝과 비교하면서 
    # 카메라 거리와 무관하게 제스쳐의 손가락끝의 거리 차이로만 측정
    #coded by Donghyeon Lim
    return sdist

def clk_len(dist, lms):

    cdist = dist(
            lms.landmark[8].x,
            lms.landmark[8].y,
            lms.landmark[12].x,
            lms.landmark[12].y)*100
    #검지 손가락 끝과 중지 손가락 끝의 거리를 잼.
    #coded by Junho Pyo

    return cdist

def brightness_dist(dist, lms):

    bdist = int(dist(
            lms.landmark[8].x,
            lms.landmark[8].y,
            lms.landmark[20].x,
            lms.landmark[20].y
            ) / (dist(
            lms.landmark[5].x,
            lms.landmark[5].y,
            lms.landmark[13].x, 
            lms.landmark[13].y) * 2))
    #검지 손가락 끝과 소지 손가락 끝의 거리를 검지,약지 마디 끝과 비교하면서 
    # 카메라 거리와 무관하게 제스쳐의 손가락끝의 거리 차이로만 측정
    #coded by Donghyeon Lim

    return bdist