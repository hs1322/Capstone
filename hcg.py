print("1801301 황찬기")

# dlib을 이용한 얼굴 인식과 불 필요한 영역 흰색 채우기
import sys
import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0, 255, 0)
line_width = 3

while True:
    ret_val, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    dets = detector(gray)

    # 이미지 크기
    height, width = gray.shape
    
    # 흑백 이미지로 마스크 생성 (검은 배경)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 얼굴이 검출되면 해당 얼굴 영역을 제외한 좌우 영역을 흰색으로 칠함
    for det in dets:
        left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()
        
        # 얼굴 영역을 제외한 좌우 영역을 흰색으로 칠함
        mask[:, left:right] = 255  # 얼굴 영역을 제외한 좌우 영역을 흰색으로

        # 얼굴 영역 표시
        cv2.rectangle(img, (left, top), (right, bottom), color_green, line_width)

    # 마스크를 이용하여 원본 이미지의 좌우 영역을 흰색으로 채움
    img[mask == 0] = 255

    # 얼굴 영역을 흰색으로 채운 이미지를 보여줌
    cv2.imshow('Face and Surrounding Area', img)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

# 창 닫기
cv2.destroyAllWindows()
