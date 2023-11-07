import rospy
import cv2
import numpy as np
import time
from os import listdir
from os.path import isfile, join
from std_msgs.msg import String
import mediapipe as mp
import math
import dlib
import sys

detector = dlib.get_frontal_face_detector()
hands_detector = mp.solutions.hands.Hands()

data_path = '/home/pi/catkin_ws/src/talker/ros_facetest/src/faces/'                # 3
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('/home/pi/catkin_ws/src/talker/ros_facetest/src/haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)
prevTime = 0  # 이전 시간을 저장할 변수
cnt=0
prev_time = 0

def callback(data):
    msg = data.data

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('chatter', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def talker(str):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(100) # 10hz
    pub.publish(str)

mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
def dist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2)) + math.sqrt(math.pow(y1-y2,2))

compareIndex = [[18,4],[6,8],[10,12],[14,16],[18,20]]
open = [False,False,False,False,False]
gesture = [[True,True,True,True,True,"back!"],
            [False,False,False,False,True,"right!"],
            [True,True,False,False,False,"left!"],
            [False,True,False,False,False,"go!"],
            [False,False,False,False,False,"stop!"]]

def calculate_distance(hand_landmark, index1, index2):
    x1, y1 = hand_landmark.landmark[index1].x, hand_landmark.landmark[index1].y
    x2, y2 = hand_landmark.landmark[index2].x, hand_landmark.landmark[index2].y
    return dist(x1, y1, x2, y2)



while True:                                                     # 4
    ret, frame = cap.read()

    # 프레임 수 계산
    currentTime = time.time()
    fps = 1 / (currentTime - prevTime) + 3
    prevTime = currentTime

    # 좌측 상단에 프레임 수 출력
    cv2.putText(frame, "FPS: {}".format(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    

    # findface="Find Face!"

    image, face = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        confidence = int(100 * (1 - (result[1]) / 300))
        display_string = str(confidence) + '% Confidence it is user'
        cv2.putText(image, display_string, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
            
            cnt+=1
            if cnt==10:
                # rospy.loginfo(findface)
                # pub.publish(findface)
                # rate.sleep()
                break
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cv2.destroyAllWindows()

mpFaceDetection = mp.solutions.face_detection
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

face_detection = mpFaceDetection.FaceDetection(min_detection_confidence=0.1)
hands_detection = mpHands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1)

color_green = (0, 255, 0)
line_width = 3

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    copy_img=img.copy()
    
    # 얼굴 검출
    dets = detector(gray)

    # 이미지 크기
    height, width = gray.shape
    
    # 흑백 이미지로 마스크 생성 (검은 배경)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 얼굴이 검출되면 해당 얼굴 영역을 제외한 좌우 영역을 흰색으로 칠함
    for det in dets:
        if len(dets)>0:
            det=dets[0]
            left, top, right, bottom = det.left()-50, det.top(), det.right()+50, det.bottom()
        
            # 얼굴 영역을 제외한 좌우 영역을 흰색으로 칠함
            mask[:, left:right] = 255  # 얼굴 영역을 제외한 좌우 영역을 흰색으로

            # 얼굴 영역 표시
            cv2.rectangle(img, (left, top), (right, bottom), color_green, line_width)

    # 마스크를 이용하여 원본 이미지의 좌우 영역을 흰색으로 채움
    img[mask == 0] = 255

    # 손 인식 코드 추가
    h,w,c = img.shape
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = my_hands.process(imgRGB)
    
    cur_time = time.time()
    FPS =  int(1/(cur_time - prev_time)) + 3
    prev_time = cur_time
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for i in range(0,5):
                open[i] = dist(handLms.landmark[0].x, handLms.landmark[0].y,
                        handLms.landmark[compareIndex[i][0]].x,handLms.landmark[compareIndex[i][0]].y)<dist(handLms.landmark[0].x, handLms.landmark[0].y,
                        handLms.landmark[compareIndex[i][1]].x,handLms.landmark[compareIndex[i][1]].y)
            print(open)
            text_x = (handLms.landmark[0].x * w)
            text_y = (handLms.landmark[0].y * h)
            for i in range(0,len(gesture)):
                flag = True
                for j in range(0,5):
                    if(gesture[i][j] != open[j]):
                        flag = False
                if(flag == True):
                    cv2.putText(img, gesture[i][5], (round(text_x)-20, round(text_y)-100),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    talker(gesture[i][5])    
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

                
    cv2.putText(img,f"FPS: {str(FPS)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    # 얼굴 영역을 흰색으로 채운 이미지,원본 이미지를 보여주는 창
    cv2.imshow('Face and Surrounding Area', img)
    cv2.imshow('Original Image', copy_img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
