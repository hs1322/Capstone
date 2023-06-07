#!/usr/bin/env python
# -*- coding: utf-8 -*-
# license removed for brevity

import rospy
import cv2
import numpy as np
import time
from std_msgs.msg import String
from os import listdir
from os.path import isfile, join

data_path = '/home/pi/catkin_ws/src/ros_facetest/src/faces/'                # 3
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

face_classifier = cv2.CascadeClassifier('/home/pi/catkin_ws/src/ros_facetest/src/haarcascade_frontalface_default.xml')

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

pub = rospy.Publisher('face', String, queue_size=10)
rospy.init_node('face_find', anonymous=True)
rate = rospy.Rate(1) # 10hz

cnt=0

while True:                                                     # 4
    ret, frame = cap.read()

    # 프레임 수 계산
    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    # 좌측 상단에 프레임 수 출력
    cv2.putText(frame, "FPS: {}".format(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    findface="Find Face!"

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
            if cnt==15:
                rospy.loginfo(findface)
                pub.publish(findface)
                rate.sleep()
                break
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
