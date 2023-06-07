import cv2
import mediapipe as mp
import math
import time
import rospy
from std_msgs.msg import String


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
    



cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
def dist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2)) + math.sqrt(math.pow(y1-y2,2))

compareIndex = [[18,4],[6,8],[10,12],[14,16],[18,20]]
open = [False,False,False,False,False]
gesture = [[True,True,True,True,True,"back!"],
            [False,False,False,False,True,"right!"],
            [True,False,False,False,False,"left!"],
            [False,True,False,False,False,"go!"],
            [False,False,False,False,False,"stop!"]]

prev_time = 0

while True:
    successs, img =  cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = my_hands.process(imgRGB)
    
    cur_time = time.time()
    FPS =  int(1/(cur_time - prev_time))
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
    cv2.imshow("HandTracking",img)
    key = cv2.waitKey(1)
    if key == 27:
        break
