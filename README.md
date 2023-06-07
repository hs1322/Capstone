# Motion Tracking Robot

# 1. 주제 : Motion Tracking Robot.
- 카메라를 이용하여 객체를 검출 -> 여러 동작을 시도 해 볼 예정입니다.

# 2. 선정 이유 
- 최근 국내 미디어 시장 중 유튜브나 인스타 라이브 등을 이용한  1인 미디어 시장이 가파르게 성장하고 있는 추세이다.
  이에 많은 사람들이 1인 미디어에 관심을 가지고 도전하는 사람들이 많아 지고 있는데, 
  1인 미디어의 특성 상 기존 거치대나 바디 캠 같은 장비를 사용하게 되며, 이 경우 사용자가 수시로 조정을 해야 하는 불편함이 있다.
  이러한 불편함을 개선 및 보완하고 나아가 1인 미디어 촬영에 도움이 되고자 이 작품을 생각하게 되었다.


# 3. 목표
- 작품의 최종 형태는 주행 로봇 위에 카메라를 설치한 중,소형 크기의 작품을 구상 하고 있습니다.
  이번 학기에는 카메라를 영상을 raspberry pi에서 영상 처리를 통해 간단한 손 모양을 인식, 
  손 모양에 따라 앞, 뒤 양옆으로 주행 로봇을 제어하며 카메라의 거리를 조절.
  주행 로봇에는 메카넘 휠을 사용하여 동작을 원할 하게 수행하며 이후 진행 상황에 따라 손동작 인식이나 음성 인식도 추가해볼 예정입니다.



# 4. 블록도
![image](https://github.com/hs1322/Capstone/assets/90660378/68f28db6-8f0c-4b60-b62d-b70d43eacd5c)


# 5. 진행사항

# 사용자 인식

1. OpenCV에서 제공하는 사전 훈련된 Haar cascades의 default.xml 파일을 사용 사용자의 얼굴 부분을 검출
2. 검출된 사용자의 얼굴을 300 * 300 pixel 사진으로 200장을 faces 폴더 안에 저장
![image](https://github.com/hs1322/Capstone/assets/90660378/d7f3a851-4eeb-47a8-883c-241b28337679)

3. 저장된 폴더내의 파일들을 가져와 배열의 저장 한 후 LBPH 알고리즘(Local-Binary-Pattern 주변의 값을 2진수로 표현 후 값을 계산)을 사용하여 사용자의 얼굴을 학습

 ※ 현재 LBPH알고리즘은 오래되어 인식율 및 성능이 좋지 못함 CNN알고리즘 사용시 얼굴의 주요 특징점을 찾기에 성능이 더 뛰어나고 정확하게 인식함 현 단계에서는 LBPH 알고리즘을 사용

![image](https://github.com/hs1322/Capstone/assets/90660378/f17f7833-fe30-41f8-99a9-019b610e3c0b)

4. 학습된 이미지 파일을 이용 웹캠을 통하여 실시간으로 사용자의 얼굴을 찾아내어 인식 후 학습된 사용자의 얼굴과 유사 또는 일치 할 시 'Find face' topic을 보냄
![image](https://github.com/hs1322/Capstone/assets/90660378/516c71e5-3e18-4f69-bde0-4cb3b12fc088)


사용자 얼굴의 이미지를 200장을 찍고 학습 후 사용자 외 얼굴은 Locked 사용자 얼굴은 UnLocked한 후 손 모양 검출로 넘어감
https://github.com/hs1322/Capstone/assets/90660378/070a5d86-6c68-4acd-9370-2651b0783c48



# 손 모양 검출
![image](https://github.com/hs1322/Capstone/assets/90660378/ff2d0525-3d36-456b-91ff-31ae05bf0aa5)

손가락 랜드마크 번호간 길이를 비교 손가락을 접었는지 폈는지 판단
ex) 검지 -> 0번~8번 길이가 0번~6번 길이보다 짧으면 접힌 걸로 판단

코드의 일부분
![image](https://github.com/hs1322/Capstone/assets/90660378/f118d931-c088-4b79-9e34-f74e7cda5a39)

학습
![image](https://github.com/hs1322/Capstone/assets/90660378/a1befcbc-d1e8-4dfc-8a9b-e01adbadcb3e)

https://github.com/hs1322/Capstone/assets/90660378/fa611796-2452-46ec-b792-7a7868b9ea0a



# topic에 따른 메카넘 휠 조절
go, back, left, right 4방향에 대한 바퀴 모터수 조절



https://github.com/hs1322/Capstone/assets/90660378/996fb51b-f784-437c-acfc-ed2ea7d1b622




손모양 검출후 손모양에 따른 topic을 Rosserial을 사용하여 arduino와 통신 진행중
