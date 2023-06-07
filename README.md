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

사용자 얼굴의 이미지를 100장을 찍고 학습 후 사용자 외 얼굴은 Locked 사용자 얼굴은 UnLocked한 후 손 모양 검출로 넘어감




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
