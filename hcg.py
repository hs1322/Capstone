print("1801301 황찬기")

# MediaPipe Face Detection을 이용한 얼굴 인식 (좌/우 눈, 코 끝, 입 중심, 좌/우 귀 이주)
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  # model_selection -> 모델 인덱스 0 = 카메라 2m 이내 부분적 모델 촬영 / 1 = 5m 이내에서 전신 모델
  # min_detection_confidenc -> 검출 성공 얼굴의 검출 모델의 신뢰값 ([0.0, 1.0]) 기본 값 0.5
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("카메라를 찾을수 없습니다.")
      # 비디오 파일 continue를 사용, 웹캠 break 사용
      break
	# 이미지를 좌우를 반전, BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
	# 성능 향상 - 이미지 작성 여부 False 설정
    image.flags.writeable = False
    results = face_detection.process(image)

    # 영상에 얼굴 감지 주석 그리기 기본값 : True.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
