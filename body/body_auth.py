"""
Software Copyright (c) 2024 Sang-Hoon Choi
Email: csh0052

Licensed under the MIT License. See LICENSE file for details.

Description:
This script performs face, hand, voice, and body authentication by continuously reading data from
various sensors, processing the data, and calculating the weighted label based on predefined weights.
It uses OpenCV for video capture, TensorFlow/Keras for face and hand recognition, and librosa for
voice recognition.

Usage:
- Ensure all necessary models and CSV files are in place.
- Run the script to start the authentication process.

Dependencies:
- OpenCV
- pandas
- numpy
- tensorflow
- sounddevice
- librosa
- mediapipe

"""

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
from datetime import datetime
import time

class BodyRecognition:
    def __init__(self):
        # Mediapipe 포즈 검출기 초기화
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.log_directory = 'auth_log'
        self.csv_filename = os.path.join(self.log_directory, 'body.csv')

        # 로그 디렉토리 생성
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        # CSV 파일이 없으면 헤더 생성
        if not os.path.exists(self.csv_filename):
            self.create_csv_file()

    def create_csv_file(self):
        df = pd.DataFrame(columns=['timestamp', 'label', 'accuracy'])
        df.to_csv(self.csv_filename, index=False)

    def save_to_csv(self, label, accuracy):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {'timestamp': timestamp, 'label': label, 'accuracy': accuracy}
        df = pd.DataFrame([data])
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)

    def recognize_body(self, frame):
        results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            # 신체 인식 결과가 있으면 처리
            label = "Body Detected"
            confidence = 1.0

            # 신체 스켈레톤 표시 (선택적)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

            # 예측 결과 CSV 파일에 저장
            self.save_to_csv(label, confidence)

            # 예측 결과 반환
            return label, confidence
        else:
            # 신체가 인식되지 않은 경우
            print("신체가 인식되지 않았습니다.")
            # 신체가 탐지되지 않았을 때 CSV 파일에 "none", 0 저장
            self.save_to_csv("none", 0)
            return "none", 0.0

    def activate(self, duration=5):
        try:
            cap = cv2.VideoCapture(0)
            start_time = time.time()  # 시작 시간 기록

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("카메라 사용 중 또는 연결 안됨")
                    break

                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > duration:  # 주어진 시간이 경과하면 종료
                    print(f"신체 인식이 {duration}초 동안 실행되었습니다.")
                    break

                label, confidence = self.recognize_body(frame)

                # 결과를 저장하거나 추가적인 처리를 여기에 추가
                print(f"Body Label: {label}, Confidence: {confidence}")

                cv2.imshow('Body Recognition', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"실시간 신체 인식 중 오류 발생: {e}")
            self.save_to_csv("none", 0)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("신체 인증 완료.")
