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
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import pandas as pd
import os
from datetime import datetime
import time

class FaceRecognition:
    def __init__(self, debug=False):
        self.debug = debug
        try:
            self.model = load_model('models/face.h5')
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.load('models/face.npy', allow_pickle=True, encoding='latin1')
        except Exception as e:
            if self.debug:
                print(f"모델 또는 라벨 인코더 로드 중 오류 발생: {e}")
            exit()

        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.image_size = (64, 64)
        self.log_directory = 'auth_log'
        self.csv_filename = os.path.join(self.log_directory, 'face.csv')

        if not os.path.exists(self.log_directory):
            if self.debug:
                print(f"Creating log directory at {self.log_directory}")
            os.makedirs(self.log_directory)

        if not os.path.exists(self.csv_filename):
            if self.debug:
                print(f"Creating log file at {self.csv_filename}")
            self.create_csv_file()

    def create_csv_file(self):
        df = pd.DataFrame(columns=['timestamp', 'label', 'accuracy'])
        df.to_csv(self.csv_filename, index=False)
        if self.debug:
            print(f"Created CSV file at {self.csv_filename}")

    def save_to_csv(self, label, accuracy):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {'timestamp': timestamp, 'label': label, 'accuracy': accuracy}
        df = pd.DataFrame([data])
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)
        if self.debug:
            print(f"Saved to CSV: {label}, Accuracy: {accuracy}")

    def recognize_face(self, frame):
        try:
            results = self.mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    face = frame[y:y + h, x:x + w]
                    face_resized = cv2.resize(face, self.image_size)
                    face_resized = face_resized.astype('float32') / 255.0
                    face_resized = np.expand_dims(face_resized, axis=0)

                    predictions = self.model.predict(face_resized)
                    max_index = np.argmax(predictions)
                    confidence = predictions[0][max_index]
                    label = self.label_encoder.inverse_transform([max_index])[0]

                    if confidence < 0.7:
                        label = "Unknown"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    self.save_to_csv(label, confidence)

                    if self.debug:
                        print(f"Detected face with label: {label}, Confidence: {confidence}")

                    return label, confidence
            else:
                if self.debug:
                    print("얼굴이 인식되지 않았습니다.")
                self.save_to_csv("none", 0)
                return "none", 0

        except Exception as e:
            if self.debug:
                print(f"얼굴 인식 중 오류 발생: {e}")
            self.save_to_csv("none", 0)
            return "Error", 0

    def activate(self, duration=5):
        try:
            cap = cv2.VideoCapture(0)
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.debug:
                        print("연구실 캠 사용 중 또는 연결 안됨")
                    break

                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > duration:
                    if self.debug:
                        print(f"얼굴 인증이 {duration}초 동안 실행되었습니다.")
                    break

                label, confidence = self.recognize_face(frame)

                if self.debug:
                    print(f"Face Label: {label}, Confidence: {confidence}")

                cv2.imshow('Face Authentication', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            if self.debug:
                print(f"실시간 얼굴 인증 중 오류 발생: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.debug:
                print("얼굴 인증 완료.")
