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
import pandas as pd
import os
from collections import Counter
from face.face_auth import FaceRecognition
from hand.hand_auth import HandRecognition
from voice.voice_auth import VoiceRecognition
from body.body_auth import BodyRecognition
import time
import threading

weights = {
    'face': 7,
    'voice': 1,
    'hand': 1,
}

log_directory = 'auth_log'
csv_files = ['face.csv', 'voice.csv', 'hand.csv']

face_recognition = FaceRecognition()
hand_recognition = HandRecognition()
voice_recognition = VoiceRecognition()

voice_thread = None

def read_latest_data():
    data = {}
    for csv_file in csv_files:
        file_path = os.path.join(log_directory, csv_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            latest_entry = df.iloc[-1]
            data[csv_file.split('.')[0]] = {
                'timestamp': latest_entry['timestamp'],
                'label': latest_entry['label'],
                'accuracy': latest_entry['accuracy']
            }
    return data

def find_duplicate_labels(data):
    labels = [info['label'] for info in data.values()]
    label_counter = Counter(labels)
    duplicates = {label: count for label, count in label_counter.items() if count > 1}
    if duplicates:
        for label, count in duplicates.items():
            relevant_variables = [var for var, info in data.items() if info['label'] == label]
            print(f"[DEBUG] 탐지된 사용자: {label}, 탐지된 요소: {count}, 요소정보: {', '.join(relevant_variables)}")
    return duplicates

def calculate_weighted_label(data):
    weighted_labels = {}
    for var, info in data.items():
        label = info['label']
        accuracy = info['accuracy']
        weight = weights[var]
        weighted_score = accuracy * weight
        if label in weighted_labels:
            weighted_labels[label] += weighted_score
        else:
            weighted_labels[label] = weighted_score

    highest_label = max(weighted_labels, key=weighted_labels.get)
    print(f"Highest Probability Label: {highest_label}, Score: {weighted_labels[highest_label]*10}%")
    return highest_label

def perform_recognition(debug=False, N_time=5):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frame_count = 0
    frame_interval = 5

    voice_thread = threading.Thread(target=voice_recognition.live_prediction, args=(1,))
    voice_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            if debug:
                print("카메라 사용 중 또는 연결 안됨")
            break

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        if frame_count % frame_interval == 0:
            frame_copy = frame.copy()
            face_label, face_confidence = face_recognition.recognize_face(frame_copy)

            frame_copy = frame.copy()
            hand_label, hand_confidence = hand_recognition.recognize_hand(frame_copy)

            if debug:
                print(f"Face Recognition: {face_label}, Confidence: {face_confidence}")
                print(f"Hand Recognition: {hand_label}, Confidence: {hand_confidence}")

        if elapsed_time > N_time:
            if debug:
                print(f"{N_time}초 동안 인증이 완료되었습니다.")
            break

        if debug:
            cv2.imshow('Combined Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    voice_recognition.stop()
    voice_thread.join()

def main_loop():
    previous_data = {}
    while True:
        perform_recognition(debug=False, N_time=5)
        current_data = read_latest_data()
        if current_data != previous_data:
            print("\nNew Data Detected, Processing...")
            duplicates = find_duplicate_labels(current_data)
            if duplicates:
                calculate_weighted_label(current_data)

        previous_data = current_data

if __name__ == "__main__":
    main_loop()
