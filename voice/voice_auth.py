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

import os
import numpy as np
import pickle
import sounddevice as sd
import pandas as pd
from datetime import datetime
import time
import librosa

class VoiceRecognition:
    def __init__(self, debug=False):
        self.debug = debug
        try:
            with open('models/voice.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            if self.debug:
                print(f"모델 로드 중 오류 발생: {e}")
            exit()

        self.log_directory = 'auth_log'
        self.csv_filename = os.path.join(self.log_directory, 'voice.csv')

        if not os.path.exists(self.log_directory):
            if self.debug:
                print(f"Creating log directory at {self.log_directory}")
            os.makedirs(self.log_directory)

        if not os.path.exists(self.csv_filename):
            if self.debug:
                print(f"Creating log file at {self.csv_filename}")
            self.create_csv_file()

        self.stop_flag = False

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

    def extract_features(self, audio, sample_rate):
        if len(audio) < sample_rate:
            padding = sample_rate - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        elif len(audio) > sample_rate:
            audio = audio[:sample_rate]

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)

    def live_prediction(self, duration=1, sample_rate=22050):
        def callback(indata, frames, time, status):
            if status and self.debug:
                print(status)
            if self.stop_flag:
                raise sd.CallbackStop()

            audio = indata.flatten()

            mfcc = self.extract_features(audio, sample_rate).reshape(1, -1)

            try:
                prediction = self.model.predict(mfcc)
                probas = self.model.predict_proba(mfcc)
                confidence = np.max(probas)

                if confidence < 0.95:
                    pass
                else:
                    if self.debug:
                        print(f"{datetime.now()} - Prediction: {prediction[0]}, Confidence: {confidence:.2f}")
                    self.save_to_csv(prediction[0], confidence)
            except Exception as e:
                if self.debug:
                    print(f"Prediction error: {e}")
                #self.save_to_csv("none", 0)# 인증못할때. none
                self.save_to_csv("none", 0)

        if self.debug:
            print("Listening...")
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate,
                            blocksize=int(sample_rate * duration)):
            while not self.stop_flag:
                sd.sleep(int(duration * 1000))

    def activate(self, duration=5):
        try:
            start_time = time.time()

            while not self.stop_flag:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time > duration:
                    if self.debug:
                        print(f"음성 인증이 {duration}초 동안 실행되었습니다.")
                    break

                self.live_prediction(duration=0.3)

        except Exception as e:
            if self.debug:
                print(f"실시간 음성 인증 중 오류 발생: {e}")

        finally:
            if self.debug:
                print("음성 인증 완료.")

    def stop(self):
        self.stop_flag = True
