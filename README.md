# UnconsciousAuth

**UnconsciousAuth**는 무자각 멀티 인증 시스템으로, 얼굴, 손, 음성 및 신체 인식을 통해 사용자 인증을 수행합니다. 이 프로젝트는 다양한 생체 인식 기술을 통합하여 사용자 인증 과정을 자동화하고, 보안을 강화합니다.

## Usage

### 설치

1. 이 저장소를 클론합니다:
    ```bash
    git clone https://github.com/KoreaSecurity/UnconsciousAuth.git
    ```

2. 필요한 라이브러리를 설치합니다:
    ```bash
    pip install -r requirements.txt
    ```

### 실행

1. 각 인증 모듈의 모델 파일과 라벨 파일이 `models/` 디렉토리에 있는지 확인합니다. 모델 파일은 다음과 같습니다:
    - `face.h5`
    - `hand.h5`
    - `voice.pkl`
    - `body.h5` (사용하는 경우)

2. 다음 명령어로 메인 스크립트를 실행합니다:
    ```bash
    python main.py
    ```

3. 프로그램은 카메라와 마이크를 통해 실시간으로 인증을 수행하며, `auth_log` 디렉토리에 인증 결과를 CSV 파일로 기록합니다.

### 모듈

- **face_auth.py**: 얼굴 인식을 통한 인증을 수행합니다.
- **hand_auth.py**: 손 인식을 통한 인증을 수행합니다.
- **voice_auth.py**: 음성 인식을 통한 인증을 수행합니다.
- **body_auth.py**: 신체 인식을 통한 인증을 수행합니다 (사용하는 경우).

### 로그 및 데이터

인증 결과는 `auth_log` 디렉토리 내의 CSV 파일로 기록됩니다:
- `face.csv`: 얼굴 인식 결과
- `hand.csv`: 손 인식 결과
- `voice.csv`: 음성 인식 결과

각 CSV 파일에는 인증 시각(`timestamp`), 인증 레이블(`label`), 그리고 인증 정확도(`accuracy`)가 포함됩니다.

### 프로젝트 구조



## Contribution

기여를 원하시는 분은 먼저 이슈를 열어 제안해 주세요. 기여를 원하시는 경우, 풀 리퀘스트를 통해 코드를 제출하실 수 있습니다. 기여에 대한 세부 사항은 `CONTRIBUTING.md` 파일을 참조해 주세요.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact:

- **Name:** Sang-Hoon Choi
- **Email:** csh0052@gmail.com

Feel free to open an issue or pull request if you encounter any bugs or have suggestions for improvement.



