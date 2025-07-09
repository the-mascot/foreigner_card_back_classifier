# 🚀 설치 및 실행 가이드

## ❗ Import 오류 해결 방법

### 1단계: 환경 확인
```bash
python test_environment.py
```

### 2단계: 패키지 설치 (오류가 발생한 경우)
```bash
python install_packages.py
```

### 3단계: 수동 설치 (위 방법이 실패한 경우)
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 필수 패키지 개별 설치
pip install tensorflow
pip install numpy
pip install opencv-python
pip install pillow
pip install matplotlib
pip install pandas
pip install seaborn
pip install scikit-learn
pip install jupyter
pip install tensorflowjs
```

### 4단계: requirements.txt 설치
```bash
pip install -r requirements.txt
```

### 5단계: 설치 확인
```bash
python test_environment.py
```

## 🔧 일반적인 문제 해결

### TensorFlow 설치 오류
```bash
# CPU 버전 설치 (호환성 문제 시)
pip uninstall tensorflow
pip install tensorflow-cpu

# 또는 특정 버전 설치
pip install tensorflow==2.13.0
```

### OpenCV 오류
```bash
# 대체 설치 방법
pip uninstall opencv-python
pip install opencv-python-headless
```

### GPU 관련 오류 (NVIDIA GPU 사용 시)
```bash
# CUDA 호환성 확인 후
pip install tensorflow[and-cuda]
```

## 📦 가상환경 사용 (권장)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ✅ 설치 완료 후 테스트

1. **환경 테스트**:
   ```bash
   python test_environment.py
   ```

2. **간단한 TensorFlow 테스트**:
   ```python
   import tensorflow as tf
   print(tf.__version__)
   print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU'))}개")
   ```

3. **프로젝트 모듈 테스트**:
   ```python
   # src 폴더를 Python 경로에 추가하고 테스트
   import sys
   sys.path.append('src')
   
   from data_utils import DataLoader
   from model import create_efficient_classifier
   print("✅ 프로젝트 모듈 import 성공!")
   ```

## 🎯 설치 후 다음 단계

1. **데이터 준비**: `data/train/` 폴더에 이미지 추가
2. **모델 훈련**: `python src/train_model.py`
3. **웹 데모**: `python web_demo/server.py`

설치 중 문제가 발생하면 단계별로 확인해보세요!
