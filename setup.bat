@echo off
REM 외국인등록증 뒷면 분류기 설정 스크립트 (Windows)

echo === 외국인등록증 뒷면 분류기 설정 ===

REM Python 버전 확인
echo Python 버전 확인 중...
python --version

REM 가상환경 생성
echo 가상환경 생성 중...
python -m venv venv

REM 가상환경 활성화
echo 가상환경 활성화 중...
call venv\Scripts\activate.bat

REM 필요한 패키지 설치
echo 필요한 패키지 설치 중...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo 설치 완료!

REM TensorFlow 설치 확인
echo TensorFlow 설치 확인 중...
python -c "import tensorflow as tf; print(f'TensorFlow 버전: {tf.__version__}')"

REM GPU 확인
echo GPU 확인 중...
python -c "import tensorflow as tf; print(f'GPU 사용 가능: {len(tf.config.list_physical_devices(\"GPU\"))}개')"

echo.
echo === 설정 완료 ===
echo 다음 단계:
echo 1. data\train\ 및 data\validation\ 폴더에 이미지 추가
echo 2. 모델 훈련: python src\train_model.py
echo 3. TensorFlow.js 변환: python src\convert_to_tfjs.py --model_path models\best_model.h5
echo 4. 웹 데모 실행: cd web_demo ^&^& python -m http.server 8000
echo.
echo Jupyter 노트북 실행: jupyter notebook notebooks\training_notebook.ipynb

pause
