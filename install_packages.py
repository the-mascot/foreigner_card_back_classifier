"""
단계별 설치 가이드
"""
import subprocess
import sys
import os

def run_command(command, description):
    """명령어 실행 및 결과 확인"""
    print(f"\n🔄 {description}")
    print(f"실행 명령어: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 성공: {description}")
            if result.stdout:
                print(f"출력: {result.stdout[:200]}...")
        else:
            print(f"❌ 실패: {description}")
            print(f"오류: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return False

def main():
    print("=== 외국인등록증 뒷면 분류기 환경 설정 ===")
    
    # 1. Python 버전 확인
    print(f"Python 버전: {sys.version}")
    
    # 2. pip 업그레이드
    if not run_command("python -m pip install --upgrade pip", "pip 업그레이드"):
        print("⚠️ pip 업그레이드에 실패했지만 계속 진행합니다.")
    
    # 3. 기본 패키지 설치
    basic_packages = [
        "numpy",
        "matplotlib", 
        "pillow",
        "opencv-python"
    ]
    
    print("\n📦 기본 패키지 설치 중...")
    for package in basic_packages:
        run_command(f"pip install {package}", f"{package} 설치")
    
    # 4. TensorFlow 설치 (별도)
    print("\n🤖 TensorFlow 설치 중...")
    tf_success = run_command("pip install tensorflow==2.15.0", "TensorFlow 설치")
    
    if not tf_success:
        print("⚠️ TensorFlow 설치 실패. CPU 버전으로 재시도...")
        run_command("pip install tensorflow-cpu==2.15.0", "TensorFlow CPU 버전 설치")
    
    # 5. 추가 패키지 설치
    additional_packages = [
        "pandas",
        "seaborn", 
        "scikit-learn",
        "jupyter"
    ]
    
    print("\n📊 추가 패키지 설치 중...")
    for package in additional_packages:
        run_command(f"pip install {package}", f"{package} 설치")
    
    # 6. TensorFlow.js 변환 도구 설치
    print("\n🌐 TensorFlow.js 변환 도구 설치 중...")
    run_command("pip install tensorflowjs", "TensorFlow.js 설치")
    
    # 7. 설치 확인
    print("\n🔍 설치 확인 중...")
    test_imports = [
        "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')",
        "import numpy as np; print(f'NumPy: {np.__version__}')",
        "import cv2; print(f'OpenCV: {cv2.__version__}')",
        "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')",
        "import PIL; print(f'Pillow: {PIL.__version__}')"
    ]
    
    for test_import in test_imports:
        run_command(f'python -c "{test_import}"', "라이브러리 확인")
    
    # 8. GPU 확인
    print("\n🖥️ GPU 확인 중...")
    run_command('python -c "import tensorflow as tf; print(f\'GPU 사용 가능: {len(tf.config.list_physical_devices(\"GPU\"))}개\')"', "GPU 확인")
    
    print("\n✅ 환경 설정 완료!")
    print("\n다음 단계:")
    print("1. python check_environment.py  # 환경 재확인")
    print("2. 데이터 준비 (data/train/ 폴더에 이미지 추가)")
    print("3. python src/train_model.py  # 모델 훈련")

if __name__ == "__main__":
    main()
