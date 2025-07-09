"""
환경 확인 스크립트
"""
import sys
import subprocess

def check_python_environment():
    print("=== Python 환경 확인 ===")
    print(f"Python 버전: {sys.version}")
    print(f"Python 경로: {sys.executable}")
    
    # 설치된 패키지 확인
    try:
        import tensorflow
        print(f"✅ TensorFlow: {tensorflow.__version__}")
    except ImportError:
        print("❌ TensorFlow가 설치되지 않음")
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy가 설치되지 않음")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib이 설치되지 않음")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV가 설치되지 않음")
    
    try:
        import PIL
        print(f"✅ Pillow: {PIL.__version__}")
    except ImportError:
        print("❌ Pillow가 설치되지 않음")

def install_requirements():
    print("\n=== 필수 패키지 설치 ===")
    requirements = [
        "tensorflow==2.15.0",
        "numpy==1.24.3",
        "opencv-python==4.8.1.78",
        "pillow==10.0.1",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "jupyter==1.0.0"
    ]
    
    for package in requirements:
        try:
            print(f"설치 중: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 설치 실패: {e}")

if __name__ == "__main__":
    check_python_environment()
    
    print("\n패키지를 설치하시겠습니까? (y/n): ", end="")
    response = input().lower()
    
    if response == 'y':
        install_requirements()
        print("\n=== 설치 후 환경 재확인 ===")
        check_python_environment()
    else:
        print("패키지 설치를 건너뜁니다.")
        print("\n수동 설치 명령어:")
        print("pip install -r requirements.txt")
