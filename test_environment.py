"""
라이브러리 import 테스트 스크립트
"""

def test_imports():
    """필수 라이브러리 import 테스트"""
    print("=== 라이브러리 Import 테스트 ===\n")
    
    tests = [
        ("TensorFlow", "import tensorflow as tf"),
        ("NumPy", "import numpy as np"),
        ("OpenCV", "import cv2"),
        ("Matplotlib", "import matplotlib.pyplot as plt"),
        ("Pillow", "from PIL import Image"),
        ("Pandas", "import pandas as pd"),
        ("Seaborn", "import seaborn as sns"),
        ("Scikit-learn", "from sklearn.metrics import classification_report"),
        ("JSON", "import json"),
        ("OS", "import os"),
        ("Datetime", "from datetime import datetime")
    ]
    
    success_count = 0
    total_count = len(tests)
    
    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"✅ {name}: 성공")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: 실패 - {e}")
        except Exception as e:
            print(f"⚠️ {name}: 오류 - {e}")
    
    print(f"\n📊 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 라이브러리가 정상적으로 import되었습니다!")
        return True
    else:
        print("⚠️ 일부 라이브러리 import에 실패했습니다.")
        print("\n해결 방법:")
        print("1. python install_packages.py 실행")
        print("2. pip install -r requirements.txt 실행")
        print("3. 가상환경 재생성")
        return False

def test_tensorflow_gpu():
    """TensorFlow GPU 사용 가능 여부 확인"""
    try:
        import tensorflow as tf
        print(f"\n🤖 TensorFlow 버전: {tf.__version__}")
        
        # GPU 확인
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU 사용 가능: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("💻 CPU 모드로 실행됩니다.")
        
        # 간단한 연산 테스트
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.matmul(x, x)
        print(f"✅ TensorFlow 연산 테스트 성공: \n{y.numpy()}")
        
    except Exception as e:
        print(f"❌ TensorFlow 테스트 실패: {e}")

def test_project_structure():
    """프로젝트 구조 확인"""
    print("\n📁 프로젝트 구조 확인:")
    
    required_dirs = [
        "data/train/foreigner_card_back",
        "data/train/other_documents", 
        "data/validation/foreigner_card_back",
        "data/validation/other_documents",
        "models",
        "src",
        "web_demo",
        "notebooks"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (누락)")

if __name__ == "__main__":
    import os
    
    # 프로젝트 루트로 이동
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 테스트 실행
    if test_imports():
        test_tensorflow_gpu()
        test_project_structure()
        
        print("\n🚀 다음 단계:")
        print("1. 데이터 준비: data/train/ 폴더에 이미지 추가")
        print("2. 모델 훈련: python src/train_model.py")
        print("3. 노트북 실행: jupyter notebook notebooks/training_notebook.ipynb")
    else:
        print("\n❌ 환경 설정이 필요합니다.")
        print("python install_packages.py를 실행해주세요.")
