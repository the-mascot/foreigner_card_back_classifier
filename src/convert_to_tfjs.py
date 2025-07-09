"""
TensorFlow.js 변환 스크립트
"""
import os
import tensorflowjs as tfjs
import tensorflow as tf
import json
from datetime import datetime

def convert_to_tfjs(model_path, output_dir):
    """TensorFlow 모델을 TensorFlow.js 형식으로 변환"""
    
    print(f"모델 로딩 중: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("모델 구조:")
    model.summary()
    
    # TensorFlow.js 변환
    print(f"TensorFlow.js로 변환 중...")
    print(f"출력 디렉토리: {output_dir}")
    
    # 양자화 옵션으로 모델 크기 줄이기
    tfjs.converters.save_keras_model(
        model, 
        output_dir,
        quantization_bytes=2,  # 16-bit 양자화로 모델 크기 줄이기
        metadata={'description': 'Foreigner Card Back Classifier'}
    )
    
    print("변환 완료!")
    
    # 변환된 파일 확인
    files = os.listdir(output_dir)
    print(f"생성된 파일: {files}")
    
    # 모델 정보 저장
    model_info = {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'model_type': 'binary_classification',
        'classes': ['other_documents', 'foreigner_card_back'],
        'conversion_date': datetime.now().isoformat(),
        'quantization': '16-bit',
        'description': 'Binary classifier for foreigner card back detection'
    }
    
    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"모델 정보 저장됨: {info_path}")
    
    return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorFlow.js 변환')
    parser.add_argument('--model_path', type=str, required=True, help='변환할 TensorFlow 모델 경로')
    parser.add_argument('--output_dir', type=str, default='models/tfjs_model', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TensorFlow.js 변환
    tfjs_dir = convert_to_tfjs(args.model_path, args.output_dir)
    
    print(f"\n=== 변환 완료 ===")
    print(f"TensorFlow.js 모델: {tfjs_dir}")
    print(f"\n웹에서 사용하는 방법:")
    print(f"1. 변환된 모델 파일들을 웹 서버에 업로드")
    print(f"2. HTML에서 tf.loadLayersModel('경로/model.json')로 로드")
    print(f"3. 예측: model.predict(preprocessed_image_tensor)")
