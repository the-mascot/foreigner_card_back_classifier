#!/bin/bash

# 전체 파이프라인 실행 스크립트

echo "=== 외국인등록증 뒷면 분류기 전체 파이프라인 ==="

# 1. 데이터 확인
echo "1. 데이터 확인 중..."
python -c "
import os
train_path = 'data/train'
val_path = 'data/validation'

def count_files(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

train_foreigner = count_files(os.path.join(train_path, 'foreigner_card_back'))
train_other = count_files(os.path.join(train_path, 'other_documents'))
val_foreigner = count_files(os.path.join(val_path, 'foreigner_card_back'))
val_other = count_files(os.path.join(val_path, 'other_documents'))

print(f'훈련 데이터: 외국인등록증 {train_foreigner}장, 기타 {train_other}장')
print(f'검증 데이터: 외국인등록증 {val_foreigner}장, 기타 {val_other}장')

if train_foreigner + train_other == 0:
    print('⚠️ 훈련 데이터가 없습니다!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "데이터를 먼저 준비해주세요."
    exit 1
fi

# 2. 모델 훈련
echo "2. 모델 훈련 시작..."
python src/train_model.py

if [ $? -ne 0 ]; then
    echo "❌ 모델 훈련 실패"
    exit 1
fi

# 3. TensorFlow.js 변환
echo "3. TensorFlow.js 변환 중..."
python src/convert_to_tfjs.py --model_path models/best_model.h5 --output_dir models/tfjs_model

if [ $? -ne 0 ]; then
    echo "❌ TensorFlow.js 변환 실패"
    exit 1
fi

# 4. 웹 데모용 파일 복사
echo "4. 웹 데모용 파일 준비 중..."
cp models/tfjs_model/* web_demo/ 2>/dev/null || echo "모델 파일을 수동으로 복사해주세요"

echo ""
echo "=== 파이프라인 완료 ==="
echo "✅ 모델 훈련 완료"
echo "✅ TensorFlow.js 변환 완료"
echo "✅ 웹 데모 준비 완료"
echo ""
echo "웹 데모 실행:"
echo "cd web_demo && python -m http.server 8000"
echo "브라우저에서 http://localhost:8000 접속"
