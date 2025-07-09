# 🚀 외국인등록증 뒷면 분류기 실행 가이드

## 📋 프로젝트 개요

이 프로젝트는 외국인등록증 뒷면을 자동으로 분류하는 AI 모델을 개발하고, 웹 브라우저에서 실시간으로 사용할 수 있도록 하는 완전한 파이프라인입니다.

## 🛠️ 설치 및 설정

### 1. 환경 설정

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

### 2. 수동 설치
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## 📁 데이터 준비

### 1. 데이터 구조
```
data/
├── train/
│   ├── foreigner_card_back/     # 외국인등록증 뒷면 이미지
│   └── other_documents/         # 기타 문서 이미지
└── validation/
    ├── foreigner_card_back/     # 검증용 외국인등록증 뒷면
    └── other_documents/         # 검증용 기타 문서
```

### 2. 권장 데이터 수량
- **외국인등록증 뒷면**: 최소 500장, 권장 1000+장
- **기타 문서**: 최소 500장, 권장 1000+장 (신분증, 여권, 운전면허증 등)
- **훈련:검증 비율**: 80:20 또는 70:30

### 3. 이미지 품질 가이드
- 다양한 각도에서 촬영
- 다양한 조명 조건
- 다양한 배경
- 흐림, 그림자 등 실제 환경 고려

## 🏋️ 모델 훈련

### 방법 1: 스크립트 실행
```bash
python src/train_model.py
```

### 방법 2: Jupyter 노트북
```bash
jupyter notebook notebooks/training_notebook.ipynb
```

### 방법 3: 전체 파이프라인
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## 🔄 TensorFlow.js 변환

### 훈련된 모델 변환
```bash
python src/convert_to_tfjs.py --model_path models/best_model.h5 --output_dir models/tfjs_model
```

### 웹 데모용 파일 복사
```bash
cp models/tfjs_model/* web_demo/
```

## 🌐 웹 데모 실행

### 방법 1: Python 서버 스크립트
```bash
python web_demo/server.py
```

### 방법 2: 직접 HTTP 서버
```bash
cd web_demo
python -m http.server 8000
```

### 방법 3: Node.js (선택사항)
```bash
cd web_demo
npx http-server -p 8000 -c-1
```

브라우저에서 `http://localhost:8000` 접속

## 📊 성능 모니터링

### 1. 훈련 중 모니터링
- TensorBoard: `tensorboard --logdir logs`
- 훈련 히스토리 그래프 자동 생성
- 조기 종료 및 학습률 감소 자동 적용

### 2. 모델 평가 지표
- **정확도 (Accuracy)**: 전체 예측 중 맞춘 비율
- **정밀도 (Precision)**: 양성 예측 중 실제 양성 비율
- **재현율 (Recall)**: 실제 양성 중 양성으로 예측한 비율
- **F1 스코어**: 정밀도와 재현율의 조화평균

## 🚀 프로덕션 배포

### 1. 웹 서버 배포
```bash
# Nginx 설정 예시
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/web_demo;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

### 2. CDN 배포
- 모델 파일을 CDN에 업로드
- `classifier.js`에서 모델 URL 수정

### 3. HTTPS 필수
- 웹캠 기능은 HTTPS에서만 작동
- Let's Encrypt로 무료 SSL 인증서 발급

## 🔧 문제 해결

### 1. 모델 성능 문제
**낮은 정확도:**
- 더 많은 데이터 수집
- 데이터 품질 개선
- 더 복잡한 모델 사용 (MobileNet 등)

**과적합:**
- 데이터 증강 강화
- 드롭아웃 비율 증가
- 정규화 강화

### 2. 웹 데모 문제
**모델 로딩 실패:**
- 모델 파일 경로 확인
- CORS 설정 확인
- 브라우저 개발자 도구로 오류 확인

**웹캠 권한 오류:**
- HTTPS 환경에서만 웹캠 사용 가능
- 브라우저 권한 설정 확인

**느린 추론 속도:**
- 모델 양자화 적용
- 이미지 크기 조정
- 브라우저 성능 확인

### 3. 일반적인 오류
**GPU 메모리 부족:**
```python
# train_model.py에서 배치 크기 조정
CONFIG['batch_size'] = 16  # 기본값 32에서 16으로 감소
```

**TensorFlow.js 변환 오류:**
```bash
# 최신 버전으로 업데이트
pip install --upgrade tensorflowjs
```

## 📈 성능 최적화

### 1. 모델 최적화
- **양자화**: 16-bit 또는 8-bit 양자화로 모델 크기 50% 감소
- **프루닝**: 불필요한 가중치 제거
- **지식 증류**: 큰 모델의 지식을 작은 모델로 전달

### 2. 웹 최적화
- **모델 캐싱**: 브라우저 캐시 활용
- **점진적 로딩**: 모델을 청크 단위로 로딩
- **웹워커**: 메인 스레드 블로킹 방지

### 3. 사용자 경험 개선
- **로딩 인디케이터**: 진행률 표시
- **오류 처리**: 사용자 친화적 오류 메시지
- **반응형 디자인**: 모바일 최적화

## 🔍 API 통합 예시

### REST API 래퍼 (Flask)
```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('models/best_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    
    # 전처리
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # 예측
    prediction = model.predict(image_array)[0][0]
    is_foreigner_card = prediction > 0.5
    confidence = prediction if is_foreigner_card else 1 - prediction
    
    return jsonify({
        'is_foreigner_card_back': bool(is_foreigner_card),
        'confidence': float(confidence),
        'raw_score': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## 📝 라이센스 및 주의사항

### 1. 데이터 개인정보보호
- 개인정보가 포함된 이미지 처리 시 GDPR, 개인정보보호법 준수
- 데이터 암호화 및 안전한 저장
- 사용자 동의 절차 필수

### 2. 모델 성능 한계
- 완벽한 정확도 보장 불가
- 실제 운영 환경에서 지속적인 모니터링 필요
- 사람의 최종 검토 권장

### 3. 보안 고려사항
- 클라이언트 사이드 검증의 한계
- 서버 사이드 추가 검증 권장
- 악의적 이미지 업로드 방지

## 🤝 기여 및 개선

### 1. 성능 개선 아이디어
- 더 많은 데이터 수집
- 앙상블 모델 적용
- 실시간 학습 (Online Learning)
- 사용자 피드백 반영

### 2. 기능 확장
- 다국가 신분증 지원
- OCR 기능 추가
- 문서 위변조 탐지
- 모바일 앱 개발

### 3. 커뮤니티 기여
- GitHub Issues를 통한 버그 리포트
- Pull Request를 통한 코드 개선
- 데이터셋 공유 (개인정보 제거 후)
- 성능 벤치마크 공유

---

## 📞 지원 및 문의

프로젝트 관련 문의사항이나 개선 제안이 있으시면 언제든 연락주세요!

**Happy Coding! 🎉**