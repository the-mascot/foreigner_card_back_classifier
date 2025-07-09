# Foreigner Card Back Classifier

외국인등록증 뒷면 분류를 위한 머신러닝 프로젝트입니다. TensorFlow.js로 변환하여 프론트엔드에서 실시간 검증이 가능합니다.

## 프로젝트 구조

```
foreigner_card_back_classifier/
├── data/                           # 데이터셋
│   ├── train/                      # 훈련 데이터
│   │   ├── foreigner_card_back/    # 외국인등록증 뒷면 이미지
│   │   └── other_documents/        # 기타 문서 이미지
│   └── validation/                 # 검증 데이터
│       ├── foreigner_card_back/
│       └── other_documents/
├── models/                         # 저장된 모델
├── src/                           # 소스 코드
├── notebooks/                     # 주피터 노트북
├── web_demo/                      # 웹 데모
└── requirements.txt               # 의존성
```

## 설치 및 실행

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 데이터 준비:
- `data/train/foreigner_card_back/`에 외국인등록증 뒷면 이미지 저장
- `data/train/other_documents/`에 기타 문서 이미지 저장

4. 모델 훈련:
```bash
python src/train_model.py
```

5. TensorFlow.js 변환:
```bash
python src/convert_to_tfjs.py
```

6. 웹 데모 실행:
```bash
python web_demo/app.py
```

## 특징

- 이진 분류 (외국인등록증 뒷면 vs 기타 문서)
- 실시간 프론트엔드 검증 가능
- 경량화된 MobileNet 기반 모델
- 웹캠/파일 업로드 지원
