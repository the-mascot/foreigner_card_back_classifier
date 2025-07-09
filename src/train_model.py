"""
모델 훈련 스크립트
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json

from data_utils import DataLoader, augment_data, visualize_samples
from model import create_mobilenet_classifier, create_efficient_classifier, get_model_summary

# 설정
CONFIG = {
    'data_dir': 'data',
    'model_dir': 'models',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'model_type': 'efficient',  # 'mobilenet', 'efficient', 'custom'
    'use_augmentation': True,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

def train_model():
    """모델 훈련 메인 함수"""
    print("=== 외국인등록증 뒷면 분류 모델 훈련 시작 ===")
    
    # 데이터 로더 생성
    print("데이터 로딩 중...")
    data_loader = DataLoader(
        data_dir=CONFIG['data_dir'],
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size']
    )
    
    # 데이터셋 생성
    train_dataset = data_loader.create_dataset('train')
    val_dataset = data_loader.create_dataset('validation')
    
    # 데이터 증강 적용 (훈련 데이터만)
    if CONFIG['use_augmentation']:
        print("데이터 증강 적용 중...")
        train_dataset = augment_data(train_dataset)
    
    # 클래스 가중치 계산
    class_weights = data_loader.get_class_weights('train')
    print(f"클래스 가중치: {class_weights}")
    
    # 데이터셋 샘플 시각화
    print("데이터셋 샘플 확인...")
    visualize_samples(train_dataset)
    
    # 모델 생성
    print(f"모델 생성 중... (타입: {CONFIG['model_type']})")
    if CONFIG['model_type'] == 'mobilenet':
        model = create_mobilenet_classifier(
            input_shape=CONFIG['img_size'] + (3,),
            num_classes=2
        )
    elif CONFIG['model_type'] == 'efficient':
        model = create_efficient_classifier(
            input_shape=CONFIG['img_size'] + (3,),
            num_classes=2
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {CONFIG['model_type']}")
    
    # 모델 요약
    get_model_summary(model)
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # 콜백 설정
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=CONFIG['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_dir'], 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 모델 훈련
    print("모델 훈련 시작...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 훈련 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 최종 모델 저장
    final_model_path = os.path.join(CONFIG['model_dir'], f'foreigner_card_classifier_{timestamp}.h5')
    model.save(final_model_path)
    print(f"최종 모델 저장됨: {final_model_path}")
    
    # 설정 및 히스토리 저장
    config_path = os.path.join(CONFIG['model_dir'], f'config_{timestamp}.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    
    history_path = os.path.join(CONFIG['model_dir'], f'history_{timestamp}.json')
    with open(history_path, 'w') as f:
        # history.history의 numpy 값들을 list로 변환
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        json.dump(history_dict, f, indent=2)
    
    # 훈련 결과 시각화
    plot_training_history(history, timestamp)
    
    # 모델 평가
    evaluate_model(model, val_dataset, timestamp)
    
    print("=== 훈련 완료 ===")
    return model, history

def plot_training_history(history, timestamp):
    """훈련 히스토리 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 손실
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # 정확도
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # 정밀도
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # 재현율
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['model_dir'], f'training_history_{timestamp}.png'))
    plt.show()

def evaluate_model(model, val_dataset, timestamp):
    """모델 평가"""
    print("=== 모델 평가 ===")
    
    # 검증 데이터에 대한 평가
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_dataset, verbose=1)
    
    # F1 스코어 계산
    f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    
    results = {
        'val_loss': float(val_loss),
        'val_accuracy': float(val_accuracy),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'f1_score': float(f1_score)
    }
    
    print(f"검증 손실: {val_loss:.4f}")
    print(f"검증 정확도: {val_accuracy:.4f}")
    print(f"검증 정밀도: {val_precision:.4f}")
    print(f"검증 재현율: {val_recall:.4f}")
    print(f"F1 스코어: {f1_score:.4f}")
    
    # 결과 저장
    results_path = os.path.join(CONFIG['model_dir'], f'evaluation_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # 모델 디렉토리 생성
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    # GPU 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 사용 가능: {len(gpus)}개")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    else:
        print("GPU 사용 불가, CPU로 훈련 진행")
    
    # 모델 훈련 실행
    model, history = train_model()
