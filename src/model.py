"""
외국인등록증 뒷면 분류 모델 정의
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple

def create_mobilenet_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    trainable_layers: int = 20
) -> Model:
    """
    MobileNetV2 기반 분류 모델 생성
    프론트엔드 배포를 위해 경량화된 모델 사용
    """
    # MobileNetV2 백본 로드
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 상위 레이어만 훈련 가능하도록 설정
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # 입력 레이어
    inputs = tf.keras.Input(shape=input_shape)
    
    # 전처리
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # 백본 모델
    x = base_model(x, training=False)
    
    # 분류 헤드
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    if num_classes == 2:
        # 이진 분류
        outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        # 다중 분류
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs)
    return model

def create_custom_cnn_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2
) -> Model:
    """
    커스텀 CNN 분류 모델 (더 가벼운 모델)
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        
        # 첫 번째 컨볼루션 블록
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # 두 번째 컨볼루션 블록
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # 세 번째 컨볼루션 블록
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # 네 번째 컨볼루션 블록
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # 분류 헤드
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # 출력 레이어
        layers.Dense(1 if num_classes == 2 else num_classes, 
                    activation='sigmoid' if num_classes == 2 else 'softmax',
                    name='predictions')
    ])
    
    return model

def create_efficient_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2
) -> Model:
    """
    TensorFlow.js 최적화를 위한 효율적인 분류 모델
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # 정규화
    x = layers.Rescaling(1./255)(inputs)
    
    # 첫 번째 블록
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # 두 번째 블록
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # 세 번째 블록
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # 네 번째 블록
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    # 분류 헤드
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # 출력
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='foreigner_card_classifier')
    return model

def get_model_summary(model: Model) -> None:
    """모델 요약 정보 출력"""
    print("모델 구조:")
    model.summary()
    
    print(f"\n총 파라미터 수: {model.count_params():,}")
    
    # 모델 크기 추정 (MB)
    param_size = model.count_params() * 4  # float32 = 4 bytes
    model_size_mb = param_size / (1024 * 1024)
    print(f"예상 모델 크기: {model_size_mb:.2f} MB")
