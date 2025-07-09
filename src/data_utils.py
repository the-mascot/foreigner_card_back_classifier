"""
데이터 전처리 및 로딩을 위한 유틸리티
"""
import tensorflow as tf
import cv2
import numpy as np
import os
from typing import Tuple, List
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224), batch_size: int = 32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
    def preprocess_image(self, image_path: str) -> tf.Tensor:
        """이미지 전처리"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    def create_dataset(self, split: str = 'train') -> tf.data.Dataset:
        """데이터셋 생성"""
        data_path = os.path.join(self.data_dir, split)
        
        # 클래스별 이미지 경로 수집
        foreigner_card_dir = os.path.join(data_path, 'foreigner_card_back')
        other_documents_dir = os.path.join(data_path, 'other_documents')
        
        image_paths = []
        labels = []
        
        # 외국인등록증 뒷면 이미지 (라벨: 1)
        if os.path.exists(foreigner_card_dir):
            for img_file in os.listdir(foreigner_card_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(foreigner_card_dir, img_file))
                    labels.append(1)
        
        # 기타 문서 이미지 (라벨: 0)
        if os.path.exists(other_documents_dir):
            for img_file in os.listdir(other_documents_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(other_documents_dir, img_file))
                    labels.append(0)
        
        # TensorFlow 데이터셋 생성
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(
            lambda x, y: (self.preprocess_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self, split: str = 'train') -> dict:
        """클래스 가중치 계산 (불균형 데이터 처리)"""
        data_path = os.path.join(self.data_dir, split)
        
        foreigner_card_count = len([f for f in os.listdir(os.path.join(data_path, 'foreigner_card_back')) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        other_documents_count = len([f for f in os.listdir(os.path.join(data_path, 'other_documents'))
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        total = foreigner_card_count + other_documents_count
        
        if total == 0:
            return {0: 1.0, 1: 1.0}
        
        weight_for_0 = total / (2.0 * other_documents_count) if other_documents_count > 0 else 1.0
        weight_for_1 = total / (2.0 * foreigner_card_count) if foreigner_card_count > 0 else 1.0
        
        return {0: weight_for_0, 1: weight_for_1}

def augment_data(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """데이터 증강"""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    
    def augment(image, label):
        return data_augmentation(image, training=True), label
    
    return dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

def visualize_samples(dataset: tf.data.Dataset, num_samples: int = 9):
    """데이터셋 샘플 시각화"""
    plt.figure(figsize=(12, 12))
    
    for i, (images, labels) in enumerate(dataset.take(1)):
        for j in range(min(num_samples, len(images))):
            plt.subplot(3, 3, j + 1)
            plt.imshow(images[j])
            class_name = "외국인등록증 뒷면" if labels[j] == 1 else "기타 문서"
            plt.title(f'{class_name}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
