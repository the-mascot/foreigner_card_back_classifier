"""
데이터 준비 도구 모음
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def setup_data_structure():
    """데이터 폴더 구조 자동 생성"""
    print("📁 데이터 폴더 구조 생성 중...")
    
    folders = [
        "data/train/foreigner_card_back",
        "data/train/other_documents", 
        "data/validation/foreigner_card_back",
        "data/validation/other_documents",
        "data/raw/foreigner_card_back",  # 원본 보관용
        "data/raw/other_documents"       # 원본 보관용
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ {folder}")
    
    # README 파일 생성
    readme_content = """# 데이터 폴더 사용법

## 📁 폴더 구조
- raw/: 원본 이미지 보관 (백업용)
- train/: 모델 훈련용 데이터 (80%)
- validation/: 모델 검증용 데이터 (20%)

## 📸 이미지 추가 방법
1. raw/ 폴더에 원본 이미지 저장
2. split_data_automatically() 함수로 자동 분할
3. python src/train_model.py로 훈련 시작

## 🎯 클래스별 설명
- foreigner_card_back/: 외국인등록증 뒷면만!
- other_documents/: 주민등록증, 여권, 운전면허증, 외국인등록증 앞면 등
"""
    
    with open("data/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📋 data/README.md 생성 완료")

def check_data_status():
    """현재 데이터 상태 확인"""
    print("=== 📊 데이터 상태 확인 ===")
    
    paths = {
        "훈련용 외국인등록증 뒷면": "data/train/foreigner_card_back",
        "훈련용 기타 문서": "data/train/other_documents",
        "검증용 외국인등록증 뒷면": "data/validation/foreigner_card_back", 
        "검증용 기타 문서": "data/validation/other_documents"
    }
    
    total_train = 0
    total_val = 0
    
    for desc, path in paths.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{desc}: {count}장")
            
            if "훈련용" in desc:
                total_train += count
            else:
                total_val += count
        else:
            print(f"{desc}: 폴더 없음")
    
    print(f"\n📈 총계:")
    print(f"  훈련 데이터: {total_train}장")
    print(f"  검증 데이터: {total_val}장")
    print(f"  전체 데이터: {total_train + total_val}장")
    
    # 권장사항
    if total_train < 400:
        print(f"\n⚠️ 훈련 데이터가 부족합니다. (권장: 800장 이상)")
    elif total_train < 800:
        print(f"\n✅ 최소 훈련 데이터 확보. (권장: 800장 이상)")
    else:
        print(f"\n🎉 충분한 훈련 데이터!")
    
    return total_train, total_val

def split_data_automatically(source_dir="data/raw", train_ratio=0.8):
    """원본 데이터를 훈련/검증으로 자동 분할"""
    print(f"🔄 데이터 자동 분할 시작 (훈련:{train_ratio*100:.0f}% / 검증:{(1-train_ratio)*100:.0f}%)")
    
    # 원본 경로
    source_foreigner = Path(source_dir) / "foreigner_card_back"
    source_other = Path(source_dir) / "other_documents"
    
    # 목표 경로
    targets = {
        "train_foreigner": Path("data/train/foreigner_card_back"),
        "train_other": Path("data/train/other_documents"),
        "val_foreigner": Path("data/validation/foreigner_card_back"),
        "val_other": Path("data/validation/other_documents")
    }
    
    # 기존 데이터 정리 (선택사항)
    response = input("기존 train/validation 데이터를 삭제하고 새로 분할하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        for target in targets.values():
            if target.exists():
                shutil.rmtree(target)
                target.mkdir(parents=True)
    
    # 폴더 생성
    for target in targets.values():
        target.mkdir(parents=True, exist_ok=True)
    
    # 외국인등록증 뒷면 분할
    if source_foreigner.exists():
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(source_foreigner.glob(ext)))
        
        random.shuffle(images)
        split_point = int(len(images) * train_ratio)
        
        print(f"📁 외국인등록증 뒷면: {len(images)}장")
        print(f"  → 훈련용: {split_point}장")
        print(f"  → 검증용: {len(images) - split_point}장")
        
        for i, img in enumerate(images):
            if i < split_point:
                shutil.copy2(img, targets["train_foreigner"])
            else:
                shutil.copy2(img, targets["val_foreigner"])
    
    # 기타 문서 분할
    if source_other.exists():
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(source_other.glob(ext)))
        
        random.shuffle(images)
        split_point = int(len(images) * train_ratio)
        
        print(f"📁 기타 문서: {len(images)}장")
        print(f"  → 훈련용: {split_point}장")
        print(f"  → 검증용: {len(images) - split_point}장")
        
        for i, img in enumerate(images):
            if i < split_point:
                shutil.copy2(img, targets["train_other"])
            else:
                shutil.copy2(img, targets["val_other"])
    
    print("✅ 데이터 분할 완료!")

def validate_images(data_dir="data"):
    """이미지 파일 유효성 검사"""
    print("🔍 이미지 파일 유효성 검사...")
    
    paths = [
        "train/foreigner_card_back",
        "train/other_documents",
        "validation/foreigner_card_back", 
        "validation/other_documents"
    ]
    
    corrupted_files = []
    small_images = []
    
    for path in paths:
        full_path = Path(data_dir) / path
        if not full_path.exists():
            continue
        
        print(f"\n📁 {path} 검사 중...")
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(list(full_path.glob(ext)))
        
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # 너무 작은 이미지 체크
                    if width < 200 or height < 200:
                        small_images.append((str(img_path), width, height))
                    
                    # 이미지 로드 테스트
                    img.verify()
                    
            except Exception as e:
                corrupted_files.append((str(img_path), str(e)))
    
    # 결과 출력
    if corrupted_files:
        print(f"\n❌ 손상된 파일 {len(corrupted_files)}개:")
        for file, error in corrupted_files:
            print(f"  {file}: {error}")
    
    if small_images:
        print(f"\n⚠️ 너무 작은 이미지 {len(small_images)}개:")
        for file, w, h in small_images:
            print(f"  {file}: {w}x{h}")
    
    if not corrupted_files and not small_images:
        print("✅ 모든 이미지가 정상입니다!")
    
    return corrupted_files, small_images

def show_sample_images(data_dir="data/train", samples_per_class=3):
    """클래스별 샘플 이미지 표시"""
    print("🖼️ 샘플 이미지 표시...")
    
    fig, axes = plt.subplots(2, samples_per_class, figsize=(15, 8))
    
    classes = ["foreigner_card_back", "other_documents"]
    class_names = ["외국인등록증 뒷면", "기타 문서"]
    
    for i, (class_dir, class_name) in enumerate(zip(classes, class_names)):
        path = Path(data_dir) / class_dir
        if path.exists():
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images.extend(list(path.glob(ext)))
            
            # 랜덤 샘플링
            if len(images) >= samples_per_class:
                samples = random.sample(images, samples_per_class)
            else:
                samples = images
            
            for j, img_path in enumerate(samples):
                if j < samples_per_class:
                    try:
                        img = Image.open(img_path)
                        axes[i, j].imshow(img)
                        axes[i, j].set_title(f"{class_name}\n{img_path.name}")
                        axes[i, j].axis('off')
                    except Exception as e:
                        axes[i, j].text(0.5, 0.5, f"로딩 실패\n{e}", 
                                       ha='center', va='center')
                        axes[i, j].axis('off')
            
            # 빈 칸 처리
            for j in range(len(samples), samples_per_class):
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig("data_samples.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 샘플 이미지를 data_samples.png로 저장했습니다.")

if __name__ == "__main__":
    print("=== 📸 데이터 준비 도구 ===")
    print("1. setup_data_structure()     # 폴더 구조 생성")
    print("2. check_data_status()        # 현재 데이터 상태 확인")  
    print("3. split_data_automatically() # 원본에서 train/val 분할")
    print("4. validate_images()          # 이미지 유효성 검사")
    print("5. show_sample_images()       # 샘플 이미지 확인")
    print("\n사용법:")
    print("python data_tools.py")
    print("그 후 원하는 함수 실행")
    
    # 폴더 구조 자동 생성
    setup_data_structure()
    
    # 현재 상태 확인
    check_data_status()
