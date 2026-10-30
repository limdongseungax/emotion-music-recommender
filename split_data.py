import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_train_val(
    train_dir: Path = Path("data/train"),
    val_dir: Path = Path("data/val"),
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    if not train_dir.exists():
        raise FileNotFoundError(f"'{train_dir}' 폴더를 찾을 수 없습니다.")

    emotion_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not emotion_dirs:
        print(f"'{train_dir}' 내부에 감정별 하위 폴더가 없습니다.")
        return

    # 중복 실행 방지 확인
    existing_val_count = sum(len(get_images(val_dir / d.name)) for d in emotion_dirs)
    if existing_val_count > 0:
        print(f"[경고] '{val_dir}' 폴더에 이미 {existing_val_count}개의 이미지가 존재합니다. 중복 분할을 건너뜁니다.")
    else:
        random.seed(seed)
        val_dir.mkdir(parents=True, exist_ok=True)

        for emotion_path in emotion_dirs:
            emotion = emotion_path.name
            target_val_path = val_dir / emotion
            target_val_path.mkdir(parents=True, exist_ok=True)

            images = get_images(emotion_path)
            num_to_move = int(len(images) * val_ratio)
            sampled_images = random.sample(images, num_to_move)

            for img_path in sampled_images:
                shutil.move(str(img_path), str(target_val_path / img_path.name))

    # 실행 후 각 폴더별 이미지 수 출력
    print("=" * 60)
    print(f"{'Emotion':<15} | {'data/train':>12} | {'data/val':>12} | {'Total':>12}")
    print("-" * 60)

    total_train = 0
    total_val = 0

    for emotion_path in emotion_dirs:
        emotion = emotion_path.name
        train_count = len(get_images(train_dir / emotion))
        val_count = len(get_images(val_dir / emotion))
        total = train_count + val_count

        total_train += train_count
        total_val += val_count

        print(f"{emotion:<15} | {train_count:>12,d} | {val_count:>12,d} | {total:>12,d}")

    print("-" * 60)
    print(
        f"{'TOTAL':<15} | {total_train:>12,d} | {total_val:>12,d} | {total_train + total_val:>12,d}"
    )
    print("=" * 60)


if __name__ == "__main__":
    split_train_val()
