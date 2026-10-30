from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 클래스 목록 (7개 감정)
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# 하이퍼파라미터 및 경로 설정
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
PLOT_PATH = MODEL_DIR / "training_history.png"

BATCH_SIZE = 64
EPOCHS = 5
IMAGE_SIZE = 96
LEARNING_RATE = 0.001
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_dataloaders(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Train 및 Validation DataLoader를 생성합니다."""
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int = 7) -> nn.Module:
    """Pretrained ResNet18 모델을 불러오고 마지막 fc 층을 교체합니다."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """1 Epoch 동안 모델을 학습하고 평균 loss와 정확도를 반환합니다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validation 데이터셋에 대해 평균 loss와 정확도를 평가합니다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def save_training_plot(
    history: dict[str, list[float]],
    save_path: Path = PLOT_PATH,
) -> None:
    """학습 완료 후 정확도 및 Loss 변화 그래프를 저장합니다."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # 1. Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], "r-o", label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 2. Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], "b-o", label="Train Accuracy")
    plt.plot(epochs_range, history["val_acc"], "r-o", label="Val Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"학습 결과 그래프가 저장되었습니다: {save_path}")


def main() -> None:
    # 1. Device 자동 감지 (GPU 우선, 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 2. 데이터 로더 준비
    train_loader, val_loader, class_names = get_dataloaders()
    print(f"클래스 목록 ({len(class_names)}개): {class_names}")
    print(f"학습 데이터: {len(train_loader.dataset):,}장 | 검증 데이터: {len(val_loader.dataset):,}장")

    # 3. 모델, 손실 함수, 옵티마이저 초기화
    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "=" * 75)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^12} | {'Val Loss':^12} | {'Val Acc':^12} | {'Time':^8}")
    print("-" * 75)

    # 4. 학습 루프
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        saved_mark = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            saved_mark = " *Best"

        print(
            f"{epoch:^8d}/{EPOCHS:<2d} | "
            f"{train_loss:^12.4f} | {train_acc:^11.2f}% | "
            f"{val_loss:^12.4f} | {val_acc:^11.2f}% | "
            f"{elapsed:^6.1f}s{saved_mark}"
        )

    print("=" * 75)
    print(f"학습 완료! 최고 Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"최고 성능 모델 저장 위치: {BEST_MODEL_PATH}")

    # 5. 학습 그래프 저장
    save_training_plot(history, PLOT_PATH)


if __name__ == "__main__":
    main()
