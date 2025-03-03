import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # ViViT 모델 입력 형태 변환
        inputs = rearrange(inputs, 'b t c h w -> b t c h w')

        # 모델 학습
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 손실 및 정확도 계산
        total_loss += loss.item()
        total_samples += labels.size(0)

        # Top-1 및 Top-5 정확도
        _, preds = outputs.topk(5, dim=1)
        correct_top1 += (preds[:, 0] == labels).sum().item()
        correct_top5 += (preds == labels.unsqueeze(1)).sum().item()

    avg_loss = total_loss / len(train_loader)
    top1_accuracy = 100 * correct_top1 / total_samples
    top5_accuracy = 100 * correct_top5 / total_samples

    return avg_loss, top1_accuracy, top5_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # ViViT 모델 입력 형태 변환
            inputs = rearrange(inputs, 'b t c h w -> b t c h w')

            # 모델 평가
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 손실 및 정확도 계산
            total_loss += loss.item()
            total_samples += labels.size(0)

            # Top-1 및 Top-5 정확도
            _, preds = outputs.topk(5, dim=1)
            correct_top1 += (preds[:, 0] == labels).sum().item()
            correct_top5 += (preds == labels.unsqueeze(1)).sum().item()

    avg_loss = total_loss / len(val_loader)
    top1_accuracy = 100 * correct_top1 / total_samples
    top5_accuracy = 100 * correct_top5 / total_samples

    return avg_loss, top1_accuracy, top5_accuracy


def plot_training_results(epochs, train_losses, val_losses, train_top1, val_top1, train_top5, val_top5):
    plt.figure(figsize=(12, 6))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_top1, label="Train Top-1 Accuracy")
    plt.plot(epochs, val_top1, label="Validation Top-1 Accuracy")
    plt.plot(epochs, train_top5, label="Train Top-5 Accuracy")
    plt.plot(epochs, val_top5, label="Validation Top-5 Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_and_evaluate(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, device):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_top1_accuracies, val_top1_accuracies = [], []
    train_top5_accuracies, val_top5_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # 한 에포크 학습
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        train_top1_accuracies.append(train_top1)
        train_top5_accuracies.append(train_top5)

        # 검증
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_top1_accuracies.append(val_top1)
        val_top5_accuracies.append(val_top5)

        # 에포크 결과 출력
        print(f"Train Loss: {train_loss:.4f} | Train Top-1 Accuracy: {train_top1:.2f}% | Train Top-5 Accuracy: {train_top5:.2f}%")
        print(f"Validation Loss: {val_loss:.4f} | Validation Top-1 Accuracy: {val_top1:.2f}% | Validation Top-5 Accuracy: {val_top5:.2f}%")
        print("-" * 50)

    # 학습 결과 시각화
    plot_training_results(
        range(1, num_epochs + 1),
        train_losses, val_losses,
        train_top1_accuracies, val_top1_accuracies,
        train_top5_accuracies, val_top5_accuracies
    )
