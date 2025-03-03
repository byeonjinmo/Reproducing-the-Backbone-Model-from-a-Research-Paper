import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, device, epochs=90, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_losses = []
    train_accuracies = []

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()   # 옵티마이저 기울기 초기화
            outputs = model(inputs)   # 순전
            loss = criterion(outputs, labels)    # 손실값 계산
            loss.backward()    # 역전
            optimizer.step()   # 최적화

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 배치별 중간 손실값 및 정확도 출력
            if batch_idx % 10 == 0:
                print(
                    f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

        scheduler.step()

        # 에포크별 손실값 및 정확도 저장
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # 에포크 결과 출력
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return train_losses, train_accuracies
