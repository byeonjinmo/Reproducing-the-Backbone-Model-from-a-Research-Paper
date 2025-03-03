import torch
import torch.nn as nn

def test_model(model, test_loader, device):
    model.eval()
    test_losses = []
    test_accuracies = []

    correct_top1 = 0
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.topk(1, 1, largest=True, sorted=True)
            total += labels.size(0)
            correct_top1 += (predicted[:, 0] == labels).sum().item()

            # 배치별 vivit_2 결과 출력
            if batch_idx % 10 == 0:
                print(f'Test Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.4f}, Accuracy: {100*correct_top1/total:.2f}%')

    # 에포크별 손실값 및 정확도 계산
    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = 100 * correct_top1 / total
    test_losses.append(epoch_loss)
    test_accuracies.append(epoch_accuracy)

    print(f'Test Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return test_losses, test_accuracies
