# import torch
# from model import AlexNet
# from dataset import load_data
# from train import train_model
# from test import test_model
# import matplotlib.pyplot as plt
#
# import matplotlib.pyplot as plt
#
# # def plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies):
# #     epochs = range(1, len(train_losses) + 1)
# #
# #     # 손실값 시각화
# #     plt.figure(figsize=(12, 6))
# #
# #     # Loss 그래프
# #     plt.subplot(1, 2, 1)
# #     plt.plot(epochs, train_losses, 'b-', label='Train Loss')  # 훈련 손실
# #     plt.plot(epochs, test_losses, 'r-', label='Test Loss')    # vivit_2 손실
# #     plt.title('Train and Test Loss')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.legend()
# #
# #     # 정확도 시각화
# #     plt.subplot(1, 2, 2)
# #     plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')  # 훈련 정확도
# #     plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')    # vivit_2 정확도
# #     plt.title('Train and Test Accuracy')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Accuracy (%)')
# #     plt.legend()
# #
# #     plt.show()
# import torch
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = AlexNet(num_classes=10).to(device)
#
#     # 데이터 경로 설정 (전체 데이터가 클래스별로 정리되어 있는 디렉토리)
#     #data_dir = 'C:/Users/user/Desktop/24년도 연구/10_11/labtop_dev/archive (8)'
#
#     # 데이터 경로 설정
#     train_data_dir = 'C:/Users/user/Desktop/24년도 연구/10_11/labtop_dev/archive (7)/train'
#     test_data_dir = 'C:/Users/user/Desktop/24년도 연구/10_11/labtop_dev/archive (7)/test'
#
#     # 데이터 로더 설정
#     train_loader = load_data(data_dir=train_data_dir, batch_size=64, is_train=True)  # 훈련 데이터
#     test_loader = load_data(data_dir=test_data_dir, batch_size=64, is_train=False)   # vivit_2 데이터
#
#     # 학습 및 vivit_2 데이터 저장용 리스트 초기화
#     train_losses, train_accuracies = [], []
#     test_losses, test_accuracies = [], []
#
#     epochs = 90  # 에포크 수 설정
#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}")
#
#         # 에포크마다 학습
#         model.train()  # 모델을 학습 모드로 설정
#         train_loss, train_accuracy = train_model(model, train_loader, device, epochs=1, lr=0.002)
#         train_losses.extend(train_loss)
#         train_accuracies.extend(train_accuracy)
#
#         # 에포크마다 vivit_2
#         model.eval()  # 모델을 평가 모드로 전환
#         with torch.no_grad():  # vivit_2 시에는 그래디언트 계산 비활성화
#             test_loss, test_accuracy = test_model(model, test_loader, device)
#             test_losses.append(test_loss[0])  # 테스트는 매 에포크마다 하나씩 결과를 기록
#             test_accuracies.append(test_accuracy[0])
#
#         print(f"Test Loss: {test_loss[0]:.4f}, Accuracy: {test_accuracy[0]:.2f}%")
#
#     # 손실값 및 정확도 시각화
#     #plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies)
#
# if __name__ == "__main__":
#     main()
import torch

from model import ResNet
from model import Bottleneck
from model import weights_init_kaiming

from dcmdataset import get_dicom_data_loaders
from train import train_model
from test import test_model
from utils import plot_loss_accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    # 가중치 초기화 적용
    model.apply(weights_init_kaiming)
    # 데이터 경로 설정 (전체 데이터가 클래스별로 정리되어 있는 디렉토리)
    data_dir = '/home/smc/Desktop/jinmo/project/dataset/Brain Tumor MRI/dicom-flair/dicom-flair'

    # DICOM 데이터 로더 설정 (훈련 및 vivit_2 데이터 분리 포함)
    train_loader, test_loader = get_dicom_data_loaders(data_dir, batch_size=64)

    # 학습 및 vivit_2 데이터 저장용 리스트 초기화
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    epochs = 70  # 에포크 수 설정
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Using device: {device}")  # GPU가 사용 중인지 확인

        # 에포크마다 학습
        model.train()  # 모델을 학습 모드로 설정
        train_loss, train_accuracy = train_model(model, train_loader, device, epochs=1, lr=0.01)
        train_losses.extend(train_loss)
        train_accuracies.extend(train_accuracy)

        # 에포크마다 vivit_2
        model.eval()  # 모델을 평가 모드로 전환
        with torch.no_grad():  # vivit_2 시에는 그래디언트 계산 비활성화
            test_loss, test_accuracy = test_model(model, test_loader, device)
            test_losses.append(test_loss[0])  # 테스트는 매 에포크마다 하나씩 결과를 기록
            test_accuracies.append(test_accuracy[0])

        print(f"Test Loss: {test_loss[0]:.4f}, Accuracy: {test_accuracy[0]:.2f}%")

    # 손실값 및 정확도 시각화
    plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies)

if __name__ == "__main__":
    main()
