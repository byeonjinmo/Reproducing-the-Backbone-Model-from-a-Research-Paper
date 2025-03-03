import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
from model.vivit import ViViT
from dataset.hmdb51_dataset import HMDB51Dataset
import argparse


def test_model(model, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_test_loss = 0
    correct_test = 0
    total_test = 0
    clip_correct_test = 0  # Clip-level Accuracy

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # ViViT 모델 입력 형태로 변환
            inputs = rearrange(inputs, 'b t c h w -> b t c h w')

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_test_loss += loss.item()
            total_test += labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            clip_correct_test += (torch.argmax(outputs, dim=1) == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_clip_accuracy = 100 * clip_correct_test / total_test

    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}% | Test Clip Accuracy: {test_clip_accuracy:.2f}%")

    return avg_test_loss, test_accuracy, test_clip_accuracy


def main():
    parser = argparse.ArgumentParser(description="Test ViViT on HMDB51 dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--num_classes", type=int, default=51, help="Number of classes in HMDB51 dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HMDB51 dataset")
    args = parser.parse_args()

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 테스트 데이터 전처리
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 테스트 데이터셋 및 데이터 로더 생성
    test_dataset = HMDB51Dataset(root_dir=args.data_dir, num_frames=args.num_frames, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ViViT 모델 초기화
    model = ViViT(
        image_size=args.image_size,
        patch_size=16,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        dropout=0.0
    )

    # 저장된 가중치 로드
    print(f"Loading model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # 테스트 수행
    avg_test_loss, test_accuracy, test_clip_accuracy = test_model(model, test_loader, device)

    # 결과 출력
    print("Final Test Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Top-1 Accuracy: {test_accuracy:.2f}%")
    print(f"Test Clip-Level Accuracy: {test_clip_accuracy:.2f}%")


if __name__ == "__main__":
    main()
