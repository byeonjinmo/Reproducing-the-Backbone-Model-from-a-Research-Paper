import os
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.hmdb51_dataset import HMDB51Dataset

def get_data_loaders(data_dir, batch_size, image_size, num_frames):
    """
    데이터 로더를 반환하는 함수.

    Args:
        data_dir (str): HMDB51 데이터셋 경로.
        batch_size (int): 배치 크기.
        image_size (int): 이미지 크기.
        num_frames (int): 비디오 클립당 프레임 수.

    Returns:
        train_loader (DataLoader): 학습 데이터 로더.
        val_loader (DataLoader): 검증 데이터 로더.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} not found.")

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 생성
    train_dataset = HMDB51Dataset(root_dir=data_dir, num_frames=num_frames, transform=transform)
    val_dataset = HMDB51Dataset(root_dir=data_dir, num_frames=num_frames, transform=val_transform)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
