import os
import pydicom
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def load_dicom_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    pixel_array = dicom.pixel_array
    pixel_array = np.clip(pixel_array, 0, 255)  # 0~255로 정규화
    image = Image.fromarray(pixel_array).convert("RGB")  # RGB로 변환
    return image

class DicomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 클래스별 폴더에서 이미지 경로 및 라벨 수집
        for label, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(".dcm"):
                        self.image_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        dicom_path = self.image_paths[idx]
        label = self.labels[idx]
        image = load_dicom_image(dicom_path)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dicom_data_loaders(data_dir, batch_size=64, train_split=0.8):
    # 전처리 설정
    transform = transforms.Compose([
        transforms.Resize(256),  # 256x256으로 크기 조정
        transforms.CenterCrop(227),  # 가운데 227x227로 크롭
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 데이터 평균
                             std=[0.229, 0.224, 0.225]),  # 데이터 표준편차
    ])

    # 전체 데이터셋 로드
    full_dataset = DicomDataset(data_dir=data_dir, transform=transform)

    # 데이터셋 크기 계산 후 분리
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size

    # 훈련/vivit_2 데이터 분할
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

