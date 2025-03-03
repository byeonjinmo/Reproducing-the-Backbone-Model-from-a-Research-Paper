import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np


class HMDB51Dataset(Dataset):
    """HMDB51 데이터셋 로드 및 전처리"""
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        Args:
            root_dir (str): HMDB51 데이터셋의 루트 디렉토리.
            num_frames (int): 비디오에서 샘플링할 프레임 수.
            transform (callable, optional): 데이터 전처리/증강 함수.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.video_paths, self.labels = self._load_metadata()

    def _load_metadata(self):
        """비디오 파일 경로와 레이블 정보를 읽어옴"""
        video_paths = []
        labels = []
        class_idx = {}
        class_dirs = sorted(os.listdir(self.root_dir))

        # 클래스 이름에 따라 인덱스 매핑
        for idx, class_name in enumerate(class_dirs):
            class_idx[class_name] = idx

        # 비디오 경로와 레이블 추출
        for class_name in class_dirs:
            class_path = os.path.join(self.root_dir, class_name)
            for video_file in os.listdir(class_path):
                video_paths.append(os.path.join(class_path, video_file))
                labels.append(class_idx[class_name])

        return video_paths, labels

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.video_paths)

    def _load_video_frames(self, video_path):
        """비디오 파일에서 프레임을 읽어 샘플링"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 균일 샘플링
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
                frames.append(frame)

        cap.release()

        # 부족한 프레임은 복제하여 채움
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return np.array(frames)

    def __getitem__(self, idx):
        """비디오 파일에서 프레임과 레이블 반환"""
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 비디오 프레임 로드 및 전처리
        frames = self._load_video_frames(video_path)
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])  # 프레임별로 transform 적용

        return frames, label
