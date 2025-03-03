import torch
from model.vivit import ViViT
from transformers import AutoModelForVideoClassification

def initialize_model(image_size, num_classes, num_frames, use_pretrained=False, hf_token=None):
    """
    ViViT 모델을 초기화하고, 필요한 경우 사전 학습된 가중치를 로드합니다.

    Args:
        image_size (int): 입력 이미지 크기.
        num_classes (int): 출력 클래스 수.
        num_frames (int): 비디오 프레임 수.
        use_pretrained (bool): 사전 학습된 가중치 사용 여부.
        hf_token (str): Hugging Face 토큰.

    Returns:
        torch.nn.Module: 초기화된 모델.
    """
    model = ViViT(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        num_frames=num_frames
    )

    if use_pretrained:
        print("Loading pretrained weights...")
        pretrained_model = AutoModelForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400", token=hf_token
        )
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
