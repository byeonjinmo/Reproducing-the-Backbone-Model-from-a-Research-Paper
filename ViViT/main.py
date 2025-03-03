import argparse
import torch
from train import train_and_evaluate
from dataset.data_loader import *
from model.model_loader import initialize_model


def main():
    # 명령줄 인자로 필요한 설정만 입력받기
    parser = argparse.ArgumentParser(description="ViViT Training Pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,  # 필수 항목 해제
        default="/home/smc/PycharmProjects/paper code/vivit/hmdb51"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    #parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for ViViT")
    parser.add_argument("--num_classes", type=int, default=51, help="Number of classes in the dataset")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for pretrained weights", default=None)
    #parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    #parser.add_argument("--early_stopping", type=int, default=None, help="Early stopping patience")
    #parser.add_argument("--clip_grad_norm", type=float, default=None, help="Clip gradients to this norm value")
    args = parser.parse_args()

    #데이터 로더 생성
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_frames=args.num_frames
    )

    # 모델 초기화
    model = initialize_model(
        image_size=args.image_size,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        #dropout_rate=args.dropout_rate,  # 드롭아웃 비율 추가
        use_pretrained=args.use_pretrained,
        hf_token=args.hf_token
    )

    # 학습 및 검증 실행
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        #use_scheduler=args.use_scheduler,
        #early_stopping=args.early_stopping,
        #clip_grad_norm=args.clip_grad_norm,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    main()
