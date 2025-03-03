import torch
print(torch.__version__)  # PyTorch 버전 출력
print(torch.cuda.is_available())  # True가 출력되면 GPU 사용 가능
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # GPU 이름 출력
