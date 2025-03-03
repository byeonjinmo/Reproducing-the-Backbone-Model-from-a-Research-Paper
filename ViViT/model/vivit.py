import torch
from torch import nn
from einops import rearrange, repeat  # 재구성을 위한 einops 함수들
from einops.layers.torch import Rearrange  # rearrange 레이어
from model.module import Attention, PreNorm, FeedForward  # 외부 모듈에서 Attention, PreNorm, FeedForward 클래스 가져옴


# Transformer 블록 정의
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer 블록 리스트로 레이어 구성
        self.norm = nn.LayerNorm(dim)  # 마지막 레이어 정규화를 위한 LayerNorm 정의

        # Transformer의 각 레이어를 쌓는 반복문
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # Attention 레이어
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # FeedForward 레이어
            ]))

    def forward(self, x):
        # 모든 Transformer 레이어에 대해 연속적으로 Attention과 FeedForward 수행
        for attn, ff in self.layers:
            x = attn(x) + x  # Attention을 수행 후 residual connection 추가
            x = ff(x) + x  # FeedForward를 수행 후 residual connection 추가
        return self.norm(x)  # 정규화 후 출력


# ViViT 모델 정의
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim=192, depth=4, heads=3, pool='cls',
                 in_channels=3, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        # 풀링 방법을 cls 또는 mean으로 지정해야 함
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 이미지 크기가 패치 크기로 나누어 떨어져야 함
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2  # 패치 개수 계산
        patch_dim = in_channels * patch_size ** 2  # 각 패치의 차원 계산

        # 패치를 임베딩하는 계층
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),  # 이미지 패치 분할
            nn.Linear(patch_dim, dim),  # 패치를 선형 변환하여 원하는 차원으로 맞춤
        )

        # 위치 임베딩 및 공간 토큰 정의
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))  # 위치 임베딩 정의
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))  # 공간 토큰 (CLS 토큰처럼 사용)
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)  # 공간 Transformer 정의

        # 시간 토큰 및 Transformer 정의
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))  # 시간 토큰 정의
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)  # 시간 Transformer 정의

        # 드롭아웃 및 풀링 방법 설정
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool  # 풀링 방식 지정 ('cls' 또는 'mean')

        # 최종 예측을 위한 MLP 헤드 정의
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 정규화 레이어
            nn.Linear(dim, num_classes)  # 최종 클래스 분류
        )

    def forward(self, x):
        # 입력을 패치 임베딩으로 변환
        x = self.to_patch_embedding(x)  # 패치 임베딩 변환
        b, t, n, _ = x.shape  # 배치 크기, 프레임 수, 패치 수, 차원으로 변환

        # 공간 CLS 토큰 생성 및 패치 임베딩에 추가
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)  # 배치와 프레임에 맞춰 토큰 반복
        x = torch.cat((cls_space_tokens, x), dim=2)  # CLS 토큰을 각 패치에 추가

        # 위치 임베딩 추가 및 드롭아웃
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        # 공간 Transformer에 입력하기 위한 재배치
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)  # 공간 Transformer를 통해 처리

        # 출력에서 첫 번째 토큰만 유지하여 시간 Transformer 입력으로 재배치
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # 시간 CLS 토큰 추가
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)  # 배치에 맞춰 CLS 토큰 반복
        x = torch.cat((cls_temporal_tokens, x), dim=1)  # 시간 CLS 토큰 추가

        # 시간 Transformer를 통해 처리
        x = self.temporal_transformer(x)

        # 풀링 방식에 따라 평균 풀링 또는 CLS 토큰 선택
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # mean 풀링 또는 CLS 토큰 선택

        # 최종 MLP 헤드를 통과하여 클래스 예측
        return self.mlp_head(x)
