import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 함수(fn)를 인자로 받아 이 인자를 사용하여 모델을 구성

    def forward(self, x, **kwargs):
        # 원본 입력 x와 함수의 출력값을 더하여 반환
        return self.fn(x, **kwargs) + x  # Residual Connection으로 학습 안정성 향상


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # Layer Normalization을 생성
        self.fn = fn  # 주어진 함수를 저장

    def forward(self, x, **kwargs):
        # 입력에 대해 정규화 후 함수(fn)를 적용
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 차원을 확장하는 선형 계층
            nn.GELU(),  # GELU 활성화 함수
            nn.Dropout(dropout),  # 드롭아웃으로 과적합 방지
            nn.Linear(hidden_dim, dim),  # 차원을 원래대로 축소하는 선형 계층
            nn.Dropout(dropout)  # 드롭아웃 적용
        )

    def forward(self, x):
        # 피드포워드 네트워크를 통해 입력 x를 변환
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 각 헤드의 차원을 계산
        project_out = not (heads == 1 and dim_head == dim)  # 헤드가 1개인 경우 투영 생략

        self.heads = heads  # 다중 헤드 수
        self.scale = dim_head ** -0.5  # 스케일링 요소로 안정성을 높임

        # Query, Key, Value 생성을 위한 선형 계층 정의
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 출력 투영을 위한 선형 계층과 드롭아웃 설정
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 필요 없으면 Identity로 대체

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 배치, 토큰 개수, 차원, 헤드 수 정의
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Query, Key, Value로 분리
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # 스케일링된 dot-product attention 계산
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)  # 소프트맥스를 사용하여 가중치 계산

        # Value와 가중치를 곱하여 출력 계산
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 원래 차원으로 재구성
        out = self.to_out(out)  # 투영 및 드롭아웃
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 다중 헤드의 차원 정의
        self.heads = heads
        self.scale = dim_head ** -0.5  # 스케일링

        # Query, Key, Value를 생성하는 선형 계층 정의
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # ReAttention의 각 헤드 간 가중치를 학습하도록 파라미터 생성
        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        # ReAttention에 대한 정규화 정의
        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        # 출력 투영 계층 정의
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 배치, 토큰 수, 차원, 헤드 수
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Query, Key, Value 생성
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Attention 가중치 계산
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # ReAttention 가중치를 각 헤드에 적용
        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)  # 정규화 적용

        # Value와 가중치를 곱하여 최종 출력 계산
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 원래 차원으로 복원
        out = self.to_out(out)
        return out


class LeFF(nn.Module):
    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()
        scale_dim = dim * scale  # 차원을 확대

        # 차원 확장 및 배치 정규화를 위한 계층 정의
        self.up_proj = nn.Sequential(
            nn.Linear(dim, scale_dim),
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(scale_dim),
            nn.GELU(),
            Rearrange('b c (h w) -> b c h w', h=14, w=14)
        )

        # Depth-wise Convolution을 적용하여 지역적 정보 학습
        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=14, w=14)
        )

        # 차원을 원래대로 축소하는 계층 정의
        self.down_proj = nn.Sequential(
            nn.Linear(scale_dim, dim),
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            Rearrange('b c n -> b n c')
        )

    def forward(self, x):
        x = self.up_proj(x)  # 차원 확장 및 정규화
        x = self.depth_conv(x)  # Depth-wise Convolution
        x = self.down_proj(x)  # 차원 축소
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 다중 헤드의 차원 정의
        project_out = not (heads == 1 and dim_head == dim)  # 출력 투영 필요 여부

        self.heads = heads
        self.scale = dim_head ** -0.5  # 스케일링

        # Query, Key, Value를 생성하는 선형 계층 정의
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 출력 투영 계층 정의
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Query, Key, Value 분리
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Query의 마지막 요소만 사용

        # dot-product Attention 계산
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # Value와 가중치를 곱하여 최종 출력 계산
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
