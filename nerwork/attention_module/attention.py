import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        raise NotImplementedError


class PixelAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x).transpose(1, 2).reshape(B, C, H, W)


class ChannelAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.qkv = nn.Sequential(
            nn.Conv2d(dim, dim * 3, kernel_size=3, bias=qkv_bias, groups=dim, padding=1),
            nn.BatchNorm2d(dim * 3)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W).transpose(0, 1)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        return self.proj(x).transpose(1, 2).reshape(B, C, H, W)
