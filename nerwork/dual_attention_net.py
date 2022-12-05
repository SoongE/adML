import torch
from timm import create_model
from torch import nn

from nerwork.attention_module.attention import ChannelAttention, PixelAttention


class DualAttentionNet(nn.Module):
    def __init__(self, backbone='resnet34', dim=512, pretrained=True):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=pretrained)
        self.backbone.global_pool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.channel_attention = ChannelAttention(dim=dim)
        self.pixel_attention = PixelAttention(dim=dim)

        self.criterion = nn.SmoothL1Loss()

    def forward(self, x, pixel_x, channel_x):
        x = self.backbone(x)
        basic_pixel_attn = self.pixel_attention(x)
        basic_channel_attn = self.channel_attention(x)

        pixel_attn = self.pixel_wise_forward(pixel_x)
        channel_attn = self.channel_wise_forward(channel_x)

        pixel_loss = self.criterion(pixel_attn, basic_pixel_attn)
        channel_loss = self.criterion(channel_attn, basic_channel_attn)

        loss = pixel_loss + channel_loss
        return loss.mean()

    def channel_wise_forward(self, x):
        x = self.backbone(x)
        channel = self.channel_attention(x)
        return channel

    def pixel_wise_forward(self, x):
        x = self.backbone(x)
        pixel = self.pixel_attention(x)
        return pixel

    def forward_attention(self, x):
        x = self.backbone(x)
        return self.pixel_attention(x), self.channel_attention(x)


if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = DualAttentionNet()
    out = model(input)

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
