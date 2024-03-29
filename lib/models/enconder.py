import torch.nn as nn
import torch
import torch.nn.functional as F


class ImageMEncoder(nn.Module):
    def __init__(self, backbone, num_feature_levels, d_model):
        super().__init__()
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, d_model),
                ))
                in_channels = d_model
            self.input_proj = nn.ModuleList(input_proj_list)

        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

    def forward(self, features):
        srcs = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))

        src_16 = self.proj(srcs[1])
        src_32 = self.upsample(F.interpolate(srcs[2], size=src_16.shape[-2:]))
        src_8 = self.downsample(srcs[0])
        src = (src_8 + src_16 + src_32) / 3
        src = src.flatten(2)

        return src.permute(0, 2, 1)


class ImageSEncoder(nn.Module):
    def __init__(self, backbone, d_model):
        super().__init__()
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone.num_channels[0], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model),
            )])

    def forward(self, features):
        src, mask = features[0].decompose()
        src = self.input_proj[0](src)

        src = src.flatten(2)

        return src.permute(0, 2, 1)


class WaveEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=cfg['input_size'],
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['layers'],
            bias=True,
            batch_first=True,
            dropout=cfg['dropout'],
            bidirectional=cfg['bilstm']
        )

    def forward(self, x, h):
        x, h = self.rnn(x, h)
        return x, h
