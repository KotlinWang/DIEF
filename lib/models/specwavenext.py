import torch
import torch.nn as nn

from .backbone import build_backbone
from .enconder import ImageMEncoder, ImageSEncoder, WaveEncoder
from .transformers import MFIE
from .trans_encoder import TransformerEncoder, TransformerEncoderLayer


class SpecWaveNeXt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg['backbone'])
        self.num_feature_levels = cfg['backbone']['num_feature_levels']

        self.img_Mencoder = ImageMEncoder(self.backbone, cfg['backbone']['num_feature_levels'], cfg['d_model'])
        self.img_Sencoder = ImageSEncoder(self.backbone, cfg['d_model'])

        # self.conv1d = nn.Conv1d(41, cfg['d_model'] // 2, kernel_size=1)
        # self.bn1 = nn.BatchNorm1d(cfg['d_model'] // 2, momentum=0.999)
        self.conv1d = nn.Conv1d(41, cfg['d_model'], kernel_size=1)
        self.bn1 = nn.BatchNorm1d(cfg['d_model'], momentum=0.999)
        self.wave_encoder = WaveEncoder(cfg['rnn'])

        self.mfie = MFIE(cfg['batch_size'], cfg['d_model'], cfg['nhead'])

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dense = nn.Linear(cfg['d_model'], cfg['d_model'])
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm1d(cfg['d_model'], momentum=0.999)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(cfg['d_model'], cfg['num_classes'])
        self.sigmoid = nn.Sigmoid()

        encoder_layer = TransformerEncoderLayer(
            256, nhead=8, dim_feedforward=256, dropout=0.1)

        self.image_encoder = TransformerEncoder(encoder_layer, 1)
        self.wave_encoder = TransformerEncoder(encoder_layer, 1)

    def forward(self, audio, imgs, hidden):
        features, pos = self.backbone(imgs)

        if self.num_feature_levels > 1:
            feat_image = self.img_Mencoder(features)
        else:
            feat_image = self.img_Sencoder(features)
        audio = self.bn1(self.conv1d(audio.permute(0, 2, 1)))
        # feat_wave, hidden = self.wave_encoder(audio.permute(0, 2, 1), hidden)

        feat_image = self.image_encoder(feat_image.permute(1, 0, 2), None, None).permute(1, 0, 2)
        feat_wave = self.wave_encoder(audio.permute(2, 0, 1), None, None).permute(1, 0, 2)

        # x1 = self.mfie(feat_wave, feat_image)
        x = self.mfie(feat_image, feat_wave)

        # x = x[:, -1, :]
        x = self.pool(x.permute(0, 2, 1))

        x = self.dropout(x.squeeze(2))
        x = self.dense(x)
        x = self.bn2(x)
        x = self.selu(x)
        x = self.out_proj(x)

        return x, hidden

def compile_model(cfg):
    return SpecWaveNeXt(cfg)
