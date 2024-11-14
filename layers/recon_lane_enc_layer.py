import torch
import torch.nn as nn


class LaneEmbeddingLayer(nn.Module):
    def __init__(self, feat_channel, encoder_channel, embed_dim=128):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(feat_channel, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim * 2, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(embed_dim * 4, embed_dim * 2, 1),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim * 2, self.encoder_channel, 1),
        )

    def forward(self, x):
        bs, n, _ = x.shape

        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n

        feature = self.second_conv(feature)  # B c n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B c
        return feature_global