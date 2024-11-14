import torch
import torch.nn as nn


class ConstDecBlock(nn.Module):
    def __init__(self, hid_dim=64, layers=1, drop=0.2):
        super().__init__()

        self.gru = nn.GRU(hid_dim * 2, hid_dim * 2, layers, batch_first=True, dropout=drop)
        self.dec = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        out, h_x = self.gru(x)
        d_x = self.dec(out)
        return d_x


class ConstDecoder(nn.Module):
    def __init__(self, hid_dim=64, layers=1, drop=0.2):
        super().__init__()

        self.speed_decode = ConstDecBlock(hid_dim, layers, drop)
        self.angle_decode = ConstDecBlock(hid_dim, layers, drop)

    def forward(self, x):
        dec_speed = self.speed_decode(x)
        dec_angle = self.angle_decode(x)

        return dec_speed, dec_angle
