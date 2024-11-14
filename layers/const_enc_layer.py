import torch
import torch.nn as nn


class ConstEncoder(nn.Module):
    def __init__(self, hid_dim=64):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim * 2)
        )

    def forward(self, x):
        x_enc = self.enc(x)
        return x_enc
