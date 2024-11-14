import torch
import torch.nn as nn


class GenDecoder(nn.Module):
    def __init__(self, embed_dim=128, traj_len=50) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.traj_len = traj_len

        self.out_traj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x_fuse):
        # [B, L, D]
        # b, l, d = x_fuse.shape
        # x_fuse = x_fuse.view(b, l * d)
        out = self.out_traj(x_fuse)
        # print(out.shape)

        return out
    