from typing import Dict
import torch.nn as nn

from neural.diffattn.diffattn_layer import DiffAttnLayer


class EmbedAndAttnBlock(nn.Module):
    def __init__(self, traj_len, embed_dim, num_heads, ff_hidden_size, depths=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.attn = nn.ModuleList([
            DiffAttnLayer(
                embed_dim, 
                num_heads, 
                ff_hidden_size,
                depth + 1,
                dropout
            ) for depth
            in range(depths)
        ])

    def forward(self, x):
        x = self.embed(x)
        for layer in self.attn:
            x = layer(x)
        return x
    

class GenEncoder(nn.Module):
    def __init__(self, embed_dim, traj_len, num_heads, ff_hidden_size, depths=2, dropout=0.2):
        super().__init__()
        self.source_traj_encode = EmbedAndAttnBlock(
            traj_len, 
            embed_dim, 
            num_heads, 
            ff_hidden_size, 
            depths, 
            dropout
        )

        self.target_traj_encode = EmbedAndAttnBlock(
            traj_len, 
            embed_dim, 
            num_heads, 
            ff_hidden_size, 
            depths, 
            dropout
        )

    def forward(self, x_src, x_tgt):
        x_src_enc = self.source_traj_encode(x_src)
        x_tgt_enc = self.target_traj_encode(x_tgt)
        return x_src_enc, x_tgt_enc
