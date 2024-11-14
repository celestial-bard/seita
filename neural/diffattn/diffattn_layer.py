import torch.nn as nn
from timm.models.layers import DropPath

from neural.diffattn.multihead_diffattn import MultiheadDiffAttn


class DiffAttnLayer(nn.Module):
    def __init__(self, emb_size, num_heads, ff_hidden_size, depth=1, dropout=0.2):
        """
        INPUT:
        emb_size - (int) embedding size of the data
        num_heads - (int) number of heads in multi head attention layer
        ff_hidden_size - (int) size of the hidden layer for the feed forward network
        dropout - (float) dropout percentage. Default value = 0.2
        """
        super().__init__()

        self.drop_path_attn = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path_ff = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.norm_attn = nn.LayerNorm(emb_size)
        self.norm_ff = nn.LayerNorm(emb_size)

        self.attn = MultiheadDiffAttn(emb_size, num_heads, depth)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_size),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_size, emb_size)
        )
    
    def forward(self, x):
        x = x + self.drop_path_attn(self.attn(self.norm_attn(x)))    # Shape = (B, N ,C)
        x = x + self.drop_path_ff(self.ff(self.norm_ff(x)))        # Shape = (B, N ,C)

        return x