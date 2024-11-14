import torch.nn as nn
from timm.models.layers import DropPath


class XBlock(nn.Module):
    def __init__(self, emb_size, num_heads, ff_hidden_size, dropout=0.2):
        super().__init__()

        self.drop_path_attn = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path_ff = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.norm_ff = nn.LayerNorm(emb_size)
        self.norm_attn_q = nn.LayerNorm(emb_size)
        self.norm_attn_kv = nn.LayerNorm(emb_size)

        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_size),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_size, emb_size)
        )
    
    def forward(self, q, k, v):
        q_n = self.norm_attn_q(q)
        k_n = self.norm_attn_kv(k)
        v_n = self.norm_attn_kv(v)

        x = q + self.drop_path_attn(self.attn(q_n, k_n, v_n)[0])    # Shape = (B, N ,C)
        x = x + self.drop_path_ff(self.ff(self.norm_ff(x)))        # Shape = (B, N ,C)

        return x
    