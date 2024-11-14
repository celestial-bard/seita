import torch
import torch.nn as nn
import torch.optim as optim

from typing import Dict
from torch import Tensor

from layers.gen_enc_layer import GenEncoder
from layers.gen_dec_layer import GenDecoder
from neural.nn.cross_attn import XBlock


class Generator(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        enc_depth=1,
        num_heads=8,
        drop_path=0.2,
        traj_len=50
    ) -> None:
        super().__init__()
        self.encoder = GenEncoder(embed_dim, traj_len, num_heads, embed_dim * 4, enc_depth, drop_path)
        self.decoder = GenDecoder(embed_dim, traj_len)

        self.xattn = XBlock(embed_dim, num_heads, embed_dim * 4, drop_path)
        # self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("gen_net.") :]: v 
            for k, v 
            in ckpt.items() 
            if k.startswith("gen_net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data: Dict[str, Tensor]):
        x_src = data['x_src']
        x_tgt = data['x_tgt']

        x_src_enc, x_tgt_enc = self.encoder(x_src, x_tgt)
        # print(x_src_enc.shape, x_tgt_enc.shape)
        # x_fuse = torch.cat((x_src_enc, x_tgt_enc), dim=-1)
        # x_fuse = self.fusion(x_fuse)
        # x_fuse = self.norm(x_fuse)

        # x_src_enc = self.xattn(x_src_enc, x_tgt_enc, x_tgt_enc)
        # x = torch.cat((x_src_enc, x_tgt_enc), dim=-1)
        # x = self.fusion(x)

        x_src_enc = self.xattn(x_src_enc, x_tgt_enc, x_tgt_enc)
        x = x_src_enc + x_tgt_enc
        x = self.norm(x)
        out = self.decoder(x)

        b, l, d = x.shape
        out = out.view(b, l, 2)
        return {'y_hat': out}
