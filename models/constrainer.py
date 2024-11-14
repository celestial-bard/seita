import torch
import torch.nn as nn
import torch.optim as optim

from typing import Dict
from torch import Tensor

from layers.const_enc_layer import ConstEncoder
from layers.const_dec_layer import ConstDecoder


class Constrainer(nn.Module):
    def __init__(self, traj_len=50, delta_t=0.1, hid_dim=64, dec_layers=2, drop=0.2):
        super().__init__()

        self.delta_t = delta_t
        self.traj_len = traj_len

        self.encoder = ConstEncoder(hid_dim)
        self.decoder = ConstDecoder(hid_dim, dec_layers, drop)
        self.norm = nn.LayerNorm(hid_dim * 2)
    
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
            k[len("net."):]: v 
            for k, v 
            in ckpt.items() 
            if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, x):
        b, l, _ = x.shape
        x_enc = self.encoder(x)
        x_enc = self.norm(x_enc)
        speed_dec, angle_dec = self.decoder(x_enc)

        return {
            'speed_hat': speed_dec.view(b, l),
            'angle_hat': angle_dec.view(b, l)
        }
