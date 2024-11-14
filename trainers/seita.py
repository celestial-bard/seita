from typing import Optional, Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MetricCollection

from metrics import MissRate, MinADE, MinFDE
from utils.optimizer import WarmupCosLR
from utils.submission import SubmissionAv2

from exp_methods.forecast_mae_api import ForecastMAE, transfer_data_batch

from utils.perturb_utils import get_dummy_attk_traj_y
from metrics.metric_utils import sort_predictions

from utils.dynamic_model import DynamicModel

from models.reconstructor import Reconstructor
from models.generator import Generator
from models.constrainer import Constrainer
from utils.perturb_utils import get_fixed_recon_traj, fix_in_perturb
from utils.data_utils import format_data_to_writable


class SeiTA(pl.LightningModule):
    def __init__(
        self,
        rec_params: Dict[str, Any],
        gen_params: Dict[str, Any],
        con_params: Dict[str, Any],
        rec_ckpt_pth: Optional[str],
        gen_ckpt_pth: Optional[str],
        con_ckpt_pth: Optional[str],
        delta_t: float = .1,
        max_dis: float = .5,
        fix_type: str = 'over',
        fuse_w_pi: bool = True
    ) -> None:
        super(SeiTA, self).__init__()

        self.rec_ckpt_pth = rec_ckpt_pth
        self.gen_ckpt_pth = gen_ckpt_pth
        self.con_ckpt_pth = con_ckpt_pth

        self.rec = Reconstructor(**rec_params)
        self.gen = Generator(**gen_params)
        self.con = Constrainer(**con_params)

        self.dm = DynamicModel(delta_t=delta_t)

        self.fuse_w_pi = fuse_w_pi
        self.fix_type = fix_type
        self.d = max_dis
        self.init_net()

    def init_net(self):
        self.rec.load_from_checkpoint(self.rec_ckpt_pth)
        self.gen.load_from_checkpoint(self.gen_ckpt_pth)
        self.con.load_from_checkpoint(self.con_ckpt_pth)

    def forward(self, data, desired: Optional[Tensor] = None):
        # data: SeiTA data
        data, _, attk = get_dummy_attk_traj_y(data, rot_angle=15.)
        if desired is not None:
            attk = desired
            data['actor_real_pos'][:, 0, 50:, :] = attk
            data['actor_pos'][:, 0, 51:, :] = attk[..., 1:, :] - attk[..., :-1, :]
            data['actor_pos'][:, 0, 50, :] = torch.zeros(2)

        ctrl = self.dm.derive_from_trajectory(attk, pad='first', w_o_time=True)
        data['actor_velocity_diff'][:, 0, 50:] = ctrl['d_v']
        data['actor_angles'][:, 0, 50:] = ctrl['th']

        rec_out = self.rec(data)
        org_traj = data['actor_real_pos'][:, 0, :50, :].clone()
        rec_traj = get_fixed_recon_traj(rec_out['y_hat'], rec_out['pi']) \
            if self.fuse_w_pi else get_fixed_recon_traj(rec_out['y_hat'])
        
        ptb_traj = self.gen({'x_src': org_traj, 'x_tgt': rec_traj})['y_hat']
        con_traj = fix_in_perturb(org_traj, ptb_traj, self.d, fix_type=self.fix_type)

        fit_ctrl = self.con(con_traj)
        return {
            'org_traj': org_traj,
            'rec_traj': rec_traj,
            'ptb_traj': ptb_traj,
            'con_traj': con_traj,
            'des_traj': attk,
            'origin': data['origin_point'],
            'theta': data['theta_at_origin'],
            **fit_ctrl
        }
    
    def forward_to_writable(self, data: Dict[str, Tensor]):
        data_sheet = self(data)
        to_write = format_data_to_writable(
            traj=data_sheet['con_traj'],
            speed=data_sheet['speed_hat'],
            angle=data_sheet['angle_hat'],
            origin=data_sheet['origin'],
            theta=data_sheet['theta']
        )

        return to_write, data_sheet
