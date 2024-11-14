import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from metrics import MissRate, MinADE, MinFDE
from utils.optimizer import WarmupCosLR
from utils.submission import SubmissionAv2

from models.generator import Generator
from exp_methods.forecast_mae_api import ForecastMAE, transfer_data_batch

from utils.perturb_utils import get_closest_actor_data, distance_between, angles_between, add_perturb
from metrics.metric_utils import sort_predictions


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        encoder_depth=1,
        num_heads=8,
        traj_len=50,
        drop_path=0.2,
        loss_scale=5,
        disc_ckpt_pth: str = None,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.gen_net = Generator(
            embed_dim=dim,
            enc_depth=encoder_depth,
            num_heads=num_heads,
            drop_path=drop_path,
            traj_len=traj_len
        )

        self.dummy_angle = 15.
        self.loss_scale = loss_scale
        self.disc_net = ForecastMAE()
        self.init_disc_net(disc_ckpt_pth)
        self.disc_net.eval()

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print(f'Loaded pretrained weights from {pretrained_weights}')

    def init_disc_net(self, ckpt_pth: str):
        self.disc_net.load_from_checkpoint(ckpt_pth)

    def forward(self, data):
        bsz = data['x'].shape[0]

        k = torch.tensor([int(i) for i in data['track_id']], device=data['x'].device)
        direct = torch.where(k % 2 == 0, torch.tensor(1), torch.tensor(-1))
        rot_rad = direct * self.dummy_angle * (torch.pi / 180.)

        cos = torch.cos(rot_rad)
        sin = torch.sin(rot_rad)
        rot_mat = torch.stack((cos, -sin, sin, cos), dim=1).view(bsz, 2, 2).to(data['x'].device)

        x_pos_bak = data['x_positions'][:, 0, ...].clone()
        x_pos_rot = data['x_positions'][:, 0, ...].clone()
        rot_x_pos = x_pos_rot @ rot_mat

        data['x'][:, 0, 1:50] = rot_x_pos[..., 1:50, :] - rot_x_pos[..., :49, :]
        data['x'][:, 0, 0, :] = torch.zeros(2)

        with torch.no_grad():
            origin_out = self.disc_net(data)
            origin_pred_traj = origin_out['y_hat']  # B, 6, L, 2
            origin_pred_pi = origin_out['pi']
        
        out = self.gen_net({
            'x_src': x_pos_bak, 
            'x_tgt': x_pos_rot
        })

        perturbed = out['y_hat']
        data['x'][:, 0, 1:50, :] = perturbed[..., 1:50, :] - perturbed[..., :49, :]
        data['x'][:, 0, 0, :] = torch.zeros(2)

        with torch.no_grad():
            perturbed_out = self.disc_net(data)
            perturbed_pred_traj = perturbed_out['y_hat']
            perturbed_pred_pi = perturbed_out['pi']

        return {
            'origin_pred_traj': origin_pred_traj,
            'perturbed_pred_traj': perturbed_pred_traj,
            'origin_pred_pi': origin_pred_pi,
            'perturbed_pred_pi': perturbed_pred_pi,
            'origin_traj': x_pos_bak,
            'perturbed_traj': perturbed,
        }

    def predict(self, data):
        pass

    def cal_loss(self, loss_cal_sheet):
        org_pred_traj, org_pred_pi = loss_cal_sheet['origin_pred_traj'], loss_cal_sheet['origin_pred_pi']
        pb_pred_traj, pb_pred_pi = loss_cal_sheet['perturbed_pred_traj'], loss_cal_sheet['perturbed_pred_pi']
        org_traj, pb_traj = loss_cal_sheet['origin_traj'], loss_cal_sheet['perturbed_traj']

        org_pred_traj_best = sort_predictions(org_pred_traj, org_pred_pi)[0][:, 0, ...]
        pb_pred_traj_best = sort_predictions(pb_pred_traj, pb_pred_pi)[0][:, 0, ...]

        attack_loss = F.smooth_l1_loss(org_pred_traj_best, pb_pred_traj_best)
        perturb_loss = F.l1_loss(org_traj, pb_traj) * self.loss_scale

        # loss = self.loss_balance * perturb_loss + (1 - self.loss_balance) * attack_loss
        loss = perturb_loss + attack_loss
        return {
            'loss': loss,
            'attack_loss': attack_loss.item(),
            'perturb_loss': perturb_loss.item()
        }

    def training_step(self, data, batch_idx):
        # data: batched SeiTA data
        loss_cal_sheet = self(data)
        losses = self.cal_loss(loss_cal_sheet)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=loss_cal_sheet['origin_traj'].shape[0]
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        loss_cal_sheet = self(data)
        losses = self.cal_loss(loss_cal_sheet)

        self.log(
            "val/attack_loss",
            losses['attack_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=loss_cal_sheet['origin_traj'].shape[0]
        )
        
        self.log(
            "val/perturb_loss",
            losses['perturb_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=loss_cal_sheet['origin_traj'].shape[0]
        )

        self.log(
            "val_loss",
            losses['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=loss_cal_sheet['origin_traj'].shape[0]
        )

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] 
                    for param_name 
                    in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] 
                    for param_name 
                    in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
