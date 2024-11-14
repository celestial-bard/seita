from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from metrics import MissRate, MinADE, MinFDE
from utils.optimizer import WarmupCosLR
from utils.submission import SubmissionAv2

from models.constrainer import Constrainer
from exp_methods.forecast_mae_api import ForecastMAE, transfer_data_batch

from utils.perturb_utils import get_closest_actor_data, distance_between, angles_between, add_perturb
from metrics.metric_utils import sort_predictions

from utils.dynamic_model import DynamicModel


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=64,
        delta_t=0.1,
        traj_len=50,
        dec_layers=1,
        drop_path=0.2,
        dyn_loss_rate=0.3,
        lr: float = 1e-3,
        epochs: int = 30,
        warmup_epochs: int = 5,
        weight_decay: float = 1e-4,
        pretrained_weights: Optional[str] = None,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.dm = DynamicModel(delta_t)
        self.dyn_loss_rate = dyn_loss_rate
        self.net = Constrainer(traj_len, delta_t, dim, dec_layers, drop_path)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print(f'Loaded pretrained weights from {pretrained_weights}')

    def forward(self, data):
        traj = data['x_positions'][:, 0, ...]
        ctrl = self.net(traj)

        pred_speed = ctrl['speed_hat']
        pred_angle = ctrl['angle_hat']

        real_speed = data['x_velocity'][:, 0, :50]
        real_angle = data['x_angles'][:, 0, :50]

        dyn_ctrl = self.dm.derive_from_trajectory(traj, pad='first')

        return {
            'pred_speed': pred_speed,
            'pred_angle': pred_angle,
            'real_speed': real_speed,
            'real_angle': real_angle,
            'dyn_speed': dyn_ctrl['v'],
            'dyn_angle': dyn_ctrl['th'],
        }

    def predict(self, data):
        pass

    def cal_loss(self, preds): 
        loss_real = F.smooth_l1_loss(preds['pred_speed'], preds['real_speed'])
        loss_real += F.smooth_l1_loss(preds['pred_angle'], preds['real_angle'])

        loss_dyn = F.mse_loss(preds['pred_speed'], preds['dyn_speed'])
        loss_dyn += F.mse_loss(preds['pred_angle'], preds['dyn_angle'])

        loss = (1 - self.dyn_loss_rate) * loss_real + self.dyn_loss_rate * loss_dyn
        
        return {
            'loss': loss,
            'real_loss': loss_real.item(),
            'dyn_loss': loss_dyn.item()
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
                batch_size=loss_cal_sheet['pred_speed'].shape[0]
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        loss_cal_sheet = self(data)
        losses = self.cal_loss(loss_cal_sheet)

        self.log(
            "val/real_loss",
            losses['real_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=loss_cal_sheet['pred_speed'].shape[0]
        )
        
        self.log(
            "val/dyn_loss",
            losses['dyn_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=loss_cal_sheet['pred_speed'].shape[0]
        )

        self.log(
            "val_loss",
            losses['loss'].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=loss_cal_sheet['pred_speed'].shape[0]
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
