from tqdm import tqdm
from pathlib import Path
from typing import Union, Any, Iterable, Optional
from trainers.seita import SeiTA

from process.av2_dataset import Av2Dataset
from process.av2_dataset import collate_fn
from utils.ray_utils import ActorHandle, ProgressBar
from utils.data_utils import format_data

from exp_methods.forecast_mae_api import fmae_collate_fn, ForecastMAE, transfer_data_batch, SubmissionAv2

import pandas as pd
import torch
import ray

from torch import nn


def to(data: Union[list, tuple, dict, torch.Tensor, Any], place: str):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to(x, place) for x in data]
    elif isinstance(data, dict):
        data = {key: to(value, place) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.to(place)
    return data


def attack_write(
    model: SeiTA, 
    targeted: bool = False, 
    disc: Optional[nn.Module] = None, 
    sub: Optional[SubmissionAv2] = None
) -> None:
    data_root = Path("datasets/av2")
    processed = 'processed'
    cached_split = 'val'

    dataset = Av2Dataset(data_root=data_root / processed, cached_split=cached_split)
    root_benign = Path('datasets/av2')
    root_attack = Path('datasets/attacked_av2')
    mode = 'val'
    data_from_path = root_benign / mode
    data_to_path = root_attack / mode

    for p_data in tqdm(dataset):
        s_id = p_data['scenario_id']
        s_pq = f'scenario_{s_id}.parquet'
        focal_id = p_data['track_id']

        data_from = data_from_path / s_id / s_pq
        data_to = data_to_path / s_id / s_pq

        df = pd.read_parquet(data_from)
        c_data = collate_fn([p_data])
        c_data = to(c_data, 'cuda')

        attack_sheet, forward_data = model.forward_to_writable(c_data)
        if targeted:
            rec_traj = forward_data['rec_traj']     # 50, 2
            c_data['actor_real_pos'][:, 0, :50, :] = rec_traj.unsqueeze(0)

            t_data = transfer_data_batch(c_data)
            desired = disc(t_data)
            desired_y, _ = sub.format_data(t_data, desired['y_hat'], desired['pi'], inference=True)

        track_rows = df[df['track_id'] == focal_id]
        to_attk = track_rows.index[:50]
        if targeted:
            to_attk_y = track_rows.index[50:]

        for k, v in attack_sheet.items():
            df.loc[to_attk, k] = v.detach().cpu().squeeze(0).numpy()
        
        if targeted:
            df.loc[to_attk_y, 'position_x'] = desired_y.squeeze(0)[0, ..., 0]
            df.loc[to_attk_y, 'position_y'] = desired_y.squeeze(0)[0, ..., 1]

        df.to_parquet(data_to)


if __name__ == '__main__':
    REC_CKPT = r'outputs/seita-reconstructor/2024-10-11/14-20-16/checkpoints/last.ckpt'
    GEN_CKPT = r'outputs/seita-generator/2024-11-05/18-58-35/checkpoints/last.ckpt'
    CON_CKPT = r'outputs/seita-constrainer/2024-11-06/20-06-45/checkpoints/last.ckpt'

    rec_params = {
        'embed_dim': 128,
        'encoder_depth': 4,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'qkv_bias': False,
        'drop_path': 0.2,
        'historical_steps': 50,
        'future_steps': 60
    }

    gen_params = {
        'embed_dim': 128,
        'enc_depth': 1,
        'num_heads': 8,
        'drop_path': 0.2,
        'traj_len': 50
    }

    con_params = {
        'traj_len': 50, 
        'delta_t': 0.1, 
        'hid_dim': 64, 
        'dec_layers': 2, 
        'drop': 0.2
    }

    model = SeiTA(
        rec_params=rec_params,
        gen_params=gen_params,
        con_params=con_params,
        rec_ckpt_pth=REC_CKPT,
        gen_ckpt_pth=GEN_CKPT,
        con_ckpt_pth=CON_CKPT,
        delta_t=0.1,
        max_dis=1.0,
        fix_type='over',
        fuse_w_pi=True
    )

    model = model.eval()
    model = model.to('cuda')

    fmae = ForecastMAE().to('cuda')
    fmae.load_from_checkpoint('exp_methods/checkpoints/model_forecast_finetune.ckpt')
    fmae = fmae.eval()

    sub = SubmissionAv2()
    attack_write(model, True, fmae, sub)

