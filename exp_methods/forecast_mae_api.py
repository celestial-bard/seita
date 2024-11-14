import os
import sys
from typing import Dict, Union

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forecast_mae', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forecast_mae'))

from exp_methods.forecast_mae.src.model.model_forecast import ModelForecast as ForecastMAE
from exp_methods.forecast_mae.src.model.trainer_forecast import Trainer as ForecastMAETrainer
from exp_methods.forecast_mae.src.datamodule.av2_dataset import collate_fn as fmae_collate_fn
from exp_methods.forecast_mae.src.datamodule.av2_dataset import Av2Dataset as FMaeDataset
from exp_methods.forecast_mae.src.utils.submission_av2 import SubmissionAv2

def transfer_data(
    data: Dict[str, Union[torch.Tensor, str]]
) -> Dict[str, Union[torch.Tensor, str]]:
    '''
    :input: data of SeiTA
    :output: data of ForecastMAE
    '''
    theta = data["theta_at_origin"]
    origin = data["origin_point"]

    rot_back_mat = torch.tensor([
        [torch.cos(-theta), -torch.sin(-theta)],
        [torch.sin(-theta), torch.cos(-theta)],
    ])

    actor_angles = ((data["actor_angles"].double() + theta + np.pi) % (2 * np.pi) - np.pi).float()
    now_theta = actor_angles[0, 49]
    now_rot_mat = torch.tensor([
        [torch.cos(now_theta), -torch.sin(now_theta)],
        [torch.sin(now_theta), torch.cos(now_theta)],
    ])

    real_pos = ((data["actor_real_pos"] @ rot_back_mat) + origin)
    now_origin = real_pos[0, 49, ...]

    pos = (real_pos - now_origin) @ now_rot_mat
    actor_centers = pos[:, 49, ...].clone()

    x_pos = pos.clone()
    x_pos[:, 50:] = x_pos[:, 50:] - x_pos[:, 49].unsqueeze(-2)
    x_pos[:, 1:50] = x_pos[:, 1:50] - x_pos[:, :49]
    x_pos[:, 0] = torch.zeros(x_pos.shape[0], 2)

    lane_positions = ((data["lane_positions"] @ rot_back_mat) + origin - now_origin) @ now_rot_mat
    lane_centers = ((data["lane_centers"] @ rot_back_mat) + origin - now_origin) @ now_rot_mat
    lane_angles = torch.atan2(
        lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
        lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
    )

    actor_angles = (actor_angles - now_theta + np.pi) % (2 * np.pi) - np.pi

    return {
        "x": x_pos[:, :50],
        "y": x_pos[:, 50:],
        "x_attr": data["actor_attr"],
        "x_positions": pos[:, :50],
        "x_centers": actor_centers,
        "x_angles": actor_angles,
        "x_velocity": data["actor_velocity"][:, :50, ...],
        "x_velocity_diff": data["actor_velocity_diff"][:, :50, ...],
        "x_padding_mask": data["actor_padding_mask"],
        "lane_positions": lane_positions,
        "lane_centers": lane_centers,
        "lane_angles": lane_angles,
        "lane_attr": data["lane_attr"],
        "lane_padding_mask": data["lane_padding_mask"],
        "is_intersections": data["is_intersections"],
        "origin": now_origin.view(-1, 2),
        "theta": now_theta.unsqueeze(-1),
        "scenario_id": data["scenario_id"],
        "track_id": data["track_id"],
        "city": data["city"],
    }


def transfer_data_batch(
    data: Dict[str, Union[torch.Tensor, str]]
) -> Dict[str, Union[torch.Tensor, str]]:
    '''
    :input: data of SeiTA (batched)
    :output: data of ForecastMAE (batched)
    '''
    batch_size, n_actors, traj_len = data["actor_angles"].shape 
    theta = data["theta_at_origin"] 
    origin = data["origin_point"]

    rot_back_mat = torch.zeros((batch_size, 2, 2), device=data["actor_angles"].device)
    rot_back_mat[:, 0, 0] = torch.cos(-theta)
    rot_back_mat[:, 0, 1] = -torch.sin(-theta)
    rot_back_mat[:, 1, 0] = torch.sin(-theta)
    rot_back_mat[:, 1, 1] = torch.cos(-theta)

    # print(data["actor_angles"].shape, theta[:, None, None].shape)
    actor_angles = (data["actor_angles"] + theta[:, None, None] + np.pi) % (2 * np.pi) - np.pi
    now_theta = actor_angles[:, 0, 49]
    now_rot_mat = torch.zeros((batch_size, 2, 2), device=data["actor_angles"].device)
    now_rot_mat[:, 0, 0] = torch.cos(now_theta)
    now_rot_mat[:, 0, 1] = -torch.sin(now_theta)
    now_rot_mat[:, 1, 0] = torch.sin(now_theta)
    now_rot_mat[:, 1, 1] = torch.cos(now_theta)

    # print(data["actor_real_pos"].shape, rot_back_mat.view(batch_size, 1, 2, 2).shape, origin.shape)
    real_pos = (data["actor_real_pos"] @ rot_back_mat.view(batch_size, 1, 2, 2)) + origin.view(batch_size, 1, 1, 2)
    now_origin = real_pos[:, 0, 49, ...]  

    pos = (real_pos - now_origin.view(batch_size, 1, 1, 2)) @ now_rot_mat.view(batch_size, 1, 2, 2)
    actor_centers = pos[:, :, 49, ...].clone()

    x_pos = pos.clone()
    # print(x_pos[..., 50:, :].shape, x_pos[..., 49, :].shape)
    x_pos[..., 50:, :] = x_pos[..., 50:, :] - x_pos[..., 49, :].unsqueeze(-2)
    x_pos[..., 1:50, :] = x_pos[..., 1:50, :] - x_pos[..., :49, :]
    x_pos[..., 0, :] = torch.zeros(batch_size, n_actors, 2, device=x_pos.device)

    # print(data["lane_positions"].shape, origin.shape, now_origin.shape)
    lane_positions = ((data["lane_positions"] @ rot_back_mat.view(batch_size, 1, 2, 2)) + 
                      origin.view(batch_size, 1, 1, 2) - now_origin.view(batch_size, 1, 1, 2)) @ now_rot_mat.view(batch_size, 1, 2, 2)
    lane_centers = ((data["lane_centers"] @ rot_back_mat.view(batch_size, 2, 2)) + 
                    origin.view(batch_size, 1, 2) - now_origin.view(batch_size, 1, 2)) @ now_rot_mat.view(batch_size, 2, 2)
    lane_angles = torch.atan2(
        lane_positions[..., 10, 1] - lane_positions[..., 9, 1],
        lane_positions[..., 10, 0] - lane_positions[..., 9, 0],
    )

    actor_angles = (actor_angles - now_theta[:, None, None] + np.pi) % (2 * np.pi) - np.pi

    return {
        "x": x_pos[..., :50, :],
        "y": x_pos[..., 50:, :],
        "x_attr": data["actor_attr"],
        "x_positions": pos[..., :50, :],
        "x_centers": actor_centers,
        "x_angles": actor_angles,
        "x_velocity": data["actor_velocity"][..., :50, :],
        "x_velocity_diff": data["actor_velocity_diff"][..., :50],
        "x_padding_mask": data["actor_padding_mask"][..., :50],
        "x_key_padding_mask": data["actor_key_padding_mask"],
        "lane_positions": lane_positions,
        "lane_centers": lane_centers,
        "lane_angles": lane_angles,
        "lane_attr": data["lane_attr"],
        "lane_padding_mask": data["lane_padding_mask"],
        "lane_key_padding_mask": data["lane_key_padding_mask"],
        "is_intersections": data["is_intersections"],
        "origin": now_origin.view(batch_size, -1),
        "theta": now_theta.unsqueeze(-1),
        "scenario_id": data["scenario_id"],
        "track_id": data["track_id"],
        # "city": data["city"],
    }

