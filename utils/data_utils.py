import torch
from torch import Tensor
import numpy as np


def format_data(traj: Tensor, origin: Tensor, theta: Tensor):
    rotate_mat = torch.tensor([
        [torch.cos(-theta), -torch.sin(-theta)],
        [torch.sin(-theta), torch.cos(-theta)],
    ], device=traj.device)

    with torch.no_grad():
        global_trajectory = (
            torch.matmul(
                traj[..., :2], 
                rotate_mat
            ) + origin.to(traj.device)
        )

    global_trajectory = global_trajectory.cpu().numpy()
    return global_trajectory


def format_data_rev(traj: Tensor, origin: Tensor, theta: Tensor):
    rot_mat = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)],
    ], device=traj.device)

    with torch.no_grad():
        local_traj = (traj[..., :2] - origin) @ rot_mat

    local_traj = local_traj.cpu().numpy()
    return local_traj


def format_data_to_writable(traj: Tensor, speed: Tensor, angle: Tensor, origin: Tensor, theta: Tensor):
    # 50, 2; 50; 2; 1
    rot_mat = torch.tensor([
        [torch.cos(-theta), -torch.sin(-theta)],
        [torch.sin(-theta), torch.cos(-theta)],
    ], device=traj.device)

    g_traj = torch.matmul(traj, rot_mat) + origin
    g_angle = (angle + theta + np.pi) % (2 * np.pi) - np.pi
    g_spd_x = speed * torch.cos(g_angle)
    g_spd_y = speed * torch.sin(g_angle)

    return {
        'velocity_x': g_spd_x,
        'velocity_y': g_spd_y,
        'heading': g_angle,
        'position_x': g_traj[..., 0],
        'position_y': g_traj[..., 1]
    }

