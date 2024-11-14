import math
import torch
from typing import Dict, Union, List, Optional, Literal
from torch import Tensor
from copy import deepcopy


def distance_between(org_traj: Tensor, recon_traj: Tensor, p=2):
    return torch.norm(org_traj - recon_traj, p=p, dim=-1)


def angles_between(org_traj: Tensor, recon_traj: Tensor):
    delta = recon_traj - org_traj
    return torch.atan2(delta[..., 1], delta[..., 0]) 


def add_perturb(org_traj: Tensor, pb_dis: Tensor, angles: Tensor):
    dx = pb_dis * torch.cos(angles)
    dy = pb_dis * torch.sin(angles)

    deltas = torch.stack((dx, dy), dim=-1)
    return org_traj + deltas


def rot_traj_and_angles(traj: Tensor, angles: Tensor, theta: Tensor, batch=False):
    if batch:
        batch_size = traj.shape[0]  
        rot_mat = torch.zeros((batch_size, 2, 2), device=traj.device)

        rot_mat[:, 0, 0] = torch.cos(-theta)
        rot_mat[:, 0, 1] = -torch.sin(-theta)
        rot_mat[:, 1, 0] = torch.sin(-theta)
        rot_mat[:, 1, 1] = torch.cos(-theta)

        rot_traj = traj @ rot_mat 
        rot_angle = angles - theta.view(batch_size, 1)
        rot_angle = (rot_angle + math.pi) % (2 * math.pi) - math.pi  

        return rot_traj, rot_angle

    rot_mat = torch.tensor([
        [torch.cos(-theta), -torch.sin(-theta)],
        [torch.sin(-theta), torch.cos(-theta)]
    ], device=traj.device)
    
    rot_traj = traj @ rot_mat
    rot_angle = angles - theta
    rot_angle = (rot_angle + math.pi) % (2 * math.pi) - math.pi
    
    return rot_traj, rot_angle


def get_closest_traj_idx(centers: Tensor, batch=False):
    if batch:
        dists = torch.sum(centers[:, 1:, ...] ** 2, dim=-1)  
        return torch.argmin(dists, dim=1) + 1 
    
    dists = torch.sum(centers[1:, ...] ** 2, dim=1)
    return torch.argmin(dists) + 1


def get_closest_actor_data(data: Dict[str, Union[Tensor, str]], batch=False):
    fixed_data = deepcopy(data)
    fixed_data['x_bak'] = fixed_data['x'].clone()
    fixed_data['y_bak'] = fixed_data['y'].clone()
    fixed_data['x_angles_bak'] = fixed_data['x_angles'].clone()
    fixed_data['x_positions_bak'] = fixed_data['x_positions'].clone()
    fixed_data['x_velocity_diff_bak'] = fixed_data['x_velocity_diff'].clone()

    if batch:
        batch_size = fixed_data['x'].shape[0]
        min_idxs = get_closest_traj_idx(data['x_centers'], batch)  
        tgt_thetas = data['x_angles'][torch.arange(batch_size), min_idxs]  
        tgt_theta = tgt_thetas[..., 49]  
        tgt_traj = data['x_positions'][torch.arange(batch_size), min_idxs] - data['x_centers'][torch.arange(batch_size), min_idxs].unsqueeze(-2)
        
        # print(tgt_traj.shape, tgt_thetas.shape, tgt_theta.shape, min_idxs.shape)
        rot_traj, rot_angles = rot_traj_and_angles(tgt_traj, tgt_thetas, tgt_theta, batch)  
        rot_y, _ = rot_traj_and_angles(data['y'][torch.arange(batch_size), min_idxs], tgt_thetas, tgt_theta, batch) 

        diff_traj = rot_traj.clone() 
        # print(diff_traj.shape)
        diff_traj[..., 1:50, :2] = rot_traj[..., 1:50, :2] - rot_traj[..., :49, :2]  
        diff_traj[..., 0, :2] = torch.tensor([0., 0.], device=diff_traj.device) 

        fixed_data['x'][:, 0, ...] = diff_traj
        fixed_data['y'][:, 0, ...] = rot_y
        fixed_data['x_angles'][:, 0, ...] = rot_angles
        fixed_data['x_positions'][:, 0, ...] = rot_traj
        fixed_data['x_velocity_diff'][:, 0, ...] = fixed_data['x_velocity_diff'][torch.arange(batch_size), min_idxs, ...]

    else:
        print('No batch')
        min_idx = get_closest_traj_idx(data['x_centers'])
        print(min_idx)
        tgt_thetas = data['x_angles'][min_idx]
        tgt_theta = data['x_angles'][min_idx, 49]
        tgt_traj = data['x_positions'][min_idx] - data['x_centers'][min_idx]

        rot_traj, rot_angles = rot_traj_and_angles(tgt_traj, tgt_thetas, tgt_theta)
        rot_y, _ = rot_traj_and_angles(data['y'][min_idx], tgt_thetas, tgt_theta)

        diff_traj = rot_traj.clone() 
        diff_traj[1:50, ...] = rot_traj[1:50, ...] - rot_traj[:49, ...]
        diff_traj[0] = torch.tensor([0., 0.], device=diff_traj.device)

        fixed_data['x'][0, ...] = diff_traj
        fixed_data['y'][0, ...] = rot_y
        fixed_data['x_angles'][0, ...] = rot_angles
        fixed_data['x_positions'][0, ...] = rot_traj
        fixed_data['x_velocity_diff'][0, ...] = fixed_data['x_velocity_diff'][min_idx, ...]

    return fixed_data


def get_dummy_attk_traj(data: Dict[str, Union[Tensor, List[str], str]], rot_angle=30.):
    bsz = data['x'].shape[0]

    k = torch.tensor([int(i) for i in data['track_id']], device=data['x'].device)
    direct = torch.where(k % 2 == 0, torch.tensor(1), torch.tensor(-1))
    rot_rad = direct * rot_angle * (torch.pi / 180)

    cos = torch.cos(rot_rad)
    sin = torch.sin(rot_rad)
    rot_mat = torch.stack((cos, -sin, sin, cos), dim=1).view(bsz, 2, 2).to(data['x'].device)

    x_pos_bak = data['x_positions'][:, 0, ...].clone()
    x_pos_rot = data['x_positions'][:, 0, ...].clone()
    rot_x_pos = x_pos_rot @ rot_mat

    data['x'][:, 0, 1:50] = rot_x_pos[..., 1:50, :] - rot_x_pos[..., :49, :]
    data['x'][:, 0, 0, :] = torch.zeros(2)
    data['y'] = data['y'] @ rot_mat

    real_org_traj = x_pos_bak
    real_attk_traj = rot_x_pos
    return data, real_org_traj, real_attk_traj


def get_dummy_attk_traj_y(data: Dict[str, Union[Tensor, List[str], str]], rot_angle=15.):
    bsz = data['actor_pos'].shape[0]

    k = torch.tensor([int(i) for i in data['track_id']], device=data['actor_pos'].device)
    direct = torch.where(k % 2 == 0, torch.tensor(1), torch.tensor(-1))
    rot_rad = direct * rot_angle * (torch.pi / 180)

    cos = torch.cos(rot_rad)
    sin = torch.sin(rot_rad)
    rot_mat = torch.stack((cos, -sin, sin, cos), dim=1).view(bsz, 2, 2).to(data['actor_pos'].device)

    x_pos_bak = data['actor_real_pos'][:, 0, 50:, :].clone()
    x_pos_rot = data['actor_real_pos'][:, 0, 50:, :].clone()
    rot_x_pos = x_pos_rot @ rot_mat

    data['actor_pos'][:, 0, 51:, :] = rot_x_pos[..., 1:, :] - rot_x_pos[..., :-1, :]
    data['actor_pos'][:, 0, 50, :] = torch.zeros(2)
    # data['y'] = data['y'] @ rot_mat

    real_org_traj = x_pos_bak
    real_attk_traj = rot_x_pos
    return data, real_org_traj, real_attk_traj


def get_fixed_recon_traj(trajs: Tensor, pi: Optional[Tensor] = None):
    if pi is None:
        return trajs.mean(dim=1)
    
    prob = pi / pi.sum(dim=1, keepdim=True)
    prob = prob.unsqueeze(-1).unsqueeze(-1)

    return (trajs * prob).sum(dim=1)


def fix_in_perturb(traj_org: Tensor, traj_ptb: Tensor, max_dis: float = .5, fix_type: Literal['over', 'all'] = 'over'):
    dis = torch.norm(traj_ptb - traj_org, dim=-1)

    if fix_type == 'over':
        mask = dis > max_dis
        scale = max_dis / dis
        scale[~mask] = 1  
        scale = scale.unsqueeze(2)
        
    elif fix_type == 'all':
        max_dists = dis.max(dim=1).values
        scale = torch.minimum(max_dis / max_dists, torch.tensor(1.0))
        scale = scale.view(-1, 1, 1)

    return traj_org + (traj_ptb - traj_org) * scale

