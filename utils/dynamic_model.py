from typing import Literal

import torch

class DynamicModel:
    def __init__(self, delta_t):
        self.delta_t = delta_t

    def inverse(self, coordinates):
        diff = torch.diff(coordinates, dim=-2)
        
        # v = ∥pt+1 − pt∥/∆t
        distances = torch.norm(diff, dim=-1)
        velocities = distances / self.delta_t
        
        # θ = arctan2(Δy, Δx)
        thetas = torch.atan2(diff[..., 1], diff[..., 0])
        
        return velocities, thetas

    def control(self, velocities, thetas):
        # a = (vt+1 - vt) / ∆t
        accels = torch.diff(velocities, dim=-1) / self.delta_t
        
        # κ = Δθ / vt
        delta_thetas = torch.diff(thetas, dim=-1)
        kappas = delta_thetas / velocities[:, :-1]
        
        return accels, kappas

    def derive_from_trajectory(self, coordinates, pad: Literal['none', 'first', 'last'] = 'none', w_o_time=True):
        velocities, thetas = self.inverse(coordinates)
        if pad == 'first':
            velocities = torch.cat((velocities[:, :1], velocities), dim=1)
            thetas = torch.cat((thetas[:, :1], thetas), dim=1)
        elif pad == 'last':
            velocities = torch.cat((velocities, velocities[:, -1:]), dim=1)
            thetas = torch.cat((thetas, thetas[:, -1:]), dim=1)

        accels, kappas = self.control(velocities, thetas)
        if pad == 'first':
            accels = torch.cat((accels[:, :1], accels), dim=1)
            kappas = torch.cat((kappas[:, :1], kappas), dim=1)
        elif pad == 'last':
            accels = torch.cat((accels, accels[:, -1:]), dim=1)
            kappas = torch.cat((kappas, kappas[:, -1:]), dim=1)

        if w_o_time:
            accels = accels * self.delta_t

        return {
            'v': velocities,
            'th': thetas,
            'd_v': accels,
            'd_th': kappas
        }
