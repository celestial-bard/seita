import torch
import abc
from typing import Dict, Union
import torch.nn as nn
import torch.nn.functional as F


class Metric:
    """
    Base class for prediction metric/loss function
    """
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self, predictions: Union[torch.Tensor, Dict], ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Main function that computes the metric
        :param predictions: Predictions generated by the model
        :param ground_truth: Ground truth labels
        :return metric: Tensor with computed value of metric.
        """
        raise NotImplementedError()
    

class LaplaceNLLLoss(Metric):
    """
    Negative log likelihood loss for ground truth goal nodes under predicted goal log-probabilities.
    """
    def __init__(self):
        self.name = 'LaplaceNLLLoss'
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def compute(self, predictions: Dict, ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        out_mu = predictions['traj']  
        out_sigma = predictions['scale']  
        gt = ground_truth  
        y = gt.repeat(6, 1, 1, 1).transpose(0, 1)
        
        out_pi = predictions['probs']  
        pred = torch.cat((out_mu, out_sigma), dim=-1)  
        l2_norm = torch.norm(out_mu - y, p=2, dim=-1)  
        l2_norm = l2_norm.sum(dim=-1)  
        best_mode = l2_norm.argmin(dim=1)  
        pred_best = pred[torch.arange(pred.shape[0]), best_mode]  
        soft_target = F.softmax(-l2_norm / pred.shape[2], dim=1).detach()  

        loc, scale = pred_best.chunk(2, dim=-1)
        
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)
        loss = 0
        for b in range(loc.shape[0]):
            nll = torch.log(2 * scale[b]) + torch.abs(gt[b] - loc[b]) / scale[b]  
            nll_mean = nll.mean()  

            cross_entropy = torch.sum(-soft_target[b] * F.log_softmax(out_pi[b], dim=-1), dim=-1)  

            loss += nll_mean + cross_entropy * 0.5
        loss_total = loss / loc.shape[0]

        return loss_total