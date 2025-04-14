import typer

import torch
import numpy as np

def triplet_loss_exp(x1, x2, x3):
    loss = -torch.log(
        torch.exp(torch.sum(x1 * x2, dim=1))
        / (torch.exp(torch.sum(x1 * x2, dim=1)) + torch.exp(torch.sum(x1 * x3, dim=1)))
    ).mean()
    return loss

def triplet_loss_l2(x1, x2, x3):
    loss = (torch.sum((x1-x2)**2, dim=1) - torch.sum((x1-x3)**2, dim=1)).mean()
    return loss

def prob_similarity_pair(x1, x2, mu1, mu2, sig):
    numerator = 0.5 * np.exp(-(x1-mu1)**2/2/(sig**2)) * np.exp(-(x2-mu1)**2/2/(sig**2)) + \
            0.5 * np.exp(-(x1-mu2)**2/2/(sig**2)) * np.exp(-(x2-mu2)**2/2/(sig**2))
    denominator = (0.5 * np.exp(-(x1-mu1)**2/2/(sig**2)) + 0.5 * np.exp(-(x1-mu2)**2/2/(sig**2))) * \
        (0.5 * np.exp(-(x2-mu1)**2/2/(sig**2)) + 0.5 * np.exp(-(x2-mu2)**2/2/(sig**2)))
    
    return np.prod(numerator/denominator)