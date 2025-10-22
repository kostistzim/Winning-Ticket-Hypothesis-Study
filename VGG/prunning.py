"""Functions for pruning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_mask(model:nn.Module)->dict:
    """Return initail masking for us."""
    current_mask={name:torch.ones_like(para) for name,para in model.state_dict().items() if 'weight' in name}
    return current_mask

def apply_mask(model:nn.Module,mask:dict)->None:
    """Masks weights based on the mask dictionary."""
    with torch.no_grad():
        for name,para in model.named_parameters():
            if name in mask:
                para.mul_(mask[name])

def global_pruning_by_percentage(model:nn.Module,percentage:int=20,current_mask:bool=False)->dict:
    """Applies pruning for VGG or ResNet.Specifically, If prunes the p lowest magnitude weights in the entire network."""
    current_weights=[]
    for name,param in model.named_parameters():
        if 'weight' in name:
            masked=param if current_mask is None else param*current_mask.get(name,1)
            current_weights += list(masked.abs().flatten().cpu().detach().numpy())

    threshold = np.percentile(current_weights, percentage)
    new_mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            masked = param if current_mask is None else param * current_mask.get(name, 1)
            new_mask[name] = (masked.abs() > threshold).float()
    return new_mask

def global_pruning_by_percentage_random(model:nn.Module,percentage:int=20,current_mask:bool=False)->dict:
    """Applies random pruning for VGG or ResNet.Specifically, If prunes the p lowest magnitude weights in the entire network."""
    weights = []
    names, shapes = [], []

    for name, p in model.named_parameters():
        if 'weight' in name:
            m = p if current_mask is None else p * current_mask.get(name, 1)
            weights.append(m.flatten())
            names.append(name)
            shapes.append(p.shape)

    weights = torch.cat([w.cpu() for w in weights])
    num_prune = int(len(weights) * percentage / 100)
    mask_flat = torch.ones(len(weights))
    mask_flat[torch.randperm(len(weights))[:num_prune]] = 0

    new_mask, start = {}, 0
    for name, shape in zip(names, shapes):
        n = torch.prod(torch.tensor(shape)).item()
        new_mask[name] = mask_flat[start:start+n].reshape(shape)
        start += n

    return new_mask

