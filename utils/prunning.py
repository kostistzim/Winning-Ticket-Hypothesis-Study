import numpy as np
import torch
import torch.nn as nn


def make_mask(model:nn.Module)->dict:
    """Return initail masking for us."""
    current_mask={name:torch.ones_like(para) for name,para in model.state_dict().items() if 'weight' in name}
    return current_mask

def global_pruning_by_percentage(model:nn.Module,percentage:int=20,current_mask:bool=False)->dict:
    """Applies pruning by percentage."""
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

