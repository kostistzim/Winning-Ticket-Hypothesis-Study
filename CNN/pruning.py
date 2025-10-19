# CNN/pruning.py

import torch
import copy

def get_initial_mask(model):
    """Creates an initial mask of all ones for the model's weights."""
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[name] = torch.ones_like(param.data)
    return mask

def apply_mask_to_weights(model, mask):
    """Applies a mask to the model's weights."""
    for name, param in model.named_parameters():
        if 'weight' in name and name in mask:
            param.data *= mask[name]

def apply_mask_to_gradients(model, mask):
    """Applies a mask to the model's gradients during training."""
    for name, param in model.named_parameters():
        if 'weight' in name and name in mask and param.grad is not None:
            param.grad.data *= mask[name]

def magnitude_prune(model, conv_prune_rate, fc_prune_rate, current_mask):
    """
    Prunes the model based on weight magnitudes, respecting the current mask.
    This implements layer-wise pruning as described in the paper.
    """
    new_mask = copy.deepcopy(current_mask)
    for name, param in model.named_parameters():
        if 'weight' in name:
            prune_rate = conv_prune_rate if 'features' in name else fc_prune_rate
            
            active_weights = param.data[current_mask[name] == 1]
            if active_weights.numel() == 0:
                continue

            threshold = torch.quantile(torch.abs(active_weights), prune_rate)
            
            prune_mask_bool = (torch.abs(param.data) <= threshold) & (current_mask[name] == 1)
            
            new_mask[name][prune_mask_bool] = 0.0
            
    return new_mask

def reset_to_initial_weights(model, initial_weights):
    """Resets the model's weights to their initial values."""
    model.load_state_dict(initial_weights)

def get_sparsity(mask):
    """Calculates the total sparsity of the network from a mask."""
    total_params, remaining_params = 0, 0
    for name in mask:
        total_params += mask[name].numel()
        remaining_params += torch.sum(mask[name]).item()
    
    sparsity = 100. * (1 - (remaining_params / total_params))
    remaining_percentage = 100. * (remaining_params / total_params)
    return sparsity, remaining_percentage