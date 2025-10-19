import torch.nn
import torch.nn.utils.prune as prune


def compute_global_sparsity(model):
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, "weight_mask"):
                mask = module.weight_mask
                total += mask.numel()
                zeros += (mask == 0).sum().item()
            else:
                w = module.weight.data
                total += w.numel()
                zeros += (w == 0).sum().item()
    return zeros / total


def layerwise_prune(model, pruning_rate=0.2, output_rate=0.1):
    """
    Iteratively prune the network layer-by-layer by a fixed rate.
    Keeps cumulative pruning across rounds (zero weights stay zero).
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            rate = output_rate if name == "fc3" else pruning_rate

            if hasattr(module, "weight_mask"):
                # Already has pruning active - need to prune MORE weights
                current_mask = module.weight_mask.detach().clone()

                # Get the actual weights (with mask applied)
                weight = module.weight.detach().abs()

                # Only consider currently non-zero weights for pruning
                active_weights = weight[current_mask == 1]

                if len(active_weights) > 0:
                    # Calculate threshold: prune 'rate' of remaining weights
                    k = int(rate * len(active_weights))
                    if k > 0:
                        threshold = torch.kthvalue(active_weights.flatten(), k).values

                        # Create new mask: keep old zeros AND newly pruned weights as zero
                        new_mask = current_mask.clone()
                        new_mask[(weight <= threshold) & (current_mask == 1)] = 0

                        # Update the mask
                        module.weight_mask.data.copy_(new_mask)
            else:
                # First time pruning this layer
                prune.l1_unstructured(module, name="weight", amount=rate)

            # Compute sparsity for reporting
            mask = module.weight_mask
            sparsity = 100.0 * (mask == 0).sum().item() / mask.numel()
            print(f"Sparsity in {name}.weight: {sparsity:.2f}% (pruned {rate * 100:.1f}% this round)")

    return model

def apply_mask(model, mask_model):
    """Copy pruning masks from mask_model into model."""
    for (name, m), (_, mask_m) in zip(model.named_modules(), mask_model.named_modules()):
        if isinstance(m, torch.nn.Linear) and hasattr(mask_m, "weight_mask"):
            prune.custom_from_mask(m, name="weight", mask=mask_m.weight_mask)