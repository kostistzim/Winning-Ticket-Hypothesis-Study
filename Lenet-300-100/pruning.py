import torch.nn
import torch.nn.utils.prune as prune


def compute_global_sparsity(model):
    total_params = 0
    zero_params = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            tensor = module.weight.data
            total_params += tensor.numel()
            zero_params += (tensor == 0).sum().item()

    print("Total params: %d" % total_params)
    print("Zero params: %d" % zero_params)
    return zero_params / total_params


def layerwise_prune(model, pruning_rate=0.2, pruning_round=1):
    """Layer-wise magnitude pruning, output layer pruned at half rate."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Set pruning amount
            rate = pruning_rate if name != "fc3" else pruning_rate / 2
            target_sparsity = 1 - (1 - rate) ** pruning_round

            # Prune and remove zero weights
            prune.l1_unstructured(module, name="weight", amount=target_sparsity)
            prune.remove(module, 'weight')


            sparsity = 100.0 * (module.weight == 0).sum().item() / module.weight.nelement()
            print(f"Sparsity in {name}.weight: {sparsity:.2f}% (target {target_sparsity * 100:.1f}%)")

    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return model


def apply_mask(model, mask_model):
    """Apply the mask from mask_model to model by zeroing out weights."""
    with torch.no_grad():
        model_layers = {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
        mask_layers = {name: m for name, m in mask_model.named_modules() if isinstance(m, torch.nn.Linear)}

        for name in model_layers:
            if name in mask_layers:
                model_layers[name].weight.data[mask_layers[name].weight == 0] = 0