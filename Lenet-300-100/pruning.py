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
    return zero_params / total_params


def layerwise_prune(model, pruning_rate=0.2):
    """Layer-wise magnitude pruning, output layer pruned at half rate."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            rate = pruning_rate if name != "fc3" else pruning_rate / 2
            prune.l1_unstructured(module, name="weight", amount=rate)
            sparsity = 100.0 * (module.weight == 0).sum().item() / module.weight.nelement()
            print(f"Sparsity in {name}.weight: {sparsity:.2f}% (target {rate*100:.1f}%)")

    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    print(f"Global sparsity: {100.0 * zeros / total:.2f}%")
    return model

def apply_mask(model, mask):
    with torch.no_grad():
        for (name, m), (_, mask_m) in zip(model.named_modules(), mask.named_modules()):
            if isinstance(m, torch.nn.Linear):
                m.weight.data[mask_m.weight == 0] = 0