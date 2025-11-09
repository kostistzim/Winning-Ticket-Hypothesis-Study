import torch
import os
import argparse
import re

def load_model_state(path):
    print(f"Loading: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model_state_dict"], checkpoint

def extract_seed_from_filename(filename):
    m = re.search(r"seed(\d+)", filename)
    return int(m.group(1)) if m else None

def extract_task_from_filename(filename):
    m = re.search(r"lottery_ticket_([a-zA-Z0-9]+)_", filename)
    return m.group(1) if m else "unknown"

def merge_masks(state_a, state_b):
    merged = {}
    for k in state_a.keys():
        wa = state_a[k]
        wb = state_b.get(k, wa.clone())
        if isinstance(wa, torch.Tensor) and wa.shape == wb.shape:
            # Keep weights active if nonzero in either model
            mask = (wa != 0) | (wb != 0)
            merged[k] = wa * mask.float()
        else:
            merged[k] = wa
    return merged

def compute_sparsity(state_dict):
    total, zeros = 0, 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel()
            zeros += (v == 0).sum().item()
    return zeros / total if total > 0 else 0

def save_ensemble_ticket(merged_state, meta_a, meta_b, task_a, task_b, seed, sparsity, save_dir="./models"):
    sparsity_str = f"{int(round(sparsity * 100))}"
    out_name = f"ensemble_ticket_{task_a}_{task_b}_sparsity{sparsity_str}_seed{seed}.pt"
    out_path = os.path.join(save_dir, out_name)
    torch.save({
        "model_state_dict": merged_state,
        "source_a": meta_a,
        "source_b": meta_b,
        "seed": seed,
        "sparsity": sparsity,
        "tasks": [task_a, task_b]
    }, out_path)
    print(f"Ensemble ticket saved to: {out_path} ({sparsity*100:.2f}% sparse)")

def main():
    parser = argparse.ArgumentParser(description="Combine two lottery tickets into an ensemble ticket.")
    parser.add_argument("--ticket_a", required=True, help="Path to first sparse model (.pt)")
    parser.add_argument("--ticket_b", required=True, help="Path to second sparse model (.pt)")
    parser.add_argument("--save_dir", default="./models", help="Directory to save ensemble ticket")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    state_a, meta_a = load_model_state(args.ticket_a)
    state_b, meta_b = load_model_state(args.ticket_b)

    task_a = extract_task_from_filename(args.ticket_a)
    task_b = extract_task_from_filename(args.ticket_b)
    seed_a = extract_seed_from_filename(args.ticket_a)
    seed_b = extract_seed_from_filename(args.ticket_b)

    if seed_a != seed_b:
        raise ValueError(f"Seeds differ ({seed_a} vs {seed_b}). Use matching seeds for ensemble creation.")
    seed = seed_a

    merged_state = merge_masks(state_a, state_b)
    sparsity = compute_sparsity(merged_state)
    save_ensemble_ticket(merged_state, meta_a, meta_b, task_a, task_b, seed, sparsity, args.save_dir)

if __name__ == "__main__":
    main()
