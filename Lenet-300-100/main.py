from pathlib import Path
import numpy as np
import argparse
import torch
from torch import optim, nn

from LeNet import LeNet
from load_dataset import *
from train import train
from eval import evaluate
from pruning import *

def set_seed(seed=42, fast_mode=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = not fast_mode
    torch.backends.cudnn.benchmark = fast_mode

def iterative_pruning(device, train_loader, val_loader, test_loader,
                      total_iterations, trial, num_rounds=20, pruning_rate=0.2,
                      random_reinit=False):
    initial_state = torch.load(f"models/trial_{trial}/lenet300_100_initial.pth", map_location=device)
    trained_state = torch.load(f"models/trial_{trial}/lenet300_100_trained.pth", map_location=device)

    pruning_model = LeNet().to(device)
    pruning_model.load_state_dict(trained_state)

    for round_idx in range(1, num_rounds + 1):
        print(f"\n=== Iterative Pruning Round {round_idx}/{num_rounds} ===")

        layerwise_prune(pruning_model, pruning_rate=pruning_rate, output_rate=0.5 * pruning_rate)
        sparsity = compute_global_sparsity(pruning_model)
        density = 1 - sparsity
        print(f"Global sparsity: {sparsity * 100:.2f}%")

        rewind_model = LeNet().to(device)
        if not random_reinit:
            rewind_model.load_state_dict(initial_state)
        else:
            print("[INFO] Randomly reinitializing weights before retraining.")

        apply_mask(rewind_model, pruning_model)

        optimizer = optim.AdamW(rewind_model.parameters(), lr=1.2e-3)
        criterion = nn.CrossEntropyLoss()
        mode_tag = "iterative_random" if random_reinit else "iterative_dense_init"

        train(
            rewind_model, device, train_loader, optimizer, criterion,
            total_iterations=total_iterations,
            val_loader=val_loader,
            test_loader=test_loader,
            evaluate_fn=evaluate,
            save_log_path=f"logs/trial_{trial}/pruned_{mode_tag}_{density * 100:.1f}.csv",
            eval_every=100,
            use_amp=True
        )

        filename = f"models/trial_{trial}/lenet300_100_{mode_tag}_density_{density * 100:.1f}.pth"
        torch.save({k: v.cpu() for k, v in rewind_model.state_dict().items()}, filename)
        pruning_model.load_state_dict(rewind_model.state_dict())

def one_shot_pruning(device, train_loader, val_loader, test_loader,
                     total_iterations, trial, num_rounds=7, pruning_rate=0.2,
                     random_reinit=False):
    initial_state = torch.load(f"models/trial_{trial}/lenet300_100_initial.pth", map_location=device)
    trained_state = torch.load(f"models/trial_{trial}/lenet300_100_trained.pth", map_location=device)

    dense_model = LeNet().to(device)
    dense_model.load_state_dict(trained_state)

    densities = [(1 - pruning_rate) ** r for r in range(1, num_rounds + 1)]

    for round_idx, density in enumerate(densities, start=1):
        target_sparsity = 1 - density
        print(f"\n--- One-Shot Pruning Round {round_idx}/{num_rounds} ---")
        print(f"Target global sparsity: {target_sparsity * 100:.2f}%")

        pruning_model = LeNet().to(device)
        pruning_model.load_state_dict(dense_model.state_dict())

        layerwise_prune(pruning_model, pruning_rate=target_sparsity)
        sparsity = compute_global_sparsity(pruning_model)
        print(f"Achieved sparsity: {sparsity * 100:.2f}%")

        rewind_model = LeNet().to(device)
        if not random_reinit:
            rewind_model.load_state_dict(initial_state)
        else:
            print("[INFO] Randomly reinitializing weights before retraining.")

        apply_mask(rewind_model, pruning_model)

        optimizer = optim.AdamW(rewind_model.parameters(), lr=1.2e-3)
        criterion = nn.CrossEntropyLoss()
        mode_tag = "oneshot_random" if random_reinit else "oneshot_dense_init"

        train(
            rewind_model, device, train_loader, optimizer, criterion,
            total_iterations=total_iterations,
            val_loader=val_loader,
            test_loader=test_loader,
            evaluate_fn=evaluate,
            save_log_path=f"logs/trial_{trial}/pruned_{mode_tag}_{density * 100:.1f}.csv",
            eval_every=100,
            use_amp=True
        )

        filename = f"models/trial_{trial}/lenet300_100_{mode_tag}_density_{density * 100:.1f}.pth"
        torch.save({k: v.cpu() for k, v in rewind_model.state_dict().items()}, filename)
        print(f"[INFO] Saved one-shot pruned model → {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--random_reinit", action="store_true")
    parser.add_argument("--mode", type=str, default="iterative", choices=["iterative", "oneshot"])
    parser.add_argument("--fast_mode", action="store_true")
    args = parser.parse_args()

    trial = args.trial
    random_reinit = args.random_reinit
    set_seed(42 + trial, fast_mode=args.fast_mode)

    Path(f"models/trial_{trial}").mkdir(parents=True, exist_ok=True)
    Path(f"logs/trial_{trial}").mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Trial: {trial}")

    model = LeNet().to(device)
    train_loader, val_loader, test_loader = load_dataset()
    optimizer = optim.AdamW(model.parameters(), lr=1.2e-3)
    criterion = nn.CrossEntropyLoss()
    total_iterations = 50000

    initial_path = f"models/trial_{trial}/lenet300_100_initial.pth"
    dense_model_path = f"models/trial_{trial}/lenet300_100_trained.pth"
    dense_log_path = f"logs/trial_{trial}/pruned_100.0.csv"

    if Path(dense_model_path).exists() and Path(initial_path).exists():
        print(f"[INFO] Found existing dense model for trial {trial}. Skipping dense retraining.")
        model.load_state_dict(torch.load(dense_model_path, map_location=device))
        model.to(device)
    else:
        print("[INFO] Dense model not found. Training from scratch...")
        if not Path(initial_path).exists():
            torch.save(model.cpu().state_dict(), initial_path)
            print(f"[INFO] Saved initial (untrained) weights to {initial_path}")
        model.to(device)
        train(model, device, train_loader, optimizer, criterion,
              total_iterations=total_iterations,
              val_loader=val_loader,
              test_loader=test_loader,
              evaluate_fn=evaluate,
              save_log_path=dense_log_path,
              eval_every=100,
              use_amp=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, dense_model_path)
        print(f"[INFO] Saved dense model to {dense_model_path}")
        model.to(device)

    if args.mode == "iterative":
        print("\n=== Starting Iterative Pruning ===")
        iterative_pruning(device, train_loader, val_loader, test_loader,
                          total_iterations, trial, num_rounds=20, pruning_rate=0.2,
                          random_reinit=random_reinit)
    elif args.mode == "oneshot":
        print("\n=== Starting One-Shot Pruning ===")
        one_shot_pruning(device, train_loader, val_loader, test_loader,
                         total_iterations, trial, num_rounds=20, pruning_rate=0.2,
                         random_reinit=random_reinit)

if __name__ == "__main__":
    main()
