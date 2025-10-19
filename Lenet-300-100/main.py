from pathlib import Path

import numpy as np
import argparse
from torch import optim, nn

from LeNet import LeNet
from load_dataset import *
from train import train
from eval import evaluate
from pruning import *

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iterative_pruning(device, train_loader, val_loader, test_loader,
                      total_iterations, trial, num_rounds=20, pruning_rate=0.2):
    initial_state = torch.load(f"models/trial_{trial}/lenet300_100_initial.pth", map_location=device)
    trained_state = torch.load(f"models/trial_{trial}/lenet300_100_trained.pth", map_location=device)

    # Model for tracking pruning masks
    pruning_model = LeNet().to(device)
    pruning_model.load_state_dict(trained_state)

    for round_idx in range(1, num_rounds+1):
        print(f"\n=== Iterative Pruning Round {round_idx}/{num_rounds} ===")

        # DEBUG CHECK
        if round_idx > 1:
            print(f"\n[DEBUG] Checking pruning_model BEFORE pruning in round {round_idx}:")
            for name, module in pruning_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Use module.weight (not .data) to get masked weights
                    # Or check the mask directly
                    if hasattr(module, 'weight_mask'):
                        mask = module.weight_mask
                        sparsity = 100.0 * (mask == 0).sum().item() / mask.numel()
                    else:
                        weight = module.weight.data
                        sparsity = 100.0 * (weight == 0).sum().item() / weight.numel()
                    print(f"  {name}.weight: {sparsity:.2f}% sparse")
            sparsity = compute_global_sparsity(pruning_model)
            print(f"  Global: {sparsity * 100:.2f}% sparse")

        # Pruning
        layerwise_prune(pruning_model, pruning_rate=pruning_rate, output_rate= 0.5 * pruning_rate)
        sparsity = compute_global_sparsity(pruning_model)
        density = 1 - sparsity
        print(f"Global sparsity: {sparsity * 100:.2f}%")

        # Apply sparsity mask to initial model
        rewind_model = LeNet().to(device)
        rewind_model.load_state_dict(initial_state)
        apply_mask(rewind_model, pruning_model)

        # Retrain
        optimizer = optim.Adam(rewind_model.parameters(), lr=1.2e-3)
        criterion = nn.CrossEntropyLoss()

        train(rewind_model, device, train_loader, optimizer, criterion,
              total_iterations=total_iterations,
              val_loader=val_loader,
              test_loader=test_loader,
              evaluate_fn=evaluate,
              save_log_path=f"logs/trial_{trial}/pruned_{density * 100:.1f}.csv")

        # Save pruned model
        filename = f"models/trial_{trial}/lenet300_100_density_{density * 100:.1f}.pth"
        torch.save(rewind_model.state_dict(), filename)

        # Update pruning model for next round
        pruning_model.load_state_dict(rewind_model.state_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0, help="Trial index")
    args = parser.parse_args()
    trial = args.trial

    # Set seed
    set_seed(42 + trial)

    # Create directories
    Path(f"models/trial_{trial}").mkdir(parents=True, exist_ok=True)
    Path(f"logs/trial_{trial}").mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Trial: {trial}")

    # Setup
    model = LeNet().to(device)
    train_loader, val_loader, test_loader = load_dataset()
    optimizer = optim.Adam(model.parameters(), lr=1.2e-3)
    criterion = nn.CrossEntropyLoss()
    total_iterations = 50000

    # Save initial weights before training
    torch.save(model.cpu().state_dict(), f"models/trial_{trial}/lenet300_100_initial.pth")
    model.to(device)

    # Train dense model and evaluate
    train(model, device, train_loader, optimizer, criterion,
          total_iterations=total_iterations,
          val_loader=val_loader,
          test_loader=test_loader,
          evaluate_fn=evaluate,
          save_log_path=f"logs/trial_{trial}/dense_training_log.csv")

    # Save trained model
    torch.save(model.cpu().state_dict(), f"models/trial_{trial}/lenet300_100_trained.pth")
    model.to(device)

    # Start iterative pruning
    print("\n=== Starting Iterative Pruning ===")
    iterative_pruning(device, train_loader, val_loader, test_loader,
                      total_iterations, trial, num_rounds=20, pruning_rate=0.2)

if __name__ == "__main__":
    main()