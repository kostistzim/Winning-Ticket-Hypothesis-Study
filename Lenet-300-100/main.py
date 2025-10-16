from pathlib import Path

import numpy as np
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


def iterative_pruning(device, train_loader, val_loader, test_loader, total_iterations, num_rounds=5, pruning_rate=0.2):
    initial_state = torch.load("models/lenet300_100_initial.pth", map_location=device)
    trained_state = torch.load("models/lenet300_100_trained.pth", map_location=device)

    # Model for tracking pruning masks
    pruning_model = LeNet().to(device)
    pruning_model.load_state_dict(trained_state)

    for round_idx in range(1, num_rounds+1):
        print(f"\n=== Iterative Pruning Round {round_idx}/{num_rounds} ===")

        # Pruning
        effective_rate = 1 - ((1 - pruning_rate) ** (1 / round_idx))
        layerwise_prune(pruning_model, pruning_rate=effective_rate)
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
              save_log_path=f"logs/pruned_{density * 100:.1f}.csv")

        # Save pruned model
        filename = f"models/lenet300_100_density_{density*100:.1f}.pth"
        torch.save(rewind_model.state_dict(), filename)

        pruning_model = rewind_model

def main():
    # Set seed
    set_seed(42)

    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Setup
    model = LeNet().to(device)
    train_loader, val_loader, test_loader = load_dataset()
    optimizer = optim.Adam(model.parameters(), lr=1.2e-3)
    criterion = nn.CrossEntropyLoss()
    total_iterations = 1000

    # Save initial weights before training
    torch.save(model.cpu().state_dict(), "models/lenet300_100_initial.pth")
    model.to(device)

    # Train dense model and evaluate
    train(model, device, train_loader, optimizer, criterion,
          total_iterations=total_iterations,
          val_loader=val_loader,
          test_loader=test_loader,
          evaluate_fn=evaluate,
          save_log_path="logs/dense_training_log.csv")

    # Save trained model
    torch.save(model.cpu().state_dict(), "models/lenet300_100_trained.pth")
    model.to(device)

    # Start iterative pruning
    print("\n=== Starting Iterative Pruning ===")
    iterative_pruning(device, train_loader, val_loader, test_loader, total_iterations, num_rounds=5, pruning_rate=0.2)

if __name__ == "__main__":
    main()