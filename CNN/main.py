# CNN/main.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import argparse
import os
import pandas as pd
import numpy as np
import random

# Import from other.py files in the same directory
from models import Conv2, Conv4, Conv6
from data import load_cifar10
from trainer import train, test
from plotter import plot_results
from pruning import (
    get_initial_mask,
    magnitude_prune,
    reset_to_initial_weights,
    apply_mask_to_weights,
    get_sparsity
)

def set_seed(seed):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # --- Argument Parsing for command-line execution ---
    parser = argparse.ArgumentParser(description='PyTorch Lottery Ticket Hypothesis for CIFAR-10')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'plot'], help='Run mode: train or plot results')
    parser.add_argument('--run_type', type=str, default='winning_ticket', choices=['winning_ticket', 'random_reinit'], help='Type of experiment to run')
    parser.add_argument('--arch', type=str, default='Conv-6', choices=['Conv-2', 'Conv-4', 'Conv-6'], help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=60, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for each pruning iteration')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--prune_iterations', type=int, default=20, help='Number of pruning iterations')
    parser.add_argument('--conv_prune_rate', type=float, default=0.15, help='Pruning rate for conv layers')
    parser.add_argument('--fc_prune_rate', type=float, default=0.20, help='Pruning rate for fc layers')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate for FC layers')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results and plots')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    args = parser.parse_args()

    # --- SET THE SEED ---
    set_seed(args.seed)
    print(f"Running with random seed: {args.seed}")

    # --- Handle Plotting Mode ---
    if args.mode == 'plot':
        plot_results(args.arch, args.output_dir)
        return

    # --- Handle Training Mode ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Device Setup
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data Loading
    trainloader, testloader = load_cifar10(args.batch_size)

    # Initialize model
    print(f"Initializing {args.arch} model with dropout rate {args.dropout_rate}...")
    if args.arch == 'Conv-2': model = Conv2(args.dropout_rate).to(device)
    elif args.arch == 'Conv-4': model = Conv4(args.dropout_rate).to(device)
    else: model = Conv6(args.dropout_rate).to(device)
        
    initial_weights = copy.deepcopy(model.state_dict())
    mask = get_initial_mask(model)
    
    results_log = []

    print(f"--- Starting LTH Experiment ({args.run_type}) for {args.arch} ---")

    for i in range(args.prune_iterations + 1):
        print(f"\n--- Pruning Iteration {i}/{args.prune_iterations} ---")
        
        sparsity, remaining_pct = get_sparsity(mask)
        print(f"Sparsity: {sparsity:.2f}% | Weights Remaining: {remaining_pct:.2f}%")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0
        print("Training the current ticket...")
        for epoch in range(1, args.epochs + 1):
            train(model, mask, trainloader, optimizer, criterion, device)
            accuracy, _ = test(model, testloader, criterion, device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            print(f'Epoch: {epoch}/{args.epochs}, Test Accuracy: {accuracy:.2f}% (Best: {best_accuracy:.2f}%)')
        
        results_log.append({
            'pruning_iteration': i,
            'sparsity_pct': sparsity,
            'weights_remaining_pct': remaining_pct,
            'test_accuracy': best_accuracy,
            'run_type': args.run_type,
            'arch': args.arch,
            'seed': args.seed
        })

        if i < args.prune_iterations:
            print("Pruning the trained network...")
            mask = magnitude_prune(model, args.conv_prune_rate, args.fc_prune_rate, mask)
            
            if args.run_type == 'winning_ticket':
                print("Resetting weights to initial values (Winning Ticket)...")
                reset_to_initial_weights(model, initial_weights)
            elif args.run_type == 'random_reinit':
                print("Re-initializing weights with new random values (Control)...")
                if args.arch == 'Conv-2': new_model = Conv2(args.dropout_rate)
                elif args.arch == 'Conv-4': new_model = Conv4(args.dropout_rate)
                else: new_model = Conv6(args.dropout_rate)
                reset_to_initial_weights(model, new_model.state_dict())
            
            apply_mask_to_weights(model, mask)

    print("\n--- Experiment Finished ---")
    
    results_df = pd.DataFrame(results_log)
    # Include the seed in the output filename to avoid overwriting results
    output_filename = os.path.join(args.output_dir, f'results_{args.arch}_{args.run_type}_seed{args.seed}.csv')
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

if __name__ == '__main__':
    main()