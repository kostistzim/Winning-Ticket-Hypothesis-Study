#!/usr/bin/env python3
# ==========================================
#  LOTTERY TICKET HYPOTHESIS FOR BERT
#  Corrected version with proper pruning + SEED support
# ==========================================

import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ==========================================
#  GLOBAL SETTINGS
# ==========================================
save_dir = "./models"
os.makedirs(save_dir, exist_ok=True)
print(f"Models will be saved to: {save_dir}")


# ==========================================
#  SEED SETTING
# ==========================================
def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to: {seed}")


# ==========================================
#  DATA PROCESSING
# ==========================================
class DatasetProcessor:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_sst2(self, examples):
        return self.tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def process_qqp(self, examples):
        return self.tokenizer(
            examples["question1"],
            examples["question2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_token_type_ids=True,  # Important for sentence pairs
        )


def load_and_process_datasets(task_name, tokenizer, batch_size=32, subset_size=1000):
    """Load and preprocess dataset"""
    processor = DatasetProcessor(tokenizer)

    if task_name == "sst2":
        dataset = load_dataset("stanfordnlp/sst2")
        process_fn = processor.process_sst2
        num_labels = 2
        label_col = "label"
    elif task_name == "qqp":
        dataset = load_dataset("glue", "qqp")
        process_fn = processor.process_qqp
        num_labels = 2
        label_col = "label"
    else:
        raise ValueError(f"Unknown task: {task_name}")

    train_dataset = dataset["train"].select(range(min(subset_size, len(dataset["train"]))))
    val_dataset = dataset["validation"].select(range(min(subset_size // 4, len(dataset["validation"]))))

    def tokenize_function(examples):
        tokenized = process_fn(examples)
        tokenized["labels"] = examples[label_col]
        return tokenized

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

    def collate_fn(batch):
        batch_dict = {
            "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
            "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        }
        # Add token_type_ids if present (for QQP)
        if "token_type_ids" in batch[0]:
            batch_dict["token_type_ids"] = torch.tensor([item["token_type_ids"] for item in batch], dtype=torch.long)
        return batch_dict

    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, num_labels


# ==========================================
#  MODEL DEFINITIONS
# ==========================================
class BertForCustomTask(nn.Module):
    """Custom BERT model with classification head"""

    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


class LotteryTicketBERT(nn.Module):
    """BERT model with lottery ticket pruning and rewinding"""

    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__()
        self.model = BertForCustomTask(model_name, num_labels)
        # Store initial weights - will be updated after warmup training
        self.init_weights = None

    def save_initial_weights(self):
        """Save weights after initial training (winning ticket initialization)"""
        self.init_weights = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def rewind_weights(self):
        """Rewind to initial weights while keeping pruning masks"""
        if self.init_weights is None:
            raise ValueError("Must call save_initial_weights() before rewinding")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.init_weights:
                    # If parameter has a mask, apply it to the rewound weights
                    if hasattr(param, '_forward_pre_hooks') and len(param._forward_pre_hooks) > 0:
                        # Module has pruning applied, so we just copy the data
                        param.data.copy_(self.init_weights[name])
                    else:
                        param.copy_(self.init_weights[name])

    def forward(self, **kwargs):
        return self.model(**kwargs)


# ==========================================
#  PRUNING WRAPPER (CORRECTED)
# ==========================================
class IterativeMagnitudePruning:
    """Iterative Magnitude Pruning using torch.nn.utils.prune"""

    def __init__(self, model, pruning_rate=0.1):
        self.model = model
        self.pruning_rate = pruning_rate
        self.current_sparsity = 0.0
        self.pruned_modules = []  # Track pruned modules

    def apply_pruning(self):
        """Apply global magnitude pruning across all eligible parameters"""
        print(f"Applying global pruning (target sparsity increment: {self.pruning_rate * 100:.1f}%)")

        # Collect all parameters to prune (only once if first iteration)
        if not self.pruned_modules:
            for module_name, module in self.model.model.named_modules():
                if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                    # Exclude LayerNorm and embedding layers
                    if "LayerNorm" not in module_name and "embeddings" not in module_name:
                        self.pruned_modules.append((module, "weight"))

        # Calculate new target sparsity
        current_sparsity = self.compute_sparsity()
        new_target_sparsity = min(current_sparsity + self.pruning_rate, 0.99)

        # Apply global unstructured pruning
        # Note: amount is relative to unpruned weights
        if current_sparsity < new_target_sparsity:
            # Calculate the fraction of remaining weights to prune
            remaining_weights = 1.0 - current_sparsity
            fraction_to_prune = (new_target_sparsity - current_sparsity) / remaining_weights

            prune.global_unstructured(
                self.pruned_modules,
                pruning_method=prune.L1Unstructured,
                amount=fraction_to_prune,
            )

        # DO NOT call prune.remove() here - we want to keep the masks!
        # The masks will accumulate across iterations

    def make_pruning_permanent(self):
        """Remove pruning reparameterization (call only at the very end)"""
        for module, name in self.pruned_modules:
            if prune.is_pruned(module):
                prune.remove(module, name)

    def compute_sparsity(self):
        """Compute current sparsity across all pruned parameters"""
        total, zeros = 0, 0
        for module, param_name in self.pruned_modules:
            param = getattr(module, param_name)
            if param is not None:
                total += param.numel()
                zeros += (param == 0).sum().item()

        if total > 0:
            self.current_sparsity = zeros / total
        return self.current_sparsity


# ==========================================
#  TRAINING & EVALUATION
# ==========================================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total


# ==========================================
#  MAIN LOOP WITH ITERATIVE PRUNING
# ==========================================
def lottery_ticket_training(
        task_name="sst2",
        model_name="bert-base-uncased",
        target_sparsity=0.9,
        pruning_rate=0.1,
        warmup_epochs=1,
        epochs_per_round=2,
        batch_size=16,
        learning_rate=2e-5,
        subset_size=1000,
        seed=42,
):
    # Set seed first
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader, num_labels = load_and_process_datasets(task_name, tokenizer, batch_size, subset_size)

    model = LotteryTicketBERT(model_name, num_labels).to(device)

    # STEP 1: Warmup training to get initial weights
    print(f"\n{'=' * 60}")
    print(f"WARMUP TRAINING ({warmup_epochs} epochs)")
    print(f"{'=' * 60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * warmup_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    for epoch in range(warmup_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Warmup Epoch {epoch + 1}/{warmup_epochs} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

    # Save these weights as the "winning ticket initialization"
    model.save_initial_weights()
    print("✓ Initial weights saved after warmup")

    # Save the dense model (0% sparsity) after warmup
    save_model_local(model, task_name, sparsity=0.0,
                     history={"sparsity": [0.0], "train_acc": [train_acc], "val_acc": [val_acc]}, seed=seed)
    print()

    # STEP 2: Iterative Magnitude Pruning
    pruner = IterativeMagnitudePruning(model, pruning_rate)
    history = {"sparsity": [0.0], "train_acc": [train_acc], "val_acc": [val_acc]}
    current_sparsity = 0.0
    iteration = 0

    while current_sparsity < target_sparsity:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"PRUNING ITERATION {iteration} | Current sparsity: {current_sparsity:.2%}")
        print(f"{'=' * 60}")

        # Prune the network
        pruner.apply_pruning()
        current_sparsity = pruner.compute_sparsity()
        print(f"New sparsity after pruning: {current_sparsity:.2%}")

        # Rewind weights to initial values (keeping masks)
        model.rewind_weights()
        print("✓ Weights rewound to initialization")

        # Train the pruned network
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs_per_round
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        for epoch in range(epochs_per_round):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{epochs_per_round} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

        history["sparsity"].append(current_sparsity)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Check if we've reached target
        if current_sparsity >= target_sparsity:
            break

    print(f"\n{'=' * 60}")
    print(f"✅ IMP COMPLETE | Final sparsity: {current_sparsity:.2%}")
    print(f"{'=' * 60}\n")

    # Make pruning permanent at the end
    pruner.make_pruning_permanent()

    # Save the final sparse model
    print("💾 Saving final sparse model...")
    save_model_local(model, task_name, current_sparsity, history, seed=seed)

    return model, history


# ==========================================
#  SAVE UTILITIES
# ==========================================
def save_model_local(model, task_name, sparsity, history=None, seed=42):
    """
    Save model checkpoint with proper state dict and mask extraction.
    """
    filename = f"lottery_ticket_{task_name}_sparsity{int(sparsity * 100)}_seed{seed}.pt"
    filepath = os.path.join(save_dir, filename)

    # ✅ FIXED: Get state dict from the inner model
    state_dict = model.model.state_dict()

    # ✅ ADDED: Extract pruning masks if they exist
    mask_dict = {}
    for module_name, module in model.model.named_modules():
        # Check if module has pruning applied
        if hasattr(module, 'weight_mask'):
            # The mask is stored as 'weight_mask' after pruning
            mask_dict[f"{module_name}.weight"] = module.weight_mask.clone()

    checkpoint = {
        "model_state_dict": state_dict,  # ✅ Now contains proper 'bert.' and 'classifier.' keys
        "mask_dict": mask_dict if mask_dict else None,  # ✅ Save masks separately
        "sparsity": sparsity,
        "seed": seed,
        "history": history or {},
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")
    print(f"  - State dict keys: {len(state_dict)}")
    print(f"  - Masks saved: {len(mask_dict)}")

    # Save JSON summary (without tensors)
    json_path = filepath.replace(".pt", "_summary.json")
    json_summary = {
        "task": task_name,
        "sparsity": sparsity,
        "seed": seed,
        "num_masks": len(mask_dict),
        "history": history or {},
    }
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Summary saved to {json_path}")
    return filepath


# ==========================================
#  MAIN EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lottery Ticket Hypothesis with torch pruning on BERT.")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "qqp"], help="Task to train on.")
    parser.add_argument("--sparsity", type=float, default=0.6, help="Target sparsity (0–1). Paper: SST-2=60%, QQP=90%")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup epochs before pruning.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs per pruning round. Paper: 3 for both SST-2 and QQP")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for quick testing. Paper: full dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Paper: 32 for both tasks")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate. Paper: 2e-5 for both tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Running {args.task.upper()} with target sparsity {args.sparsity} | Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    # Use full dataset if subset not specified
    subset_size = args.subset if args.subset else (67360 if args.task == "sst2" else 363872)

    model, history = lottery_ticket_training(
        task_name=args.task,
        target_sparsity=args.sparsity,
        warmup_epochs=args.warmup,
        epochs_per_round=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_size=subset_size,
        seed=args.seed,
    )

    # Note: Models are saved inside lottery_ticket_training()
    # - Dense model (0% sparsity) saved after warmup
    # - Sparse model (target sparsity) saved at the end

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history["sparsity"], history["val_acc"], marker="o", linewidth=2, markersize=8, label="Validation")
    plt.plot(history["sparsity"], history["train_acc"], marker="s", linewidth=2, markersize=8, label="Training")
    plt.xlabel("Sparsity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"BERT Lottery Ticket Hypothesis ({args.task.upper()}) - Seed {args.seed}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_dir,
                             f"results_plot_{args.task}_sparsity{int(args.sparsity * 100)}_seed{args.seed}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()