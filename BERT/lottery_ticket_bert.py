#!/usr/bin/env python3
# ==========================================
#  LOTTERY TICKET HYPOTHESIS FOR BERT
#  Clean executable Python version
#  Works locally (Mac/Windows/Linux)
# ==========================================

import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# Optional: Disable CarbonTracker on unsupported systems
try:
    from carbontracker.tracker import CarbonTracker
    tracker_available = True
except Exception:
    print("⚠️ CarbonTracker not available; will skip energy tracking.")
    tracker_available = False


# ==========================================
#  GLOBAL SETTINGS
# ==========================================
save_dir = "./lottery_ticket_models"
os.makedirs(save_dir, exist_ok=True)
print(f"Models will be saved to: {save_dir}")


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
        )

    def process_wnli(self, examples):
        return self.tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
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
    elif task_name == "wnli":
        dataset = load_dataset("glue", "wnli")
        process_fn = processor.process_wnli
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
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
            "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        }

    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, val_loader, num_labels


# ==========================================
#  MODEL CLASSES
# ==========================================
class LotteryTicketBERT(nn.Module):
    """BERT model with lottery ticket masking"""

    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.masks = {}
        self.init_weights = {}
        self._store_init_weights()

    def _store_init_weights(self):
        for name, param in self.model.named_parameters():
            self.init_weights[name] = param.data.clone()

    def rewind_weights(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.init_weights:
                    if name in self.masks:
                        param.data.copy_(self.init_weights[name].to(param.device) * self.masks[name])
                    else:
                        param.data.copy_(self.init_weights[name].to(param.device))

    def compute_mask_statistics(self):
        total_params = sum(mask.numel() for mask in self.masks.values())
        pruned_params = sum((mask == 0).sum().item() for mask in self.masks.values())
        sparsity = pruned_params / total_params if total_params > 0 else 0
        return {"sparsity": sparsity, "total_params": total_params, "remaining_params": total_params - pruned_params}

    def apply_masks(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name].to(param.device)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def to(self, device):
        super().to(device)
        self.masks = {k: v.to(device) for k, v in self.masks.items()}
        self.init_weights = {k: v.to(device) for k, v in self.init_weights.items()}
        return self


class IterativeMagnitudePruning:
    """Implements Iterative Magnitude Pruning"""

    def __init__(self, model, pruning_rate=0.2):
        self.model = model
        self.pruning_rate = pruning_rate

    def global_magnitude_pruning(self, target_sparsity):
        all_weights = []
        weight_names = []

        for name, param in self.model.model.named_parameters():
            if "weight" in name and "LayerNorm" not in name:
                if name not in self.model.masks:
                    self.model.masks[name] = torch.ones_like(param.data)
                masked_weights = param.data * self.model.masks[name]
                all_weights.append(masked_weights.abs().flatten())
                weight_names.append(name)

        if not all_weights:
            return

        all_weights_tensor = torch.cat(all_weights)
        k = int(all_weights_tensor.numel() * target_sparsity)
        if k <= 0:
            return

        threshold = torch.topk(all_weights_tensor, k, largest=False).values.max()

        for name, param in self.model.model.named_parameters():
            if name in weight_names:
                weight_magnitude = param.data.abs()
                new_mask = (weight_magnitude > threshold)
                self.model.masks[name] &= new_mask

    def iterative_pruning_step(self, current_sparsity, target_sparsity):
        next_sparsity = min(current_sparsity + self.pruning_rate * (1 - current_sparsity), target_sparsity)
        self.global_magnitude_pruning(next_sparsity)
        return next_sparsity


# ==========================================
#  TRAINING + EVALUATION
# ==========================================
def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.model.named_parameters():
                if name in model.masks and param.grad is not None:
                    param.grad *= model.masks[name]

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total


# ==========================================
#  MAIN LOTTERY TICKET TRAINING LOOP
# ==========================================
def lottery_ticket_training(
    task_name="sst2",
    model_name="bert-base-uncased",
    target_sparsity=0.9,
    pruning_rate=0.2,
    epochs_per_round=3,
    batch_size=16,
    learning_rate=2e-5,
    subset_size=1000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, val_loader, num_labels = load_and_process_datasets(task_name, tokenizer, batch_size, subset_size)

    model = LotteryTicketBERT(model_name, num_labels).to(device)
    pruner = IterativeMagnitudePruning(model, pruning_rate)

    history = {"sparsity": [], "train_acc": [], "val_acc": []}
    current_sparsity, iteration = 0.0, 0

    tracker = None
    if tracker_available:
        try:
            tracker = CarbonTracker(epochs=-1, components="cpu", ignore_warnings=True)
        except Exception as e:
            print(f"⚠️ CarbonTracker disabled: {e}")
            tracker = None

    print(f"\n{'='*50}\nStarting IMP for {task_name}\nTarget sparsity: {target_sparsity}\n{'='*50}")

    while current_sparsity < target_sparsity:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---\nCurrent sparsity: {current_sparsity:.2%}")
        model.rewind_weights()
        model.apply_masks()

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs_per_round
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        for epoch in range(epochs_per_round):
            if tracker:
                tracker.epoch_start()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_acc = evaluate(model, val_loader, device)
            if tracker:
                tracker.epoch_end()
            print(f"Epoch {epoch+1}/{epochs_per_round} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        history["sparsity"].append(current_sparsity)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if current_sparsity < target_sparsity:
            current_sparsity = pruner.iterative_pruning_step(current_sparsity, target_sparsity)
            stats = model.compute_mask_statistics()
            print(f"After pruning - Sparsity: {stats['sparsity']:.2%}")

    if tracker:
        tracker.stop()

    print(f"\n{'='*50}\nIMP Complete!\nFinal sparsity: {model.compute_mask_statistics()['sparsity']:.2%}")
    print(f"Final validation accuracy: {val_acc:.2%}\n{'='*50}")
    return model, history


# ==========================================
#  SAVE & LOAD UTILITIES
# ==========================================
def save_model_local(model, task_name, sparsity, history=None):
    filename = f"lottery_ticket_{task_name}_sparsity{int(sparsity*100)}.pt"
    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        "model_state_dict": model.model.state_dict(),
        "masks": model.masks,
        "init_weights": model.init_weights,
        "sparsity_stats": model.compute_mask_statistics(),
        "task_name": task_name,
        "target_sparsity": sparsity,
        "history": history or {},
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved: {filepath}")

    summary_path = filepath.replace(".pt", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"task_name": task_name, "sparsity": sparsity, "stats": checkpoint["sparsity_stats"], "history": history},
            f,
            indent=2,
        )
    print(f"Summary saved to: {summary_path}")
    return filepath


# ==========================================
#  MAIN EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    task = "sst2"
    sparsity = 0.7
    print(f"\n{'='*60}\nRunning experiment: {task} with target sparsity {sparsity}\n{'='*60}\n")

    model, history = lottery_ticket_training(
        task_name=task,
        target_sparsity=sparsity,
        epochs_per_round=2,  # short demo
        subset_size=500,
        batch_size=16,
    )

    save_model_local(model, task, sparsity, history)

    # Plot results
    plt.figure(figsize=(6, 4))
    plt.plot(history["sparsity"], history["val_acc"], marker="o", label="Validation")
    plt.plot(history["sparsity"], history["train_acc"], marker="s", label="Training")
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.title("BERT Lottery Ticket: Accuracy vs Sparsity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(save_dir, f"results_plot_{task}_sparsity{int(sparsity*100)}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to: {plot_path}")
