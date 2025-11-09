#!/usr/bin/env python3
# ==========================================
#  FEW-SHOT TRANSFER TRAINING FOR LOTTERY TICKETS
#  (Reuses modules from lottery_ticket_training.py)
# ==========================================

import os
import json
import torch
import argparse
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import reusable parts
from lottery_ticket_bert import (
    set_seed,
    BertForCustomTask,
    load_and_process_datasets,
    train_epoch,
    evaluate,
)

# ==========================================
#  FEW-SHOT TRAINING FUNCTION
# ==========================================
def fewshot_transfer(
    checkpoint_path,
    target_task,
    mode="headonly",
    seed=42,
    fewshot_size=128,
    batch_size=16,
    learning_rate=2e-5,
    epochs=3,
    save_dir="./fewshot_models",
):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Few-shot fine-tuning | Target: {target_task.upper()} | Mode: {mode} | Seed: {seed}")
    print(f"{'='*60}\n")

    # --- Load checkpoint ---
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Load 128 examples from target task ---
    train_loader, val_loader, num_labels = load_and_process_datasets(
        task_name=target_task,
        tokenizer=tokenizer,
        batch_size=batch_size,
        subset_size=fewshot_size,
    )

    # --- Initialize model and load weights ---
    model = BertForCustomTask(model_name, num_labels)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # --- Freeze if head-only mode ---
    if mode == "headonly":
        for p in model.bert.parameters():
            p.requires_grad = False
        print("🧊 Head-only mode: encoder frozen.")
    else:
        print("🔥 Full fine-tuning mode: all parameters trainable.")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # --- Train ---
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

    # --- Save few-shot model + summary ---
    base = os.path.basename(checkpoint_path).replace(".pt", "")
    out_model = f"fewshot_{target_task}_from_{base}_{mode}.pt"
    out_model_path = os.path.join(save_dir, out_model)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint": checkpoint_path,
            "target_task": target_task,
            "mode": mode,
            "seed": seed,
            "train_acc": train_acc,
            "val_acc": val_acc,
        },
        out_model_path,
    )

    # Summary JSON
    out_json = out_model_path.replace(".pt", ".json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "source_checkpoint": checkpoint_path,
                "target_task": target_task,
                "mode": mode,
                "seed": seed,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "examples": fewshot_size,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Saved few-shot model to {out_model_path}")
    print(f"📄 Summary saved to {out_json}\n")

# ==========================================
#  MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot fine-tuning for transfer experiments.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (.pt)")
    parser.add_argument("--target", required=True, choices=["sst2", "qqp"], help="Target task for transfer")
    parser.add_argument("--mode", choices=["headonly", "full"], default="headonly", help="Training mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="./fewshot_models", help="Output directory")
    parser.add_argument("--fewshot_size", type=int, default=128, help="Number of examples from target task")
    args = parser.parse_args()

    fewshot_transfer(
        checkpoint_path=args.checkpoint,
        target_task=args.target,
        mode=args.mode,
        seed=args.seed,
        fewshot_size=args.fewshot_size,
        save_dir=args.save_dir,
    )
