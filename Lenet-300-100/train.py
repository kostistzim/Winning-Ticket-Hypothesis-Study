from tqdm import tqdm
import torch
import pandas as pd

def train(
    model, device, train_loader, optimizer, criterion,
    total_iterations=50000, val_loader=None, test_loader=None,
    evaluate_fn=None, eval_every=100, save_log_path=None,
    use_amp=True
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    iteration, epoch, running_loss = 0, 0, 0.0
    logs = []

    best_val_loss = float("inf")
    best_iter = 0
    best_metrics = {"train_acc": 0.0, "val_acc": 0.0, "test_acc": 0.0, "val_loss": float("inf")}

    while iteration < total_iterations:
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}", ncols=100)
        for data, target in progress_bar:
            if iteration >= total_iterations:
                break

            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(data)
                loss = criterion(outputs, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            iteration += 1

            if iteration % 50 == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", iters=iteration)

            if (evaluate_fn is not None) and (iteration % eval_every == 0):
                model.eval()
                with torch.no_grad():
                    val_loss, val_acc = evaluate_fn(model, device, val_loader, criterion)
                    test_acc = None
                    if test_loader is not None:
                        _, test_acc = evaluate_fn(model, device, test_loader, criterion)

                logs.append((iteration, None, val_loss, val_acc, test_acc))
                print(f"[Iter {iteration}/{total_iterations}] "
                      f"ValLoss={val_loss:.4f} | ValAcc={val_acc:.2f}% "
                      f"{'(TestAcc=' + f'{test_acc:.2f}%)' if test_acc is not None else ''}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iter = iteration
                    best_metrics.update({"val_loss": val_loss, "val_acc": val_acc, "test_acc": test_acc})

                model.train()

        epoch += 1

    avg_loss = running_loss / total_iterations
    print(f"Training completed — {iteration} iterations, avg loss {avg_loss:.4f}")
    print(f"💡 Minimum val_loss at iter {best_iter}: "
          f"ValLoss={best_metrics['val_loss']:.4f} | "
          f"ValAcc={best_metrics['val_acc']:.2f}% | "
          f"TestAcc={best_metrics['test_acc']:.2f}%")

    if save_log_path and len(logs) > 0:
        df = pd.DataFrame(logs, columns=["iteration", "train_acc", "val_loss", "val_acc", "test_acc"])
        df = pd.concat([
            df,
            pd.DataFrame([{
                "iteration": f"# MinValLoss (iter={best_iter})",
                "train_acc": best_metrics["train_acc"],
                "val_loss": best_metrics["val_loss"],
                "val_acc": best_metrics["val_acc"],
                "test_acc": best_metrics["test_acc"],
            }])
        ], ignore_index=True)
        df.to_csv(save_log_path, index=False)
        print(f"📄 Saved training log → {save_log_path}")

        summary_path = save_log_path.replace(".csv", "_summary.csv")
        pd.DataFrame([{
            "min_val_loss_iter": best_iter,
            "val_loss": best_metrics["val_loss"],
            "val_acc": best_metrics["val_acc"],
            "test_acc": best_metrics["test_acc"],
        }]).to_csv(summary_path, index=False)
        print(f"📄 Saved summary → {summary_path}")

    return avg_loss
