from tqdm import tqdm
import pandas as pd


def train(model, device, train_loader, optimizer, criterion,
          total_iterations=50000, val_loader=None, test_loader=None,
          evaluate_fn=None, eval_every=100, save_log_path=None):
    # Call model train function
    model.train()
    iteration = 0
    epoch = 0
    running_loss = 0.0
    logs = []

    while iteration < total_iterations:
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", ncols=100)
        for batch_idx, (data, target) in progress_bar:
            if iteration >= total_iterations:
                break

            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iteration += 1

            # Update tqdm display text
            progress_bar.set_postfix(loss=loss.item(), iters=iteration)

            if (evaluate_fn is not None) and (iteration % eval_every == 0):
                val_loss, val_acc = evaluate_fn(model, device, val_loader, criterion)
                test_acc = None
                if test_loader is not None:
                    _, test_acc = evaluate_fn(model, device, test_loader, criterion)

                logs.append((iteration, val_acc, test_acc))
                print(f"[Iter {iteration}/{total_iterations}] "
                      f"ValAcc={val_acc:.2f}% "
                      f"{'(TestAcc=' + f'{test_acc:.2f}%)' if test_acc is not None else ''}")

        epoch += 1


    average_loss = running_loss / total_iterations
    print(f"Training completed — {iteration} iterations, avg loss {average_loss:.4f}")

    if save_log_path and len(logs) > 0:
        df = pd.DataFrame(logs, columns=["iteration", "val_acc", "test_acc"])
        df.to_csv(save_log_path, index=False)
        print(f"📄 Saved training log → {save_log_path}")


    return average_loss
