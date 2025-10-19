# CNN/trainer.py

import torch
import torch.nn as nn

# Import from the pruning.py file in the same directory
from pruning import apply_mask_to_gradients

def train(model, mask, trainloader, optimizer, criterion, device):
    """Standard training loop for one epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # Apply mask to gradients to ensure pruned weights don't get updated
        apply_mask_to_gradients(model, mask)
        optimizer.step()

def test(model, testloader, criterion, device):
    """Standard evaluation loop."""
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)
    return accuracy, test_loss