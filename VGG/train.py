
"""To train the vgg model on cifar"""
import numpy as np
import torch
import torch.nn as nn

def train(model:nn.Module, train_loader, criterion,optimiser,epoch,device):
    """Traing vgg on cifar."""

    model.train()

    for batch_id,(inputs,targets) in enumerate(train_loader):

        inputs.to(device)
        targets.to(device)
        outputs=model(inputs)
        loss=criterion(outputs,targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print("Srep: ",batch_id+1," Loss: ",loss.item())
    
def save_checkpoint(state,lr,wd,model_name='vgg'):
    file_path=f'models/{model_name}_lr={lr}_wd={wd}.pth'
    torch.save(state, file_path)
    
    