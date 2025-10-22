"""To train the vgg model on cifar."""
from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from carbontracker.tracker import CarbonTracker
from tqdm import tqdm
from vgg import VGG

from utils.eval import AverageMeter,test,get_dataloaders


def train(model:nn.Module, train_loader, criterion,optimiser,device,iteration,args):
    """Traing vgg on cifar."""

    model.train()

    for batch_id,(inputs,targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs=model(inputs)
        loss=criterion(outputs,targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # print("Srep: ",batch_id+1," Loss: ",loss.item())
        iteration+=1
        if iteration in [30000,90000]:
            save_checkpoint(model.state_dict(),iteration,args.lr,args.wd,args.seed)

    return iteration

def save_checkpoint(state,iteration,lr,wd,seed,model_name='vgg'):
    file_path=f'models/{model_name}_seed={seed}_iter={iteration}_lr={lr}_wd={wd}.pth'
    torch.save(state, file_path)

def adjust_learning_rate(optimizer, epoch,args,lr):
    if epoch in args.schedule:
        lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)


    epochs=args.epochs
    tracker = CarbonTracker(epochs=epochs,
                            log_dir='carbontracker/',
                            log_file_prefix=F"vgg_seed={args.seed}_lr={args.lr}_wd={args.wd}")
    device=args.device

    if not os.path.exists('models/'):
        os.makedirs('models/')
    if not os.path.exists('carbontracker/'):
        os.makedirs('carbontracker/')

    model=VGG()
    model.to(device)
    train_loader,test_loader=get_dataloaders(args)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    iteration=0
    lr=args.lr
    for epoch in tqdm(range(epochs),desc="Training model"):
        tracker.epoch_start()
        adjust_learning_rate(optimiser, epoch,args,lr)

        iteration=train(model,train_loader,criterion,optimiser,device,iteration,args)

        if (epoch+1)%40==0:
            test_acc=test(test_loader,model,device)
            print("Accuracy for epoch, ",epoch,' is ',test_acc)
        tracker.epoch_end()

    tracker.stop()
    save_checkpoint(model.state_dict(),iteration,args.lr,args.wd,args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=160, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--bs', default=64, type=int, metavar='N',
                        help='train and testing batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    main(args)
