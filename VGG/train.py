"""To train the vgg model on cifar."""
from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from vgg import VGG


class AverageMeter(object):
    """Computes and stores the average and current value. Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262."""
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model:nn.Module, train_loader, criterion,optimiser,device,iteration,args):
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
        iteration+=1
        if iteration in [30000,90000]:
            save_checkpoint(model.state_dict(),iteration,args.lr,args.wd,args.seed)

    return iteration

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k. Imported from https://github.com/pytorch/metrics."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(testloader, model,device):

    top1 = AverageMeter()
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        outputs = model(inputs)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
    return top1.avg

def save_checkpoint(state,iteration,lr,wd,seed,model_name='vgg'):
    file_path=f'models/{model_name}_seed={seed}_iter={iteration}_lr={lr}_wd={wd}.pth'
    torch.save(state, file_path)

def get_dataloaders(args):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = datasets.CIFAR10

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    return trainloader,testloader

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
    device=args.device
    if not os.path.exists('models/'):
        os.makedirs('models/')
    model=VGG()
    model.to(device)
    train_loader,test_loader=get_dataloaders(args)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    iteration=0
    lr=args.lr
    for epoch in tqdm(epochs,desc="Training model"):
        adjust_learning_rate(optimiser, epoch,args,lr)

        iteration=train(model,train_loader,criterion,optimiser,device,iteration,args)

        if (epoch+1)%10==0:
            test_acc=test(test_loader,model,device)
            print("Accuracy for epoch, ",epoch,' is ',test_acc)


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
