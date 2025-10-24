import random
import torch
import torch.nn as nn
import torch.optim as optim
from eval import AverageMeter,test,get_dataloaders
from train import train
from prunning import make_mask,apply_mask,global_pruning_by_percentage,global_pruning_by_percentage_random,get_sparsity
from vgg import VGG
import torchvision.datasets as datasets
import argparse
from tqdm import tqdm
import pandas as pd
from carbontracker.tracker import CarbonTracker

def random_reinitialize(model, mask, init_fn):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                new_param = torch.empty_like(param)
                if new_param.ndimension() < 2:
                    nn.init.uniform_(new_param, -0.01, 0.01)
                else:
                    init_fn(new_param)
                param.data = new_param * mask[name]
    return model

def main(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # tracker = CarbonTracker(epochs=args.prune_iterations,
                            # log_dir='carbontracker/',
                            # log_file_prefix=F"vgg_seed={args.seed}_lr={args.lr}_wd={args.wd}")
    model =VGG().to(args.device)
    if args.iterations!=0:
        model_path=f'models/{args.model_name}_seed={args.seed}_iter={args.iterations}_lr={args.lr}_wd={args.wd}.pth'
        initial_weights=torch.load(model_path)
    else:
        initial_weights=model.state_dict()

    current_mask=make_mask(model)
    results_log = []
    masksused=[]

    trainloader,testloader=get_dataloaders(args)

    for i in tqdm(range(args.prune_iterations)):
        # tracker.epoch_start()

        if args.prunning=='random':
            model=model._initialize_weights()
        else:
            model.load_state_dict(initial_weights)
        sparsity, remaining_pct = get_sparsity(current_mask)
        print(f"Sparsity: {sparsity:.2f}% | Weights Remaining: {remaining_pct:.2f}%")
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        for _epoch in tqdm(range(args.epochs),'traning model'):
            _=train(model,trainloader,criterion,optimiser,args.device,0,args,mask=current_mask)
        masksused.append(current_mask)
        acc=test(testloader,model,args.device)
        print('Accuracy: ',acc)
        results_log.append({
            'pruning_iteration':i,
            'sparcity_pct':sparsity,
            'weight_remaining_pct':remaining_pct,
            'acc':acc
        })
        current_mask=global_pruning_by_percentage(model,0.2,current_mask)
        apply_mask(model,current_mask)
        
        # tracker.epoch_end()
    # tracker.stop()
    if args.iterations!=0:
        accuracy_save_path=f'VGG/results/accuracy/{args.prunning}_results_seed={args.seed}_iter={args.iterations}_lr={args.lr}_wd={args.wd}_epochs={args.epochs}.csv'
    else:
        accuracy_save_path=f'VGG/results/accuracy/{args.prunning}_results_seed={args.seed}_lr={args.lr}_wd={args.wd}_epochs={args.epochs}.csv'
    df=pd.DataFrame(results_log)
    df.to_csv(accuracy_save_path,index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    parser.add_argument('--prunning', default='global', type=str,
                        help='how to prune',choices=['global','random'])
    parser.add_argument('--model_name', default='vgg', type=str,
                        help='Model name')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--iterations', type=int,default=0,
                        help='iteration')
    parser.add_argument('--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--prune_iterations', default=20, type=int, metavar='N',
                        help='number of pruneing_steps to run')
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