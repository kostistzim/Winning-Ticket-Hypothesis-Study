import random
import torch
from utils.eval import AverageMeter,test,get_dataloaders
from utils.prunning import make_mask,apply_mask,global_pruning_by_percentage,global_pruning_by_percentage_random
from vgg import VGG
import torchvision.datasets as datasets
import argparse

def main(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model =VGG().to(args.device)
    model_path=f'models/{args.model_name}_seed={args.seed}_iter={args.iteration}_lr={args.lr}_wd={args.wd}.pth'
    model_weights=torch.load(model_path)

    current_mask=make_mask(model)
    trial_accs=[]
    masksused=[]

    _,testloader=get_dataloaders(args)

    for _ in range(0,30):
        model.load_state_dict(model_weights)
        masksused.append(current_mask)
        apply_mask(model,current_mask)
        acc=test(testloader,model,args.device)
        if args.prunning=='global':
            current_mask=global_pruning_by_percentage(model,20,current_mask)
        elif args.prunning=='random':
            current_mask=global_pruning_by_percentage_random(model,20,current_mask)

        trial_accs.append(acc)
    save_path=f'VGG/{args.prunning}_results_seed={args.seed}_iter={args.iteration}_lr={args.lr}_wd={args.wd}.pth'
    torch.save({'masksused':masksused,'trial_accs':trial_accs},save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    parser.add_argument('--prunning', default='global', type=str,
                        help='how to prune',choices=['global','random'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--iteration', type=int,
                        help='iteration')
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