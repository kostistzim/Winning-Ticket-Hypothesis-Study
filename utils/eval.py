import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

def test(testloader, model,device):

    model.eval()
    accuracy=[]
    model.eval()
    top1=AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        outputs = model(inputs)
        predictions=outputs.data.argmax(dim=1)
        labels=targets.data
        correct = (predictions == labels).sum().item()
        accuracy=100.0 * correct / labels.size(0)
        top1.update(accuracy, inputs.size(0))
    return top1.avg

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