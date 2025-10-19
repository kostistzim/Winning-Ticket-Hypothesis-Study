import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(batch_size):
    """Downloads and loads the CIFAR-10 dataset."""
    
    # We must convert the image to a tensor and normalize it.
    # These are the standard mean and std dev values for CIFAR-10.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Using the 'batch_size' argument here for consistency
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False) 
    
    return trainloader, testloader