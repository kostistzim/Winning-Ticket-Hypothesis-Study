import torch
from torchvision import datasets, transforms

def load_dataset():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Split training dataset in train and validation sets
    train_set, val_set = torch.utils.data.random_split(train_dataset, [55000, 5000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=60, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, val_loader, test_loader