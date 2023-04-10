import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms 

def get_ds(data, size=1, split=False):
    
    assert data in ['mnist','cifar10'], "unsupported data"
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]) 
    
    if data=='mnist':
        tval_data = datasets.MNIST(
            root="./datasets",
            train=True,
            download=True,
            transform=transform
        )

        test_data = datasets.MNIST(
            root="./datasets",
            train=False,
            download=True,
            transform=transform
        )

    elif data=='cifar10':
        tval_data = datasets.CIFAR10(
            root="./datasets",
            train=True,
            download=True,
            transform=transform
        )

        test_data = datasets.CIFAR10(
            root="./datasets",
            train=False,
            download=True,
            transform=transform
        )

    if split:
        train_data, val_data = random_split(tval_data, [int(0.8*len(tval_data)), len(tval_data)-int(0.8*len(tval_data))], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_data, size)
        valid_loader = DataLoader(val_data, size)
        test_loader = DataLoader(test_data, size)
        return train_loader, valid_loader, test_loader
    
    else:
        train_loader = DataLoader(tval_data, size, shuffle=False)
        test_loader = DataLoader(test_data, size, shuffle=False)
        return train_loader, test_loader
