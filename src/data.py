import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST

URL = 'https://github.com/medmnist/medmnist/raw/master/'

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    train_ds = PneumoniaMNIST(split='train', transform=transform, download=True)
    test_ds  = PneumoniaMNIST(split='test',  transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader