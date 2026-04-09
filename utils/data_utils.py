import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_mnist(root='./data', train=True, img_size=32):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    return datasets.MNIST(root, train=train, download=True, transform=transform)


def get_usps(root='./data', train=True, img_size=32):
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    return datasets.USPS(root, train=train, download=True, transform=transform)


def get_svhn(root='./data', train=True, img_size=32):
    split = 'train' if train else 'test'
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    return datasets.SVHN(root, split=split, download=True, transform=transform)


def get_office31(root='./data/office31', domain='amazon', img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    path = os.path.join(root, domain, 'images')
    return datasets.ImageFolder(path, transform=transform)


def get_office_home(root='./data/office_home', domain='Art', img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    path = os.path.join(root, domain)
    return datasets.ImageFolder(path, transform=transform)


def get_pacs(root='./data/pacs', domain='photo', img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    path = os.path.join(root, domain)
    return datasets.ImageFolder(path, transform=transform)


def get_unnormalized_transform(img_size=224):
    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])


def get_paired_loaders(source_dataset, target_dataset, batch_size=32, num_workers=2):
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return source_loader, target_loader
