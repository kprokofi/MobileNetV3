# -*- coding: UTF-8 -*-

'''
Image dataset loader
'''

from torchvision import transforms, datasets
import os
import torch
import cv2 as cv
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as scio


class ImageTxt(Dataset):
    def __init__(self, root, mode='train', transform=None) -> None:

        self.root = root
        self.transform = transform

        if mode == 'train':
            self.annot = self.root + '/train.txt'
            self.data = self.load_annotation(
                self.annot,
                self.root,
            )

        if mode == 'val':
            self.annot = self.root + '/val.txt'
            self.data = self.load_annotation(
                self.annot,
                self.root,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        image, label = self.data[indx]
        image = cv.imread(image)
        assert image is not None
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, int(label)

    @staticmethod
    def load_annotation(annot_path, data_dir, dataset_id=0):
        out_data = []
        for line in open(annot_path):
            parts = line.strip().split(' ')
            if len(parts) != 2:
                print("line doesn't fits pattern. Expected: 'relative_path/to/image label'")
                continue
            rel_image_path, label_str = parts
            full_image_path = os.path.join(data_dir, rel_image_path)
            if not os.path.exists(full_image_path):
                continue

            label = int(label_str)
            out_data.append((full_image_path, label))
        return out_data

def Cifar10DataLoader(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    }

    image_datasets = {}
    image_datasets['train'] = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def Cifar100DataLoader(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
    }

    image_datasets = {}
    image_datasets['train'] = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def ImageNetDataLoader(args):
    # data transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC2012_img_train'), transform=data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'ILSVRC2012_img_val'), transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def TinyImageNetDataLoader(args):
    # data transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(56),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(56),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def SVHNDataLoader(args):
    from SVHN import SVHN
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4309, 0.4302, 0.4463), (0.1965, 0.1983, 0.1994))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4524, 0.4525, 0.4690), (0.2194, 0.2266, 0.2285))
        ])
    }

    image_datasets = {}
    image_datasets['train'] = SVHN(root=os.path.join(args.data_dir, 'SVHN'), split='train', download=False, transform=data_transforms['train'])
    image_datasets['val'] = SVHN(root=os.path.join(args.data_dir, 'SVHN'), split='test', download=False, transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def CustomDatasetFolder(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),
            transforms.RandomCrop(args.resolution, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.resolution,args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
    }


    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def CustomDatasetTxt(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resolution,args.resolution)),
            transforms.RandomCrop(args.resolution, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resolution,args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
    }


    image_datasets = {}
    image_datasets['train'] = ImageTxt(root=args.data_dir,mode='train', transform=data_transforms['train'])
    image_datasets['val'] = ImageTxt(root=args.data_dir, mode='val', transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False,
                    num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val']}

    return dataloders

def dataloaders(args):
    dataset = args.dataset.lower()
    if dataset == 'imagenet':
        return ImageNetDataLoader(args)
    elif dataset == 'tinyimagenet':
        return TinyImageNetDataLoader(args)
    elif dataset == 'cifar10':
        return Cifar10DataLoader(args)
    elif dataset == 'cifar100':
        return Cifar100DataLoader(args)
    elif dataset == 'svhn':
        return SVHNDataLoader(args)
    elif dataset == 'custom_folder':
        return CustomDatasetFolder(args)
    elif dataset == 'custom_txt':
        return CustomDatasetTxt(args)
