import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

class ImagenetDataLoader(torch.utils.data.DataLoader):
    def __next__(self):
        try:
            super().__next__(self)
        except OSError:
            return (None, None)


class DataLoader(object):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--dset',type=str, help='MNIST,CIFAR, ...',default="MNIST")

    def __init__(self, args, test):
        self.args = args
        self.test = test
        if(args.dset=="MNIST"):
            self.mnest_loader()
            self.valid_iterator = iter(self.valid_data_loader)
        elif(args.dset.startswith("cifar")):
            self.cifar_loader()
            self.valid_iterator = iter(self.valid_data_loader)
        elif args.dset=='SVHN':
            self.svhn_loader()
            self.valid_iterator = iter(self.valid_data_loader)
        elif(args.dset=='IMAGENET'):
            self.imagenet_loader()
            self.valid_iterator = iter(self.valid_data_loader)
        else:
            self.cifar_loader()
            self.valid_iterator = iter(self.valid_data_loader)

    def __iter__(self):
        return iter(self.data_loader)

    def valid_batch(self):
        try:
            return next(self.valid_iterator)
        except StopIteration:
            self.valid_iterator = iter(self.valid_data_loader)
            return next(self.valid_iterator)

    def __len__(self):
        return len(self.data_loader)

    def mnest_loader(self):
        def flatten(x):
            return x.view((-1))

        transform_l = []
        transform_l.append(transforms.ToTensor())
        if self.args.network_type == 'auto_fc':
            transform_l.append(transforms.Lambda(flatten))
        transform=transforms.Compose(transform_l)

        if self.test:
            data_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transform),
                batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        else:
            data_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,transform=transform),
                batch_size=self.args.batch_size, shuffle=True, drop_last=True, num_workers=2)
            self.valid_data_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transform),
                batch_size=self.args.batch_size, shuffle=False, drop_last=True)


        self.data_loader = data_loader

    def manifold_loader(self):
        if self.args.seed != -1:
            seed = self.args.seed
        else:
            seed = np.random.randint(np.power(2,31))
            self.args.seed = seed
        train_dataset=ManifoldSampler(self.args, seed)
        valid_dataset=ManifoldSampler(self.args, seed)

        self.data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=self.args.batch_size, shuffle=True, drop_last=True, num_workers=2)
        self.valid_data_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=2)

    def svhn_loader(self):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        trainer=not self.test
        dataset=datasets.SVHN('../datasets/data_svhn',split='train' if trainer else 'test',download=True, transform=transform_train if (not self.test) else transform_test)
        valid_dataset=datasets.SVHN('../datasets/data_svhn',split='test',download=True, transform=transform_test)

        self.data_loader=torch.utils.data.DataLoader(dataset,batch_size=self.args.batch_size, shuffle=not self.test, drop_last=True, num_workers=4)
        self.valid_data_loader= torch.utils.data.DataLoader(valid_dataset,batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=4)


    def cifar_loader(self):
        """
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        """

        """
        transform_l = []
        transform_l.append(transforms.ToTensor())
        transform_l.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if self.args.network_type == 'auto_fc':
            transform_l.append(transforms.Lambda(flatten))
        transform=transforms.Compose(transform_l)
        """

        #transforms taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        transform_train = transforms.Compose([
            #transforms.RandomRotation(15),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainer=not self.test
        if self.args.dset=='CIFAR100':
            dataset=datasets.CIFAR100('../datasets/data_cifar100',train=trainer,download=True, transform=transform_train if (not self.test) else transform_test)
            valid_dataset=datasets.CIFAR100('../datasets/data_cifar100',train=False,download=False, transform=transform_test)
        else:
            dataset=datasets.CIFAR10('../datasets/data_cifar',train=trainer,download=True, transform=transform_train if (not self.test) else transform_test)
            valid_dataset=datasets.CIFAR10('../datasets/data_cifar',train=False,download=False, transform=transform_test)

        self.data_loader=torch.utils.data.DataLoader(dataset,batch_size=self.args.batch_size, shuffle=not self.test, drop_last=True, num_workers=4)
        self.valid_data_loader= torch.utils.data.DataLoader(valid_dataset,batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=4)


    def imagenet_loader(self):
        traindir = os.path.join('/scratch','IMAGENET','UNZIPPED', 'train')
        valdir = os.path.join('/scratch','IMAGENET', 'UNZIPPED', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_dataset = datasets.ImageFolder(
            valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        )

        self.data_loader = ImagenetDataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True ,num_workers=8, pin_memory=True)
    
        self.valid_data_loader = ImagenetDataLoader(
            valid_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)

