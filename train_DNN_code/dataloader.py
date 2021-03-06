import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from typing import Tuple, List

from torchvision.transforms import functional as F

class SyntheticGaussianDataset(torchvision.datasets.CIFAR10):

    def __init__(self, data_num):
        self.data = torch.rand(data_num,3,32,32)
        self.data = self.data / torch.max(self.data)
        self.targets = torch.randint(low=0, high=9, size=(data_num,))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        
        img = F.normalize(img,(0.1307,),(0.3081,))
       
        return img, target


    def __len__(self) -> int:
        return len(self.data)

def get_data_loaders(args):
    if args.trainloader and args.testloader:
        assert os.path.exists(args.trainloader), 'trainloader does not exist'
        assert os.path.exists(args.testloader), 'testloader does not exist'
        trainloader = torch.load(args.trainloader)
        testloader = torch.load(args.testloader)
        return trainloader, testloader

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.raw_data:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        if not args.noaug:
            # with data augmentation
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # no data agumentation
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.ngpu else {}
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transform_test)
    elif args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    return trainloader, testloader, trainset


def get_synthetic_gaussian_data_loaders(args):
    trainset = SyntheticGaussianDataset(data_num=50000)
    testset = SyntheticGaussianDataset(data_num=1000)


     # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=False
        #sampler=torch.utils.data.SubsetRandomSampler(np.arange(args.subset))
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        shuffle=False
        #sampler=torch.utils.data.SubsetRandomSampler(np.arange(args.subset))
        )
   

    return train_loader, test_loader
