import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import numbers
import random
from typing import Tuple, List

from torchvision.transforms import functional as F

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
    num_classes = 10
    num_train_samples = 50000
    num_test_samples  = 1000   

    train_data = []
    for _ in range(num_train_samples):
        for class_i in range(1,num_classes+1):
            train_data.append([class_i * torch.rand(3,32,32), class_i])

    test_data = []
    for _ in range(num_test_samples):
        for class_i in range(1,num_classes+1):
            test_data.append([class_i * torch.rand(3,32,32), class_i])


     # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False
        #sampler=torch.utils.data.SubsetRandomSampler(np.arange(args.subset))
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False
        #sampler=torch.utils.data.SubsetRandomSampler(np.arange(args.subset))
        )
   

    return train_loader, test_loader
   
