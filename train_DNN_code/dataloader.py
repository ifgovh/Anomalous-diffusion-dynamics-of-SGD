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


    kwargs = {'num_workers': 2, 'pin_memory': True} if args.ngpu else {}
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transform_test)
    elif args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                               transform=transform_test)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    return trainloader, testloader, trainset


def get_synthetic_gaussian_data_loaders(args):
    num_classes = 10
    dim_data = args.gauss_dimension 
    num_train_samples = args.gauss_train_samples
    num_test_samples  = args.gauss_test_samples        
    scale = args.gauss_scale
    means = np.random.multivariate_normal(np.zeros(dim_data), np.identity(dim_data), size=num_classes)         
    means = scale * means / np.linalg.norm(means, ord=2, keepdims=True, axis=1)

    min_distance = 100.0
    for i in range(num_classes):
        for j in range(1+i, num_classes):
             if np.linalg.norm(means[i] - means[j]) < min_distance:        
                 min_distance = np.linalg.norm(means[i] - means[j])

    sigma = 0.05 * min_distance/np.sqrt(dim_data) 
    train_data = []
    class_num = 0 
    for mean in means: 
         samples = np.random.multivariate_normal(mean, sigma * np.identity(dim_data), size=num_train_samples // num_classes)
         for sample in samples:  
             train_data.append([sample, class_num])
         class_num += 1         

    test_data = []
    class_num = 0   
    for mean in means:
        samples = np.random.multivariate_normal(mean, sigma * np.identity(dim_data), size=num_test_samples // num_classes)
        for sample in samples:
             test_data.append([sample, class_num])  
        class_num += 1

     # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False
        #sampler=torch.utils.data.SubsetRandomSampler(np.arange(args.subset))
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False
        #sampler=torch.utils.data.SubsetRandomSampler(np.arange(args.subset))
        )
   

    return train_loader, test_loader
   
