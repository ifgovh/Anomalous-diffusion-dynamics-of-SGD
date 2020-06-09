"""
    Calculate the hessian matrix of the projected surface and their eigen values.
"""

import argparse
import copy
import numpy as np
import h5py
import torch
import time
import socket
import os
import sys
import torchvision
import torch.nn as nn
import dataloader
import model_loader

from hessian_eigenthings import compute_hessian_eigenthings
import scipy.io as sio


###############################################################
####                        MAIN
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hessian_eigenthings')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--datapath', default='data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='./trained_nets/resnet14_sgd_lr=0.1_bs=512_wd=0_mom=0_save_epoch=1', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--max_epoch', type=int, default=500, help='the maximum epoch')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')    

    # eig parameters
    parser.add_argument('--num_eigenthings', type=int, default=5, help='compute how many top eigenvalues/eigenvectors')

    args = parser.parse_args()

    torch.manual_seed(123)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------   

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()        
        print('Use GPU %d of %d GPUs on %s' %
              (torch.cuda.current_device(), gpu_count, socket.gethostname()))

    for epoch in range(0,args.max_epoch,5):
        #--------------------------------------------------------------------------
        # Load models and extract parameters
        #--------------------------------------------------------------------------
        net = model_loader.load(args.dataset, args.model, args.model_folder+'/model_' + str(epoch) + '.t7')
        if args.ngpu > 1:
            # data parallel with multiple GPUs on a single node
            net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
       
        
        #--------------------------------------------------------------------------
        # Setup dataloader
        #--------------------------------------------------------------------------
        # download CIFAR10 if it does not exit
        if args.dataset == 'cifar10':
            torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)


        trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
                                    args.batch_size, args.threads, args.raw_data,
                                    args.data_split, args.split_idx,
                                    args.trainloader, args.testloader)

        #--------------------------------------------------------------------------
        # Setup loss function
        #--------------------------------------------------------------------------
        if args.loss_name == 'crossentropy':
            loss = torch.nn.functional.cross_entropy
        else:
            raise Exception('Add your loss function here')

        #--------------------------------------------------------------------------
        # Start the computation
        #--------------------------------------------------------------------------        

        eigenvals, eigenvecs = compute_hessian_eigenthings(net, trainloader,
                                                       loss, args.num_eigenthings,False,"power_iter",True,args.batch_size)

        #--------------------------------------------------------------------------
        # save results
        #--------------------------------------------------------------------------
        
        sio.savemat(args.model_folder + '/eigendata_' + str(epoch) + '.mat',
                                mdict={'eigenvals': eigenvals,'eigenvecs': eigenvecs}
                                )

