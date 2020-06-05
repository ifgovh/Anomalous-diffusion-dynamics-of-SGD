from __future__ import print_function
import os
import random
import numpy as np
import argparse
import scipy.io as sio
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel

import model_loader as ml

import get_gradient_weight.net_plotter as net_plotter
from get_gradient_weight.gradient_noise import get_layerWise_norms, get_grads, alpha_estimator, alpha_estimator2, compute_hessian

import train_DNN_code.model_loader as model_loader
from train_DNN_code.dataloader import get_data_loaders

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Training with save all transient state in one epoch
def train_save(trainloader, net, criterion, optimizer, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    grads = []
    sub_loss = []
    sub_weights = []

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # get gradient
            grad = get_grads(net).cpu()
            grads.append(grad)
            optimizer.step()

            # record tiny steps in every epoch
            sub_loss.append(loss.item())
            w = net_plotter.get_weights(net) # initial parameters
            for j in range(len(w)):
                w[j] = w[j].cpu().numpy()
            sub_weights.append(w)

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            loss.backward()

            # get gradient
            grad = get_grads(net).cpu()
            grads.append(grad)

            optimizer.step()

            # record tiny steps in every epoch
            sub_loss.append(loss.item())
            import pdb; pdb.set_trace()
            w = net_plotter.get_weights(net) # initial parameters
            for j in range(len(w)):
                w[j] = w[j].cpu().numpy()
            sub_weights.append(w)

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    M = len(grads[0]) # total number of parameters
    grads = torch.cat(grads).view(-1, M)
    mean_grad = grads.sum(0) / (batch_idx + 1) # divided by # batchs
    noise_norm = (grads - mean_grad).norm(dim=1)
    
    N = M * (batch_idx + 1) 

    for i in range(1, 1 + int(math.sqrt(N))):
        if N%i == 0:
            m = i
    alpha = alpha_estimator(m, (grads - mean_grad).view(-1, 1))
    
    del grads
    del mean_grad

    hist_noise = [noise_norm.numpy(), alpha.numpy()]

    w,g = get_layerWise_norms(net)

    layerWise_norms = [w,g]  

    return train_loss/total, 100 - 100.*correct/total, hist_noise, layerWise_norms, sub_weights, sub_loss


def test_save(testloader, net, criterion, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    sub_loss = []

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            sub_loss.append(loss.item())
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            sub_loss.append(loss.item())
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total, sub_loss

# Training without save all transient state in one epoch
def train(trainloader, net, criterion, optimizer, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    grads = []    

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # get gradient
            grad = get_grads(net).cpu()
            grads.append(grad)
            optimizer.step()

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            loss.backward()

            # get gradient
            grad = get_grads(net).cpu()
            grads.append(grad)

            optimizer.step()            

            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    M = len(grads[0]) # total number of parameters
    grads = torch.cat(grads).view(-1, M)
    mean_grad = grads.sum(0) / (batch_idx + 1) # divided by # batchs
    noise_norm = (grads - mean_grad).norm(dim=1)
    
    N = M * (batch_idx + 1) 

    for i in range(1, 1 + int(math.sqrt(N))):
        if N%i == 0:
            m = i
    alpha = alpha_estimator(m, (grads - mean_grad).view(-1, 1))
    
    del grads
    del mean_grad

    hist_noise = [noise_norm.numpy(), alpha.numpy()]

    w,g = get_layerWise_norms(net)

    layerWise_norms = [w,g]  

    return train_loss/total, 100 - 100.*correct/total, hist_noise, layerWise_norms


def test(testloader, net, criterion, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total

def name_save_folder(args):
    save_folder = args.model + '_' + str(args.optimizer) + '_lr=' + str(args.lr)
    if args.lr_decay != 0.1:
        save_folder += '_lr_decay=' + str(args.lr_decay)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mom=' + str(args.momentum)
    save_folder += '_save_epoch=' + str(args.save_epoch)
    if args.loss_name != 'crossentropy':
        save_folder += '_loss=' + str(args.loss_name)
    if args.noaug:
        save_folder += '_noaug'
    if args.raw_data:
        save_folder += '_rawdata'
    if args.label_corrupt_prob > 0:
        save_folder += '_randlabel=' + str(args.label_corrupt_prob)
    if args.ngpu > 1:
        save_folder += '_ngpu=' + str(args.ngpu)
    if args.idx:
        save_folder += '_idx=' + str(args.idx)

    return save_folder

if __name__ == '__main__':
    # Training options
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')
    parser.add_argument('--weight_decay', default=0, type=float)#0.0005
    parser.add_argument('--momentum', default=0, type=float)#0.9
    parser.add_argument('--epochs', default=5000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save', default='trained_nets',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=1, type=int, help='save every save_epochs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')

    # model parameters
    parser.add_argument('--model', '-m', default='resnet20')#vgg9
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # data parameters
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--noaug', default=False, action='store_true', help='no data augmentation')
    parser.add_argument('--label_corrupt_prob', type=float, default=0.0)
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')

    args = parser.parse_args()

    print('\nLearning Rate: %f' % args.lr)
    print('\nDecay Rate: %f' % args.lr_decay)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Current devices: ' + str(torch.cuda.current_device()))
        print('Device count: ' + str(torch.cuda.device_count()))

    # Set the seed for reproducing the results
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
        cudnn.benchmark = True

    lr = args.lr  # current learning rate
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    save_folder = name_save_folder(args)
    if not os.path.exists('trained_nets/' + save_folder):
        os.makedirs('trained_nets/' + save_folder)

    f = open('trained_nets/' + save_folder + '/log.out', 'a')

    trainloader, testloader, trainset = get_data_loaders(args)

    if args.label_corrupt_prob and not args.resume_model:
        torch.save(trainloader, 'trained_nets/' + save_folder + '/trainloader.dat')
        torch.save(testloader, 'trained_nets/' + save_folder + '/testloader.dat')

    # Model
    if args.resume_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model)
        net = model_loader.load(args.model)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        net = model_loader.load(args.model)
        print(net)
        init_params(net)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_opt:
        checkpoint_opt = torch.load(args.resume_opt)
        optimizer.load_state_dict(checkpoint_opt['optimizer'])

    # record the performance of initial model
    if not args.resume_model:
        train_loss, train_err = test(trainloader, net, criterion, use_cuda)
        test_loss, test_err = test(testloader, net, criterion, use_cuda)
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (0, train_loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        state = {
            'acc': 100 - test_err,
            'epoch': 0,
            'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict()
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'trained_nets/' + save_folder + '/model_0.t7')
        torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_0.t7')

    # training logs per iteration
    training_history = []
    testing_history = []
    weight_grad_history = []
    # noise logs less frequently
    noise_norm_history_TRAIN = []

    for epoch in range(start_epoch, args.epochs + 1):
        print(epoch)        

        # Save checkpoint.
        if epoch == 1 or epoch % args.save_epoch == 0 or epoch == 150:
            loss, train_err, hist_noise, layerWise_norms, sub_weights, sub_loss = train_save(trainloader, net, criterion, optimizer, use_cuda)
            test_loss, test_err, test_sub_loss = test_save(testloader, net, criterion, use_cuda)

            # save loss and weights in each tiny step in every epoch
            sio.savemat('trained_nets/' + save_folder + '/model_' + str(epoch) + '_sub_loss_w.mat',
                                mdict={'sub_weights': sub_weights,'sub_loss': sub_loss, 'test_sub_loss': test_sub_loss},
                                )            
        else:
            loss, train_err, hist_noise, layerWise_norms = train(trainloader, net, criterion, optimizer, use_cuda)
            test_loss, test_err = test(testloader, net, criterion, use_cuda)

        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        # validation acc
        acc = 100 - test_err

        # record training history (starts at initial point)
        training_history.append([loss, 100 - train_err])
        testing_history.append([test_loss, acc])
        weight_grad_history.append(layerWise_norms)            
        # grandient noise            
        noise_norm_history_TRAIN.append(hist_noise) 

        # save state for landscape on every epoch
        state = {
            'acc': acc,
            'epoch': epoch,
            'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict(),
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
        torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')

                   

        # if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
        #     lr *= args.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= args._lr_decay

    f.close()

    sio.savemat('trained_nets/' + save_folder + '/' + args.model + '_gradient_noise.mat',
                        mdict={'training_history': training_history,'testing_history': testing_history,'train_noise_norm': noise_norm_history_TRAIN,'weight_grad_history': weight_grad_history},
                        )

    #--------------------------------------------------------------------------
    # Load weights and save them in a mat file (temporal unit: epoch)
    #--------------------------------------------------------------------------
    all_weights = []
    for i in range(0,args.epochs+1,args.save_epoch):
        model_file = 'trained_nets/' + save_folder + '/' + 'model_' + str(i) + '.t7'
        net = ml.load('cifar10', args.model, model_file)
        w = net_plotter.get_weights(net) # initial parameters
        #s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
        #import pdb; pdb.set_trace()        
        for j in range(len(w)):
            w[j] = w[j].cpu().numpy()

        all_weights.append(w)

    sio.savemat('trained_nets/' + save_folder + '/' + args.model + 'all_weights.mat',
                            mdict={'weight': all_weights},
                            )
