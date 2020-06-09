import os
import train_DNN_code.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = train_DNN_code.load(model_name, model_file, data_parallel)
    return net
