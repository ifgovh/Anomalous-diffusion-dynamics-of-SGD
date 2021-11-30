import os
import torch, torchvision
import train_DNN_code.models.vgg as vgg
import train_DNN_code.models.resnet as resnet
import train_DNN_code.models.densenet as densenet
import train_DNN_code.models.Alexnet as Alexnet
import train_DNN_code.models.vit as ViT

# map between model name and function
models = {
    'vgg9'                  : vgg.VGG9,
    'densenet121'           : densenet.DenseNet121,
    'resnet18'              : resnet.ResNet18,
    'resnet18_noshort'      : resnet.ResNet18_noshort,
    'resnet34'              : resnet.ResNet34,
    'resnet34_noshort'      : resnet.ResNet34_noshort,
    'resnet50'              : resnet.ResNet50,
    'resnet50_noshort'      : resnet.ResNet50_noshort,
    'resnet101'             : resnet.ResNet101,
    'resnet101_noshort'     : resnet.ResNet101_noshort,
    'resnet152'             : resnet.ResNet152,
    'resnet152_noshort'     : resnet.ResNet152_noshort,
    'resnet14'              : resnet.ResNet14,
    'resnet14_noshort'      : resnet.ResNet14_noshort,
    'resnet20'              : resnet.ResNet20,
    'resnet20_noshort'      : resnet.ResNet20_noshort,
    'resnet32_noshort'      : resnet.ResNet32_noshort,
    'resnet44_noshort'      : resnet.ResNet44_noshort,
    'resnet50_16_noshort'   : resnet.ResNet50_16_noshort,
    'resnet56'              : resnet.ResNet56,
    'resnet56_noshort'      : resnet.ResNet56_noshort,
    'resnet110'             : resnet.ResNet110,
    'resnet110_noshort'     : resnet.ResNet110_noshort,
    'wrn56_2'               : resnet.WRN56_2,
    'wrn56_2_noshort'       : resnet.WRN56_2_noshort,
    'wrn56_4'               : resnet.WRN56_4,
    'wrn56_4_noshort'       : resnet.WRN56_4_noshort,
    'wrn56_8'               : resnet.WRN56_8,
    'wrn56_8_noshort'       : resnet.WRN56_8_noshort,
    'wrn110_2_noshort'      : resnet.WRN110_2_noshort,
    'wrn110_4_noshort'      : resnet.WRN110_4_noshort,
    'alex'                  : Alexnet.alexnet,
    'fc3'                   : Alexnet.fc3,
    'fc20'                  : Alexnet.fc20,
    'fc56'                  : Alexnet.fc56,
    'fc110'                 : Alexnet.fc110,
    'simplenet'             : Alexnet.simplenet,
    'wrn14_2'               : resnet.WRN14_2,
    'wrn14_2_noshort'       : resnet.WRN14_2_noshort,
    'wrn14_4'               : resnet.WRN14_4,
    'wrn14_4_noshort'       : resnet.WRN14_4_noshort,
    'wrn14_8'               : resnet.WRN14_8,
    'wrn14_8_noshort'       : resnet.WRN14_8_noshort,
    'resnet_mnist'        : resnet.ResNet_mnist,
    'resnet14_nobatchnorm'  : resnet.ResNet14_nobatchnorm,
    'resnet14_noshort_nobatchnorm':resnet.ResNet14_noshort_nobatchnorm,
    'vit'                   : vit.vit_cifar10,
    
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
