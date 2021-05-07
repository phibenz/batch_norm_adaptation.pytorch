from __future__ import division

import os, shutil
import numpy as np
import torch
import torchvision.models as models

from networks.resnet_cifar import cifar_resnet20, cifar_resnet56
from networks.resnet_cifar_madry import ResNet18 as ResNet18_Cifar
from networks.resnet_cifar_madry import ResNet50 as ResNet50_Cifar
from networks.vgg_cifar import vgg16, vgg16_bn, vgg19, vgg19_bn
from networks.wrn_cifar import wrn_28_10_drop
from networks.densenet_cifar import densenet_bc_l190_k40
from networks.resnext_cifar import resnext_16x64d, resnext_8x64d
from networks.resnet_imagenet import wide_resnet101_2, resnext101_32x8d

def get_network(model_arch, input_size, num_classes, num_channels, pretrained=True):
    
    #### CIFAR-10 models ####
    if model_arch == "alexnet_cifar10":
        if pretrained:
            net = alexnet(pretrained='cifar10')
        else:
            net = alexnet()            
    elif model_arch == "resnet20_cifar10":
        if pretrained:
            net = cifar_resnet20(pretrained='cifar10')
        else:
            net = cifar_resnet20()
    elif model_arch == "resnet56_cifar10":
        if pretrained:
            net = cifar_resnet56(pretrained='cifar10')
        else:
            net = cifar_resnet56()
    elif model_arch == "vgg19_bn_cifar10":
        if pretrained:
            net = vgg19_bn(pretrained='cifar10')
        else:
            net = vgg19_bn()
    elif model_arch == 'wrn_28_10_drop_cifar10':
        if pretrained:
            net = wrn_28_10_drop(pretrained='cifar10')
        else:
            net = wrn_28_10_drop()
    elif model_arch == 'densenet_bc_l190_k40_cifar10':
        net = densenet_bc_l190_k40(pretrained='cifar10')
    elif model_arch == 'resnext_8x64d_cifar10':
        net = resnext_8x64d(pretrained='cifar10')
    elif model_arch == 'resnet18_cifar10':
        net = ResNet18_Cifar(pretrained='cifar10')
    elif model_arch == 'resnet50_cifar10':
        net = ResNet50_Cifar(pretrained='cifar10')

    #### CIFAR-100 models ####
    elif model_arch == "resnet20_cifar100":
        net = cifar_resnet20(pretrained='cifar100')
    elif model_arch == "resnet56_cifar100":
        net = cifar_resnet56(pretrained='cifar100')
    elif model_arch == "vgg19_bn_cifar100":
        net = vgg19_bn(pretrained='cifar100')
    elif model_arch == 'wrn_28_10_drop_cifar100':
        net = wrn_28_10_drop(pretrained='cifar100')
    elif model_arch == 'densenet_bc_l190_k40_cifar100':
        net = densenet_bc_l190_k40(pretrained='cifar100')
    elif model_arch == 'resnext_16x64d_cifar100':
        net = resnext_16x64d(pretrained='cifar100')
    elif model_arch == 'resnext_8x64d_cifar100':
        net = resnext_8x64d(pretrained='cifar100')

    #### ImageNet models ####
    elif model_arch == "resnet18":
        net = models.resnet18(pretrained=True)
    elif model_arch == "resnet50":
        net = models.resnet50(pretrained=True)
    elif model_arch == "vgg19_bn":
        net = models.vgg19_bn(pretrained=True)
    elif model_arch == "densenet161":
        net = models.densenet161(pretrained=True)
    elif model_arch == "resnext101_32x8d":
        net = resnext101_32x8d(pretrained=True)
    elif model_arch == "wide_resnet101_2":
        net = wide_resnet101_2(pretrained=True)
    else:
        raise ValueError("Network {} not supported".format(model_arch))
    return net

def set_bn(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.momentum = None
        m.num_batches_tracked = torch.tensor(0)

def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==True, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def get_num_non_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==False, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])
