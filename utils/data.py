import os
import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfunc

from config.config import DATA_PATH, IMAGENET_PATH


class DatasetFromTorchTensor(Dataset):
    def __init__(self, data, target, transform=None):
        # Data type handling must be done beforehand. It is too difficult at this point.
        self.data = data
        self.target = target
        if len(self.target.shape)==1:
            self.target = target.long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = tfunc.to_pil_image(x)
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def get_data_specs(dataset):
    mean = [0., 0., 0.]
    std = [1., 1., 1.]
    if dataset.startswith('cifar100'):
        num_classes = 100
        img_size = 32
        num_channels = 3
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset.startswith('cifar10'):
        num_classes = 10
        img_size = 32
        num_channels = 3
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset.startswith('imagenet'):
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_size = 224
        num_channels = 3
    else:
        raise ValueError()

    return num_classes, (mean, std), img_size, num_channels


def get_transforms(dataset, augmentation=True):
    _, (mean, std), img_size, _ = get_data_specs(dataset)
    if dataset.startswith('cifar10'):
        if augmentation:
            train_transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(img_size, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                     ])
        else:
            train_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
    
    elif dataset == 'imagenet':
        if augmentation:
            train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        else:
            train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        
        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    
    elif dataset.startswith('imagenetc_'):
        if augmentation:    
            train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        else:
            train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    else:
      raise ValueError
    
    return train_transform, test_transform
    

def get_data(dataset, train_transform, test_transform, severity=1):
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)

    elif dataset == 'cifar100':
        train_data = dset.CIFAR100(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(DATA_PATH, train=False, transform=test_transform, download=True)
    
    elif dataset == 'imagenet':
        # Data loading code
        traindir = os.path.join(IMAGENET_PATH, 'train')
        valdir = os.path.join(IMAGENET_PATH, 'val')
        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=valdir, transform=test_transform)
    
    elif dataset.startswith('imagenetc_'):
        if dataset == 'imagenetc_brightness':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'brightness', str(severity))
        elif dataset == 'imagenetc_contrast':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'contrast', str(severity))
        elif dataset == 'imagenetc_defocus_blur':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'defocus_blur', str(severity))
        elif dataset == 'imagenetc_elastic_transform':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'elastic_transform', str(severity))
        elif dataset == 'imagenetc_fog':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'fog', str(severity))
        elif dataset == 'imagenetc_frost':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'frost', str(severity))
        elif dataset == 'imagenetc_gaussian_blur':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'gaussian_blur', str(severity))
        elif dataset == 'imagenetc_gaussian_noise':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'gaussian_noise', str(severity))
        elif dataset == 'imagenetc_glass_blur':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'glass_blur', str(severity))
        elif dataset == 'imagenetc_impulse_noise':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'impulse_noise', str(severity))
        elif dataset == 'imagenetc_jpeg_compression':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'jpeg_compression', str(severity))
        elif dataset == 'imagenetc_motion_blur':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'motion_blur', str(severity))
        elif dataset == 'imagenetc_pixelate':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'pixelate', str(severity))
        elif dataset == 'imagenetc_saturate':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'saturate', str(severity))
        elif dataset == 'imagenetc_shot_noise':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'shot_noise', str(severity))
        elif dataset == 'imagenetc_snow':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'snow', str(severity))
        elif dataset == 'imagenetc_spatter':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'spatter', str(severity))
        elif dataset == 'imagenetc_speckle_noise':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'speckle_noise', str(severity))
        elif dataset == 'imagenetc_zoom_blur':
            valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'zoom_blur', str(severity))
        train_data = dset.ImageFolder(root=valdir, transform=train_transform)
        test_data = dset.ImageFolder(root=valdir, transform=test_transform)

    elif dataset.startswith('cifar10c_'):    # Blur
        if dataset == 'cifar10c_gaussian_noise':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'gaussian_noise.npy')
        elif dataset == 'cifar10c_speckle':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'speckle_noise.npy')
        elif dataset == 'cifar10c_shot':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'shot_noise.npy')
        elif dataset == 'cifar10c_impulse':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'impulse_noise.npy')
        elif dataset == 'cifar10c_contrast':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'contrast.npy')
        elif dataset == 'cifar10c_elastic':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'elastic_transform.npy')
        elif dataset == 'cifar10c_pixelate':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'pixelate.npy')
        elif dataset == 'cifar10c_jpeg':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'jpeg_compression.npy')
        elif dataset == 'cifar10c_saturate':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'saturate.npy')
        elif dataset == 'cifar10c_snow':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'snow.npy')
        elif dataset == 'cifar10c_fog':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'fog.npy')
        elif dataset == 'cifar10c_brightness':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'brightness.npy')
        elif dataset == 'cifar10c_defocus':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'defocus_blur.npy')
        elif dataset == 'cifar10c_frost':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'frost.npy')
        elif dataset == 'cifar10c_spatter':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'spatter.npy')
        elif dataset == 'cifar10c_glass':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'glass_blur.npy')
        elif dataset == 'cifar10c_motion':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'motion_blur.npy')
        elif dataset == 'cifar10c_zoom':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'zoom_blur.npy')
        elif dataset == 'cifar10c_gaussian_blur':
            data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', 'gaussian_blur.npy')
        else: 
            raise ValueError

        train_imgs = torch.tensor(np.transpose(np.load(data_path), (0,3,1,2)))
        train_labels = torch.tensor(np.load(os.path.join(DATA_PATH, 'CIFAR-10-C', 'labels.npy')))
        train_data = DatasetFromTorchTensor(train_imgs, train_labels, transform=train_transform)
        test_data = DatasetFromTorchTensor(train_imgs, train_labels, transform=test_transform)

    elif dataset.startswith('cifar100c_'):
        if dataset == 'cifar100c_gaussian_noise':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'gaussian_noise.npy')
        elif dataset == 'cifar100c_speckle':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'speckle_noise.npy')
        elif dataset == 'cifar100c_shot':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'shot_noise.npy')
        elif dataset == 'cifar100c_impulse':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'impulse_noise.npy')
        elif dataset == 'cifar100c_contrast':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'contrast.npy')
        elif dataset == 'cifar100c_elastic':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'elastic_transform.npy')
        elif dataset == 'cifar100c_pixelate':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'pixelate.npy')
        elif dataset == 'cifar100c_jpeg':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'jpeg_compression.npy')
        elif dataset == 'cifar100c_saturate':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'saturate.npy')
        elif dataset == 'cifar100c_snow':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'snow.npy')
        elif dataset == 'cifar100c_fog':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'fog.npy')
        elif dataset == 'cifar100c_brightness':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'brightness.npy')
        elif dataset == 'cifar100c_defocus':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'defocus_blur.npy')
        elif dataset == 'cifar100c_frost':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'frost.npy')
        elif dataset == 'cifar100c_spatter':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'spatter.npy')
        elif dataset == 'cifar100c_glass':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'glass_blur.npy')
        elif dataset == 'cifar100c_motion':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'motion_blur.npy')
        elif dataset == 'cifar100c_zoom':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'zoom_blur.npy')
        elif dataset == 'cifar100c_gaussian_blur':
            data_path = os.path.join(DATA_PATH, 'CIFAR-100-C', 'gaussian_blur.npy')
        else: 
            raise ValueError

        train_imgs = torch.tensor(np.transpose(np.load(data_path), (0,3,1,2)))
        train_labels = torch.tensor(np.load(os.path.join(DATA_PATH, 'CIFAR-100-C', 'labels.npy')))
        train_data = DatasetFromTorchTensor(train_imgs, train_labels, transform=train_transform)
        test_data = DatasetFromTorchTensor(train_imgs, train_labels, transform=test_transform)

    else:
        raise ValueError
    return train_data, test_data
