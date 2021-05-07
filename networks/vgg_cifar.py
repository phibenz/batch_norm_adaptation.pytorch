'''VGG11/13/16/19 in Pytorch.'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


cfg = {
    'VGG8' : [64, 'M', 128, 'M', 256, 'M', 512, 'M', 'L', 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'L', 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'L', 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 'L', 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 'L', 'M'],
}


pretrained_settings = {
    "cifar10": {
        'vgg19_bn': 'https://www.dropbox.com/s/a178my3jz8mm1tz/vgg19_bn_best-0672e2002d8598ae0c00046dc5e6b83ec497b11624e11d9579c7cd7068e4f1b3.pth?dl=1',
        'num_classes': 10
    },
    "cifar100": {
        'vgg19_bn': 'https://www.dropbox.com/s/5qddrhgo8nkyno9/vgg19_bn_best-e7ae5ba1ea968bb0a1736eee7287f61ef3cae8c688e9699d7805c8d129886474.pth?dl=1',
        'num_classes': 100
    }
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, in_channels=3, normalization="bn"):
        super(VGG, self).__init__()
        
        self.normalization = normalization
        self.in_channels = in_channels
        
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.zero_()
                
        
    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)
        pre_out = out.view(out.size(0), -1)
        final = self.classifier(pre_out)
        return final

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'L':
                if self.normalization == "in":
                    layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
                               nn.InstanceNorm2d(512, affine=True)]
                elif self.normalization == "bn":
                    layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
                               nn.BatchNorm2d(512)]
                elif self.normalization == "ln":
                    layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
                               nn.GroupNorm(num_groups=1, num_channels=512)]
                elif self.normalization == "gn":
                    layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
                               nn.GroupNorm(num_groups=32, num_channels=512)]
                elif self.normalization == None:
                    layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)]
                else:
                    raise ValueError
            else:
                if self.normalization == "in":
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.InstanceNorm2d(x, affine=True),
                               nn.ReLU(inplace=True)]
                elif self.normalization == "bn":
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                elif self.normalization == "ln":
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.GroupNorm(num_groups=1, num_channels=x),
                               nn.ReLU(inplace=True)]
                elif self.normalization == "gn":
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.GroupNorm(num_groups=32, num_channels=x),
                               nn.ReLU(inplace=True)]
                elif self.normalization == None:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                else:
                    raise ValueError
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)] # Move into forward pass
        return nn.Sequential(*layers)

def vgg16(pretrained=None, **kwargs):
    """VGG 19-layer model
    """
    if pretrained is None:
        model = VGG('VGG16', normalization=None, **kwargs)
    else:
        model = VGG('VGG16', normalization=None, num_classes=pretrained_settings[pretrained]['num_classes'])
        # model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['vgg19']))
        model = torch.nn.DataParallel(model, device_ids=list(range(1)))
        model.load_state_dict(torch.load(pretrained_settings[pretrained]['vgg16'])['state_dict'])
    return model

def vgg16_bn(pretrained=None, **kwargs):
    """VGG 19-layer model with batch normalization"""
    if pretrained is None:
        model = VGG('VGG16', normalization="bn", **kwargs)
    else:
        model = VGG('VGG16', normalization="bn", num_classes=pretrained_settings[pretrained]['num_classes'])
        # model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['vgg19_bn']))
        model = torch.nn.DataParallel(model, device_ids=list(range(1)))
        model.load_state_dict(torch.load(pretrained_settings[pretrained]['vgg16_bn'])['state_dict'])
    return model

def vgg19(pretrained=None, **kwargs):
    """VGG 19-layer model
    """
    if pretrained is None:
        model = VGG('VGG19', normalization=None, **kwargs)
    else:
        model = VGG('VGG19', normalization=None, num_classes=pretrained_settings[pretrained]['num_classes'])
        # model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['vgg19']))
        model.load_state_dict(torch.load(pretrained_settings[pretrained]['vgg19']))
    return model

def vgg19_bn(pretrained=None, **kwargs):
    """VGG 19-layer model with batch normalization"""
    if pretrained is None:
        model = VGG('VGG19', normalization="bn", **kwargs)
    else:
        model = VGG('VGG19', normalization="bn", num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['vgg19_bn']))
    return model
