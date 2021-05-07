import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def compute_mce(corruption_accs, corruptions, alexnet_err):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  avg = []
  for i in range(len(corruptions)):
    avg.append(np.mean(corruption_accs[corruptions[i]])*100.)
    avg_err = 1 - np.mean(corruption_accs[corruptions[i]])
    ce = 100 * avg_err / alexnet_err[i]
    mce += ce / 15
  return np.mean(avg), mce


CIFAR_CORRUPTIONS = [
    'gaussian_noise', 'shot', 'impulse', 'defocus', 
    'glass', 'motion','zoom', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic', 'pixelate',
    'jpeg'
#     'speckle', 'saturate', 'spatter','gaussian_blur'
]

CORRUPTIONS_IMAGENET = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

ALEXNET_ERR_CIFAR10 = [0.2584, 0.25102, 0.30084, 0.30662, 
                       0.30328, 0.37378, 0.34576, 0.30212, 0.32398, 0.37776, 
                       0.26932, 0.51958, 0.3035, 0.25248, 
                       0.24516]

ALEXNET_ERR_CIFAR100 = [0.61448, 0.6007, 0.66594, 0.61368, 
                        0.61268, 0.65894, 0.63438, 0.66192, 0.68484, 0.68874, 
                        0.62692, 0.75924, 0.6183, 0.57462, 
                        0.58216]

ALEXNET_ERR_IMAGENET = [0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
                      0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
                      0.606500]


## CIFAR 10 
# VGG19BN
result_path = './results/cifar10_resnet20_cifar10/results_default.csv'
## CIFAR100
# result_path = './results/cifar100_resnet20_cifar100/results_default.csv'
## ImageNet
# result_path = './results/imagenet_resnet18/results_default.csv'

df = pd.read_csv(result_path, names=['dataset', 'severity', 'adapted', 'top1', 'top5', 'error'])

corruption_accs_before = {}
corruption_accs_after = {}
for c in CIFAR_CORRUPTIONS:
# for c in CORRUPTIONS_IMAGENET:
    c_name = 'cifar10c_' + c
    # c_name = 'cifar100c_' + c
    # c_name = 'imagenetc_' + c
    print(c_name)
    accs_before = (df.loc[(df['dataset'] == c_name) & (df['adapted'] == False) ]['top1']/100.).tolist()
    accs_after = (df.loc[(df['dataset'] == c_name) & (df['adapted'] == True) ]['top1']/100.).tolist()
    print(np.mean(accs_before))
    print(np.mean(accs_after))
    assert len(accs_before)==5, print(len(accs_before), c_name)
    assert len(accs_after)==5, print(len(accs_after), c_name)
    corruption_accs_before[c] = accs_before
    corruption_accs_after[c] = accs_after

avg_before, mce_before = compute_mce(corruption_accs_before, CIFAR_CORRUPTIONS, ALEXNET_ERR_CIFAR10)
avg_after, mce_after = compute_mce(corruption_accs_after, CIFAR_CORRUPTIONS, ALEXNET_ERR_CIFAR10)
# avg_before, mce_before = compute_mce(corruption_accs_before, CIFAR_CORRUPTIONS, ALEXNET_ERR_CIFAR100)
# avg_after, mce_after = compute_mce(corruption_accs_after, CIFAR_CORRUPTIONS, ALEXNET_ERR_CIFAR100)
# avg_before, mce_before = compute_mce(corruption_accs_before, CORRUPTIONS_IMAGENET, ALEXNET_ERR_IMAGENET)
# avg_after, mce_after = compute_mce(corruption_accs_after, CORRUPTIONS_IMAGENET, ALEXNET_ERR_IMAGENET)

print()
print("Average before:", avg_before)
print("Average after:", avg_after)
print("Difference:", avg_after-avg_before)
print()
print("MCE before:", mce_before)
print("MCE after:", mce_after)
print("Differenece:", mce_before - mce_after)