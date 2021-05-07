# Revisiting Batch Normalization for Improving Corruption Robustness
This is the repository accompanying our WACV 2021 paper [Revisiting Batch Normalization for Improving Corruption Robustness](https://arxiv.org/pdf/2010.03630.pdf).

## Config
Edit the paths in `./config/config.py` according to your environment.

## Datasets
Download the corruption datasets from [here](https://github.com/hendrycks/robustness) and extract them into your `DATA_PATH` set in the `./config/config.py`.
The [ImageNet](http://www.image-net.org/) dataset should be preprocessed, such that the validation images are located in labeled subfolders as for the training set. You can have a look at this [bash-script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) if you did not process your data already. Set the paths in your `config.py`.
```
IMAGENET_PATH = "/path/to/Data/ImageNet"
```
For CIFAR10 and CIFAR100 the `torchvision` datasets were used. 

## Models
Checkpoints for the models should be automatically downloaded when used.
For ImageNet, the `torchvision` models were used. The checkpoints for CIFAR10 and CIFAR100 kudos go to the awesome repositories of [bearpaw](https://github.com/bearpaw/pytorch-classification) and [chenyaofo](https://github.com/chenyaofo/pytorch-cifar-models).

## Run
Run `bash ./run_<dataset>.sh` to run the respective adaptation code. The bash script should be easy to adapt to perform different experiments. The results of the three scripts are included in the results folder. 

## Evaluation
The accuracy and mCE can be evaluated with `python3 eval.py`. You can adapt the paths inside the file to evaluate your results. 

## Docker 
We used the pytorch docker container `pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel` with Pytorch 1.0.1 for our experiments.
```
docker pull pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel
```

## Concurrent Work
Also check out the concurrent work by Schneider et al. [Improving robustness against common corruptions by covariate shift adaptation](https://proceedings.neurips.cc/paper/2020/file/85690f81aadc1749175c187784afc9ee-Paper.pdf). 

## Citation
```
@inproceedings{benz2021revisiting,
  title={Revisiting batch normalization for improving corruption robustness},
  author={Benz, Philipp and Zhang, Chaoning and Karjauv, Adil and Kweon, In So},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={494--503},
  year={2021}
}
```