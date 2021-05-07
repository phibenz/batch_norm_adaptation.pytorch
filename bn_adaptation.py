import os, sys, csv, random
import pickle
import torch
import argparse

from utils.utils import get_model_path, get_logger
from utils.data import get_data_specs, get_data, get_transforms
from utils.network import get_network, set_parameter_requires_grad, set_bn
from utils.training import validate


def get_statistics(net):
    mean_paras = []
    var_paras = []
    for layer in net.modules():
       if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
           mean_paras.append(layer.running_mean.clone())
           var_paras.append(layer.running_var.clone())
    return mean_paras, var_paras


def parse_arguments():
    parser = argparse.ArgumentParser(description='Adapting BN')
    # Evaluation dataset
    parser.add_argument('--evaluation_dataset', required=True, help='Dataset(s) to evaluate on')
    parser.add_argument('--severity', type=int, default=3, help='Severity of the evaluation dataset, if applicable (default: 3)')
    parser.add_argument('--adapt_bn', type=eval, default="True", choices=[True, False], help='Run additonal evaluation fo evaluating the BN (default: True)')
    parser.add_argument('--adaptation_dataset', required=True, help='Dataset to perform the batch norm adaptation on')
    parser.add_argument('--adaptation_batch_size', type=int, default=32, help='Batch size for the adaptation (default: 32)')
    parser.add_argument('--postfix', default='default', help='Postfix to append to the result filename (default: default)')
    # Net & Data
    parser.add_argument('--dataset', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'], help='Used dataset to train the initial model (default: imagenet)')
    parser.add_argument('--arch', default='vgg16', help='Used model architecture (default: vgg16)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoin to resume from (default: None)')
    parser.add_argument('--seed', type=int, default=333, help='Seed used in the generation process (default: 333)')
    parser.add_argument('--evaluate_before_adaptation', action='store_true', help='Evaluate before the adaptation process (default: False)')
    # Optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=6, help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = torch.cuda.is_available()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    return args


def main():
    args = parse_arguments()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # Get the model path 
    args.model_path = get_model_path(dataset_name=args.dataset, network_arch=args.arch)

    # Get result path
    args.result_path = os.path.join(args.model_path, args.adaptation_dataset + "_" + str(args.severity))
    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path)
    results_file = os.path.join(args.model_path, 'results' + '_{}'.format(args.postfix) + '.csv')

    args.stats_folder = os.path.join(args.model_path, 'statistics')
    if not os.path.isdir(args.stats_folder):
        os.makedirs(args.stats_folder)

    logger = get_logger(args.result_path)
    logger.info('#### Evaluation ####')

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info('{} : {}'.format(key, value))

    
    ###### Getting the adaptation dataset ######
    num_classes, (mean, std), img_size, num_channels = get_data_specs(dataset=args.adaptation_dataset)

    # Overwriting mean, std for special cases 
    if args.arch in ['resnet20_cifar100', 'resnet56_cifar100']:
        mean = [0.5071, 0.4865, 0.4409]  # https://github.com/chenyaofo/CIFAR-pretrained-models/
        std = [0.2009, 0.1984, 0.2023] # https://github.com/chenyaofo/CIFAR-pretrained-models/
    
    train_transform, test_transform = get_transforms(dataset=args.adaptation_dataset,
                                                    augmentation=True)

    logger.info('Getting adaptation dataset: {}'.format(args.adaptation_dataset))
    _, data_adapt = get_data(args.adaptation_dataset,
                                train_transform=train_transform, 
                                test_transform=test_transform,
                                severity=args.severity)

    if args.adaptation_dataset.startswith('cifar10c_') or args.adaptation_dataset.startswith('cifar100c_'):
        start_idx = 10000*(args.severity-1)
        print('Selecting data subset severity {} idxs: {} to {}'.format(args.severity, start_idx, start_idx+10000))
        data_adapt.data = data_adapt.data[start_idx:start_idx+10000]
        data_adapt.target = data_adapt.target[start_idx:start_idx+10000]

    data_adapt_loader = torch.utils.data.DataLoader(data_adapt,
                                                    batch_size=args.adaptation_batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ############## Network ##################
    net = get_network(args.arch, img_size, num_classes, num_channels)
 
    # net = torch.nn.DataParallel(net, device_ids=list(range(1)))
    net.eval()

    if args.resume != None:
        assert os.path.isfile(args.resume)
        network_data = torch.load(args.resume)
        net.load_state_dict(network_data['state_dict'])
        logger.info('Model restored from : {}'.format(args.resume))

    # Setting the momentum term in BN to None
    net.apply(set_bn)

    # Statistics dictionary to store
    statistics = {}
    statistics['mean'] = {}
    statistics['var'] = {}

    mean_paras_before, var_paras_before = get_statistics(net)
    statistics['mean']['before'] = mean_paras_before
    statistics['var']['before'] = var_paras_before

    if args.use_cuda:
        net.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    ######## Evaluation dataset ############ 
    csvfile = open(results_file, 'a', newline='')
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    if args.evaluate_before_adaptation:
        train_transform, test_transform = get_transforms(dataset=args.evaluation_dataset,
                                                        augmentation=True)

        _, data_test = get_data(args.evaluation_dataset,
                                train_transform=train_transform, 
                                test_transform=test_transform,
                                severity=args.severity)


        if args.evaluation_dataset.startswith('cifar10c_') or args.evaluation_dataset.startswith('cifar100c_'):
            start_idx = 10000*(args.severity-1)
            print('Selecting data subset severity {} idxs: {} to {}'.format(args.severity, start_idx, start_idx+10000))
            data_test.data = data_test.data[start_idx:start_idx+10000]
            data_test.target = data_test.target[start_idx:start_idx+10000]

        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True)

        logger.info('Evaluation before adaptation on: {}'.format(args.evaluation_dataset))
        top1, top5, loss = validate(data_test_loader, net, criterion, use_cuda=args.use_cuda)
        logger.info("Eval:\tTop1: {}\tTop5: {}\tLoss: {}".format(top1, top5, loss))
        csvwriter.writerow([args.evaluation_dataset, args.severity, False, top1, top5, loss])

    if args.adapt_bn:
        logger.info('Adapting BN statistics...')
        # Fixing all network parameters
        set_parameter_requires_grad(net, requires_grad=False)
        net.train()

        data_adapt_iter = iter(data_adapt_loader)
        with torch.no_grad():
            # Adapt only to one batch
            input, target = next(data_adapt_iter)
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            output = net(input)
        
        net.eval()

        # Storing adapted statistics 
        mean_paras, var_paras = get_statistics(net)
        statistics['mean']['after'] = mean_paras
        statistics['var']['after'] = var_paras
        
        # Saving the statistics
        stats_filename = args.adaptation_dataset + "_" + str(args.severity) + "_" + args.postfix + ".pkl"
        stats_filepath = os.path.join(args.stats_folder, stats_filename)
        logger.info("Storing statistics to :{}".format(stats_filepath))
        with open(stats_filepath, "wb") as stats_file:
            pickle.dump(statistics, stats_file)

        train_transform, test_transform = get_transforms(dataset=args.evaluation_dataset,
                                                    augmentation=True)

        _, data_test = get_data(args.evaluation_dataset,
                                train_transform=train_transform, 
                                test_transform=test_transform,
                                severity=args.severity)

        if args.evaluation_dataset.startswith('cifar10c_') or args.evaluation_dataset.startswith('cifar100c_'):
            start_idx = 10000*(args.severity-1)
            print('Selecting data subset severity {} idxs: {} to {}'.format(args.severity, start_idx, start_idx+10000))
            data_test.data = data_test.data[start_idx:start_idx+10000]
            data_test.target = data_test.target[start_idx:start_idx+10000]

        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True)

        logger.info('Evaluation after adaptation on: {}'.format(args.evaluation_dataset))
        top1_adapt_bn, top5_adapt_bn, loss_adapt_bn = validate(data_test_loader, net, criterion, use_cuda=args.use_cuda)
        logger.info("Adapt BN:\tTop1: {}\tTop5: {}\tLoss: {}".format(top1_adapt_bn, top5_adapt_bn, loss_adapt_bn))
        csvwriter.writerow([args.evaluation_dataset, args.severity, True, top1_adapt_bn, top5_adapt_bn, loss_adapt_bn])

    csvfile.close()

if __name__ == '__main__':
    main()