# -*- coding: UTF-8 -*-

'''
Train the model
Ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
from mobileNetV3 import MobileNetV3
import argparse
import copy
from math import cos, pi
import pickle
import os.path as osp
import warnings
from collections import OrderedDict
from functools import partial
from pprint import pformat

from statistics import *
from EMA import EMA
from metrics import evaluate_classification
from LabelSmoothing import LabelSmoothingLoss
# from DataLoader import dataloaders
from ResultWriter import ResultWriter
from CosineLR import *
from Mixup import mixup_data, mixup_criterion

def load_checkpoint(fpath):
    r"""Loads checkpoint. Imported from openvinotoolkit/deep-object-reid.


    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers):
    if discarded_layers:
        print(
            '** The following layers are discarded '
            'due to unmatched keys or layer size: {}'.
            format(pformat(discarded_layers))
        )
    if unmatched_layers:
        print(
            '** The following layers were not loaded from checkpoint: {}'.
            format(pformat(unmatched_layers))
        )

def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def load_pretrained_weights(model, file_path='', pretrained_dict=None, extra_prefix=''):
    r"""Loads pretrianed weights to model. Imported from openvinotoolkit/deep-object-reid.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        file_path (str): path to pretrained weights.
    """
    def _remove_prefix(key, prefix):
        prefix = prefix + '.'
        if key.startswith(prefix):
            key = key[len(prefix):]
        return key

    if file_path:
        check_isfile(file_path)
    checkpoint = (load_checkpoint(file_path)
                       if not pretrained_dict
                       else pretrained_dict)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        k = extra_prefix + _remove_prefix(k, 'module')
        # print(model_dict[k].size(), v.size())
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    message = file_path if file_path else "pretrained dict"
    unmatched_layers = sorted(set(model_dict.keys()) - set(new_state_dict))
    if len(matched_layers) == 0:
        print(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually'.format(message)
        )
        _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

        raise RuntimeError(f'The pretrained weights {message} cannot be loaded')
    print(
        'Successfully loaded pretrained weights from "{}"'.
        format(message)
    )
    _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

def train(args, model, dataloader, loader_len, criterion, optimizer, scheduler, use_gpu, epoch, ema=None, save_file_name='train.csv'):
    '''
    train the model
    '''
    # save result every epoch
    resultWriter = ResultWriter(args.save_path, save_file_name)
    if epoch == 0:
        resultWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5', 'lr'])

    # use gpu or not
    device = torch.device('cuda' if use_gpu else 'cpu')

    # statistical information
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # update lr here if using stepLR
    if args.lr_decay == 'step':
        scheduler.step(epoch)

    # Set model to training mode
    model.train()

    end = time.time()

    # Iterate over data
    for i, (inputs, labels) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        if args.mixup:
            # using mixup
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            acc1_a, acc5_a = accuracy(outputs, labels_a, topk=(1, 5))
            acc1_b, acc5_b = accuracy(outputs, labels_b, topk=(1, 5))
            # measure accuracy and record loss
            acc1 = lam * acc1_a + (1 - lam) * acc1_b
            acc5 = lam * acc5_a + (1 - lam) * acc5_b
        else:
            # normal forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        # zero the parameter gradients
        optimizer.zero_grad()

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # backward + optimize
        loss.backward()
        if args.lr_decay == 'cos':
            # update lr here if using cosine lr decay
            scheduler.step(epoch * loader_len + i)
        elif args.lr_decay == 'sgdr':
            # update lr here if using sgdr
            scheduler.step(epoch + i / loader_len)
        optimizer.step()
        if args.ema_decay > 0:
            # EMA update after training(every iteration)
            ema.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # write training result to file
    resultWriter.write_csv([epoch, losses.avg, top1.avg.item(), top5.avg.item(), scheduler.optimizer.param_groups[0]['lr']])
    # there is a bug in get_lr() if using pytorch 1.1.0, see https://github.com/pytorch/pytorch/issues/22107
    # so here we don't use get_lr()
    # print('lr:%.6f' % scheduler.get_lr()[0])
    print('lr:%.6f' % scheduler.optimizer.param_groups[0]['lr'])
    print('Train ***    Loss:{losses.avg:.2e}    Acc@1:{top1.avg:.2f}    Acc@5:{top5.avg:.2f}'.format(losses=losses, top1=top1, top5=top5))

    if epoch % args.save_epoch_freq == 0 and epoch != 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))

def validate(args, model, dataloader, loader_len, criterion, use_gpu, epoch, ema=None, save_file_name='val.csv'):
    '''
    validate the model
    '''

    # save result every epoch
    resultWriter = ResultWriter(args.save_path, save_file_name)
    if epoch == 0:
        resultWriter.create_csv(['epoch', 'mAP', 'top-1', 'top-5'])

    if args.ema_decay > 0:
        # apply EMA at validation stage
        ema.apply_shadow()
    # Set model to evaluate mode
    model.eval()
    top_k, m_ap = evaluate_classification(dataloader, model, use_gpu, topk=(1,5))
    top1,top5 = top_k

    if args.ema_decay > 0:
        # restore the origin parameters after val
        ema.restore()
    # write val result to file
    resultWriter.write_csv([epoch, m_ap, top1, top5])

    print(' Val  ***    mAP:{mAP:.2f}    Acc@1:{top1:.2f}    Acc@5:{top5:.2f}'.format(mAP=m_ap*100, top1=top1*100, top5=top5*100))

    if epoch % args.save_epoch_freq == 0 and epoch != 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))

    top1_acc = top1
    mAP_acc = m_ap

    return top1_acc, mAP_acc

def train_model(args, model, dataloader, loaders_len, criterion, optimizer, scheduler, use_gpu):
    '''
    train the model
    '''
    since = time.time()

    ema = None
    # exponential moving average
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)
        ema.register()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    correspond_top5 = 0.0

    for epoch in range(args.start_epoch, args.num_epochs):

        epoch_time = time.time()
        train(args, model, dataloader['train'], loaders_len['train'], criterion, optimizer, scheduler, use_gpu, epoch, ema)
        top1_acc, mAP_acc = validate(args, model, dataloader['val'], loaders_len['val'], criterion, use_gpu, epoch, ema)
        epoch_time = time.time() - epoch_time
        print('Time of epoch-[{:d}/{:d}] : {:.0f}h {:.0f}m {:.0f}s\n'.format(epoch, args.num_epochs, epoch_time // 3600, (epoch_time % 3600) // 60, epoch_time % 60))

        # deep copy the model if it has higher top-1 accuracy
        if top1_acc > best_acc:
            best_acc = top1_acc
            correspond_mAP = mAP_acc
            if args.ema_decay > 0:
                ema.apply_shadow()
            best_model_wts = copy.deepcopy(model.state_dict())
            if args.ema_decay > 0:
                ema.restore()

    print(os.path.split(args.save_path)[-1])
    print('Best val top-1 Accuracy: {:4f}'.format(best_acc))
    print('Corresponding mAP : {:4f}'.format(correspond_mAP))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save best model weights
    if args.save:
        torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model_wts-' + '{:.2f}'.format(best_acc) + '.pth'))
    return model

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='PyTorch implementation of MobileNetV3')
    # Root catalog of images
    parser.add_argument('--data-dir', type=str, default='/mnt/pai_share/datasets/classification/SVHN')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--resolution', type=int, default=240)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    #parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--print-freq', type=int, default=1000)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='./output')
    parser.add_argument('-save', default=False, action='store_true', help='save model or not')
    parser.add_argument('--resume', type=str, default='pretrained/best_model_wts-67.52.pth', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='The decay of exponential moving average ')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='The dataset to be trained')
    parser.add_argument('-dali', default=False, action='store_true', help='Using DALI or not')
    parser.add_argument('--mode', type=str, default='large', help='large or small MobileNetV3')
    # parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--width-multiplier', type=float, default=1.0, help='width multiplier')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--lr-decay', type=str, default='step', help='learning rate decay method, step, cos or sgdr')
    parser.add_argument('--step-size', type=int, default=3, help='step size in stepLR()')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma in stepLR()')
    parser.add_argument('--lr-min', type=float, default=0, help='minium lr using in CosineWarmupLR')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='warmup epochs using in CosineWarmupLR')
    parser.add_argument('--T-0', type=int, default=10, help='T_0 in CosineAnnealingWarmRestarts')
    parser.add_argument('--T-mult', type=int, default=2, help='T_mult in CosineAnnealingWarmRestarts')
    parser.add_argument('--decay-rate', type=float, default=1, help='decay rate in CosineAnnealingWarmRestarts')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--bn-momentum', type=float, default=0.1, help='momentum in BatchNorm2d')
    parser.add_argument('-use-seed', default=False, action='store_true', help='using fixed random seed or not')
    parser.add_argument('--seed', type=int, default=5, help='random seed')
    parser.add_argument('-deterministic', default=False, action='store_true', help='torch.backends.cudnn.deterministic')
    parser.add_argument('-nbd', default=False, action='store_true', help='no bias decay')
    parser.add_argument('-zero-gamma', default=False, action='store_true', help='zero gamma in BatchNorm2d when init')
    parser.add_argument('-mixup', default=False, action='store_true', help='mixup or not')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='alpha used in mixup')
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()

    args.lr_decay = args.lr_decay.lower()
    args.dataset = args.dataset.lower()
    args.optimizer = args.optimizer.lower()

    # folder to save what we need in this type: MobileNetV3-mode-dataset-width_multiplier-dropout-lr-batch_size-ema_decay-label_smoothing
    folder_name = ['MobileNetV3', args.mode, args.dataset]
    if args.lr_decay == 'step':
        folder_name.append(args.lr_decay+str(args.step_size)+'&'+str(args.gamma))
    elif args.lr_decay == 'cos':
        folder_name.append(args.lr_decay+str(args.warmup_epochs) + '&' + str(args.lr_min))
    elif args.lr_decay == 'sgdr':
        folder_name.append(args.lr_decay+str(args.T_0)+'&'+str(args.T_mult)+'&'+str(args.warmup_epochs)+'&'+str(args.decay_rate))
    folder_name = '-'.join(folder_name)
    args.save_path = os.path.join(args.save_path, folder_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # set random seed
    if args.use_seed:
        print('Using fixed random seed')
        torch.manual_seed(args.seed)
    else:
        print('do not use fixed random seed')
    if use_gpu:
        if args.use_seed:
            torch.cuda.manual_seed(args.seed)
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        print('torch.backends.cudnn.deterministic:' + str(args.deterministic))

    # different input size and number of classes for different datasets
    assert args.dataset in {'imagenet', 'tinyimagenet',
                                'cifar100', 'cifar10', 'svhn',
                                'custom_folder', 'custom_txt'}
    if args.dataset == 'imagenet':
        input_size = 224
        num_class = 1000
    elif args.dataset == 'tinyimagenet':
        input_size = 56
        num_class = 200
    if args.dataset == 'cifar100':
        input_size = 32
        num_class = 100
    elif args.dataset == 'cifar10' or args.dataset == 'svhn':
        input_size = 32
        num_class = 10
    if args.dataset == 'custom_folder':
        input_size = args.resolution
        num_class = args.num_classes
    elif args.dataset == 'custom_txt':
        input_size = args.resolution
        num_class = args.num_classes

    # read data
    # dataloaders = dataloaders(args)
    if args.dali and (args.dataset == 'tinyimagenet' or args.dataset == 'imagenet'):
        if args.dataset == 'imagenet':
            from DALIDataLoader import get_dali_imageNet_train_loader, get_dali_imageNet_val_loader
            train_loader, train_loader_len = get_dali_imageNet_train_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            val_loader, val_loader_len = get_dali_imageNet_val_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            dataloaders = {'train' : train_loader, 'val' : val_loader}
            loaders_len = {'train': train_loader_len, 'val' : val_loader_len}
        elif args.dataset == 'tinyimagenet':
            from DALIDataLoader import get_dali_tinyImageNet_train_loader, get_dali_tinyImageNet_val_loader
            train_loader, train_loader_len = get_dali_tinyImageNet_train_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            val_loader, val_loader_len = get_dali_tinyImageNet_val_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            dataloaders = {'train' : train_loader, 'val' : val_loader}
            loaders_len = {'train': train_loader_len, 'val' : val_loader_len}
    else:
        from DataLoader import dataloaders
        loaders = dataloaders(args)
        train_loader = loaders['train']
        train_loader_len = len(train_loader)
        val_loader = loaders['val']
        val_loader_len = len(val_loader)
        dataloaders = {'train' : train_loader, 'val' : val_loader}
        loaders_len = {'train': train_loader_len, 'val' : val_loader_len}

    # get model
    model = MobileNetV3(mode=args.mode, classes_num=num_class, input_size=input_size,
                    width_multiplier=args.width_multiplier, dropout=args.dropout,
                    BN_momentum=args.bn_momentum, zero_gamma=args.zero_gamma)

    if args.resume:
        if os.path.isfile(args.resume):
            load_pretrained_weights(model, args.resume)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        if torch.cuda.device_count() > 1:
            print("cuda devices: ", torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
        model.to(torch.device('cuda'))
    else:
        model.to(torch.device('cpu'))

    if args.label_smoothing > 0:
        # using Label Smoothing
        criterion = LabelSmoothingLoss(num_class, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        if args.nbd:
            from NoBiasDecay import noBiasDecay
            optimizer_ft = optim.SGD(
                # no bias decay
                noBiasDecay(model, args.lr, args.weight_decay),
                momentum=0.9)
        else:
            optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer_ft = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_decay == 'step':
        # Decay LR by a factor of 0.99 every 3 epoch
        lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_decay == 'cos':
        lr_scheduler = CosineWarmupLR(optimizer=optimizer_ft, epochs=args.num_epochs, iter_in_one_epoch=loaders_len['train'], lr_min=args.lr_min, warmup_epochs=args.warmup_epochs)
    elif args.lr_decay == 'sgdr':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer_ft, T_0=args.T_0, T_mult=args.T_mult, warmup_epochs=args.warmup_epochs, decay_rate=args.decay_rate)

    if args.eval:
        cmc, m_ap = evaluate_classification(dataloaders['val'], model, use_gpu)
        print(f'evaluated model: {cmc}, mAP: {m_ap}')
        exit()
    model = train_model(args=args,
                        model=model,
                        dataloader=dataloaders,
                        loaders_len=loaders_len,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=lr_scheduler,
                        use_gpu=use_gpu)
