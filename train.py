#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import sys
import time
import warnings
from random import seed as rnd_seed
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, SequentialLR, LinearLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.multiprocessing import set_sharing_strategy, get_context

from fpcnn.data import IdTargetData, StructData
from fpcnn.data import collate_pool, get_train_val_test_loader
from fpcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='EOSNet: Embedded Overlap Structures for Graph Neural Networks')
# Explicitly add arguments for data options
parser.add_argument('root_dir', metavar='ROOT_DIR', type=str,
                    help='Path to the root directory')
parser.add_argument('--max_num_nbr', default=12, type=int,
                    help='Maximum number of neighbors (default: 12)')
parser.add_argument('--radius', default=8.0, type=float,
                    help='The cutoff radius for searching neighbors (default: 8.0)')
parser.add_argument('--dmin', default=0.5, type=float,
                    help='Minimum distance of GDF (default: 0.5)')
parser.add_argument('--step', default=0.1, type=float,
                    help='Step size of GDF (default: 0.1)')
parser.add_argument('--var', default=1.0, type=float,
                    help='Variance of GDF (default: 1.0)')
parser.add_argument('--nx', default=256, type=int,
                    help='Maximum number of neighbors to construct '
                    'the Gaussian overlap matrix for atomic Fingerprint (default: 256)')
parser.add_argument('--lmax', default=0, type=int,
                    help='Integer to control whether using s orbitals only '
                    'or both s and p orbitals for calculating the Guassian '
                    'overlap matrix. 0 for s orbitals only, other integers '
                    'will indicate that using both s and p orbitals. (default: 0)')
parser.add_argument('--random_seed', default=42, type=int,
                    help='Random seed (default: 42)')
parser.add_argument('--save_to_disk', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='Save data to disk (default: False)')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression',
                    help='complete a regression or '
                    'classification task (default: regression)')
parser.add_argument('--update-bond', action='store_true',
                    help='Enable bond feature updates in the model')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--disable-mps', action='store_true',
                    help='Disable Apple Metal Acceleration')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of epochs for warm-up scheduler')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--drop-last', action='store_true',
                    help='Drop the last incomplete batch if the dataset '
                    'size is not divisible by the batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N',
                    help='milestones for scheduler (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--fine-tune', default='', type=str, metavar='PATH',
                    help='path to pre-trained model for fine-tuning (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default 0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default 1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()
args.mps = not args.disable_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built()

if args.cuda:
    device = torch.device("cuda")
    torch.set_default_device(device)
elif args.mps:
    device = torch.device("mps")
    torch.set_default_device(device)
else:
    device = torch.device("cpu")
    torch.set_default_device(device)

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error, class_weights
    # Load IdTargetData
    id_target_dataset = IdTargetData(root_dir=args.root_dir,
                                     random_seed=args.random_seed)
    
    # Get train/val/test splits using IdTargetData
    class_weights, id_train_loader, id_val_loader, id_test_loader = get_train_val_test_loader(
        dataset=id_target_dataset,
        classification=True if args.task == 'classification' else False,
        collate_fn=collate_pool,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        drop_last=args.drop_last,
        multiprocessing_context=get_context('spawn') if args.workers > 0 else None,
        return_test=True)

    # If save_to_disk is True, pre-process and save all data
    if args.save_to_disk:
        struct_dataset = StructData(
            id_prop_data=id_target_dataset.id_prop_data,
            root_dir=args.root_dir,
            max_num_nbr=args.max_num_nbr,
            radius=args.radius,
            dmin=args.dmin,
            step=args.step,
            var=args.var,
            nx=args.nx,
            lmax=args.lmax,
            batch_size=args.batch_size,
            drop_last=args.drop_last,
            save_to_disk=True
        )
        # Clear the memory after saving
        struct_dataset.clear_cache()
        struct_dataset = None

    # Create a temporary StructData instance to get feature dimensions
    temp_struct_dataset = StructData(
        id_prop_data=[id_target_dataset.id_prop_data[0]],
        root_dir=args.root_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step,
        var=args.var,
        nx=args.nx,
        lmax=args.lmax,
        batch_size=1,
        drop_last=False,
        save_to_disk=False
    )
    
    # Get feature dimensions from the first structure
    structures, _, _ = temp_struct_dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    
    # Clear temporary dataset
    temp_struct_dataset = None

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(0.0)
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(id_target_dataset) < 1000:
            warnings.warn('Dataset has less than 1000 data points. '
                          'Lower accuracy is expected. ')
            sample_indices = range(len(id_target_dataset))
        else:
            sample_size = 1000 + int(0.1 * (len(id_target_dataset) - 1000))
            sample_indices = sample(range(len(id_target_dataset)), sample_size)
        
        # Use id_target_dataset directly
        sample_data_list = [id_target_dataset[i] for i in sample_indices]
        sample_target = torch.tensor([target for _, target in sample_data_list], dtype=torch.float)
        normalizer = Normalizer(sample_target)

    # build model
    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        h_fea_len=args.h_fea_len,
        n_conv=args.n_conv,
        n_h=args.n_h,
        classification=True if args.task == 'classification' else False
    )

    if args.cuda:
        device = torch.device("cuda")
        model.to(device)
        normalizer.to(device)
    elif args.mps:
        device = torch.device("mps")
        model.to(device)
        normalizer.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)
        normalizer.to(device)

    # define loss func and optimizer
    if args.task == 'classification':
        loss_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.NLLLoss(weight=loss_weights, reduction='mean')
    else:
        criterion = nn.MSELoss(reduction='mean')
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally load pre-trained model for fine-tuning
    if args.fine_tune:
        if os.path.isfile(args.fine_tune):
            print("=> loading pre-trained model for fine-tuning '{}'".format(args.fine_tune))
            checkpoint = torch.load(args.fine_tune, map_location=device)
            
            pretrained_args = checkpoint.get('args', {})
            pretrained_task = pretrained_args.get('task', 'unknown')
            if pretrained_task != args.task:
                print("ERROR: Task mismatch! Pre-trained model task: '{}', current task: '{}'".format(
                    pretrained_task, args.task))
                print("Fine-tuning requires the same task type. Terminating.")
                sys.exit(1)
            
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pre-trained model '{}' (task: {})".format(args.fine_tune, pretrained_task))
            print("=> Initializing new normalizer for fine-tuning dataset")
        else:
            print("=> no pre-trained model found at '{}'".format(args.fine_tune))
            sys.exit(1)
    
    # optionally resume from a checkpoint
    elif args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    # main_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    main_scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    # main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=0.01*args.lr)
    # main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, threshold=0.01, threshold_mode='abs')
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(id_train_loader, model, criterion, optimizer, epoch, normalizer)
        
        # evaluate on validation set
        val_loss, mae_error = validate(id_val_loader, model, criterion, normalizer, test=False)
        
        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()
        # scheduler.step(val_loss)

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best, fine_tune=bool(args.fine_tune))

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    if args.fine_tune:
        best_model_file = 'model_ft_best.pth.tar'
    else:
        best_model_file = 'model_best.pth.tar'
    best_checkpoint = torch.load(best_model_file)
    model.load_state_dict(best_checkpoint['state_dict'])
    
    # Save training set predictions
    print('---------Saving Training & Testing Set Results---------------')
    if args.fine_tune:
        train_results_file = 'ft_train_results.csv'
        test_results_file = 'ft_test_results.csv'
    else:
        train_results_file = 'train_results.csv'
        test_results_file = 'test_results.csv'
    
    if args.save_to_disk:
        validate(id_train_loader, model, criterion, normalizer, test=True, filename=train_results_file)
    else:
        test_size = len(id_test_loader.dataset)
        subset_indices = torch.randperm(len(id_train_loader.dataset))[:test_size]
        subset_train_dataset = torch.utils.data.Subset(id_train_loader.dataset, subset_indices)
        subset_train_loader = torch.utils.data.DataLoader(
            subset_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=args.workers > 0,
            collate_fn=collate_pool,
            pin_memory=args.cuda
        )
        
        # Evaluate on subset of training data
        validate(subset_train_loader, model, criterion, normalizer, test=True, filename=train_results_file)

    # Save test set predictions
    validate(id_test_loader, model, criterion, normalizer, test=True, filename=test_results_file)


def train(id_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # Create StructData instance with training data
    train_data = [(sid, target) for sid, target in id_loader.dataset.id_prop_data]
    struct_dataset = StructData(
        id_prop_data=train_data,
        root_dir=args.root_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step,
        var=args.var,
        nx=args.nx,
        lmax=args.lmax,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        save_to_disk=False
    )

    # switch to train mode
    model.train()

    end = time.time()

    for i, (targets, struct_ids) in enumerate(id_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Get batch data
        batch_data = []
        for sid, target in zip(struct_ids, targets):
            idx = next(i for i, (s, _) in enumerate(struct_dataset.id_prop_data) if s == sid)
            batch_data.append(struct_dataset[idx])
        
        # Pass complete tuples to collate_pool
        input, _, _ = collate_pool(batch_data)
        
        if args.cuda:
            input_var = (Variable(input[0].to("cuda", non_blocking=True)),
                        Variable(input[1].to("cuda", non_blocking=True)),
                        input[2].to("cuda", non_blocking=True),
                        [crys_idx.to("cuda", non_blocking=True) for crys_idx in input[3]])
        elif args.mps:
            input_var = (Variable(input[0].to("mps", non_blocking=False)),
                         Variable(input[1].to("mps", non_blocking=False)),
                         input[2].to("mps", non_blocking=False),
                         [crys_idx.to("mps", non_blocking=False) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])

        # Convert targets to tensor
        target = torch.tensor([float(t) for t in targets], dtype=torch.float)
        target = target.view(-1, 1)
        target = target.to(device)

        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.to("cuda", non_blocking=True))
        elif args.mps:
            target_var = Variable(target_normed.to("mps", non_blocking=False))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data, target)
            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(id_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(id_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )
    
    # Clean up
    struct_dataset.clear_cache()
    struct_dataset = None

def validate(id_loader, model, criterion, normalizer, test=False, filename=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_struct_ids = []

    # Create StructData instance with validation/test data
    val_data = [(sid, target) for sid, target in id_loader.dataset.id_prop_data]
    struct_dataset = StructData(
        id_prop_data=val_data,
        root_dir=args.root_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step,
        var=args.var,
        nx=args.nx,
        lmax=args.lmax,
        batch_size=args.batch_size,
        drop_last=args.drop_last,
        save_to_disk=False
    )

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (targets, struct_ids) in enumerate(id_loader):
        # Get batch data
        batch_data = []
        for sid, target in zip(struct_ids, targets):
            idx = next(i for i, (s, _) in enumerate(struct_dataset.id_prop_data) if s == sid)
            batch_data.append(struct_dataset[idx])
        
        # Pass complete tuples to collate_pool
        input, _, _ = collate_pool(batch_data)
        
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].to("cuda", non_blocking=True)),
                             Variable(input[1].to("cuda", non_blocking=True)),
                             input[2].to("cuda", non_blocking=True),
                             [crys_idx.to("cuda", non_blocking=True) for crys_idx in input[3]])
                target = target.to("cuda", non_blocking=True)
        elif args.mps:
            with torch.no_grad():
                input_var = (Variable(input[0].to("mps", non_blocking=False)),
                             Variable(input[1].to("mps", non_blocking=False)),
                             input[2].to("mps", non_blocking=False),
                             [crys_idx.to("mps", non_blocking=False) for crys_idx in input[3]])
                target = target.to("mps", non_blocking=False)
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

        # Convert targets to tensor
        target = torch.tensor([float(t) for t in targets], dtype=torch.float)
        target = target.view(-1, 1)
        target = target.to(device)

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.to("cuda", non_blocking=True))
        elif args.mps:
            with torch.no_grad():
                target_var = Variable(target_normed.to("mps", non_blocking=False))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data), target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data)
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_struct_ids += struct_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_struct_ids += struct_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(id_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(id_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))
    
    # Clean up
    struct_dataset.clear_cache()
    struct_dataset = None

    if test and filename:
        star_label = '**'
        import csv
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            for struct_id, target, pred in zip(test_struct_ids, test_targets,
                                               test_preds):
                writer.writerow((struct_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return losses.avg, mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return losses.avg, auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        if isinstance(tensor, torch.Tensor):
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
        else:
            self.mean = tensor
            self.std = 1.0

    def norm(self, tensor):
        if isinstance(self.mean, torch.Tensor):
            return (tensor - self.mean) / self.std
        return tensor

    def denorm(self, normed_tensor):
        if isinstance(self.mean, torch.Tensor):
            return normed_tensor * self.std + self.mean
        return normed_tensor

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
    
    def to(self, device):
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        return self

def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        class_weights_dict = {class_idx: weight for class_idx,
                              weight in zip(np.unique(target_label), class_weights)}
        sample_weight = [class_weights_dict[class_idx] for class_idx in target_label]
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='weighted',
            sample_weight=sample_weight, zero_division=np.nan)
        try: # Handle "Only one class present in y_true" Error MSG
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1],
                                              average='weighted', sample_weight=sample_weight)
        except ValueError:
            auc_score = 0.0
        accuracy = metrics.accuracy_score(target_label, pred_label,
                                          sample_weight=sample_weight)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fine_tune=False, filename='checkpoint.pth.tar'):
    """Save model checkpoint with different filenames for fine-tuning mode. """
    if fine_tune:
        checkpoint_file = 'checkpoint_ft.pth.tar'
        best_model_file = 'model_ft_best.pth.tar'
    else:
        checkpoint_file = filename
        best_model_file = 'model_best.pth.tar'
    
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    os.system('ulimit -n 4096')
    # Only set file_system sharing if having memory/stability issues
    # set_sharing_strategy('file_system')
    warnings.filterwarnings("ignore", category=UserWarning, message=".*epoch parameter in `scheduler.step\(\)`.*")
    
    # For future reproducibility
    seed = 42       # Your favorite seed, if all fail, try 3047 (https://arxiv.org/pdf/2109.08203)
    np.random.seed(seed)
    rnd_seed(seed)
    torch.manual_seed(seed)
    
    main()