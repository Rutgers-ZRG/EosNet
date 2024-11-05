#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import sys
import csv

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fpcnn.data import StructData
from fpcnn.data import collate_pool
from fpcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('structpath', help='path to the directory of structure files.')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--save_to_disk', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='Save data to disk (default: False)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--disable-mps', action='store_true',
                    help='Disable MPS')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test', action='store_true', default=False,
                    help='output predicted targets to CSV file')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()
args.mps = not args.disable_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built()

if model_args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, model_args, best_mae_error

    # Automatically generate id_prop.csv if it doesn't exist
    id_prop_path = os.path.join(args.structpath, 'id_prop.csv')
    if not os.path.exists(id_prop_path):
        print(f"Generating id_prop.csv in {args.structpath}")
        struct_ids = []
        for filename in os.listdir(args.structpath):
            if filename.endswith('.vasp'):
                struct_ids.append(filename.replace('.vasp', ''))
        if not struct_ids:
            raise ValueError(f"No .vasp files found in {args.structpath}")
            
        # Generate dummy targets
        if model_args.task == 'classification':
            targets = np.random.randint(0, 2, size=len(struct_ids))
        else:
            targets = np.random.rand(len(struct_ids))
        
        # Write to id_prop.csv
        with open(id_prop_path, 'w') as f:
            for sid, target in zip(struct_ids, targets):
                f.write(f'{sid},{target}\n')

    # Load id_prop.csv without shuffling
    with open(id_prop_path) as f:
        reader = csv.reader(f, delimiter=',')
        id_prop_data = [row for row in reader]
    
    # Create a simple class to mimic IdTargetData behavior without shuffling
    class SimpleIdTargetData:
        def __init__(self, data):
            self.id_prop_data = data
        def __len__(self):
            return len(self.id_prop_data)
        def __getitem__(self, idx):
            struct_id, target = self.id_prop_data[idx]
            return struct_id, float(target)
    
    # Create id_target_data without shuffling
    id_target_data = SimpleIdTargetData(id_prop_data)

    # Clean up any existing cached data
    processed_file = os.path.join(args.structpath, 'processed_data.npz')
    if os.path.exists(processed_file):
        print(f"Removing existing cached file: {processed_file}")
        os.remove(processed_file)

    # load data
    dataset = StructData(id_prop_data=id_target_data,
                         root_dir=args.structpath,
                         max_num_nbr=model_args.max_num_nbr,
                         radius=model_args.radius,
                         dmin=model_args.dmin,
                         step=model_args.step,
                         var=model_args.var,
                         nx=model_args.nx,
                         lmax=model_args.lmax,
                         save_to_disk=args.save_to_disk)

    def ordered_collate(data_list):
        _, _, struct_ids = zip(*data_list)
        batch_data = collate_pool(data_list)
        
        return batch_data[0], batch_data[1], list(struct_ids)

    test_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             collate_fn=ordered_collate,
                             pin_memory=args.cuda)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task ==
                                'classification' else False)
    
    # Initialize normalizer
    if model_args.task == 'classification':
        normalizer = Normalizer(0.0)
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        normalizer = Normalizer(0.0)

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

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    validate(test_loader, model, normalizer, test=args.test)


def validate(val_loader, model, normalizer, test=False):
    # Always initialize these variables
    test_preds = []
    test_struct_ids = []
    
    # switch to evaluate mode
    model.eval()

    for _, (input, _, batch_struct_ids) in enumerate(val_loader):
        with torch.no_grad():
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

        # compute output
        output = model(*input_var)
        
        # Store predictions
        if model_args.task == 'regression':
            pred = normalizer.denorm(output.data.cpu())
            test_preds += pred.view(-1).tolist()
        else:
            pred = torch.exp(output.data.cpu())
            test_preds += pred[:, 1].tolist()
        test_struct_ids += batch_struct_ids

    # Output predictions
    if test:
        # Save to CSV
        print("\nSaving predictions to target_predicted.csv")
        with open('target_predicted.csv', 'w') as f:
            for struct_id, pred in zip(test_struct_ids, test_preds):
                f.write(f'{struct_id},{pred}\n')
    else:
        # Print to screen
        print("\nPredictions:")
        for struct_id, pred in zip(test_struct_ids, test_preds):
            if model_args.task == 'classification':
                print(f"{struct_id},{pred:.16f}")
            else:
                print(f"{struct_id},{pred:.16f}")


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
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='weighted')
        try: # Handle "Only one class present in y_true" Error MSG
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        except ValueError:
            auc_score = 0.0
        accuracy = metrics.accuracy_score(target_label, pred_label)
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()