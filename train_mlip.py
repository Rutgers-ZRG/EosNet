#!/usr/bin/env python3
"""Train EOSNet MLIP on extxyz data with energy/forces/stress."""

import argparse
import os
import sys
import time
import warnings
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list

from eosnet.mlip import EOSNetMLIP
from eosnet.fp import get_lfp_from_ase_neighbors


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train EOSNet MLIP on extxyz data')
    parser.add_argument('data_path', type=str,
                        help='Path to extxyz file or directory containing data.extxyz')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--energy-weight', type=float, default=1.0)
    parser.add_argument('--force-weight', type=float, default=10.0)
    parser.add_argument('--stress-weight', type=float, default=0.0,
                        help='Weight for stress loss (0 to disable)')
    parser.add_argument('--radial-cutoff', type=float, default=5.0)
    parser.add_argument('--gom-cutoff', type=float, default=6.0)
    parser.add_argument('--natx', type=int, default=64,
                        help='Max GOM neighbors per atom')
    parser.add_argument('--n-gom-features', type=int, default=32)
    parser.add_argument('--no-gom', action='store_true',
                        help='Disable GOM features (pure e3nn baseline)')
    parser.add_argument('--irreps-hidden', type=str,
                        default='32x0e+16x1o+8x2e')
    parser.add_argument('--max-ell', type=int, default=2)
    parser.add_argument('--n-conv', type=int, default=3)
    parser.add_argument('--num-radial-basis', type=int, default=16)
    parser.add_argument('--energy-hidden', type=int, default=128)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='.',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-cuda', action='store_true')
    return parser.parse_args()


def load_dataset(data_path):
    """Load extxyz dataset. Returns list of ASE Atoms with energy/forces."""
    if os.path.isdir(data_path):
        candidates = ['data.extxyz', 'train.extxyz', 'dataset.xyz']
        for c in candidates:
            p = os.path.join(data_path, c)
            if os.path.isfile(p):
                data_path = p
                break
        else:
            raise FileNotFoundError(
                f"No extxyz file found in {data_path}")

    print(f"Loading {data_path}...")
    frames = ase_read(data_path, index=':')
    print(f"Loaded {len(frames)} structures")

    # Validate: need energy and forces
    # ASE 3.27+ puts extxyz energy/forces into SinglePointCalculator
    valid = []
    for i, atoms in enumerate(frames):
        energy = (atoms.info.get('energy')
                  or atoms.info.get('REF_energy')
                  or atoms.info.get('total_energy'))
        forces = (atoms.arrays.get('forces')
                  or atoms.arrays.get('REF_forces'))
        stress = (atoms.info.get('stress')
                  or atoms.info.get('REF_stress'))

        # Fall back to SinglePointCalculator results (ASE 3.27+)
        if atoms.calc is not None:
            r = getattr(atoms.calc, 'results', {})
            if energy is None:
                energy = r.get('energy')
            if forces is None:
                forces = r.get('forces')
            if stress is None:
                stress = r.get('stress')

        if energy is not None and forces is not None:
            atoms.info['_energy'] = float(energy)
            atoms.arrays['_forces'] = np.array(forces, dtype=np.float64)
            if stress is not None:
                atoms.info['_stress'] = np.array(stress, dtype=np.float64)
            valid.append(atoms)
    print(f"{len(valid)} structures with energy+forces")
    return valid


def build_graph(atoms, cutoff, gom_cutoff=6.0, natx=64, use_gom=True):
    """Build edge graph from ASE Atoms, with precomputed GOM fingerprints.

    GOM eigenvalues are computed once here and cached — NOT recomputed
    every forward pass.
    """
    i_idx, j_idx, D_vec, S_vec = neighbor_list('ijDS', atoms, cutoff)

    nat = len(atoms)
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32)
    edge_index = torch.stack([
        torch.tensor(i_idx, dtype=torch.long),
        torch.tensor(j_idx, dtype=torch.long)
    ])
    # Store shift vectors so edge_vec can be recomputed from positions
    edge_shifts = torch.tensor(S_vec, dtype=torch.float32)  # (E, 3) lattice shift indices

    energy = torch.tensor(atoms.info['_energy'], dtype=torch.float32)
    forces = torch.tensor(atoms.arrays['_forces'], dtype=torch.float32)

    stress = None
    if '_stress' in atoms.info:
        stress = torch.tensor(atoms.info['_stress'], dtype=torch.float32)

    # Precompute GOM fingerprints (cached, not recomputed per epoch)
    gom_fp = None
    if use_gom:
        gom_i, gom_j, gom_D = neighbor_list('ijD', atoms, gom_cutoff)
        gom_fp = get_lfp_from_ase_neighbors(
            atoms.get_positions(), atoms.numbers,
            gom_i, gom_j, gom_D,
            cutoff=gom_cutoff, natx=natx,
            device='cpu', dtype=torch.float64
        ).float()  # (nat, natx)

    return {
        'atomic_numbers': atomic_numbers,
        'positions': positions,
        'cell': cell,
        'edge_index': edge_index,
        'edge_shifts': edge_shifts,
        'energy': energy,
        'forces': forces,
        'stress': stress,
        'nat': nat,
        'gom_fp': gom_fp,
    }


def train_step(model, batch, optimizer, energy_w, force_w, stress_w, device):
    """Train on a single structure (no batching across structures)."""
    model.train()
    optimizer.zero_grad()

    positions = batch['positions'].to(device).requires_grad_(True)
    cell = batch['cell'].to(device)
    atomic_numbers = batch['atomic_numbers'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_shifts = batch['edge_shifts'].to(device)
    gom_fp = batch['gom_fp'].to(device) if batch['gom_fp'] is not None else None

    # Recompute edge_vec from positions so autograd can compute forces
    edge_src, edge_dst = edge_index
    edge_vec = positions[edge_dst] - positions[edge_src] + edge_shifts @ cell

    target_energy = batch['energy'].to(device)
    target_forces = batch['forces'].to(device)
    nat = batch['nat']

    compute_stress = stress_w > 0 and batch['stress'] is not None

    output = model(
        atomic_numbers, positions, cell,
        edge_index, edge_vec,
        compute_forces=True,
        compute_stress=compute_stress,
        gom_fp=gom_fp,
    )

    # Energy loss (per atom)
    energy_loss = ((output['energy'] / nat - target_energy / nat) ** 2)

    # Force loss (per component)
    force_loss = ((output['forces'] - target_forces) ** 2).mean()

    loss = energy_w * energy_loss + force_w * force_loss

    # Stress loss
    if compute_stress and 'stress' in output:
        target_stress = batch['stress'].to(device)
        stress_loss = ((output['stress'] - target_stress) ** 2).mean()
        loss = loss + stress_w * stress_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    return {
        'loss': loss.item(),
        'energy_loss': energy_loss.item(),
        'force_loss': force_loss.item(),
        'energy_mae': abs(output['energy'].item() / nat
                          - target_energy.item() / nat),
        'force_mae': (output['forces'] - target_forces).abs().mean().item(),
    }


def validate(model, val_data, device):
    """Validate on a dataset."""
    model.eval()
    energy_maes = []
    force_maes = []

    for batch in val_data:
        positions = batch['positions'].to(device).requires_grad_(True)
        cell = batch['cell'].to(device)
        atomic_numbers = batch['atomic_numbers'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_shifts = batch['edge_shifts'].to(device)
        gom_fp = batch['gom_fp'].to(device) if batch['gom_fp'] is not None else None
        nat = batch['nat']

        # Recompute edge_vec from positions for autograd
        edge_src, edge_dst = edge_index
        edge_vec = positions[edge_dst] - positions[edge_src] + edge_shifts @ cell

        with torch.enable_grad():
            output = model(
                atomic_numbers, positions, cell,
                edge_index, edge_vec,
                compute_forces=True,
                compute_stress=False,
                gom_fp=gom_fp,
            )

        target_energy = batch['energy'].to(device)
        target_forces = batch['forces'].to(device)

        energy_maes.append(
            abs(output['energy'].item() / nat - target_energy.item() / nat))
        force_maes.append(
            (output['forces'] - target_forces).abs().mean().item())

    return {
        'energy_mae': np.mean(energy_maes),
        'force_mae': np.mean(force_maes),
    }


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load data
    frames = load_dataset(args.data_path)

    # Shuffle and split
    indices = np.random.permutation(len(frames))
    n_train = int(args.train_ratio * len(frames))
    n_val = int(args.val_ratio * len(frames))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, "
          f"{len(test_idx)} test")

    # Pre-build graphs
    print("Building graphs...")
    use_gom = not args.no_gom
    def _build(i):
        return build_graph(frames[i], args.radial_cutoff,
                           gom_cutoff=args.gom_cutoff, natx=args.natx,
                           use_gom=use_gom)
    train_data = [_build(i) for i in train_idx]
    val_data = [_build(i) for i in val_idx]
    test_data = [_build(i) for i in test_idx]

    # Build model on CPU first (e3nn Gate float64 issue)
    prev_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else None
    model = EOSNetMLIP(
        irreps_hidden=args.irreps_hidden,
        max_ell=args.max_ell,
        n_conv=args.n_conv,
        num_radial_basis=args.num_radial_basis,
        radial_cutoff=args.radial_cutoff,
        n_gom_features=args.n_gom_features,
        gom_cutoff=args.gom_cutoff,
        natx=args.natx,
        energy_hidden=args.energy_hidden,
        use_gom=not args.no_gom,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"GOM features: {'enabled' if not args.no_gom else 'disabled'}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=args.patience // 3,
                                   min_lr=1e-6)

    # Resume
    start_epoch = 0
    best_val_mae = float('inf')
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_val_mae = ckpt.get('best_val_mae', float('inf'))

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Shuffle training data
        perm = np.random.permutation(len(train_data))

        epoch_losses = []
        epoch_e_mae = []
        epoch_f_mae = []

        for i, idx in enumerate(perm):
            batch = train_data[idx]
            metrics = train_step(
                model, batch, optimizer,
                args.energy_weight, args.force_weight, args.stress_weight,
                device
            )
            epoch_losses.append(metrics['loss'])
            epoch_e_mae.append(metrics['energy_mae'])
            epoch_f_mae.append(metrics['force_mae'])

        # Validation
        val_metrics = validate(model, val_data, device)
        scheduler.step(val_metrics['force_mae'])

        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % args.print_freq == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"loss {np.mean(epoch_losses):.4f} | "
                  f"E_MAE {np.mean(epoch_e_mae)*1000:.1f} meV | "
                  f"F_MAE {np.mean(epoch_f_mae)*1000:.1f} meV/Å | "
                  f"val_E {val_metrics['energy_mae']*1000:.1f} meV | "
                  f"val_F {val_metrics['force_mae']*1000:.1f} meV/Å | "
                  f"lr {lr:.1e} | {dt:.1f}s")

        # Save best
        is_best = val_metrics['force_mae'] < best_val_mae
        if is_best:
            best_val_mae = val_metrics['force_mae']
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'mlip_best.pth'))
        else:
            no_improve += 1

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_mae': best_val_mae,
            'args': vars(args),
        }, os.path.join(args.save_dir, 'mlip_checkpoint.pth'))

        # Early stopping
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Test
    print("\n--- Test Set Evaluation ---")
    best_ckpt = torch.load(os.path.join(args.save_dir, 'mlip_best.pth'),
                            map_location=device)
    model.load_state_dict(best_ckpt['model'])
    test_metrics = validate(model, test_data, device)
    print(f"Test E_MAE: {test_metrics['energy_mae']*1000:.1f} meV/atom")
    print(f"Test F_MAE: {test_metrics['force_mae']*1000:.1f} meV/Å")

    # Save test results
    results = {
        'test_energy_mae_meV': test_metrics['energy_mae'] * 1000,
        'test_force_mae_meV_A': test_metrics['force_mae'] * 1000,
        'best_epoch': best_ckpt['epoch'],
        'n_train': len(train_data),
        'n_val': len(val_data),
        'n_test': len(test_data),
        'args': vars(args),
    }
    with open(os.path.join(args.save_dir, 'mlip_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.save_dir}/mlip_results.json")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
