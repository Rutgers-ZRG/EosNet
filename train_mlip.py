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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--energy-weight', type=float, default=1.0)
    parser.add_argument('--force-weight', type=float, default=10.0)
    parser.add_argument('--stress-weight', type=float, default=0.0,
                        help='Weight for stress loss (0 to disable)')
    parser.add_argument('--radial-cutoff', type=float, default=5.0)
    parser.add_argument('--gom-cutoff', type=float, default=6.0)
    parser.add_argument('--natx', type=int, default=32,
                        help='Max GOM neighbors per atom')
    parser.add_argument('--n-gom-features', type=int, default=32)
    parser.add_argument('--no-gom', action='store_true',
                        help='Disable GOM features (pure e3nn baseline)')
    parser.add_argument('--orbital', type=str, default='s',
                        choices=['s', 'sp'],
                        help='GOM orbital type: s (default) or sp (s+p)')
    parser.add_argument('--irreps-hidden', type=str,
                        default='32x0e+16x1o+8x2e')
    parser.add_argument('--max-ell', type=int, default=2)
    parser.add_argument('--n-conv', type=int, default=3)
    parser.add_argument('--num-radial-basis', type=int, default=16)
    parser.add_argument('--energy-hidden', type=int, default=128)
    parser.add_argument('--train-file', type=str, default='',
                        help='Separate train extxyz file')
    parser.add_argument('--val-file', type=str, default='',
                        help='Separate val extxyz file')
    parser.add_argument('--test-file', type=str, default='',
                        help='Separate test extxyz file')
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
    parser.add_argument('--use-cuequivariance', action='store_true',
                        help='Use cuequivariance fused CUDA kernels (3x speed, 2x memory)')
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


def build_graph(atoms, cutoff, gom_cutoff=6.0, natx=64, use_gom=True,
                orbital='s'):
    """Build edge graph from ASE Atoms, with precomputed GOM fingerprints.

    GOM eigenvalues are computed once here and cached — NOT recomputed
    every forward pass.
    """
    from eosnet.fp import get_lfp

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

    # Precompute GOM neighbor topology (indices + shifts, reused every epoch)
    # Actual GOM eigenvalues computed differentiably in forward pass
    gom_data = None
    if use_gom:
        from eosnet.fp.rcov import get_rcov
        gom_i, gom_j, gom_D, gom_S = neighbor_list('ijDS', atoms, gom_cutoff)

        rcov_all = get_rcov(torch.tensor(atoms.numbers),
                            dtype=torch.float32).numpy()

        # For each atom, collect neighbors sorted by distance
        # Slot 0 = self, slots 1..max_nbr = neighbors
        nbr_counts = np.bincount(gom_i, minlength=nat)
        max_sphere = int(nbr_counts.max()) + 1 if len(gom_i) > 0 else 1
        max_n = min(max_sphere, natx)

        nbr_idx = np.zeros((nat, max_n), dtype=np.int64)
        nbr_shifts = np.zeros((nat, max_n, 3), dtype=np.float32)
        nbr_rcov = np.zeros((nat, max_n), dtype=np.float32)
        n_sphere = np.zeros(nat, dtype=np.int64)

        for iat in range(nat):
            # Self entry at slot 0
            nbr_idx[iat, 0] = iat
            nbr_shifts[iat, 0] = 0.0
            nbr_rcov[iat, 0] = rcov_all[iat]

            mask = (gom_i == iat)
            n_nbr = mask.sum()
            if n_nbr == 0:
                n_sphere[iat] = 1
                continue

            local_d2 = (gom_D[mask] ** 2).sum(axis=1)
            order = np.argsort(local_d2)
            n_keep = min(n_nbr, max_n - 1)
            order = order[:n_keep]

            nbr_idx[iat, 1:1+n_keep] = gom_j[mask][order]
            nbr_shifts[iat, 1:1+n_keep] = gom_S[mask][order]
            nbr_rcov[iat, 1:1+n_keep] = rcov_all[gom_j[mask][order]]
            n_sphere[iat] = 1 + n_keep

        gom_data = {
            'nbr_idx': torch.tensor(nbr_idx, dtype=torch.long),
            'nbr_shifts': torch.tensor(nbr_shifts, dtype=torch.float32),
            'nbr_rcov': torch.tensor(nbr_rcov, dtype=torch.float32),
            'n_sphere': torch.tensor(n_sphere, dtype=torch.long),
            'self_rcov': torch.tensor(rcov_all, dtype=torch.float32),
        }

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
        'gom_data': gom_data,
    }


def collate_batch(graphs, device):
    """Collate multiple structure graphs into a single batched graph.

    Concatenates atoms and edges with offset indices, enabling GPU-efficient
    mini-batch training without padding.
    """
    all_positions = []
    all_atomic_numbers = []
    all_edge_src = []
    all_edge_dst = []
    all_edge_shifts = []
    all_cells = []
    all_energies = []
    all_forces = []
    all_batch_idx = []
    all_nats = []

    # GOM neighbor topology (for differentiable GOM)
    all_gom_nbr_idx = []
    all_gom_nbr_shifts = []
    all_gom_nbr_rcov = []
    all_gom_n_sphere = []
    all_gom_self_rcov = []
    has_gom_data = False
    all_stresses = []
    has_stress = False

    atom_offset = 0
    for i, g in enumerate(graphs):
        nat = g['nat']
        all_positions.append(g['positions'])
        all_atomic_numbers.append(g['atomic_numbers'])
        all_forces.append(g['forces'])
        all_energies.append(g['energy'])
        all_nats.append(nat)
        all_batch_idx.append(torch.full((nat,), i, dtype=torch.long))

        # Stress (Voigt 6-component)
        if g.get('stress') is not None:
            s = g['stress']
            if s.numel() == 9:
                # 3x3 matrix → Voigt: xx, yy, zz, yz, xz, xy
                s3 = s.reshape(3, 3)
                s = torch.stack([s3[0,0], s3[1,1], s3[2,2],
                                 s3[1,2], s3[0,2], s3[0,1]])
            all_stresses.append(s)
            has_stress = True
        else:
            all_stresses.append(torch.zeros(6, dtype=torch.float32))

        # Offset edge indices
        src, dst = g['edge_index']
        all_edge_src.append(src + atom_offset)
        all_edge_dst.append(dst + atom_offset)
        all_edge_shifts.append(g['edge_shifts'])
        all_cells.append(g['cell'])

        # GOM neighbor data — offset nbr_idx by atom_offset
        if g['gom_data'] is not None:
            has_gom_data = True
            gd = g['gom_data']
            all_gom_nbr_idx.append(gd['nbr_idx'] + atom_offset)
            all_gom_nbr_shifts.append(gd['nbr_shifts'])
            all_gom_nbr_rcov.append(gd['nbr_rcov'])
            all_gom_n_sphere.append(gd['n_sphere'])
            all_gom_self_rcov.append(gd['self_rcov'])

        atom_offset += nat

    positions = torch.cat(all_positions).to(device).requires_grad_(True)
    atomic_numbers = torch.cat(all_atomic_numbers).to(device)
    edge_index = torch.stack([
        torch.cat(all_edge_src), torch.cat(all_edge_dst)
    ]).to(device)
    forces = torch.cat(all_forces).to(device)
    energies = torch.stack(all_energies).to(device)
    batch_idx = torch.cat(all_batch_idx).to(device)
    nats = torch.tensor(all_nats, dtype=torch.float32, device=device)

    # Recompute edge_vec from positions for autograd
    edge_src, edge_dst = edge_index
    edge_batch = batch_idx[edge_src]

    cells = torch.stack(all_cells).to(device)
    edge_shifts = torch.cat(all_edge_shifts).to(device)

    edge_cell = cells[edge_batch]
    shift_cart = torch.einsum('ei,eij->ej', edge_shifts, edge_cell)
    edge_vec = positions[edge_dst] - positions[edge_src] + shift_cart

    # Collate GOM neighbor topology
    gom_data = None
    if has_gom_data:
        # Pad nbr arrays to same max_nbr across batch
        max_nbrs = [g.shape[1] for g in all_gom_nbr_idx]
        max_nbr = max(max_nbrs)

        def pad_2d(tensors, max_col, fill=0):
            result = []
            for t in tensors:
                if t.shape[1] < max_col:
                    pad = torch.full((t.shape[0], max_col - t.shape[1]),
                                     fill, dtype=t.dtype)
                    result.append(torch.cat([t, pad], dim=1))
                else:
                    result.append(t)
            return torch.cat(result)

        def pad_3d(tensors, max_col):
            result = []
            for t in tensors:
                if t.shape[1] < max_col:
                    pad = torch.zeros(t.shape[0], max_col - t.shape[1], 3,
                                      dtype=t.dtype)
                    result.append(torch.cat([t, pad], dim=1))
                else:
                    result.append(t)
            return torch.cat(result)

        gom_data = {
            'nbr_idx': pad_2d(all_gom_nbr_idx, max_nbr).to(device),
            'nbr_shifts': pad_3d(all_gom_nbr_shifts, max_nbr).to(device),
            'nbr_rcov': pad_2d(all_gom_nbr_rcov, max_nbr).to(device),
            'n_sphere': torch.cat(all_gom_n_sphere).to(device),
            'self_rcov': torch.cat(all_gom_self_rcov).to(device),
        }

    stresses = None
    if has_stress:
        stresses = torch.stack(all_stresses).to(device)  # (B, 6)

    edge_shifts_cat = torch.cat(all_edge_shifts).to(device)

    return {
        'positions': positions,
        'atomic_numbers': atomic_numbers,
        'edge_index': edge_index,
        'edge_vec': edge_vec,
        'edge_shifts': edge_shifts_cat,
        'forces': forces,
        'energies': energies,
        'stresses': stresses,
        'batch_idx': batch_idx,
        'nats': nats,
        'gom_data': gom_data,
        'cells': cells,
        'n_structures': len(graphs),
    }


def train_step(model, graphs, optimizer, energy_w, force_w, stress_w, device):
    """Train on a mini-batch of structures."""
    model.train()
    optimizer.zero_grad()

    batch = collate_batch(graphs, device)
    use_stress = stress_w > 0

    output = model(
        batch['atomic_numbers'], batch['positions'], None,
        batch['edge_index'], batch['edge_vec'],
        compute_forces=True,
        compute_stress=use_stress,
        batch_idx=batch['batch_idx'],
        n_structures=batch['n_structures'],
        gom_data=batch['gom_data'],
        cells=batch['cells'],
        edge_shifts=batch.get('edge_shifts'),
    )

    # Energy loss: per-atom MSE averaged over structures
    pred_energy_per_atom = output['energy_per_struct'] / batch['nats']
    target_energy_per_atom = batch['energies'] / batch['nats']
    energy_loss = ((pred_energy_per_atom - target_energy_per_atom) ** 2).mean()

    # Force loss: per-component MSE
    force_loss = ((output['forces'] - batch['forces']) ** 2).mean()

    loss = energy_w * energy_loss + force_w * force_loss

    # Stress loss
    stress_mae = 0.0
    if use_stress and 'stress' in output and batch.get('stresses') is not None:
        pred_stress = output['stress']  # (B, 6) Voigt
        target_stress = batch['stresses']  # (B, 6) Voigt
        stress_loss = ((pred_stress - target_stress) ** 2).mean()
        loss = loss + stress_w * stress_loss
        with torch.no_grad():
            stress_mae = (pred_stress - target_stress).abs().mean().item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    with torch.no_grad():
        energy_mae = (pred_energy_per_atom - target_energy_per_atom).abs().mean().item()
        force_mae = (output['forces'] - batch['forces']).abs().mean().item()

    return {
        'loss': loss.item(),
        'energy_mae': energy_mae,
        'force_mae': force_mae,
        'stress_mae': stress_mae,
    }


def validate(model, val_data, device, batch_size=32, compute_stress=False):
    """Validate on a dataset using mini-batches."""
    model.eval()
    energy_maes = []
    force_maes = []
    stress_maes = []

    for start in range(0, len(val_data), batch_size):
        graphs = val_data[start:start + batch_size]
        batch = collate_batch(graphs, device)

        with torch.enable_grad():
            output = model(
                batch['atomic_numbers'], batch['positions'], None,
                batch['edge_index'], batch['edge_vec'],
                compute_forces=True,
                compute_stress=compute_stress,
                batch_idx=batch['batch_idx'],
                n_structures=batch['n_structures'],
                gom_data=batch['gom_data'],
                cells=batch['cells'],
                edge_shifts=batch.get('edge_shifts'),
            )

        with torch.no_grad():
            pred_epa = output['energy_per_struct'] / batch['nats']
            target_epa = batch['energies'] / batch['nats']
            energy_maes.append(
                (pred_epa - target_epa).abs().mean().item())
            force_maes.append(
                (output['forces'] - batch['forces']).abs().mean().item())
            if compute_stress and 'stress' in output and batch.get('stresses') is not None:
                stress_maes.append(
                    (output['stress'] - batch['stresses']).abs().mean().item())

    result = {
        'energy_mae': np.mean(energy_maes),
        'force_mae': np.mean(force_maes),
    }
    if stress_maes:
        result['stress_mae'] = np.mean(stress_maes)
    return result


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
    if args.train_file and args.val_file and args.test_file:
        # Separate train/val/test files
        train_frames = load_dataset(args.train_file)
        val_frames = load_dataset(args.val_file)
        test_frames = load_dataset(args.test_file)
        print(f"Split: {len(train_frames)} train, {len(val_frames)} val, "
              f"{len(test_frames)} test")
    else:
        # Single file, auto-split
        frames = load_dataset(args.data_path)
        indices = np.random.permutation(len(frames))
        n_train = int(args.train_ratio * len(frames))
        n_val = int(args.val_ratio * len(frames))
        train_frames = [frames[i] for i in indices[:n_train]]
        val_frames = [frames[i] for i in indices[n_train:n_train + n_val]]
        test_frames = [frames[i] for i in indices[n_train + n_val:]]
        print(f"Split: {len(train_frames)} train, {len(val_frames)} val, "
              f"{len(test_frames)} test")

    # Pre-build graphs
    print("Building graphs...")
    use_gom = not args.no_gom
    lseg = 4 if args.orbital == 'sp' else 1
    gom_input_dim = args.natx * lseg

    def _build(atoms):
        return build_graph(atoms, args.radial_cutoff,
                           gom_cutoff=args.gom_cutoff, natx=args.natx,
                           use_gom=use_gom, orbital=args.orbital)
    train_data = [_build(a) for a in train_frames]
    val_data = [_build(a) for a in val_frames]
    test_data = [_build(a) for a in test_frames]

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
        gom_input_dim=gom_input_dim,
        orbital=args.orbital,
        use_cuequivariance=args.use_cuequivariance,
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
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
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

        # Mini-batch training
        for start in range(0, len(perm), args.batch_size):
            batch_indices = perm[start:start + args.batch_size]
            graphs = [train_data[i] for i in batch_indices]
            metrics = train_step(
                model, graphs, optimizer,
                args.energy_weight, args.force_weight, args.stress_weight,
                device
            )
            epoch_losses.append(metrics['loss'])
            epoch_e_mae.append(metrics['energy_mae'])
            epoch_f_mae.append(metrics['force_mae'])

        # Validation
        use_stress = args.stress_weight > 0
        val_metrics = validate(model, val_data, device, compute_stress=use_stress)
        scheduler.step(val_metrics['force_mae'])

        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % args.print_freq == 0 or epoch == 0:
            msg = (f"Epoch {epoch+1:4d} | "
                   f"loss {np.mean(epoch_losses):.4f} | "
                   f"E_MAE {np.mean(epoch_e_mae)*1000:.1f} meV | "
                   f"F_MAE {np.mean(epoch_f_mae)*1000:.1f} meV/Å | "
                   f"val_E {val_metrics['energy_mae']*1000:.1f} meV | "
                   f"val_F {val_metrics['force_mae']*1000:.1f} meV/Å")
            if 'stress_mae' in val_metrics:
                msg += f" | val_S {val_metrics['stress_mae']*1000:.1f} meV/ų"
            msg += f" | lr {lr:.1e} | {dt:.1f}s"
            print(msg)

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
                            map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    test_metrics = validate(model, test_data, device,
                            compute_stress=args.stress_weight > 0)
    print(f"Test E_MAE: {test_metrics['energy_mae']*1000:.1f} meV/atom")
    print(f"Test F_MAE: {test_metrics['force_mae']*1000:.1f} meV/Å")
    if 'stress_mae' in test_metrics:
        print(f"Test S_MAE: {test_metrics['stress_mae']*1000:.1f} meV/ų")

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
