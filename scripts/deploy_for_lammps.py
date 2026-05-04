#!/usr/bin/env python3
"""Export EOSNet MLIP to TorchScript for LAMMPS deployment.

Usage:
    python deploy_for_lammps.py --checkpoint models/mlip_fe_exp3_sfw20.pth \
                                --output eosnet_deployed.pt \
                                --test-data test.extxyz

The exported model takes flattened tensor arguments (no dicts) and returns
total energy as a scalar. Forces are computed via torch.autograd.grad on the
C++ side (following the pair_nequip pattern).
"""

import argparse
import sys
import os
import numpy as np
import torch
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eosnet.mlip import EOSNetMLIP
from eosnet.fp.rcov import get_rcov


class EOSNetForLAMMPS(torch.nn.Module):
    """Wrapper that flattens gom_data dict into positional tensor args.

    torch.jit.trace cannot handle dict inputs, so we unpack gom_data
    into individual tensors. The C++ pair_style passes these directly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        batch_idx: torch.Tensor,
        cells: torch.Tensor,
        gom_nbr_idx: torch.Tensor,
        gom_nbr_shifts: torch.Tensor,
        gom_nbr_rcov: torch.Tensor,
        gom_n_sphere: torch.Tensor,
        gom_self_rcov: torch.Tensor,
    ) -> torch.Tensor:
        gom_data = {
            'nbr_idx': gom_nbr_idx,
            'nbr_shifts': gom_nbr_shifts,
            'nbr_rcov': gom_nbr_rcov,
            'n_sphere': gom_n_sphere,
            'self_rcov': gom_self_rcov,
        }
        out = self.model(
            atomic_numbers, positions, cell, edge_index, edge_vec,
            compute_forces=False, compute_stress=False,
            batch_idx=batch_idx, n_structures=1,
            gom_data=gom_data, cells=cells,
        )
        return out['energy']


def build_example_inputs(atoms, model_config, device):
    """Build example tensors for tracing from an ASE Atoms object."""
    cutoff = model_config['radial_cutoff']
    gom_cutoff = model_config['gom_cutoff']
    natx = model_config['natx']

    i_idx, j_idx, D_vec, S_vec = neighbor_list('ijDS', atoms, cutoff)
    nat = len(atoms)

    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    positions.requires_grad_(True)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32, device=device)
    edge_index = torch.stack([
        torch.tensor(i_idx, dtype=torch.long),
        torch.tensor(j_idx, dtype=torch.long),
    ]).to(device)
    edge_shifts = torch.tensor(S_vec, dtype=torch.float32, device=device)

    # Compute edge_vec
    cells = cell.unsqueeze(0)
    batch_idx = torch.zeros(nat, dtype=torch.long, device=device)
    src, dst = edge_index
    edge_batch = batch_idx[src]
    edge_cell = cells[edge_batch]
    shift_cart = torch.einsum('ei,eij->ej', edge_shifts, edge_cell)
    edge_vec = positions[dst] - positions[src] + shift_cart

    # Build GOM neighbor topology
    gom_i, gom_j, gom_D, gom_S = neighbor_list('ijDS', atoms, gom_cutoff)
    rcov_all = get_rcov(torch.tensor(atoms.numbers), dtype=torch.float32).numpy()
    nbr_counts = np.bincount(gom_i, minlength=nat)
    max_sphere = int(nbr_counts.max()) + 1 if len(gom_i) > 0 else 1
    max_n = min(max_sphere, natx)

    nbr_idx = np.zeros((nat, max_n), dtype=np.int64)
    nbr_shifts_arr = np.zeros((nat, max_n, 3), dtype=np.float32)
    nbr_rcov = np.zeros((nat, max_n), dtype=np.float32)
    n_sphere = np.zeros(nat, dtype=np.int64)

    for iat in range(nat):
        nbr_idx[iat, 0] = iat
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
        nbr_shifts_arr[iat, 1:1+n_keep] = gom_S[mask][order]
        nbr_rcov[iat, 1:1+n_keep] = rcov_all[gom_j[mask][order]]
        n_sphere[iat] = 1 + n_keep

    return (
        atomic_numbers,
        positions,
        cell,
        edge_index,
        edge_vec,
        batch_idx,
        1,  # n_structures
        cells,
        torch.tensor(nbr_idx, dtype=torch.long, device=device),
        torch.tensor(nbr_shifts_arr, dtype=torch.float32, device=device),
        torch.tensor(nbr_rcov, dtype=torch.float32, device=device),
        torch.tensor(n_sphere, dtype=torch.long, device=device),
        torch.tensor(rcov_all, dtype=torch.float32, device=device),
    )


def main():
    parser = argparse.ArgumentParser(description='Export EOSNet to TorchScript')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint .pth')
    parser.add_argument('--output', default='eosnet_deployed.pt', help='Output .pt file')
    parser.add_argument('--test-data', help='Test extxyz for tracing + validation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model architecture args (must match checkpoint)
    parser.add_argument('--irreps-hidden', default='32x0e+16x1o+8x2e')
    parser.add_argument('--max-ell', type=int, default=2)
    parser.add_argument('--n-conv', type=int, default=3)
    parser.add_argument('--num-radial-basis', type=int, default=16)
    parser.add_argument('--radial-cutoff', type=float, default=5.0)
    parser.add_argument('--n-gom-features', type=int, default=32)
    parser.add_argument('--gom-cutoff', type=float, default=6.0)
    parser.add_argument('--natx', type=int, default=32)
    parser.add_argument('--energy-hidden', type=int, default=128)
    parser.add_argument('--orbital', default='s')
    parser.add_argument('--use-cuequivariance', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model_config = {
        'radial_cutoff': args.radial_cutoff,
        'gom_cutoff': args.gom_cutoff,
        'natx': args.natx,
    }

    # gom_input_dim: when n_gom_features=0 (no projection), raw GOM is natx-dimensional
    gom_input_dim = args.n_gom_features if args.n_gom_features > 0 else args.natx

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
        use_gom=True,
        gom_input_dim=gom_input_dim,
        orbital=args.orbital,
        use_cuequivariance=args.use_cuequivariance,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build wrapper
    wrapper = EOSNetForLAMMPS(model)
    wrapper.eval()

    # Get example data for tracing
    if args.test_data:
        from ase.io import read as ase_read
        atoms = ase_read(args.test_data, index='0')
    else:
        # Create a dummy BCC Fe structure
        from ase.build import bulk
        atoms = bulk('Fe', 'bcc', a=2.87) * (2, 2, 2)
    print(f"Tracing with {len(atoms)}-atom structure")

    example_inputs = build_example_inputs(atoms, model_config, device)

    # Reference energy (eager mode)
    with torch.no_grad():
        ref_energy = wrapper(*example_inputs).item()
    print(f"Reference energy: {ref_energy:.6f} eV")

    # Trace
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs)

    # Validate traced model
    with torch.no_grad():
        traced_energy = traced(*example_inputs).item()
    diff = abs(traced_energy - ref_energy)
    print(f"Traced energy: {traced_energy:.6f} eV (diff: {diff:.2e})")
    assert diff < 1e-4, f"Energy mismatch too large: {diff}"

    # Test forces via autograd
    fresh_inputs = list(example_inputs)
    fresh_pos = fresh_inputs[1].clone().detach().requires_grad_(True)
    fresh_inputs[1] = fresh_pos
    # Recompute edge_vec with fresh positions
    src, dst = fresh_inputs[3]
    edge_batch = fresh_inputs[5][src]
    edge_cell = fresh_inputs[7][edge_batch]
    shift_cart = torch.einsum('ei,eij->ej',
                              torch.tensor(np.array(neighbor_list('S', atoms, args.radial_cutoff)),
                                           dtype=torch.float32, device=device),
                              edge_cell)
    fresh_inputs[4] = fresh_pos[dst] - fresh_pos[src] + shift_cart
    fresh_inputs = tuple(fresh_inputs)

    energy = traced(*fresh_inputs)
    forces = -torch.autograd.grad(energy, fresh_pos, create_graph=False)[0]
    print(f"Forces: shape={forces.shape}, max={forces.abs().max():.4f} eV/Å")

    # Save
    traced.save(args.output)
    file_size = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({file_size:.1f} MB)")
    print("DONE — model ready for LAMMPS pair_eosnet")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
