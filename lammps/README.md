# pair_eosnet — LAMMPS pair_style for EOSNet MLIP

LAMMPS interface for EOSNet, an e3nn equivariant MLIP with differentiable
GOM (Gaussian Overlap Matrix) fingerprints.

## Overview

EOSNet computes per-atom energies using:
1. **e3nn backbone**: Equivariant message passing with tensor products (edges at 5.0 Å)
2. **GOM fingerprints**: Eigenvalues of Gaussian overlap matrices (neighbors at 6.0 Å)
3. **Autograd forces**: F = -dE/dr computed via PyTorch autograd

The pair_style constructs both neighbor topologies from LAMMPS's single neighbor
list (built at the larger GOM cutoff), calls the traced TorchScript model, and
extracts forces via autograd.

## Build

### Prerequisites

- LAMMPS (stable release)
- PyTorch / libtorch (matching the training version, e.g., 2.5.x)
- CUDA toolkit 12.x (for GPU acceleration)
- CMake 3.18+

### Option A: Build within LAMMPS

```bash
cd lammps/build
cmake ../cmake \
  -D CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
  -D PKG_ML-SNAP=yes \
  -D PAIR_EOSNET_DIR=/path/to/EosNet-dev/lammps
make -j$(nproc)
```

### Option B: Build as standalone plugin

```bash
cd EosNet-dev/lammps
mkdir build && cd build
cmake .. \
  -D LAMMPS_SOURCE_DIR=/path/to/lammps/src \
  -D CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
make -j$(nproc)
```

## Model Export

Export a trained EOSNet checkpoint to TorchScript:

```bash
python scripts/deploy_for_lammps.py \
  --checkpoint models/mlip_fe_exp3_sfw20.pth \
  --output eosnet_deployed.pt \
  --test-data test.extxyz \
  --use-cuequivariance  # optional, for GPU acceleration
```

The export script:
1. Wraps the model to flatten dict inputs into positional tensors
2. Traces with example data via `torch.jit.trace`
3. Validates energy and forces against eager mode
4. Saves a `.pt` file (~3.6 MB)

## LAMMPS Usage

```
units           metal
atom_style      atomic
newton          off          # Required

pair_style      eosnet
pair_coeff      * * eosnet_deployed.pt 5.0 6.0 32 Fe

# With default cutoffs (5.0 / 6.0 / 32):
pair_coeff      * * eosnet_deployed.pt Fe

# Multi-element:
pair_coeff      * * eosnet_deployed.pt 5.0 6.0 32 Fe Si
```

### pair_coeff arguments

```
pair_coeff * * <model.pt> [e3nn_cutoff gom_cutoff natx] <element1> [element2] ...
```

| Argument | Default | Description |
|----------|---------|-------------|
| model.pt | required | TorchScript model file |
| e3nn_cutoff | 5.0 | Edge cutoff for e3nn message passing (Å) |
| gom_cutoff | 6.0 | GOM neighbor cutoff (Å) |
| natx | 32 | Max GOM neighbors per atom |
| elements | required | Element symbols matching LAMMPS atom types |

### Requirements

- `newton off` (NequIP-style, forces only on local atoms)
- `units metal` (eV, Å, ps)
- Single MPI rank (multi-GPU via future domain decomposition support)

## Performance

Benchmarked on L40S-48GB GPU with BCC Fe (natx=32, cuequivariance):

| N_atoms | ms/step | ms/atom | GPU Memory |
|---------|---------|---------|------------|
| 2,048   | 63      | 0.031   | 7.0 GB     |
| 4,000   | 149     | 0.037   | 13.6 GB    |
| 8,788   | 367     | 0.042   | 30.3 GB    |

Max atoms per GPU: ~8,800 on L40S-48GB.

## Environment Variables

- `EOSNET_DEBUG=1`: Print edge count and energy at each step

## Files

```
lammps/
├── pair_eosnet.h          # Header
├── pair_eosnet.cpp         # Implementation
├── CMakeLists.txt          # Build system
├── in.eosnet_example       # Example LAMMPS input
└── README.md               # This file

scripts/
├── deploy_for_lammps.py    # Model export script
└── test_torchscript.py     # TorchScript compatibility test
```

## Architecture

```
LAMMPS neighbor list (gom_cutoff = 6.0 Å)
    │
    ├── Filter at e3nn_cutoff (5.0 Å) ──→ edge_index + cell_shifts
    │                                       │
    └── GOM topology (natx=32 nearest) ──→ nbr_idx + nbr_shifts + rcov
                                            │
                    ┌───────────────────────┘
                    ▼
            TorchScript Model (EOSNetForLAMMPS)
            ┌─────────────────────────────────┐
            │  positions (requires_grad=True)  │
            │       │                          │
            │  edge_vec = pos[dst]-pos[src]    │
            │       + cell_shift @ cell        │
            │       │                          │
            │  e3nn backbone ──→ node features │
            │       │                          │
            │  GOM eigvalsh ──→ gom features   │
            │       │                          │
            │  concat ──→ MLP ──→ atom_energy  │
            │       │                          │
            │  sum ──→ total_energy            │
            └────────────┬────────────────────┘
                         │
            torch::autograd::grad(E, positions)
                         │
                    forces = -dE/dr
```
