#!/bin/bash
#SBATCH --job-name=eosnet_export
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=export_%j.out
#SBATCH --error=export_%j.err

set -e

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate nequip

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

cd /scratch/lz432/eosnet_compile_test

echo "=== Step 1: Export model to TorchScript ==="
# This checkpoint was trained with:
#   natx=64, n_gom_features=0 (raw GOM), orbital='s', gom_input_dim=64
#   e3nn (not cuequivariance)
#   force_weight=20
#   energy_readout input = 32 scalars + 64 raw gom = 96
#
# Run deploy inline to avoid script caching issues
python -c "
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/lz432/eosnet_compile_test')

import torch
import numpy as np
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list
from eosnet.mlip import EOSNetMLIP
from eosnet.fp.rcov import get_rcov

# Build model matching checkpoint architecture
model = EOSNetMLIP(
    irreps_hidden='32x0e+16x1o+8x2e',
    max_ell=2, n_conv=3, num_radial_basis=16,
    radial_cutoff=5.0, n_gom_features=0, gom_cutoff=6.0,
    natx=64, energy_hidden=128,
    use_gom=True, gom_input_dim=64,
    orbital='s',
    use_cuequivariance=False,
).cuda()

# Check readout shape before loading
print(f'readout[0] weight shape: {model.energy_readout[0].weight.shape}')
print(f'gom_proj: {model.gom_proj}')
print(f'natx={model.natx}, gom_input_dim={model.gom_input_dim}')

# Load checkpoint
ckpt = torch.load('/scratch/lz432/eosnet_compile_test/mlip_fe_exp3_sfw20.pth', map_location='cuda')
sd = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(sd)
model.eval()
print(f'Checkpoint loaded OK, params={sum(p.numel() for p in model.parameters()):,}')

# Wrapper for tracing (flattens gom_data dict -> positional args)
class EOSNetForLAMMPS(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, atomic_numbers, positions, cell, edge_index, edge_vec,
                batch_idx, cells,
                gom_nbr_idx, gom_nbr_shifts, gom_nbr_rcov, gom_n_sphere, gom_self_rcov):
        gom_data = {
            'nbr_idx': gom_nbr_idx, 'nbr_shifts': gom_nbr_shifts,
            'nbr_rcov': gom_nbr_rcov, 'n_sphere': gom_n_sphere, 'self_rcov': gom_self_rcov,
        }
        out = self.model(
            atomic_numbers, positions, cell, edge_index, edge_vec,
            compute_forces=False, compute_stress=False,
            batch_idx=batch_idx, n_structures=1,
            gom_data=gom_data, cells=cells,
        )
        return out['energy']

wrapper = EOSNetForLAMMPS(model).cuda().eval()

# Build example from test data
atoms = ase_read('/home/lz432/allegro_Fe_dmft_finetune/test.extxyz', index='0')
nat = len(atoms)
device = torch.device('cuda')

i_idx, j_idx, D_vec, S_vec = neighbor_list('ijDS', atoms, 5.0)
atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device, requires_grad=True)
cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32, device=device)
edge_index = torch.stack([torch.tensor(i_idx, dtype=torch.long), torch.tensor(j_idx, dtype=torch.long)]).to(device)
edge_shifts = torch.tensor(S_vec, dtype=torch.float32, device=device)
cells = cell.unsqueeze(0)
batch_idx = torch.zeros(nat, dtype=torch.long, device=device)
src, dst = edge_index
edge_batch = batch_idx[src]
edge_cell = cells[edge_batch]
shift_cart = torch.einsum('ei,eij->ej', edge_shifts, edge_cell)
edge_vec = positions[dst] - positions[src] + shift_cart

# GOM neighbors
gom_i, gom_j, gom_D, gom_S = neighbor_list('ijDS', atoms, 6.0)
rcov_all = get_rcov(torch.tensor(atoms.numbers), dtype=torch.float32).numpy()
nbr_counts = np.bincount(gom_i, minlength=nat)
max_sphere = int(nbr_counts.max()) + 1
max_n = min(max_sphere, 64)

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

gom_nbr_idx = torch.tensor(nbr_idx, dtype=torch.long, device=device)
gom_nbr_shifts = torch.tensor(nbr_shifts_arr, dtype=torch.float32, device=device)
gom_nbr_rcov = torch.tensor(nbr_rcov, dtype=torch.float32, device=device)
gom_n_sphere = torch.tensor(n_sphere, dtype=torch.long, device=device)
gom_self_rcov = torch.tensor(rcov_all, dtype=torch.float32, device=device)

# Reference
with torch.no_grad():
    ref_E = wrapper(atomic_numbers, positions, cell, edge_index, edge_vec,
                     batch_idx, cells,
                     gom_nbr_idx, gom_nbr_shifts, gom_nbr_rcov, gom_n_sphere, gom_self_rcov).item()
print(f'Reference energy: {ref_E:.6f} eV')

# Trace
print('Tracing...')
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (atomic_numbers, positions, cell, edge_index, edge_vec,
                                        batch_idx, cells,
                                        gom_nbr_idx, gom_nbr_shifts, gom_nbr_rcov,
                                        gom_n_sphere, gom_self_rcov))

with torch.no_grad():
    traced_E = traced(atomic_numbers, positions, cell, edge_index, edge_vec,
                       batch_idx, cells,
                       gom_nbr_idx, gom_nbr_shifts, gom_nbr_rcov,
                       gom_n_sphere, gom_self_rcov).item()
print(f'Traced energy: {traced_E:.6f} eV (diff: {abs(traced_E - ref_E):.2e})')

# Test forces
pos2 = positions.clone().detach().requires_grad_(True)
edge_vec2 = pos2[dst] - pos2[src] + shift_cart
E2 = traced(atomic_numbers, pos2, cell, edge_index, edge_vec2,
             batch_idx, cells,
             gom_nbr_idx, gom_nbr_shifts, gom_nbr_rcov,
             gom_n_sphere, gom_self_rcov)
forces = -torch.autograd.grad(E2, pos2)[0]
print(f'Forces: max={forces.abs().max():.4f} eV/A')

# Save
output = '/scratch/lz432/eosnet_lammps/eosnet_deployed.pt'
traced.save(output)
fsize = os.path.getsize(output) / 1e6
print(f'Saved: {output} ({fsize:.1f} MB)')
print('Model export SUCCESS')
"

echo ""
echo "=== Step 2: Generate LAMMPS data file from test structure ==="
python -c "
import numpy as np
from ase.build import bulk
from ase.io import write as ase_write

# Use a small cubic BCC Fe supercell (conventional cell, orthogonal)
atoms = bulk('Fe', 'bcc', a=2.87, cubic=True) * (3, 3, 3)
nat = len(atoms)
print(f'Writing LAMMPS data file: {nat} atoms, cell={atoms.cell.array.diagonal()}')

# Use ASE's built-in LAMMPS writer for correct format
ase_write('/scratch/lz432/eosnet_lammps/fe_bcc.data', atoms, format='lammps-data')
print('Done: /scratch/lz432/eosnet_lammps/fe_bcc.data')
"

echo ""
echo "=== Step 3: Verify exported model loads correctly ==="
python -c "
import torch
model = torch.jit.load('/scratch/lz432/eosnet_lammps/eosnet_deployed.pt', map_location='cuda')
print(f'Model loaded on CUDA: {model}')
print(f'File size: {__import__(\"os\").path.getsize(\"/scratch/lz432/eosnet_lammps/eosnet_deployed.pt\")/1e6:.1f} MB')
"

echo ""
echo "DONE — model and data ready for LAMMPS test"
