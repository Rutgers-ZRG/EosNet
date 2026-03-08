"""EOSNet MLIP: e3nn equivariant backbone + GOM fingerprints for energy/forces.

Per-atom energy prediction with forces via autograd.
GOM eigenvalues provide orbital-overlap inductive bias as additional node features.
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

from .model import RadialBasis, scatter_sum
from .fp.gom import build_gom_s_batched, build_gom_sp_batched, _eigvalsh_batched_adaptive
from .fp.rcov import get_rcov
from .fp.cutoff import cutoff_amplitude
from .fp.neighbors import get_ixyz


def compute_gom_differentiable(positions, gom_nbr_idx, gom_nbr_shifts,
                                gom_nbr_rcov, gom_n_sphere, gom_self_rcov,
                                cells, batch_idx, gom_cutoff, natx,
                                orbital='s'):
    """Compute GOM eigenvalues differentiably from live positions.

    Neighbor topology is precomputed and fixed; positions are live tensors
    with requires_grad=True so autograd flows through to forces.

    Args:
        positions: (N_total, 3) all atom positions (requires_grad)
        gom_nbr_idx: (N_total, max_nbr) neighbor atom indices (global)
        gom_nbr_shifts: (N_total, max_nbr, 3) lattice shift vectors (fractional)
        gom_nbr_rcov: (N_total, max_nbr) covalent radii of neighbors
        gom_n_sphere: (N_total,) actual neighbor count per atom (including self)
        gom_self_rcov: (N_total,) covalent radii of center atoms
        cells: (n_struct, 3, 3) lattice vectors
        batch_idx: (N_total,) structure index per atom
        gom_cutoff: cutoff radius
        natx: fingerprint dimension

    Returns:
        (N_total, natx) eigenvalues, descending, differentiable w.r.t. positions
    """
    N = positions.shape[0]
    max_nbr = gom_nbr_idx.shape[1]
    device = positions.device

    # Self position: slot 0 in the neighbor sphere
    # Neighbor positions: positions[nbr_idx] + shifts @ cell
    nbr_pos_raw = positions[gom_nbr_idx]  # (N, max_nbr, 3) — differentiable!

    # Cartesian shift: for each (i, j), shift_cart[i,j] = gom_nbr_shifts[i,j] @ cells[batch_idx[i]]
    atom_cells = cells[batch_idx]  # (N, 3, 3)
    shift_cart = torch.einsum('ijk,ikl->ijl', gom_nbr_shifts, atom_cells)

    nbr_pos = nbr_pos_raw + shift_cart  # (N, max_nbr, 3)

    # Build rxyz_padded: slot 0 = self, slots 1..max_nbr = neighbors
    # Already structured this way in precomputation
    rxyz_padded = nbr_pos  # (N, max_nbr, 3)

    # Compute squared distances for cutoff amplitude
    center_pos = positions.unsqueeze(1)  # (N, 1, 3)
    d_vec = rxyz_padded - center_pos
    d2 = (d_vec ** 2).sum(-1)  # (N, max_nbr)

    # Cutoff amplitudes (differentiable)
    amp_padded = cutoff_amplitude(d2, gom_cutoff)  # (N, max_nbr)

    # rcov_padded: slot 0 = self_rcov, rest from precomputed
    rcov_padded = gom_nbr_rcov  # (N, max_nbr)

    # Build GOMs and eigendecompose
    if orbital == 'sp':
        lseg = 4
        goms = build_gom_sp_batched(rxyz_padded, rcov_padded, amp_padded, gom_n_sphere)
    else:
        lseg = 1
        goms = build_gom_s_batched(rxyz_padded, rcov_padded, amp_padded, gom_n_sphere)

    eigvals = _eigvalsh_batched_adaptive(goms)  # ascending
    eigvals = eigvals.flip(-1)  # descending

    fp_dim = natx * lseg
    n_eig = eigvals.shape[1]

    # Pad/trim to fp_dim
    if n_eig >= fp_dim:
        eigvals = eigvals[:, :fp_dim]
    else:
        pad = torch.zeros(N, fp_dim - n_eig, device=device, dtype=eigvals.dtype)
        eigvals = torch.cat([eigvals, pad], dim=1)

    # Zero padding artifacts
    keep_dim = gom_n_sphere.clamp(max=natx).unsqueeze(1) * lseg
    cols = torch.arange(fp_dim, device=device).unsqueeze(0)
    keep = cols < keep_dim
    eigvals = eigvals * keep.to(eigvals.dtype)

    return eigvals


class MLIPInteractionBlock(nn.Module):
    """e3nn tensor product interaction with gated nonlinearity."""

    def __init__(self, irreps_in, irreps_sh, irreps_out, num_radial,
                 fc_hidden=64):
        super().__init__()

        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out
                                     if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out
                                    if ir.l > 0])
        num_gates = sum(mul for mul, ir in irreps_gated)
        irreps_gates = o3.Irreps(f"{num_gates}x0e") if num_gates > 0 \
            else o3.Irreps("")

        act_scalars = [torch.nn.functional.silu] * len(irreps_scalars)
        act_gates = [torch.sigmoid] * len(irreps_gates)

        self.gate = Gate(
            irreps_scalars, act_scalars,
            irreps_gates, act_gates,
            irreps_gated
        )

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_sh, self.gate.irreps_in,
            shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(num_radial, fc_hidden),
            nn.SiLU(),
            nn.Linear(fc_hidden, self.tp.weight_numel)
        )

        self.self_interaction = o3.Linear(irreps_in, irreps_out)

    def forward(self, node_fea, edge_sh, edge_radial,
                edge_src, edge_dst, num_nodes):
        self_fea = self.self_interaction(node_fea)
        tp_weights = self.fc(edge_radial)
        messages = self.tp(node_fea[edge_src], edge_sh, tp_weights)
        aggregated = scatter_sum(messages, edge_dst, num_nodes)
        out = self.gate(aggregated)
        return out + self_fea


class EOSNetMLIP(nn.Module):
    """EOSNet MLIP: e3nn backbone + GOM features → per-atom energy.

    Forces computed via autograd: F = -dE/dr.
    Stress computed via autograd on strain.

    Architecture:
        positions → GOM eigenvalues → proj → gom_fea (scalars)
        positions → e3nn edges → tensor products → equivariant node features
        scalar node features + gom_fea → concat → per-atom energy MLP → sum → E
    """

    def __init__(self,
                 num_species=100,
                 embedding_dim=32,
                 irreps_hidden="32x0e+16x1o+8x2e",
                 max_ell=2,
                 n_conv=3,
                 num_radial_basis=16,
                 radial_cutoff=5.0,
                 fc_hidden=64,
                 n_gom_features=32,
                 gom_cutoff=6.0,
                 natx=64,
                 energy_hidden=128,
                 use_gom=True,
                 gom_input_dim=None,
                 orbital='s'):
        super().__init__()

        self.radial_cutoff = radial_cutoff
        self.gom_cutoff = gom_cutoff
        self.orbital = orbital
        self.use_gom = use_gom
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_ell)
        if gom_input_dim is None:
            gom_input_dim = natx

        # Count scalar channels
        num_scalars = sum(mul for mul, ir in self.irreps_hidden if ir.l == 0)

        # Species embedding → scalar features
        self.species_embedding = nn.Embedding(num_species, embedding_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(embedding_dim, num_scalars),
            nn.SiLU(),
            nn.Linear(num_scalars, num_scalars)
        )

        # Radial basis for e3nn edges
        self.radial_basis = RadialBasis(num_radial_basis, radial_cutoff)

        # e3nn interaction blocks
        irreps_input = o3.Irreps(f"{num_scalars}x0e")
        self.convs = nn.ModuleList()
        irreps_in = irreps_input
        for _ in range(n_conv):
            self.convs.append(MLIPInteractionBlock(
                irreps_in=irreps_in,
                irreps_sh=self.irreps_sh,
                irreps_out=self.irreps_hidden,
                num_radial=num_radial_basis,
                fc_hidden=fc_hidden,
            ))
            irreps_in = self.irreps_hidden

        # Extract scalar indices from hidden irreps
        self._scalar_indices = []
        offset = 0
        for mul, ir in self.irreps_hidden:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 0:
                self._scalar_indices.extend(range(offset, offset + dim))
            offset += dim
        self._scalar_indices = torch.LongTensor(self._scalar_indices)
        num_output_scalars = len(self._scalar_indices)

        # GOM features
        self.natx = natx
        self.gom_input_dim = gom_input_dim
        if use_gom:
            if n_gom_features > 0:
                # Learned projection from eigenvalues
                self.gom_proj = nn.Linear(gom_input_dim, n_gom_features)
                readout_in = num_output_scalars + n_gom_features
            else:
                # Feed raw eigenvalues directly (no compression)
                self.gom_proj = None
                readout_in = num_output_scalars + gom_input_dim
        else:
            readout_in = num_output_scalars

        # Per-atom energy readout
        self.energy_readout = nn.Sequential(
            nn.Linear(readout_in, energy_hidden),
            nn.SiLU(),
            nn.Linear(energy_hidden, energy_hidden),
            nn.SiLU(),
            nn.Linear(energy_hidden, 1)
        )

        # Per-species energy shift (learnable)
        self.energy_shift = nn.Embedding(num_species, 1)
        nn.init.zeros_(self.energy_shift.weight)

    def forward(self, atomic_numbers, positions, cell,
                edge_index, edge_vec,
                compute_forces=True, compute_stress=False,
                gom_fp=None, batch_idx=None, n_structures=None,
                gom_data=None, cells=None):
        """Forward pass supporting both single structure and batched input.

        Args:
            atomic_numbers: (N,) integer atomic numbers (N = total atoms in batch)
            positions: (N, 3) Cartesian positions
            cell: (3, 3) lattice vectors (single structure only)
            edge_index: (2, E) [src, dst] edge indices
            edge_vec: (E, 3) edge displacement vectors
            compute_forces: if True, compute forces via autograd
            compute_stress: if True, compute stress via autograd
            gom_fp: (N, dim) precomputed GOM eigenvalues (static, no grad)
            batch_idx: (N,) structure index for each atom (for batched training)
            n_structures: number of structures in batch
            gom_data: dict with precomputed GOM neighbor topology for
                      differentiable computation (if provided, overrides gom_fp)
            cells: (n_struct, 3, 3) lattice vectors (for batched + differentiable GOM)

        Returns:
            dict with 'energy', 'forces', 'energy_per_struct' (if batched)
        """
        num_nodes = positions.shape[0]
        device = positions.device

        # Enable gradients for force computation
        if compute_forces and not positions.requires_grad:
            positions.requires_grad_(True)

        # Strain for stress computation (single structure only)
        if compute_stress:
            strain = torch.zeros(3, 3, device=device, dtype=positions.dtype,
                                 requires_grad=True)
            eye = torch.eye(3, device=device, dtype=positions.dtype)
            deformation = eye + strain
            positions = positions @ deformation.T
            cell = cell @ deformation.T
            edge_src, edge_dst = edge_index
            edge_vec = positions[edge_dst] - positions[edge_src]

        edge_src, edge_dst = edge_index

        # Edge features
        edge_length = edge_vec.norm(dim=-1)
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec,
            normalize=True, normalization='component')
        edge_radial = self.radial_basis(edge_length)

        # Node embedding
        node_fea = self.species_embedding(atomic_numbers)
        node_fea = self.node_proj(node_fea)

        # e3nn message passing
        for conv in self.convs:
            node_fea = conv(node_fea, edge_sh, edge_radial,
                            edge_src, edge_dst, num_nodes)

        # Extract scalar features
        scalar_idx = self._scalar_indices.to(device)
        scalar_fea = node_fea[:, scalar_idx]

        # GOM features
        if self.use_gom:
            if gom_data is not None:
                # Differentiable GOM: compute from live positions
                gom_eigvals = compute_gom_differentiable(
                    positions,
                    gom_data['nbr_idx'], gom_data['nbr_shifts'],
                    gom_data['nbr_rcov'], gom_data['n_sphere'],
                    gom_data['self_rcov'],
                    cells, batch_idx,
                    self.gom_cutoff,
                    self.natx,
                    orbital=self.orbital)
                gom_fp = gom_eigvals.float()
            elif gom_fp is None:
                gom_fp = self._compute_gom_onthefly(
                    positions, cell, atomic_numbers)
            if self.gom_proj is not None:
                gom_fea = self.gom_proj(gom_fp)
            else:
                gom_fea = gom_fp
            combined = torch.cat([scalar_fea, gom_fea], dim=-1)
        else:
            combined = scalar_fea

        # Per-atom energy
        atom_energy = self.energy_readout(combined).squeeze(-1)
        atom_energy = atom_energy + self.energy_shift(atomic_numbers).squeeze(-1)

        # Total energy (sum over all atoms or per-structure)
        if batch_idx is not None:
            energy_per_struct = scatter_sum(
                atom_energy.unsqueeze(-1), batch_idx, n_structures).squeeze(-1)
            energy = energy_per_struct.sum()
        else:
            energy = atom_energy.sum()
            energy_per_struct = energy.unsqueeze(0)

        result = {'energy': energy, 'atom_energy': atom_energy,
                  'energy_per_struct': energy_per_struct}

        # Forces via autograd
        if compute_forces:
            forces = -torch.autograd.grad(
                energy, positions,
                create_graph=self.training,
                retain_graph=True
            )[0]
            result['forces'] = forces

        # Stress via autograd (single structure only)
        if compute_stress:
            volume = torch.abs(torch.det(cell))
            stress_voigt = torch.autograd.grad(
                energy, strain,
                create_graph=self.training,
                retain_graph=True
            )[0]
            stress = torch.stack([
                stress_voigt[0, 0], stress_voigt[1, 1], stress_voigt[2, 2],
                stress_voigt[1, 2], stress_voigt[0, 2], stress_voigt[0, 1]
            ]) / volume
            result['stress'] = stress

        return result

    def _compute_gom_onthefly(self, positions, cell, atomic_numbers):
        """Compute GOM fingerprints on-the-fly (for inference/calculator)."""
        import numpy as np
        from .fp import get_lfp

        lat = cell.detach().cpu().numpy()
        rxyz = positions.detach().cpu().numpy()
        znucl_list = []
        types = []
        for z in atomic_numbers.cpu().tolist():
            if z not in znucl_list:
                znucl_list.append(z)
            types.append(znucl_list.index(z) + 1)
        cell_tuple = (lat, rxyz, np.array(types), np.array(znucl_list))
        fp_raw = get_lfp(cell_tuple, cutoff=6.0, natx=self.natx,
                         device='cpu', dtype=torch.float64).float()
        return fp_raw.to(positions.device)
