"""EOSNet MLIP: e3nn equivariant backbone + GOM fingerprints for energy/forces.

Per-atom energy prediction with forces via autograd.
GOM eigenvalues provide orbital-overlap inductive bias as additional node features.
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

from .model import RadialBasis, scatter_sum
from .fp.gom import build_gom_s_batched, _eigvalsh_batched_adaptive
from .fp.rcov import get_rcov
from .fp.cutoff import cutoff_amplitude
from .fp.neighbors import get_ixyz


class GOMFeatures(nn.Module):
    """Compute differentiable GOM fingerprint features for all atoms.

    Uses torch-fplib's batched GOM construction + eigendecomposition.
    Output: per-atom GOM eigenvalue vector (l=0 scalars).
    """

    def __init__(self, gom_cutoff=6.0, natx=64, n_gom_features=32):
        super().__init__()
        self.gom_cutoff = gom_cutoff
        self.natx = natx
        self.n_gom_features = n_gom_features
        # Linear projection from raw eigenvalues to fixed-size feature
        self.proj = nn.Linear(natx, n_gom_features)

    def forward(self, positions, cell, atomic_numbers):
        """Compute GOM features for all atoms in a structure.

        Args:
            positions: (nat, 3) atomic positions, requires_grad=True
            cell: (3, 3) lattice vectors
            atomic_numbers: (nat,) integer atomic numbers

        Returns:
            (nat, n_gom_features) GOM feature vectors
        """
        nat = positions.shape[0]
        device = positions.device
        dtype = positions.dtype

        # Get covalent radii
        rcov_all = get_rcov(atomic_numbers, device=device, dtype=dtype)

        # Neighbor search (periodic images within GOM cutoff)
        cutoff2 = self.gom_cutoff ** 2
        ixyz = get_ixyz(cell, self.gom_cutoff)

        # Build shift vectors
        arange = torch.arange(-ixyz, ixyz + 1, device=device, dtype=dtype)
        grid = torch.stack(
            torch.meshgrid(arange, arange, arange, indexing='ij'), dim=-1)
        shifts = grid.reshape(-1, 3)
        shift_vecs = shifts @ cell  # (n_shifts, 3)

        # All images: (n_shifts, nat, 3)
        images = positions[None, :, :] + shift_vecs[:, None, :]

        # Distances: (nat, n_shifts*nat)
        d_vec = positions[:, None, None, :] - images[None, :, :, :]
        d2 = (d_vec ** 2).sum(-1)  # (nat, n_shifts, nat)
        n_shifts = shifts.shape[0]
        d2_flat = d2.reshape(nat, n_shifts * nat)
        images_flat = images.reshape(n_shifts * nat, 3)
        rcov_flat = rcov_all.repeat(n_shifts)

        # Mask within cutoff
        d2_masked = d2_flat.clone()
        d2_masked[d2_flat > cutoff2] = float('inf')

        # Sort and take top natx neighbors
        sorted_d2, sort_idx = d2_masked.sort(dim=1)
        max_n = min(self.natx, (sorted_d2 < float('inf')).sum(dim=1).max().item())
        sorted_d2 = sorted_d2[:, :max_n]
        sort_idx = sort_idx[:, :max_n]

        # Gather positions and rcov
        rxyz_padded = images_flat[sort_idx.reshape(-1)].reshape(nat, max_n, 3)
        rcov_padded = rcov_flat[sort_idx.reshape(-1)].reshape(nat, max_n)

        # Cutoff amplitudes
        amp_padded = cutoff_amplitude(sorted_d2, self.gom_cutoff)

        # Count actual neighbors
        n_sphere = (sorted_d2 < cutoff2 + 1e-6).sum(dim=1).clamp(max=self.natx)

        # Build GOMs and eigendecompose (batched)
        goms = build_gom_s_batched(rxyz_padded, rcov_padded, amp_padded, n_sphere)
        eigvals = _eigvalsh_batched_adaptive(goms)  # (nat, max_n), ascending
        eigvals = eigvals.flip(-1)  # descending

        # Pad/trim to natx
        if max_n >= self.natx:
            eigvals = eigvals[:, :self.natx]
        else:
            pad = torch.zeros(nat, self.natx - max_n,
                              device=device, dtype=dtype)
            eigvals = torch.cat([eigvals, pad], dim=1)

        # Zero padding artifacts
        cols = torch.arange(self.natx, device=device).unsqueeze(0)
        keep = cols < n_sphere.clamp(max=self.natx).unsqueeze(1)
        eigvals = eigvals * keep.to(dtype)

        # Project to feature dimension
        gom_fea = self.proj(eigvals.float())
        return gom_fea


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
                 use_gom=True):
        super().__init__()

        self.radial_cutoff = radial_cutoff
        self.use_gom = use_gom
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_ell)

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
        if use_gom:
            self.gom = GOMFeatures(
                gom_cutoff=gom_cutoff, natx=natx,
                n_gom_features=n_gom_features)
            readout_in = num_output_scalars + n_gom_features
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
                compute_forces=True, compute_stress=False):
        """Forward pass.

        Args:
            atomic_numbers: (nat,) integer atomic numbers
            positions: (nat, 3) Cartesian positions
            cell: (3, 3) lattice vectors
            edge_index: (2, E) [src, dst] edge indices
            edge_vec: (E, 3) edge displacement vectors
            compute_forces: if True, compute forces via autograd
            compute_stress: if True, compute stress via autograd

        Returns:
            dict with 'energy', 'forces' (optional), 'stress' (optional)
        """
        num_nodes = positions.shape[0]
        device = positions.device

        # Enable gradients for force computation
        if compute_forces and not positions.requires_grad:
            positions.requires_grad_(True)

        # Strain for stress computation
        if compute_stress:
            # Apply symmetric strain to cell and positions
            strain = torch.zeros(3, 3, device=device, dtype=positions.dtype,
                                 requires_grad=True)
            eye = torch.eye(3, device=device, dtype=positions.dtype)
            deformation = eye + strain
            positions = positions @ deformation.T
            cell = cell @ deformation.T
            # Recompute edge vectors under strain
            edge_src, edge_dst = edge_index
            edge_vec = positions[edge_dst] - positions[edge_src]
            # TODO: apply PBC shift vectors under strain if needed

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
            gom_fea = self.gom(positions, cell, atomic_numbers)
            combined = torch.cat([scalar_fea, gom_fea], dim=-1)
        else:
            combined = scalar_fea

        # Per-atom energy
        atom_energy = self.energy_readout(combined).squeeze(-1)
        atom_energy = atom_energy + self.energy_shift(atomic_numbers).squeeze(-1)

        # Total energy
        energy = atom_energy.sum()

        result = {'energy': energy, 'atom_energy': atom_energy}

        # Forces via autograd
        if compute_forces:
            forces = -torch.autograd.grad(
                energy, positions,
                create_graph=self.training,
                retain_graph=True
            )[0]
            result['forces'] = forces

        # Stress via autograd
        if compute_stress:
            volume = torch.abs(torch.det(cell))
            stress_voigt = torch.autograd.grad(
                energy, strain,
                create_graph=self.training,
                retain_graph=True
            )[0]
            # Convert to Voigt notation: xx, yy, zz, yz, xz, xy
            stress = torch.stack([
                stress_voigt[0, 0], stress_voigt[1, 1], stress_voigt[2, 2],
                stress_voigt[1, 2], stress_voigt[0, 2], stress_voigt[0, 1]
            ]) / volume
            result['stress'] = stress

        return result
