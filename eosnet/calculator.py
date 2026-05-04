"""ASE Calculator for EOSNet MLIP."""

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from .mlip import EOSNetMLIP
from .fp.rcov import get_rcov


class EOSNetCalculator(Calculator):
    """ASE Calculator wrapping EOSNet MLIP.

    Usage:
        calc = EOSNetCalculator('mlip_best.pth')
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

    For large systems (>1000 atoms), enable gradient checkpointing:
        calc = EOSNetCalculator('mlip_best.pth', gradient_checkpointing=True)
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path, device=None,
                 gradient_checkpointing=False,
                 use_cuequivariance=False, **kwargs):
        super().__init__(**kwargs)

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device,
                          weights_only=False)
        model_args = ckpt.get('args', {})

        # Build model
        self.model = EOSNetMLIP(
            irreps_hidden=model_args.get('irreps_hidden', '32x0e+16x1o+8x2e'),
            max_ell=model_args.get('max_ell', 2),
            n_conv=model_args.get('n_conv', 3),
            num_radial_basis=model_args.get('num_radial_basis', 16),
            radial_cutoff=model_args.get('radial_cutoff', 5.0),
            n_gom_features=model_args.get('n_gom_features', 32),
            gom_cutoff=model_args.get('gom_cutoff', 6.0),
            natx=model_args.get('natx', 32),
            energy_hidden=model_args.get('energy_hidden', 128),
            use_gom=not model_args.get('no_gom', False),
            gradient_checkpointing=gradient_checkpointing,
            use_cuequivariance=use_cuequivariance,
        )
        # strict=False allows loading e3nn-trained weights into cuequivariance model
        # (TP buffers like w3j differ between backends but all learnable params match)
        self.model.load_state_dict(ckpt['model'], strict=not use_cuequivariance)
        self.model.to(self.device)
        self.model.eval()
        self.cutoff = model_args.get('radial_cutoff', 5.0)
        self.gom_cutoff = model_args.get('gom_cutoff', 6.0)
        self.natx = model_args.get('natx', 32)
        self.use_gom = not model_args.get('no_gom', False)

    def _build_gom_data(self, atoms, positions, cell, batch_idx):
        """Build differentiable GOM neighbor topology on GPU.

        Precomputes neighbor indices and shifts (CPU), then transfers to GPU.
        Actual GOM eigenvalues computed differentiably in the forward pass
        using live positions — so forces include GOM contribution.
        """
        nat = len(atoms)
        gom_i, gom_j, gom_D, gom_S = neighbor_list(
            'ijDS', atoms, self.gom_cutoff)

        rcov_all = get_rcov(
            torch.tensor(atoms.numbers), dtype=torch.float32).numpy()

        nbr_counts = np.bincount(gom_i, minlength=nat)
        max_sphere = int(nbr_counts.max()) + 1 if len(gom_i) > 0 else 1
        max_n = min(max_sphere, self.natx)

        nbr_idx = np.zeros((nat, max_n), dtype=np.int64)
        nbr_shifts = np.zeros((nat, max_n, 3), dtype=np.float32)
        nbr_rcov = np.zeros((nat, max_n), dtype=np.float32)
        n_sphere = np.zeros(nat, dtype=np.int64)

        for iat in range(nat):
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

        dev = self.device
        return {
            'nbr_idx': torch.tensor(nbr_idx, dtype=torch.long, device=dev),
            'nbr_shifts': torch.tensor(nbr_shifts, dtype=torch.float32, device=dev),
            'nbr_rcov': torch.tensor(nbr_rcov, dtype=torch.float32, device=dev),
            'n_sphere': torch.tensor(n_sphere, dtype=torch.long, device=dev),
            'self_rcov': torch.tensor(rcov_all, dtype=torch.float32, device=dev),
        }

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Build graph — use shift vectors so edge_vec is differentiable
        i_idx, j_idx, S_vec = neighbor_list('ijS', self.atoms, self.cutoff)

        atomic_numbers = torch.tensor(
            self.atoms.numbers, dtype=torch.long, device=self.device)
        positions = torch.tensor(
            self.atoms.get_positions(), dtype=torch.float32,
            device=self.device).requires_grad_(True)
        cell = torch.tensor(
            np.array(self.atoms.cell), dtype=torch.float32,
            device=self.device)
        edge_index = torch.stack([
            torch.tensor(i_idx, dtype=torch.long, device=self.device),
            torch.tensor(j_idx, dtype=torch.long, device=self.device),
        ])
        edge_shifts = torch.tensor(S_vec, dtype=torch.float32, device=self.device)

        compute_stress = 'stress' in properties

        # Build GOM neighbor topology (differentiable path)
        gom_data = None
        cells = None
        batch_idx = torch.zeros(len(self.atoms), dtype=torch.long, device=self.device)
        if self.use_gom:
            gom_data = self._build_gom_data(
                self.atoms, positions, cell, batch_idx)
            cells = cell.unsqueeze(0)

        with torch.enable_grad():
            edge_src, edge_dst = edge_index
            shift_cart = edge_shifts @ cell
            edge_vec = positions[edge_dst] - positions[edge_src] + shift_cart

            output = self.model(
                atomic_numbers, positions, cell,
                edge_index, edge_vec,
                compute_forces='forces' in properties,
                compute_stress=compute_stress,
                batch_idx=batch_idx,
                n_structures=1,
                gom_data=gom_data,
                cells=cells,
            )

        self.results['energy'] = output['energy'].item()
        if 'forces' in output:
            self.results['forces'] = output['forces'].detach().cpu().numpy()
        if 'stress' in output:
            self.results['stress'] = output['stress'].detach().cpu().numpy()
