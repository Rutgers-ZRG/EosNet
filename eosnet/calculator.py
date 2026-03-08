"""ASE Calculator for EOSNet MLIP."""

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from .mlip import EOSNetMLIP


class EOSNetCalculator(Calculator):
    """ASE Calculator wrapping EOSNet MLIP.

    Usage:
        calc = EOSNetCalculator('mlip_best.pth')
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path, device=None, **kwargs):
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
            natx=model_args.get('natx', 64),
            energy_hidden=model_args.get('energy_hidden', 128),
            use_gom=not model_args.get('no_gom', False),
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()
        self.cutoff = model_args.get('radial_cutoff', 5.0)

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Build graph
        i_idx, j_idx, D_vec = neighbor_list('ijD', self.atoms, self.cutoff)

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
        edge_vec = torch.tensor(
            D_vec, dtype=torch.float32, device=self.device)

        compute_stress = 'stress' in properties

        with torch.no_grad():
            # Need grad for forces
            with torch.enable_grad():
                output = self.model(
                    atomic_numbers, positions, cell,
                    edge_index, edge_vec,
                    compute_forces='forces' in properties,
                    compute_stress=compute_stress,
                )

        self.results['energy'] = output['energy'].item()
        if 'forces' in output:
            self.results['forces'] = output['forces'].detach().cpu().numpy()
        if 'stress' in output:
            self.results['stress'] = output['stress'].detach().cpu().numpy()
