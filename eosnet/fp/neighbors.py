"""Periodic neighbor search for GOM fingerprints.

Enumerates periodic images within cutoff for each atom.
"""

import torch
import math
from .rcov import get_rcov
from .cutoff import cutoff_amplitude


def get_ixyz(lat: torch.Tensor, cutoff: float) -> int:
    """Compute periodic image range from lattice vectors.

    Uses lattice metric eigenvalues to determine how many periodic
    images are needed to capture all neighbors within cutoff.

    Args:
        lat: (3, 3) lattice matrix (rows are lattice vectors)
        cutoff: cutoff radius

    Returns:
        ixyz: integer, search range in each lattice direction
    """
    lat_cpu = lat.detach().cpu().to(torch.float64)
    lat2 = lat_cpu @ lat_cpu.T  # metric tensor
    eigvals = torch.linalg.eigvalsh(lat2)
    # Smallest lattice vector length ~ 1/sqrt(max eigenvalue of inverse)
    # = sqrt(1/min eigenvalue of metric)
    # But fplib2 uses: ixyz = int(sqrt(1/max(eigvals)) * cutoff) + 1
    # This is the reciprocal: shortest reciprocal-space extent
    # C libfp uses w[0] (min eigenvalue from dsyev, ascending order).
    # Min eigenvalue = shortest lattice direction → needs most images.
    ixyz = int(math.sqrt(1.0 / eigvals.min().item()) * cutoff) + 1
    return ixyz


def find_neighbors(lat, rxyz, types, znucl, cutoff, natx=300,
                   dtype=torch.float64, device=None):
    """Find all neighbors within cutoff for each atom, including periodic images.

    Args:
        lat: (3, 3) lattice vectors (Cartesian, Angstroms)
        rxyz: (nat, 3) atomic positions (Cartesian, Angstroms)
        types: (nat,) 1-indexed atom type integers
        znucl: (ntyp,) atomic numbers for each type
        cutoff: cutoff radius in Angstroms
        natx: maximum neighbors per atom
        dtype: float dtype
        device: target device

    Returns:
        dict with keys for each atom iat:
            'rxyz_sphere': list of (n_sphere, 3) tensors — neighbor positions
            'rcov_sphere': list of (n_sphere,) tensors — covalent radii
            'amp': list of (n_sphere,) tensors — cutoff amplitudes
            'n_sphere': list of int — neighbor counts
    """
    lat = torch.as_tensor(lat, dtype=dtype, device=device)
    rxyz = torch.as_tensor(rxyz, dtype=dtype, device=device)
    types = torch.as_tensor(types, dtype=torch.long, device=device)
    znucl = torch.as_tensor(znucl, dtype=torch.long, device=device)

    nat = len(rxyz)
    cutoff2 = cutoff ** 2
    ixyz = get_ixyz(lat, cutoff)

    # Precompute atomic numbers for each atom
    atom_Z = znucl[types - 1]  # types are 1-indexed
    rcov_all = get_rcov(atom_Z, device=device, dtype=dtype)

    # Build translation vectors
    shifts = []
    for ix in range(-ixyz, ixyz + 1):
        for iy in range(-ixyz, ixyz + 1):
            for iz in range(-ixyz, ixyz + 1):
                shifts.append([ix, iy, iz])
    shifts = torch.tensor(shifts, dtype=dtype, device=device)  # (n_shifts, 3)
    shift_vecs = shifts @ lat  # (n_shifts, 3)

    # For each atom, find all neighbors
    all_rxyz_sphere = []
    all_rcov_sphere = []
    all_amp = []
    all_n_sphere = []

    for iat in range(nat):
        xi = rxyz[iat]  # (3,)

        # All images of all atoms: (n_shifts, nat, 3)
        images = rxyz[None, :, :] + shift_vecs[:, None, :]  # (n_shifts, nat, 3)
        d_vec = images - xi[None, None, :]  # (n_shifts, nat, 3)
        d2 = (d_vec ** 2).sum(-1)  # (n_shifts, nat)

        # Mask: within cutoff
        mask = d2 <= cutoff2  # (n_shifts, nat)

        # Flatten and gather
        flat_d2 = d2[mask]
        flat_pos = images.reshape(-1, 3)[mask.reshape(-1)]

        # Covalent radii for neighbors
        # jat indices from mask
        jat_indices = torch.arange(nat, device=device).unsqueeze(0).expand(len(shifts), -1)
        flat_jat = jat_indices[mask]
        flat_rcov = rcov_all[flat_jat]

        n_sphere = flat_d2.shape[0]
        if n_sphere > natx:
            # Sort by distance and keep closest natx
            _, sort_idx = flat_d2.sort()
            sort_idx = sort_idx[:natx]
            flat_d2 = flat_d2[sort_idx]
            flat_pos = flat_pos[sort_idx]
            flat_rcov = flat_rcov[sort_idx]
            n_sphere = natx

        # Compute amplitudes
        amp = cutoff_amplitude(flat_d2, cutoff)

        all_rxyz_sphere.append(flat_pos)
        all_rcov_sphere.append(flat_rcov)
        all_amp.append(amp)
        all_n_sphere.append(n_sphere)

    return {
        'rxyz_sphere': all_rxyz_sphere,
        'rcov_sphere': all_rcov_sphere,
        'amp': all_amp,
        'n_sphere': all_n_sphere,
    }


def find_neighbors_vectorized(lat, rxyz, types, znucl, cutoff, natx=300,
                               dtype=torch.float64, device=None):
    """Fully vectorized neighbor search — no Python atom loop.

    Returns padded tensors suitable for batched GOM construction.

    Args:
        Same as find_neighbors.

    Returns:
        rxyz_padded: (nat, max_n, 3) padded neighbor positions
        rcov_padded: (nat, max_n) padded covalent radii
        amp_padded: (nat, max_n) padded amplitudes (0 for padding)
        n_sphere: (nat,) neighbor counts
    """
    lat = torch.as_tensor(lat, dtype=dtype, device=device)
    rxyz = torch.as_tensor(rxyz, dtype=dtype, device=device)
    types = torch.as_tensor(types, dtype=torch.long, device=device)
    znucl = torch.as_tensor(znucl, dtype=torch.long, device=device)

    nat = len(rxyz)
    cutoff2 = cutoff ** 2
    ixyz = get_ixyz(lat, cutoff)

    atom_Z = znucl[types - 1]
    rcov_all = get_rcov(atom_Z, device=device, dtype=dtype)

    # Build all shift vectors
    arange = torch.arange(-ixyz, ixyz + 1, device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(arange, arange, arange, indexing='ij'), dim=-1)
    shifts = grid.reshape(-1, 3)  # (n_shifts, 3)
    shift_vecs = shifts @ lat  # (n_shifts, 3)

    # All images of all atoms: (n_shifts, nat, 3)
    images = rxyz[None, :, :] + shift_vecs[:, None, :]

    # Distance from every atom to every image: (nat, n_shifts, nat)
    # d[iat, s, jat] = |rxyz[iat] - images[s, jat]|^2
    d_vec = rxyz[:, None, None, :] - images[None, :, :, :]  # (nat, n_shifts, nat, 3)
    d2 = (d_vec ** 2).sum(-1)  # (nat, n_shifts, nat)

    # Flatten shift and jat dims: (nat, n_shifts*nat)
    n_shifts = shifts.shape[0]
    d2_flat = d2.reshape(nat, n_shifts * nat)
    images_flat = images.reshape(n_shifts * nat, 3)

    # rcov for each image: repeat rcov_all for each shift
    rcov_flat = rcov_all.repeat(n_shifts)  # (n_shifts * nat,)

    # Mask within cutoff
    mask = d2_flat <= cutoff2  # (nat, n_shifts*nat)

    # Count neighbors per atom
    n_sphere = mask.sum(dim=1)  # (nat,)
    max_n = min(n_sphere.max().item(), natx)

    # For each atom, gather the closest neighbors (sorted by distance)
    # Set masked-out distances to infinity for sorting
    d2_masked = d2_flat.clone()
    d2_masked[~mask] = float('inf')

    # Sort each atom's neighbors by distance
    sorted_d2, sort_idx = d2_masked.sort(dim=1)  # (nat, n_shifts*nat)

    # Take top max_n
    sorted_d2 = sorted_d2[:, :max_n]  # (nat, max_n)
    sort_idx = sort_idx[:, :max_n]  # (nat, max_n)

    # Gather positions and rcov
    rxyz_padded = images_flat[sort_idx.reshape(-1)].reshape(nat, max_n, 3)
    rcov_padded = rcov_flat[sort_idx.reshape(-1)].reshape(nat, max_n)

    # Amplitudes (zero for padding — inf distances give zero amplitude)
    amp_padded = cutoff_amplitude(sorted_d2, cutoff)  # (nat, max_n)

    # Clamp n_sphere to natx
    n_sphere = n_sphere.clamp(max=natx)

    return rxyz_padded, rcov_padded, amp_padded, n_sphere
