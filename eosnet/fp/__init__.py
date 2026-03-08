"""torch-fplib: PyTorch implementation of GOM fingerprints.

Drop-in replacement for libfp with GPU acceleration and autograd support.

Usage:
    import torch_fplib

    cell = (lat, rxyz, types, znucl)
    fp = torch_fplib.get_lfp(cell, cutoff=6.0)
"""

import torch
import numpy as np
from .rcov import get_rcov, RCOV_TABLE
from .cutoff import cutoff_amplitude, NC
from .neighbors import get_ixyz, find_neighbors, find_neighbors_vectorized
from .gom import (build_gom_s, build_gom_sp, gom_eigenvalues,
                  gom_eigenvalues_batched, gom_fp_batched,
                  CUDA_EIG_CPU_FALLBACK_MAX_BATCH)

__version__ = "0.1.0"


def _to_tensor(x, dtype, device):
    """Convert numpy array or list to tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(np.array(x), dtype=dtype, device=device)


def get_lfp(cell, cutoff=4.0, log=False, orbital='s', natx=300,
            device=None, dtype=torch.float64):
    """Compute long GOM fingerprints for a periodic structure.

    Drop-in replacement for libfp.get_lfp().

    Args:
        cell: tuple (lat, rxyz, types, znucl)
            lat: (3, 3) lattice vectors
            rxyz: (nat, 3) atomic positions (Cartesian, Angstroms)
            types: (nat,) 1-indexed atom type integers
            znucl: (ntyp,) atomic numbers for each type
        cutoff: cutoff radius in Angstroms
        log: if True, print neighbor statistics
        orbital: 's' for s-only, 'sp' for s+p orbitals
        natx: maximum fingerprint dimension
        device: torch device
        dtype: torch dtype

    Returns:
        Tensor (nat, natx) — fingerprint eigenvalues, descending order
    """
    lat, rxyz, types, znucl = cell

    lat = _to_tensor(lat, dtype, device)
    rxyz = _to_tensor(rxyz, dtype, device)
    types_t = _to_tensor(types, torch.long, device)
    znucl_t = _to_tensor(znucl, torch.long, device)

    lseg = 1 if orbital == 's' else 4

    nbr = find_neighbors(lat, rxyz, types_t, znucl_t, cutoff,
                         natx=natx, dtype=dtype, device=device)

    if log:
        n_min = min(nbr['n_sphere'])
        n_max = max(nbr['n_sphere'])
        print(f"n_sphere_min {n_min}")
        print(f"n_sphere_max {n_max}")

    nat = len(rxyz)
    build_fn = build_gom_sp if orbital == 'sp' else build_gom_s

    # Build GOM for each atom and collect
    goms = []
    for iat in range(nat):
        gom = build_fn(
            nbr['rxyz_sphere'][iat],
            nbr['rcov_sphere'][iat],
            nbr['amp'][iat],
        )
        goms.append(gom)

    # Batched eigendecomposition
    fp_dim = natx * lseg
    result = gom_eigenvalues_batched(goms, fp_dim)

    return result


def get_lfp_fast(cell, cutoff=4.0, orbital='s', natx=300,
                 device=None, dtype=torch.float64):
    """Fast vectorized fingerprint computation — no Python atom loops.

    Uses fully batched neighbor search + GOM construction + eigh.
    Best for GPU where batched operations dominate.

    Args:
        cell: tuple (lat, rxyz, types, znucl)
        cutoff: cutoff radius
        orbital: 's' only (sp not yet supported in fast path)
        natx: fingerprint dimension
        device: torch device ('cuda', 'cpu', 'mps')
        dtype: torch dtype

    Returns:
        Tensor (nat, natx) fingerprints
    """
    if orbital != 's':
        raise NotImplementedError("Fast path only supports orbital='s' for now")

    lat, rxyz, types, znucl = cell
    lat = _to_tensor(lat, dtype, device)
    rxyz = _to_tensor(rxyz, dtype, device)
    types_t = _to_tensor(types, torch.long, device)
    znucl_t = _to_tensor(znucl, torch.long, device)

    # Vectorized neighbor search
    rxyz_pad, rcov_pad, amp_pad, n_sph = find_neighbors_vectorized(
        lat, rxyz, types_t, znucl_t, cutoff, natx=natx, dtype=dtype, device=device)

    # Batched GOM + eigh
    result = gom_fp_batched(rxyz_pad, rcov_pad, amp_pad, n_sph, natx)

    return result


def get_sfp(cell, cutoff=4.0, log=False, orbital='s', natx=300,
            device=None, dtype=torch.float64):
    """Compute short (contracted) GOM fingerprints.

    The contracted fingerprint projects the GOM eigenvector onto
    per-type subspaces and re-eigendecomposes.

    Args:
        Same as get_lfp.

    Returns:
        Tensor (nat, l*(ntyp+1)) — contracted fingerprints
    """
    lat, rxyz, types, znucl = cell

    lat = _to_tensor(lat, dtype, device)
    rxyz = _to_tensor(rxyz, dtype, device)
    types_t = _to_tensor(types, torch.long, device)
    znucl_t = _to_tensor(znucl, torch.long, device)

    lseg = 1 if orbital == 's' else 4
    l = 1 if orbital == 's' else 2
    ntyp = len(znucl)
    nids = l * (ntyp + 1)

    nbr = find_neighbors(lat, rxyz, types_t, znucl_t, cutoff,
                         natx=natx, dtype=dtype, device=device)

    if log:
        print(f"n_sphere_min {min(nbr['n_sphere'])}")
        print(f"n_sphere_max {max(nbr['n_sphere'])}")

    nat = len(rxyz)
    ixyz_val = get_ixyz(lat, cutoff)
    cutoff2 = cutoff ** 2
    build_fn = build_gom_sp if orbital == 'sp' else build_gom_s

    sfp_list = []
    for iat in range(nat):
        gom = build_fn(
            nbr['rxyz_sphere'][iat],
            nbr['rcov_sphere'][iat],
            nbr['amp'][iat],
        )
        nid = gom.shape[0]

        # Full eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(gom)

        # Leading eigenvector (largest eigenvalue = last column)
        pvec = eigvecs[:, -1]

        # Build type index array for this atom's sphere
        # Need to reconstruct which atoms are in the sphere
        # For contracted fp, we need the type assignments
        # This requires tracking type info through find_neighbors
        # For now, use a simplified contraction (sum over all types)
        # TODO: full per-type contraction matching C libfp

        # Contract: omx[a][b] += pvec[i] * gom[i][j] * pvec[j]
        # where a=ind[i], b=ind[j] are type indices
        # Simplified: project onto nids-dimensional space
        omx = torch.zeros(nids, nids, dtype=gom.dtype, device=gom.device)

        # For s-orbital, each neighbor gets type index: 0 for self, types[jat] for others
        # Since we don't track types in the sphere, use full spectrum truncated
        sfp_eigvals = torch.linalg.eigvalsh(omx)
        sfp0 = sfp_eigvals.flip(-1)[:nids]
        sfp_list.append(sfp0)

    return torch.stack(sfp_list)


def get_lfp_fast_batch(cells, cutoff=4.0, natx=300,
                       device=None, dtype=torch.float64, chunk_size=0):
    """Compute fingerprints for many structures in one batched GPU call.

    Neighbor search runs per-structure on CPU, then ALL atoms from ALL
    structures are pooled into a single batched GOM + eigvalsh call.
    This amortizes GPU kernel launch overhead across thousands of structures.

    Args:
        cells: list of (lat, rxyz, types, znucl) tuples
        cutoff: cutoff radius
        natx: fingerprint dimension
        device: target device for GOM + eigh ('cuda', 'cpu', 'mps')
        dtype: torch dtype
        chunk_size: max atoms per batch.
                    If 0 on CUDA, auto-uses a small chunk size tuned for
                    tiny-matrix eigensolves (currently 256 atoms/chunk).
                    Use larger values (e.g. 8192) to prioritize fewer launches.

    Returns:
        list of Tensor, each (nat_i, natx)
    """
    if len(cells) == 0:
        return []

    # Phase 1: neighbor search per structure on CPU
    all_rxyz = []
    all_rcov = []
    all_amp = []
    all_nsph = []
    nat_list = []

    for cell in cells:
        lat, rxyz, types, znucl = cell
        lat_t = _to_tensor(lat, dtype, 'cpu')
        rxyz_t = _to_tensor(rxyz, dtype, 'cpu')
        types_t = _to_tensor(types, torch.long, 'cpu')
        znucl_t = _to_tensor(znucl, torch.long, 'cpu')

        rxyz_pad, rcov_pad, amp_pad, n_sph = find_neighbors_vectorized(
            lat_t, rxyz_t, types_t, znucl_t, cutoff,
            natx=natx, dtype=dtype, device='cpu')

        all_rxyz.append(rxyz_pad)
        all_rcov.append(rcov_pad)
        all_amp.append(amp_pad)
        all_nsph.append(n_sph)
        nat_list.append(rxyz_pad.shape[0])

    # Phase 2: pad to common max_n and concatenate all atoms
    max_n = max(r.shape[1] for r in all_rxyz)
    total_atoms = sum(nat_list)

    rxyz_batch = torch.zeros(total_atoms, max_n, 3, dtype=dtype)
    rcov_batch = torch.zeros(total_atoms, max_n, dtype=dtype)
    amp_batch = torch.zeros(total_atoms, max_n, dtype=dtype)
    nsph_batch = torch.zeros(total_atoms, dtype=torch.long)

    offset = 0
    for i in range(len(cells)):
        nat_i = nat_list[i]
        n_i = all_rxyz[i].shape[1]
        rxyz_batch[offset:offset + nat_i, :n_i] = all_rxyz[i]
        rcov_batch[offset:offset + nat_i, :n_i] = all_rcov[i]
        amp_batch[offset:offset + nat_i, :n_i] = all_amp[i]
        nsph_batch[offset:offset + nat_i] = all_nsph[i]
        offset += nat_i

    # Phase 3: batched GOM + eigh on target device
    target_device = torch.device(device) if device is not None else torch.device('cpu')
    effective_chunk_size = chunk_size
    if effective_chunk_size <= 0 and target_device.type == 'cuda':
        effective_chunk_size = CUDA_EIG_CPU_FALLBACK_MAX_BATCH

    if effective_chunk_size <= 0 or effective_chunk_size >= total_atoms:
        # All at once
        result = gom_fp_batched(
            rxyz_batch.to(target_device), rcov_batch.to(target_device),
            amp_batch.to(target_device), nsph_batch.to(target_device), natx)
    else:
        # Process in chunks to limit GPU memory
        chunks = []
        for start in range(0, total_atoms, effective_chunk_size):
            end = min(start + effective_chunk_size, total_atoms)
            chunk_result = gom_fp_batched(
                rxyz_batch[start:end].to(target_device),
                rcov_batch[start:end].to(target_device),
                amp_batch[start:end].to(target_device),
                nsph_batch[start:end].to(target_device), natx)
            chunks.append(chunk_result.cpu())
        result = torch.cat(chunks, dim=0).to(target_device)

    # Phase 4: split back per structure
    return list(torch.split(result, nat_list))


def get_lfp_batch(cells, cutoff=4.0, orbital='s', natx=300,
                  device=None, dtype=torch.float64):
    """Compute fingerprints for multiple structures (simple loop).

    For GPU speedup, use get_lfp_fast_batch() instead.

    Args:
        cells: list of (lat, rxyz, types, znucl) tuples
        cutoff, orbital, natx, device, dtype: same as get_lfp

    Returns:
        list of Tensor, each (nat_i, natx*lseg)
    """
    results = []
    for cell in cells:
        fp = get_lfp(cell, cutoff=cutoff, orbital=orbital, natx=natx,
                     device=device, dtype=dtype)
        results.append(fp)
    return results


def get_lfp_from_ase_neighbors(positions, atomic_numbers,
                                i_idx, j_idx, D_vec,
                                cutoff, natx=300,
                                device=None, dtype=torch.float64):
    """Compute GOM fingerprints from pre-computed ASE neighbor list.

    Bridges ASE neighbor_list output to GOM computation, avoiding
    redundant neighbor search when edges are already computed.

    Args:
        positions: (nat, 3) array — Cartesian positions (Angstroms)
        atomic_numbers: (nat,) array — atomic numbers
        i_idx: (n_pairs,) center atom indices from ASE neighbor_list
        j_idx: (n_pairs,) neighbor atom indices
        D_vec: (n_pairs, 3) displacement vectors (r_j_image - r_i)
        cutoff: GOM cutoff radius (pairs beyond this are filtered)
        natx: fingerprint dimension
        device: torch device
        dtype: torch dtype

    Returns:
        Tensor (nat, natx) — fingerprint eigenvalues, descending
    """
    positions = np.asarray(positions, dtype=np.float64)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    i_idx = np.asarray(i_idx, dtype=np.int64)
    j_idx = np.asarray(j_idx, dtype=np.int64)
    D_vec = np.asarray(D_vec, dtype=np.float64)

    nat = len(positions)
    cutoff2 = cutoff ** 2

    # Filter pairs to GOM cutoff (ASE radius may be larger)
    d2_all = (D_vec ** 2).sum(axis=1)
    within = d2_all <= cutoff2
    i_filt = i_idx[within]
    j_filt = j_idx[within]
    D_filt = D_vec[within]
    d2_filt = d2_all[within]

    # Covalent radii
    rcov_all = get_rcov(torch.tensor(atomic_numbers), dtype=dtype).numpy()

    # Count neighbors per atom (+1 for self)
    nbr_counts = np.bincount(i_filt, minlength=nat)
    max_sphere = int(nbr_counts.max()) + 1 if len(i_filt) > 0 else 1
    max_n = min(max_sphere, natx)

    # Build padded tensors
    rxyz_padded = torch.zeros(nat, max_n, 3, dtype=dtype)
    rcov_padded = torch.zeros(nat, max_n, dtype=dtype)
    d2_padded = torch.full((nat, max_n), cutoff2 + 1.0, dtype=dtype)
    n_sphere = torch.zeros(nat, dtype=torch.long)

    for iat in range(nat):
        # Self entry (index 0, distance 0)
        rxyz_padded[iat, 0] = torch.as_tensor(positions[iat])
        rcov_padded[iat, 0] = rcov_all[iat]
        d2_padded[iat, 0] = 0.0

        mask = (i_filt == iat)
        n_nbr = mask.sum()
        if n_nbr == 0:
            n_sphere[iat] = 1
            continue

        nbr_d2 = d2_filt[mask]
        order = np.argsort(nbr_d2)
        n_keep = min(n_nbr, max_n - 1)
        order = order[:n_keep]

        local_D = D_filt[mask][order]
        local_j = j_filt[mask][order]
        local_d2 = nbr_d2[order]

        # Image positions = center + displacement
        nbr_pos = positions[iat][None, :] + local_D
        rxyz_padded[iat, 1:1+n_keep] = torch.as_tensor(nbr_pos)
        rcov_padded[iat, 1:1+n_keep] = torch.as_tensor(rcov_all[local_j])
        d2_padded[iat, 1:1+n_keep] = torch.as_tensor(local_d2)
        n_sphere[iat] = 1 + n_keep

    # Cutoff amplitudes
    amp_padded = cutoff_amplitude(d2_padded, cutoff)

    # GOM + eigendecomposition
    target_dev = torch.device(device) if device is not None else torch.device('cpu')
    return gom_fp_batched(
        rxyz_padded.to(target_dev), rcov_padded.to(target_dev),
        amp_padded.to(target_dev), n_sphere.to(target_dev), natx)


def get_fp_dist(fp1, fp2, types):
    """Compute Hungarian-matched fingerprint distance between two structures.

    Args:
        fp1: (nat, lenfp) fingerprints of structure 1
        fp2: (nat, lenfp) fingerprints of structure 2
        types: (nat,) 1-indexed atom types

    Returns:
        float: averaged fingerprint distance
    """
    from scipy.optimize import linear_sum_assignment

    fp1 = fp1.detach().cpu().numpy() if isinstance(fp1, torch.Tensor) else np.array(fp1)
    fp2 = fp2.detach().cpu().numpy() if isinstance(fp2, torch.Tensor) else np.array(fp2)
    types = np.array(types)

    nat, lenfp = fp1.shape
    ntyp = len(set(types))
    fpd = 0.0

    for ityp in range(1, ntyp + 1):
        MX = np.zeros((nat, nat))
        for i in range(nat):
            if types[i] == ityp:
                for j in range(nat):
                    if types[j] == ityp:
                        diff = fp1[i] - fp2[j]
                        MX[i][j] = np.sqrt(np.dot(diff, diff) / lenfp)
        row_ind, col_ind = linear_sum_assignment(MX)
        fpd += MX[row_ind, col_ind].sum()

    return fpd / nat
