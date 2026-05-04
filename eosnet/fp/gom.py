"""Gaussian Overlap Matrix (GOM) construction and eigendecomposition.

Core fingerprint computation: build overlap matrices from Gaussian-type
orbitals centered on atoms, then extract eigenvalue spectrum.

Reference: Zhu et al., J. Chem. Phys. 144, 034203 (2016)
"""

import torch

# cuSOLVER batched eigensolver threshold: fast for n <= ~32, slow above.
# Benchmark on L40S (Apr 2026): 20x20 GPU 34x faster than CPU fallback,
# 40x40 GPU 16x slower.  32 is a safe threshold.
CUDA_EIG_GPU_MAX_N = 32


def _eigh_safe(mat):
    """Eigendecomposition with CPU fallback for MPS."""
    if mat.device.type == 'mps':
        result = torch.linalg.eigh(mat.cpu())
        return result.eigenvalues.to(mat.device), result.eigenvectors.to(mat.device)
    result = torch.linalg.eigh(mat)
    return result.eigenvalues, result.eigenvectors


def _eigvalsh_safe(mat):
    """Eigenvalues with CPU fallback for MPS."""
    if mat.device.type == 'mps':
        return torch.linalg.eigvalsh(mat.cpu()).to(mat.device)
    return torch.linalg.eigvalsh(mat)


def _eigvalsh_batched_adaptive(mats: torch.Tensor) -> torch.Tensor:
    """Batched eigenvalues with adaptive backend.

    For CUDA: cuSOLVER batched eigensolver is fast for small matrices
    (n <= ~32) but catastrophically slow for larger ones.  We stay on
    GPU when n <= CUDA_EIG_GPU_MAX_N (fast path) and fall back to CPU
    LAPACK otherwise.
    """
    if mats.device.type == 'mps':
        return torch.linalg.eigvalsh(mats.cpu()).to(mats.device)

    if mats.device.type == 'cuda':
        _, n, _ = mats.shape
        if n <= CUDA_EIG_GPU_MAX_N:
            # Small matrices: GPU batched eigensolver is efficient
            return torch.linalg.eigvalsh(mats)
        else:
            # Large matrices: cuSOLVER is very slow, CPU LAPACK wins
            return torch.linalg.eigvalsh(mats.cpu()).to(mats.device)

    return torch.linalg.eigvalsh(mats)


def build_gom_s(rxyz_sphere: torch.Tensor,
                rcov_sphere: torch.Tensor,
                amp: torch.Tensor) -> torch.Tensor:
    """Build GOM for s-orbitals only (lseg=1).

    Overlap integral: O_ij = [4r * rcov_i * rcov_j]^(3/2) * exp(-r * d_ij^2)
    where r = 0.5 / (rcov_i^2 + rcov_j^2)

    Weighted: aom_ij = O_ij * amp_i * amp_j

    Args:
        rxyz_sphere: (n, 3) positions of atoms in the neighbor sphere
        rcov_sphere: (n,) covalent radii
        amp: (n,) cutoff amplitudes

    Returns:
        (n, n) symmetric overlap matrix
    """
    n = rxyz_sphere.shape[0]
    d = rxyz_sphere[:, None, :] - rxyz_sphere[None, :, :]  # (n, n, 3)
    d2 = (d ** 2).sum(-1)  # (n, n)

    rc_i = rcov_sphere[:, None]  # (n, 1)
    rc_j = rcov_sphere[None, :]  # (1, n)
    r = 0.5 / (rc_i ** 2 + rc_j ** 2)  # (n, n)

    overlap = (4.0 * r * rc_i * rc_j).sqrt() ** 3 * torch.exp(-d2 * r)
    gom = overlap * amp[:, None] * amp[None, :]
    return gom


def build_gom_sp(rxyz_sphere: torch.Tensor,
                 rcov_sphere: torch.Tensor,
                 amp: torch.Tensor) -> torch.Tensor:
    """Build GOM for s+p orbitals (lseg=4).

    4n x 4n matrix with s-s, s-p, p-s, p-p blocks per atom pair.

    Args:
        rxyz_sphere: (n, 3) positions
        rcov_sphere: (n,) covalent radii
        amp: (n,) cutoff amplitudes

    Returns:
        (4n, 4n) symmetric overlap matrix
    """
    n = rxyz_sphere.shape[0]
    d = rxyz_sphere[:, None, :] - rxyz_sphere[None, :, :]  # (n, n, 3)
    d2 = (d ** 2).sum(-1)  # (n, n)

    rc_i = rcov_sphere[:, None]  # (n, 1)
    rc_j = rcov_sphere[None, :]  # (1, n)
    r = 0.5 / (rc_i ** 2 + rc_j ** 2)  # (n, n)
    amp_ij = amp[:, None] * amp[None, :]  # (n, n)

    # s-s overlap: same as build_gom_s
    sji = (4.0 * rc_i * rc_j).sqrt() ** 3 * torch.exp(-d2 * r)
    ss = (4.0 * r * rc_i * rc_j).sqrt() ** 3 * torch.exp(-d2 * r) * amp_ij

    # Build 4n x 4n matrix
    om = rxyz_sphere.new_zeros(4 * n, 4 * n)

    # s-s block: om[4i, 4j]
    om[0::4, 0::4] = ss

    # s-p block: om[4i, 4j+k] for k=1,2,3
    stv_sp = (8.0 ** 0.5) * rc_j * r * sji * amp_ij  # (n, n)
    for k in range(3):
        om[0::4, (k + 1)::4] = stv_sp * d[:, :, k]

    # p-s block: om[4i+k, 4j] for k=1,2,3
    stv_ps = -(8.0 ** 0.5) * rc_i * r * sji * amp_ij  # (n, n)
    for k in range(3):
        om[(k + 1)::4, 0::4] = stv_ps * d[:, :, k]

    # p-p block: om[4i+k1, 4j+k2]
    stv_pp = -8.0 * rc_i * rc_j * r * r * sji * amp_ij  # (n, n)
    inv_2r = 0.5 / r  # (n, n)
    for k1 in range(3):
        for k2 in range(3):
            val = stv_pp * d[:, :, k1] * d[:, :, k2]
            if k1 == k2:
                val = val - stv_pp * inv_2r
            om[(k1 + 1)::4, (k2 + 1)::4] = val

    return om


def gom_eigenvalues(gom: torch.Tensor, natx: int) -> torch.Tensor:
    """Compute eigenvalues of GOM matrix, sorted descending, padded to natx.

    Args:
        gom: (n, n) symmetric overlap matrix
        natx: output dimension (pad with zeros)

    Returns:
        (natx,) eigenvalue tensor, largest first
    """
    eigvals = _eigvalsh_safe(gom)  # ascending
    # Reverse to descending (match C libfp convention)
    eigvals = eigvals.flip(-1)
    # Pad to natx
    n = eigvals.shape[0]
    if n < natx:
        pad = eigvals.new_zeros(natx - n)
        eigvals = torch.cat([eigvals, pad])
    elif n > natx:
        eigvals = eigvals[:natx]
    return eigvals


def gom_eigenvalues_batched(goms: list, natx: int) -> torch.Tensor:
    """Batched eigendecomposition of multiple GOM matrices.

    Pads all matrices to the same size and runs a single batched eigh.

    Args:
        goms: list of (n_i, n_i) GOM matrices
        natx: output fingerprint dimension

    Returns:
        (batch, natx) eigenvalue tensor
    """
    if len(goms) == 0:
        return torch.zeros(0, natx)

    sizes = [g.shape[0] for g in goms]
    max_size = max(sizes)
    device = goms[0].device
    dtype = goms[0].dtype

    # Pad to max_size and stack
    batched = torch.zeros(len(goms), max_size, max_size, device=device, dtype=dtype)
    for i, g in enumerate(goms):
        n = g.shape[0]
        batched[i, :n, :n] = g

    # Batched eigendecomposition
    all_eigvals = _eigvalsh_batched_adaptive(batched)  # (batch, max_size), ascending

    # Reverse to descending
    all_eigvals = all_eigvals.flip(-1)

    # Trim/pad to natx
    if max_size >= natx:
        result = all_eigvals[:, :natx]
    else:
        pad = torch.zeros(len(goms), natx - max_size, device=device, dtype=dtype)
        result = torch.cat([all_eigvals, pad], dim=1)

    # Zero out padding artifacts (eigenvalues from padded zero rows)
    for i, n in enumerate(sizes):
        if n < natx:
            result[i, n:] = 0.0

    return result


def build_gom_s_batched(rxyz_padded, rcov_padded, amp_padded, n_sphere):
    """Build s-orbital GOMs for all atoms at once — fully batched.

    Args:
        rxyz_padded: (B, max_n, 3) padded neighbor positions
        rcov_padded: (B, max_n) padded covalent radii
        amp_padded: (B, max_n) padded amplitudes (0 for padding)
        n_sphere: (B,) actual neighbor counts

    Returns:
        (B, max_n, max_n) batched GOM matrices
    """
    # (B, max_n, 1, 3) - (B, 1, max_n, 3) = (B, max_n, max_n, 3)
    d = rxyz_padded[:, :, None, :] - rxyz_padded[:, None, :, :]
    d2 = (d ** 2).sum(-1)  # (B, max_n, max_n)

    rc_i = rcov_padded[:, :, None]  # (B, max_n, 1)
    rc_j = rcov_padded[:, None, :]  # (B, 1, max_n)

    # Avoid division by zero for padding (rcov=0)
    denom = rc_i ** 2 + rc_j ** 2
    denom = denom.clamp(min=1e-30)
    r = 0.5 / denom  # (B, max_n, max_n)

    overlap = (4.0 * r * rc_i * rc_j).clamp(min=0.0).sqrt() ** 3 * torch.exp(-d2 * r)
    gom = overlap * amp_padded[:, :, None] * amp_padded[:, None, :]

    return gom


def build_gom_sp_batched(rxyz_padded, rcov_padded, amp_padded, n_sphere):
    """Build s+p orbital GOMs for all atoms at once — fully batched.

    Args:
        rxyz_padded: (B, max_n, 3) padded neighbor positions
        rcov_padded: (B, max_n) padded covalent radii
        amp_padded: (B, max_n) padded amplitudes (0 for padding)
        n_sphere: (B,) actual neighbor counts

    Returns:
        (B, 4*max_n, 4*max_n) batched GOM matrices
    """
    B, max_n, _ = rxyz_padded.shape

    # (B, max_n, max_n, 3)
    d = rxyz_padded[:, :, None, :] - rxyz_padded[:, None, :, :]
    d2 = (d ** 2).sum(-1)  # (B, max_n, max_n)

    rc_i = rcov_padded[:, :, None]  # (B, max_n, 1)
    rc_j = rcov_padded[:, None, :]  # (B, 1, max_n)

    denom = (rc_i ** 2 + rc_j ** 2).clamp(min=1e-30)
    r = 0.5 / denom  # (B, max_n, max_n)
    amp_ij = amp_padded[:, :, None] * amp_padded[:, None, :]  # (B, max_n, max_n)

    sji = (4.0 * rc_i * rc_j).clamp(min=0.0).sqrt() ** 3 * torch.exp(-d2 * r)

    # s-s: same as build_gom_s_batched
    ss = (4.0 * r * rc_i * rc_j).clamp(min=0.0).sqrt() ** 3 * torch.exp(-d2 * r) * amp_ij

    # Build 4n x 4n matrix
    om = rxyz_padded.new_zeros(B, 4 * max_n, 4 * max_n)

    # s-s block
    om[:, 0::4, 0::4] = ss

    # s-p block
    stv_sp = (8.0 ** 0.5) * rc_j * r * sji * amp_ij  # (B, max_n, max_n)
    for k in range(3):
        om[:, 0::4, (k + 1)::4] = stv_sp * d[:, :, :, k]

    # p-s block
    stv_ps = -(8.0 ** 0.5) * rc_i * r * sji * amp_ij
    for k in range(3):
        om[:, (k + 1)::4, 0::4] = stv_ps * d[:, :, :, k]

    # p-p block
    stv_pp = -8.0 * rc_i * rc_j * r * r * sji * amp_ij
    inv_2r = 0.5 / r.clamp(min=1e-30)
    for k1 in range(3):
        for k2 in range(3):
            val = stv_pp * d[:, :, :, k1] * d[:, :, :, k2]
            if k1 == k2:
                val = val - stv_pp * inv_2r
            om[:, (k1 + 1)::4, (k2 + 1)::4] = val

    return om


def gom_fp_batched(rxyz_padded, rcov_padded, amp_padded, n_sphere, natx,
                   orbital='s'):
    """Full pipeline: build GOMs + eigendecompose, all batched.

    Args:
        rxyz_padded: (B, max_n, 3) padded neighbor positions
        rcov_padded: (B, max_n) padded covalent radii
        amp_padded: (B, max_n) padded amplitudes
        n_sphere: (B,) neighbor counts
        natx: output fingerprint dim
        orbital: 's' for s-only, 'sp' for s+p orbitals

    Returns:
        (B, natx) for s-only, (B, 4*natx) for s+p — eigenvalues, descending
    """
    if orbital == 'sp':
        lseg = 4
        goms = build_gom_sp_batched(rxyz_padded, rcov_padded, amp_padded, n_sphere)
    else:
        lseg = 1
        goms = build_gom_s_batched(rxyz_padded, rcov_padded, amp_padded, n_sphere)

    fp_dim = natx * lseg

    # Batched eigendecomposition
    eigvals = _eigvalsh_batched_adaptive(goms)  # ascending

    # Descending
    eigvals = eigvals.flip(-1)

    mat_n = goms.shape[1]
    # Trim/pad to fp_dim
    if mat_n >= fp_dim:
        result = eigvals[:, :fp_dim].contiguous()
    else:
        pad = torch.zeros(eigvals.shape[0], fp_dim - mat_n,
                          device=eigvals.device, dtype=eigvals.dtype)
        result = torch.cat([eigvals, pad], dim=1)

    # Zero out padding artifacts without Python/device sync loops.
    # Keep entries j < n_sphere[i] * lseg, zero the rest.
    keep_counts = (n_sphere.clamp(max=natx) * lseg).to(result.device)
    cols = torch.arange(fp_dim, device=result.device).unsqueeze(0)
    keep_mask = cols < keep_counts.unsqueeze(1)
    result = result * keep_mask.to(result.dtype)

    return result
