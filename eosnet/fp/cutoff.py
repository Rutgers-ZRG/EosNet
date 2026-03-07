"""Smooth polynomial cutoff envelope for GOM fingerprints.

Matches C libfp: NC=2 (fingerprint.h #define NC 2).
"""

import torch

NC = 2  # cutoff polynomial order (matches C libfp)


def cutoff_amplitude(d2: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Compute smooth cutoff amplitude from squared distances.

    amp = (1 - d^2 * fc)^NC  where  fc = 1/(2*NC*wc^2), wc = cutoff/sqrt(2*NC)

    Simplifies to: amp = (1 - d^2 / cutoff^2)^NC

    Args:
        d2: squared distances, shape (...)
        cutoff: cutoff radius in Angstroms

    Returns:
        amplitude tensor, same shape as d2. Zero for d > cutoff.
    """
    cutoff2 = cutoff ** 2
    # wc = cutoff / sqrt(2*NC)  =>  fc = 1/(2*NC*wc^2) = 1/cutoff^2
    # So amp = (1 - d2 * fc)^NC = (1 - d2/cutoff^2)^NC
    x = 1.0 - d2 / cutoff2
    # Smooth masking: clamp to zero for d > cutoff
    x = x.clamp(min=0.0)
    return x ** NC
