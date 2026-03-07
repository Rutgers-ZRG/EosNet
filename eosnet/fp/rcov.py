"""Covalent radii table for GOM fingerprints.

Values in Angstroms, indexed by atomic number Z (0..112).
Source: fplib / rcovdata.py (Rutgers-ZRG).
"""

import torch

# fmt: off
_RCOV_ANGSTROM = [
    1.00,  # 0  X  (placeholder)
    0.37,  # 1  H
    0.32,  # 2  He
    1.34,  # 3  Li
    0.90,  # 4  Be
    0.82,  # 5  B
    0.77,  # 6  C
    0.75,  # 7  N
    0.73,  # 8  O
    0.71,  # 9  F
    0.69,  # 10 Ne
    1.54,  # 11 Na
    1.30,  # 12 Mg
    1.18,  # 13 Al
    1.11,  # 14 Si
    1.06,  # 15 P
    1.02,  # 16 S
    0.99,  # 17 Cl
    0.97,  # 18 Ar
    1.96,  # 19 K
    1.74,  # 20 Ca
    1.44,  # 21 Sc
    1.36,  # 22 Ti
    1.25,  # 23 V
    1.27,  # 24 Cr
    1.39,  # 25 Mn
    1.25,  # 26 Fe
    1.26,  # 27 Co
    1.21,  # 28 Ni
    1.38,  # 29 Cu
    1.31,  # 30 Zn
    1.26,  # 31 Ga
    1.22,  # 32 Ge
    1.19,  # 33 As
    1.16,  # 34 Se
    1.14,  # 35 Br
    1.10,  # 36 Kr
    2.11,  # 37 Rb
    1.92,  # 38 Sr
    1.62,  # 39 Y
    1.48,  # 40 Zr
    1.37,  # 41 Nb
    1.45,  # 42 Mo
    1.56,  # 43 Tc
    1.26,  # 44 Ru
    1.35,  # 45 Rh
    1.31,  # 46 Pd
    1.53,  # 47 Ag
    1.48,  # 48 Cd
    1.44,  # 49 In
    1.41,  # 50 Sn
    1.38,  # 51 Sb
    1.35,  # 52 Te
    1.33,  # 53 I
    1.30,  # 54 Xe
    2.25,  # 55 Cs
    1.98,  # 56 Ba
    1.80,  # 57 La
    1.63,  # 58 Ce
    1.76,  # 59 Pr
    1.74,  # 60 Nd
    1.73,  # 61 Pm
    1.72,  # 62 Sm
    1.68,  # 63 Eu
    1.69,  # 64 Gd
    1.68,  # 65 Tb
    1.67,  # 66 Dy
    1.66,  # 67 Ho
    1.65,  # 68 Er
    1.64,  # 69 Tm
    1.70,  # 70 Yb
    1.60,  # 71 Lu
    1.50,  # 72 Hf
    1.38,  # 73 Ta
    1.46,  # 74 W
    1.59,  # 75 Re
    1.28,  # 76 Os
    1.37,  # 77 Ir
    1.28,  # 78 Pt
    1.44,  # 79 Au
    1.49,  # 80 Hg
    1.48,  # 81 Tl
    1.47,  # 82 Pb
    1.46,  # 83 Bi
    1.45,  # 84 Po
    1.47,  # 85 At
    1.42,  # 86 Rn
    2.23,  # 87 Fr
    2.01,  # 88 Ra
    1.86,  # 89 Ac
    1.75,  # 90 Th
    1.69,  # 91 Pa
    1.70,  # 92 U
    1.71,  # 93 Np
    1.72,  # 94 Pu
    1.66,  # 95 Am
    1.66,  # 96 Cm
    1.68,  # 97 Bk
    1.68,  # 98 Cf
    1.65,  # 99 Es
    1.67,  # 100 Fm
    1.73,  # 101 Md
    1.76,  # 102 No
    1.61,  # 103 Lr
    1.57,  # 104 Rf
    1.49,  # 105 Db
    1.43,  # 106 Sg
    1.41,  # 107 Bh
    1.34,  # 108 Hs
    1.29,  # 109 Mt
    1.28,  # 110 Ds
    1.21,  # 111 Rg
    1.22,  # 112 Cn
]
# fmt: on

RCOV_TABLE = torch.tensor(_RCOV_ANGSTROM, dtype=torch.float64)


def get_rcov(atomic_numbers, device=None, dtype=torch.float64):
    """Look up covalent radii by atomic number.

    Args:
        atomic_numbers: integer tensor or list of atomic numbers
        device: target device
        dtype: target dtype

    Returns:
        Tensor of covalent radii in Angstroms
    """
    if not isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
    table = RCOV_TABLE.to(device=device, dtype=dtype)
    return table[atomic_numbers.long()]
