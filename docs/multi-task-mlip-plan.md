# Multi-Task MLIP + Property Prediction: Design Plan

**Date**: March 4, 2026
**Authors**: Li Zhu, with AI assistance
**Status**: Planning — awaiting PI direction on implementation priority

---

## Motivation

Current MLIPs (MACE, NequIP, Allegro, MatterSim) predict only energy and forces. Electronic properties (band gap, metallicity) and mechanical properties (elastic moduli) require separate DFT calculations on selected MD snapshots. This creates a bottleneck:

```
Run MLIP-MD (10^6 steps) → select snapshots → DFT on each → properties
       fast                    manual              expensive
```

**Goal**: Predict material properties (band gap, elastic moduli, metal/nonmetal classification) simultaneously with energy/forces during MD, at negligible extra cost.

---

## Core Idea

Couple a standard equivariant MLIP backbone (for energy/forces) with an EOSnet-style GOM fingerprint branch (for property prediction) in a single model:

```
                    Crystal Structure
                          |
              +-----------+-----------+
              |                       |
    Equivariant MLIP backbone    GOM Fingerprint branch
    (Allegro/MACE/NequIP)       (PyTorch reimplementation)
              |                       |
        atom embeddings          GOM eigenvalues
              |                       |
              +--------> Fusion <-----+
                          |
              +-----------+-----------+
              |                       |
         Energy head             Property head
         (per-atom E, F)         (band gap, K, G, etc.)
```

**Key insight**: The MLIP backbone already learns good structural representations. The GOM branch adds orbital overlap information that is critical for electronic properties but not captured by standard equivariant features.

---

## Why EOSnet Alone Won't Work as MLIP

We discussed three obstacles (March 4, 2026):

1. **Force accuracy**: GOM eigenvalue gradients can be noisy (eigenvalue reordering under small perturbations). Equivariant models have smooth features by construction.

2. **Cost**: GOM eigendecomposition is O(M^3) per atom per step. Spherical harmonics + tensor products are O(M * L^2) — cheaper for same expressiveness.

3. **Ecosystem**: Existing MLIPs have years of optimization (CUDA kernels, LAMMPS integration, multi-GPU). Rebuilding from scratch has uncertain ROI.

**Conclusion**: Don't compete head-on with MACE/NequIP for PES fitting. Instead, leverage EOSnet's unique strength (orbital overlap encoding) for **property prediction during MD**.

---

## PyTorch GOM Implementation

**Decision (March 4, 2026)**: Reimplement GOM in PyTorch rather than parallelize C fplib.

**Rationale**:
- Everything stays on GPU — no CPU-GPU data transfer per MD step
- `torch.linalg.eigh` is differentiable — autograd handles forces for free
- Batched eigendecomposition on GPU is competitive with OpenMP-parallel CPU for typical GOM sizes (50-100 neighbors)
- Any model architecture change automatically gets correct gradients

**Implementation sketch**:
```python
def build_gom_batched(positions, neighbors, cov_radii, cutoff, n=2):
    """
    Build GOM for all atoms in a batch, fully on GPU.

    Args:
        positions: (N_atoms, 3) atomic positions
        neighbors: (N_atoms, M) neighbor indices within GOM cutoff
        cov_radii: (N_atoms,) covalent radii
        cutoff: float, GOM cutoff radius
        n: int, cutoff function nonlinearity (Eq 1b in paper)

    Returns:
        eigenvalues: (N_atoms, M) GOM eigenvalues — node features
    """
    # Gaussian widths from covalent radii: alpha_i = 0.5 / r_cov_i^2
    alpha = 0.5 / cov_radii**2

    # Pairwise displacements and distances
    r_ij = positions[neighbors] - positions[:, None]  # (N, M, 3)
    d_ij = torch.norm(r_ij, dim=-1)                   # (N, M)

    # Smooth cutoff function: fc(r) = (1 - r^2/rcut^2)^n
    fc = (1 - (d_ij / cutoff)**2).clamp(min=0)**n     # (N, M)

    # GTO overlap integrals (s-orbital case):
    # <phi_i|phi_j> = (2*ai*aj/(ai+aj))^(3/4) * exp(-ai*aj/(ai+aj) * |ri-rj|^2)
    ai = alpha[:, None].expand_as(d_ij)                # center atom
    aj = alpha[neighbors]                               # neighbor atoms
    prefactor = (2 * ai * aj / (ai + aj))**(3/4)
    exponent = -ai * aj / (ai + aj) * d_ij**2
    overlap = prefactor * torch.exp(exponent)           # (N, M, M) after outer product

    # Build full GOM: O[i,j] = fc(r_im) * <phi_i|phi_j> * fc(r_jm)
    gom = fc[:, :, None] * overlap * fc[:, None, :]    # (N, M, M)

    # Batched eigendecomposition
    eigenvalues = torch.linalg.eigh(gom).eigenvalues   # (N, M)

    return eigenvalues
```

**Known issue**: `torch.linalg.eigh` backward is numerically unstable for degenerate eigenvalues. Fix: add Lorentzian broadening to the denominator in the backward pass, or use a custom backward with regularization.

---

## Practical Applications

### Tier 1: High-impact, directly relevant to group

**1. Metallization under dynamic compression**
- Compress Fe/Fe-Si alloys in MD, track band gap closing in real-time
- Directly supports Earth's core project (NSF Collab w/ Hemley)
- Shock experiments measure conductivity jumps but lack atomic-level mechanism
- Validation: compare predicted metallization pressure against experiment

**2. Structure search with property constraints (CRISP integration)**
- During CSP relaxation, simultaneously score structures for target electronic properties
- Multi-objective ranking: energy AND band gap/stability
- Huge speedup over: relax → rank by energy → DFT band gap on top N candidates

### Tier 2: Broader materials science

**3. Phase-change materials (PCM) for memory devices**
- GST alloys (Ge-Sb-Te): amorphous↔crystalline switching
- Track resistivity contrast during crystallization MD
- Industry relevance: Intel Optane, Samsung Z-NAND

**4. Photovoltaic degradation**
- Halide perovskites (CsPbI3, MAPbI3) at operating temperature
- Band gap drift during defect formation / ion migration
- Band gap drift = measurable efficiency loss

**5. Metal-insulator transitions**
- VO2 (340K transition), compressed H2 metallization
- Most publishable demonstration: show band gap evolution matches known transition

---

## Architecture Options

### Option A: Shared backbone + separate heads
```
Structure → Equivariant backbone → shared atom embeddings
                                        |
                              +---------+---------+
                              |                   |
                         Energy head          Property head
                         (sum → E)            (pool → Eg, K, G)
```
- Simplest. Property head uses learned representations from MLIP training.
- Risk: MLIP backbone optimized for energy may not encode band-gap-relevant info.

### Option B: Dual encoder + fusion (recommended)
```
Structure → Equivariant backbone → h_equi  ─┐
         → GOM branch           → h_gom   ─┤→ concat → Property head
                                             └───────→ Energy head (h_equi only)
```
- GOM branch explicitly provides orbital overlap info for properties.
- Energy head uses only equivariant features (proven to work).
- Property head fuses both — gets the best of both worlds.
- GOM branch can be lightweight (no message passing, just eigenvalues + linear projection).

### Option C: GOM as auxiliary loss (knowledge distillation)
```
Structure → Equivariant backbone → atom embeddings
                                        |
                              +---------+---------+
                              |         |         |
                         Energy     Property   GOM reconstruction
                         head       head       head (auxiliary loss)
```
- Train backbone to predict GOM fingerprints as auxiliary task.
- At inference, drop the GOM reconstruction head — no eigendecomp cost at all.
- Injects orbital info during training without runtime penalty.
- Trade-off: property prediction quality depends on how well backbone can encode GOM info.

---

## Training Strategy

### Multi-task loss
```
L = w_E * L_energy + w_F * L_forces + w_S * L_stress + w_prop * L_property
```

### Data requirements
- Energy/forces/stress: standard DFT datasets (MP, OQMD, Alexandria)
- Band gap: Materials Project (19k structures, already used in EOSnet)
- Elastic moduli: Materials Project (5k structures)
- Can train jointly or in stages (pre-train MLIP → fine-tune with property data)

### Staged training approach
1. Pre-train equivariant backbone on energy/forces (standard MLIP training)
2. Freeze backbone, train GOM branch + property head on property data
3. Fine-tune end-to-end with combined loss

---

## Paper Concept

**Title**: "Simultaneous Molecular Dynamics and Electronic Property Prediction via Orbital-Overlap-Informed Graph Neural Networks"

**Key demonstration**:
- Well-known metal-insulator transition (VO2 at 340K or compressed H2)
- Show band gap evolution during MD matches experimental transition
- Validate against explicit DFT snapshots at selected frames
- Show 1000x speedup over AIMD for property tracking

**Unique contribution**: First MLIP that provides on-the-fly electronic property prediction during MD, enabled by coupling equivariant representations with GOM orbital overlap features.

---

## Connection to Existing Group Projects

| Project | How this helps |
|---------|---------------|
| MLIP Active Learning | GOM features as additional training signal; property-aware active learning |
| DMFT-MLIP (Fe) | Predict electronic transitions during Fe MD at core conditions |
| CRISP (CSP) | Multi-objective structure search: energy + properties simultaneously |
| B-C Clathrates | Screen for superconducting clathrates with on-the-fly Eg prediction |
| Carbon transitions | Track band gap during graphite→diamond MD (semimetal→insulator) |

---

## GOM vs e3nn: Comparison and Integration Strategy

**Discussion (March 4, 2026)**: Are GOM fingerprints redundant with e3nn features?

### What each computes

| | GOM (fplib) | e3nn |
|---|---|---|
| **Basis** | Gaussian-type orbitals (physical, chemistry-motivated) | Spherical harmonics (mathematical, abstract) |
| **Chemical info** | Covalent radii hardcoded in Gaussian width | Learned from data |
| **Many-body** | All at once via matrix eigenspectrum (one operation) | Built up layer-by-layer via tensor products |
| **Output type** | Invariant (scalars only) | Equivariant (scalars + vectors + tensors) |
| **Radial encoding** | Orbital overlap: exp(-αiαj/(αi+αj)|rij|²) | Learned Bessel/polynomial basis |

### Answer: Complementary, not redundant

- e3nn CAN in principle learn what GOM captures — but GOM provides orbital overlap as free inductive bias
- GOM captures full spectral structure of neighbor overlap graph in one eigendecomposition; e3nn needs multiple layers for equivalent body order
- e3nn's strength (equivariance, smooth gradients) is exactly what GOM lacks
- For electronic properties (band gap), GOM's orbital overlap bias is valuable; for energy/forces, e3nn is superior

### Integration: e3nn replaces CGCNN backbone, GOM stays as input features

```
Node features at input:  one-hot(Z) + GOM eigenvalues  ← keep (fplib C, computed once)
                                |
                    e3nn conv layers (tensor products)  ← replace CGCNN
                                |
                    invariant readout → FC → property
```

GOM eigenvalues are l=0 scalars → plug directly into e3nn scalar channel. Equivariance of higher-l channels preserved.

---

## Implementation Roadmap

### Phase 1: EOSnet v2 — e3nn backbone (immediate, high ROI)
- Replace CGCNN message passing with e3nn/MACE-style tensor product layers
- Keep fplib C for GOM features (computed once per structure, no forces needed)
- Keep existing data pipeline and benchmarks
- **Key experiment**: Does GOM + e3nn > GOM + CGCNN? Does GOM + e3nn > pure e3nn?
- Publishable as EOSnet v2

### Phase 2: PyTorch GOM (when ready for MLIP)
- Reimplement GOM in PyTorch for GPU + autograd
- Only needed when building multi-task MLIP (forces require differentiable GOM)
- Validate against fplib C output
- Benchmark GPU batched eigh vs CPU parallel

### Phase 3: Multi-task MLIP + property prediction
- Implement dual encoder architecture (Option B)
- Equivariant backbone for energy/forces + GOM branch for properties
- Train on combined energy + property datasets
- Demonstrate on metal-insulator transition MD

### Phase 4: Applications and paper
- VO2 or compressed H2 metallization demonstration
- Integration with CRISP for multi-objective CSP
- Fe alloy electronic transitions at core conditions
