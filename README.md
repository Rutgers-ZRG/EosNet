# EOSNet: Embedded Overlap Structures for Graph Neural Networks

<p align="center">
  <img src="https://raw.githubusercontent.com/Rutgers-ZRG/EosNet/master/EOSnet_TOC.png" width="65%" alt="EOSNet TOC">
</p>

EOSNet uses atomic-centered Gaussian Overlap Matrix (GOM) fingerprints as node features in a graph neural network to predict material properties from crystal structures.

**Key idea**: GOM eigenvalues encode orbital overlap information — a physics-motivated representation that captures chemical bonding character beyond simple distance-based descriptors.

## Models

EOSNet provides two backbone architectures:

| Model | Backbone | Description |
|-------|----------|-------------|
| **EOSNet v1** (CGCNN) | Crystal graph convolution | GOM fingerprints + CGCNN message passing |
| **EOSNet v2** (e3nn) | Equivariant tensor products | GOM fingerprints + e3nn spherical harmonics |

Select with `--model-type e3nn` (default) or `--model-type cgcnn`.

## Installation

### Prerequisites

- Python 3.10+
- [PyTorch](https://pytorch.org) >= 2.0
- [e3nn](https://github.com/e3nn/e3nn)

GOM fingerprints are computed natively in PyTorch (`eosnet/fp/`) — no external C library needed.

### Setup

```bash
conda create -n eosnet python=3.10 pip
conda activate eosnet

# Install dependencies
pip install numpy scipy ase scikit-learn pymatgen
pip install torch torchvision
pip install e3nn
```

## Data Format

EOSNet supports two input formats:

### POSCAR format (default)

Create a directory with:

```
root_dir/
├── id_prop.csv      # ID, target_value (one per line)
├── id0.vasp         # POSCAR files
├── id1.vasp
└── ...
```

### Extended XYZ format

```
root_dir/
├── id_prop.csv      # ID, target_value
└── structures.xyz   # All structures in extended XYZ
```

Use `convert_to_extxyz.py` to convert POSCAR directories to extended XYZ.

## Training

```bash
# Train with e3nn backbone (default)
python train.py root_dir

# Train with CGCNN backbone (v1)
python train.py root_dir --model-type cgcnn

# Full example with recommended settings
python train.py root_dir \
    --task regression \
    --epochs 500 \
    --batch-size 64 \
    --optim Adam \
    --lr 1e-3 \
    --warmup-epochs 20 \
    --lr-milestones 100 200 400 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --n-conv 3 \
    --n-h 1 \
    | tee train_log.txt
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | `e3nn` | Model backbone: `e3nn` or `cgcnn` |
| `--task` | `regression` | `regression` or `classification` |
| `--epochs` | `200` | Number of training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--optim` | `SGD` | Optimizer: `SGD` or `Adam` |
| `--lr` | `0.01` | Initial learning rate |
| `--n-conv` | `3` | Number of convolution layers |
| `--nx` | `256` | Max neighbors for GOM fingerprint |
| `--lmax` | `0` | `0` for s-orbitals only, `1` for s+p |
| `--radius` | `8.0` | Neighbor cutoff radius (Å) |
| `--save_to_disk` | `False` | Pre-process and cache data to disk |
| `--fine-tune` | — | Path to pre-trained model for fine-tuning |
| `--resume` | — | Path to checkpoint for resuming training |
| `--data-format` | `auto` | Input format: `auto`, `extxyz`, or `vasp` |

e3nn-specific options:

| Flag | Default | Description |
|------|---------|-------------|
| `--irreps-hidden` | `32x0e+16x1o+8x2e` | Hidden irreducible representations |
| `--max-ell` | `2` | Max spherical harmonic order |
| `--num-radial-basis` | `16` | Number of radial basis functions |

Run `python train.py -h` for all options.

### Output files

- `model_best.pth.tar` — Best model (by validation MAE or AUC)
- `checkpoint.pth.tar` — Latest checkpoint
- `train_results.csv` / `test_results.csv` — Predictions (ID, target, predicted)

## Prediction

```bash
python predict.py model_best.pth.tar --test root_dir
```

## How to Cite

If you use EOSNet, please cite:

```bibtex
@article{taoEOSnetEmbeddedOverlap2025,
  title   = {EOSnet: Embedded Overlap Structures for Graph Neural Networks
             in Predicting Material Properties},
  author  = {Tao, Shuo and Zhu, Li},
  journal = {J. Phys. Chem. Lett.},
  volume  = {16},
  pages   = {717--724},
  year    = {2025},
  doi     = {10.1021/acs.jpclett.4c03179}
}
```

GOM fingerprint methodology:

```bibtex
@article{sadeghiMetricsMeasuringDistances2013,
  title   = {Metrics for Measuring Distances in Configuration Spaces},
  author  = {Sadeghi, Ali and Ghasemi, S. Alireza and Schaefer, Bastian
             and Mohr, Stephan and Lill, Markus A. and Goedecker, Stefan},
  journal = {J. Chem. Phys.},
  volume  = {139},
  pages   = {184118},
  year    = {2013},
  doi     = {10.1063/1.4828704}
}
```

Fingerprint library:

```bibtex
@article{zhuFingerprintBasedMetric2016,
  title   = {A Fingerprint Based Metric for Measuring Similarities
             of Crystalline Structures},
  author  = {Zhu, Li and Amsler, Maximilian and Fuhrer, Tobias and Schaefer, Bastian
             and Faraji, Somayeh and Rostami, Samare and Ghasemi, S. Alireza
             and Sadeghi, Ali and Grauzinyte, Migle and Wolverton, Chris
             and Goedecker, Stefan},
  journal = {J. Chem. Phys.},
  volume  = {144},
  pages   = {034203},
  year    = {2016},
  doi     = {10.1063/1.4940026}
}
```

CGCNN framework:

```bibtex
@article{PhysRevLett.120.145301,
  title     = {Crystal Graph Convolutional Neural Networks for an Accurate
               and Interpretable Prediction of Material Properties},
  author    = {Xie, Tian and Grossman, Jeffrey C.},
  journal   = {Phys. Rev. Lett.},
  volume    = {120},
  pages     = {145301},
  year      = {2018},
  doi       = {10.1103/PhysRevLett.120.145301}
}
```

## License

The v1 (CGCNN) backbone is based on the [CGCNN](https://github.com/txie-93/cgcnn) framework by Tian Xie.
