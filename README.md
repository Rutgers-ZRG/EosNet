# Fingerprint Graph Neural Networks

**Note**: This FpGNN package is inherited from the [CGCNN](https://github.com/txie-93/cgcnn) framework, and there are some major changes.

## Change log

- Using atomic-centered Gaussian Overlap Matrix (GOM) Fingerprint vectors as atomic features
- Switch reading pymatgen structures from CIF to POSCAR
- Add `drop_last` in `torch.utils.data.DataLoader`
- Take data imbalance into account for classification job
- Clip `lfp` (Long FP) and `sfp` (Contracted FP) length for arbitrary crystal structures
- Add MPS support to accelerate training on MacOS, for details see [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html) and [Apple Metal acceleration](https://developer.apple.com/metal/pytorch/) \
  **Note**: For classification jobs you may need to modify [line 227 in WeightedRandomSampler](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py#L227) to `weights_tensor = torch.as_tensor(weights, dtype=torch.float32 if weights.device.type == "mps" else torch.float64)` when using MPS backend. To maximize the efficiency of training while using MPS backend, you may want to use only single core (`--workers 0`) of the CPU to load the dataset.
- Switching from [Python3 implementation](https://github.com/Tack-Tau/fplib3/) of the Fingerprint Library to [C implementation](https://github.com/zhuligs/fplib) to improve speed. \
  To install this C version you need to modify the `setup.py` in `fplib/fppy`
  ```python
  lapack_dir=["$HOME/miniforge3/envs/fplibenv/lib"]
  lapack_lib=['openblas']
  extra_link_args = ["-Wl,-rpath,$HOME/miniforge3/envs/fplibenv/lib"]
  .
  .
  .
  include_dirs = [source_dir, "$HOME/miniforge3/envs/fplibenv/include"]
  ```
  Also set the corresponding `DYLD_LIBRARY_PATH` in your `.bashrc` file as:
  ```bash
  export DYLD_LIBRARY_PATH="$HOME/miniforge3/envs/fplibenv/lib:$DYLD_LIBRARY_PATH"
  ```
  Then install LAPACK using `conda`:
  ```bash
  conda create -n fplibenv python=3.10 pip ; conda activate fplibenv
  python3 -m pip install -U pip setuptools wheel
  conda install conda-forge::lapack
  cd fplib/fppy/ ; python3 -m pip install -e .
  ```
  For the remaining FpGNN dependecies follow the original instruction. \
  **Note**: Currently only `lmax=0` is supported in the C version 

This package is based on the [Crystal Graph Convolutional Neural Networks]((https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)) that takes an arbitary crystal structure to predict material properties. 

The package provides two major functions:

- Train a FpGNN model with a customized dataset.
- Predict material properties of new crystals with a pre-trained FpGNN model.

##  Dependencies

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- ~~[Numba](https://numba.pydata.org/)~~ (Numba is no longer needed since we are switching from `fplib3` to `fplib_c`)

If you are new to Python, please [conda](https://conda.io/docs/index.html) to manage Python packages and environments.

```bash
conda activate fplibenv
python3 -m pip install numpy>=1.21.4 Scipy>=1.8.0 ase==3.22.1
python3 -m pip install scikit-learn torch==2.2.2 torchvision==0.17.2 pymatgen==2024.3.1
```
The above environment has been tested stable for both M-chip MacOS and CentOS clusters

## Check your strcuture files before use FpGNN

To catch the erroneous POSCAR file you can use the following `check_fp.py` in the `root_dir`:
```python

#!/usr/bin/env python3

import os
import sys
import numpy as np
from functools import reduce
import fplib
from ase.io import read as ase_read

def get_ixyz(lat, cutoff):
    lat = np.ascontiguousarray(lat)
    lat2 = np.dot(lat, np.transpose(lat))
    vec = np.linalg.eigvals(lat2)
    ixyz = int(np.sqrt(1.0/max(vec))*cutoff) + 1
    ixyz = np.int32(ixyz)
    return ixyz

def check_n_sphere(rxyz, lat, cutoff, natx):
    
    ixyzf = get_ixyz(lat, cutoff)
    ixyz = int(ixyzf) + 1
    nat = len(rxyz)
    cutoff2 = cutoff**2

    for iat in range(nat):
        xi, yi, zi = rxyz[iat]
        n_sphere = 0
        for jat in range(nat):
            for ix in range(-ixyz, ixyz+1):
                for iy in range(-ixyz, ixyz+1):
                    for iz in range(-ixyz, ixyz+1):
                        xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                        yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                        zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                        d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > natx:
                                raise ValueError()


def read_types(cell_file):
    buff = []
    with open(cell_file) as f:
        for line in f:
            buff.append(line.split())
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    return types

if __name__ == "__main__":
    current_dir = './'
    for filename in os.listdir(current_dir):
        f = os.path.join(current_dir, filename)
        if os.path.isfile(f) and os.path.splitext(f)[-1].lower() == '.vasp':
            atoms = ase_read(f)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            chem_nums = list(atoms.numbers)
            znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
            typ = len(znucl_list)
            znucl = np.array(znucl_list, int)
            types = read_types(f)
            cell = (lat, rxyz, types, znucl)

            natx = int(256)
            lmax = int(0)
            cutoff = np.float64(int(np.sqrt(8.0))*3) # Shorter cutoff for GOM
            
            try:
                check_n_sphere(rxyz, lat, cutoff, natx)
            except ValueError:
                print(str(filename) + " is glitchy !")
            
            if len(rxyz) != len(types) or len(set(types)) != len(znucl):
                print(str(filename) + " is glitchy !")
            else:
                fp = fplib.get_lfp(cell, cutoff=cutoff, natx=natx, log=False) # Long Fingerprint
                # fplib.get_sfp(cell, cutoff=cutoff, natx=natx, log=False)   # Contracted Fingerprint         
```

## Usage

### Define a customized dataset 

To input crystal structures to FpGNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [POSCAR](https://www.vasp.at/wiki/index.php/POSCAR) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)

2. `ID.vasp` a [POSCAR](https://www.vasp.at/wiki/index.php/POSCAR) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.vasp
├── id1.vasp
├── ...
```

### Train a GNN model

Before training a new GNN model, you will need to:

- Define a customized dataset at `root_dir` to store the structure-property relations of interest.

Then, in directory `FpGNN`, you can train a GNN model for your customized dataset by:

```bash
python3 train.py root_dir
```

For detailed info of setting tags you can run

```bash
python3 train.py -h
```

```bash
python3 train.py --task regression --workers 31 --epochs 1000 --batch-size 64 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1  root_dir | tee FpGNN_log.txt
```

To resume from a previous `checkpoint`

```bash
python3 train.py --resume ./checkpoint.pth.tar --task regression --workers 31 --epochs 1000 --batch-size 64 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1  root_dir >> FpGNN_log.txt
```

After training, you will get three files in `FpGNN` directory.

- `model_best.pth.tar`: stores the GNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the GNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Predict material properties with a pre-trained GNN model

In directory `FpGNN`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar root_dir
```

**Note**: you need to put some random numbers in `id_prop.csv` and the `struct_id`s are the structures you want to predict.

## How to cite

Please cite the following work if you want to use FpGNN:

For CGCNN framework, please cite:
```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
```

If you use [Python3 implementation](https://github.com/Tack-Tau/fplib3/) of the Fingerprint Library, please cite:
```
@article{taoAcceleratingStructuralOptimization2024,
  title = {Accelerating Structural Optimization through Fingerprinting Space Integration on the Potential Energy Surface},
  author = {Tao, Shuo and Shao, Xuecheng and Zhu, Li},
  year = {2024},
  month = mar,
  journal = {J. Phys. Chem. Lett.},
  volume = {15},
  number = {11},
  pages = {3185--3190},
  doi = {10.1021/acs.jpclett.4c00275},
  url = {https://pubs.acs.org/doi/10.1021/acs.jpclett.4c00275}
}
```

If you use [C implementation](https://github.com/zhuligs/fplib) of the Fingerprint Library, please cite:
```
@article{zhuFingerprintBasedMetric2016,
  title = {A Fingerprint Based Metric for Measuring Similarities of Crystalline Structures},
  author = {Zhu, Li and Amsler, Maximilian and Fuhrer, Tobias and Schaefer, Bastian and Faraji, Somayeh and Rostami, Samare and Ghasemi, S. Alireza and Sadeghi, Ali and Grauzinyte, Migle and Wolverton, Chris and Goedecker, Stefan},
  year = {2016},
  month = jan,
  journal = {The Journal of Chemical Physics},
  volume = {144},
  number = {3},
  pages = {034203},
  doi = {10.1063/1.4940026},
  url = {https://doi.org/10.1063/1.4940026}
}
```

For GOM Fingerprint methodology, please cite:
```
@article{sadeghiMetricsMeasuringDistances2013,
  title = {Metrics for Measuring Distances in Configuration Spaces},
  author = {Sadeghi, Ali and Ghasemi, S. Alireza and Schaefer, Bastian and Mohr, Stephan and Lill, Markus A. and Goedecker, Stefan},
  year = {2013},
  month = nov,
  journal = {The Journal of Chemical Physics},
  volume = {139},
  number = {18},
  pages = {184118},
  doi = {10.1063/1.4828704},
  url = {https://pubs.aip.org/aip/jcp/article/317391}
}
```
