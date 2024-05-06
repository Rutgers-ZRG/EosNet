# Fingerprint Graph Neural Networks

**Note**: This version is a personal modification from the [original repo](https://github.com/txie-93/cgcnn), and there are some major changes.

## Change log

- Using [Fingerprint](https://github.com/Tack-Tau/fplib3/) (FP) vectors as atomic features
- Switch reading pymatgen structures from CIF to POSCAR
- Add `drop_last` in `torch.utils.data.DataLoader`
- Take data imbalance into account for classification job
- Clip `lfp` (Long FP) and `sfp` (Contracted FP) length for arbitrary crystal structures

This software is based on the Crystal Graph Convolutional Neural Networks (CGCNN) that takes an arbitary crystal structure to predict material properties. 

The package provides two major functions:

- Train a CGCNN model with a customized dataset.
- Predict material properties of new crystals with a pre-trained CGCNN model.

The following paper describes the details of the CGCNN framework:

[Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)

##  Dependencies

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [Numba](https://numba.pydata.org/)

If you are new to Python, please [conda](https://conda.io/docs/index.html) to manage Python packages and environments.

```bash
conda create -n fpgnn python=3.8 pip ; conda activate fpgnn
python3 -m pip install -U pip setuptools wheel
python3 -m pip install numpy==1.25.0 numba==0.58.0 ase==3.22.1
python3 -m pip install scikit-learn torch==2.2.2 torchvision==0.17.2 pymatgen==2024.3.1
```

## Usage

### Define a customized dataset 

To input crystal structures to CGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

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

### Train a CGCNN model

Before training a new CGCNN model, you will need to:

- Define a customized dataset at `root_dir` to store the structure-property relations of interest.

Then, in directory `FpGNN`, you can train a CGCNN model for your customized dataset by:

```bash
python3 train.py root_dir
```

For detailed info of setting tags you can run

```bash
python3 train.py -h
```
or alternatively

Following is a demo of how to use `train.py`

```bash
python3 train.py --task regression --workers 31 --epochs 1000 --batch-size 64 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1  root_dir | tee FpGNN_log.txt
```

To resume from a previous `checkpoint`

```bash
python3 train.py --resume ./checkpoint.pth.tar --task regression --workers 31 --epochs 1000 --batch-size 64 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1  root_dir | tee FpGNN_log.txt
```

After training, you will get three files in `cgcnn` directory.

- `model_best.pth.tar`: stores the CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Predict material properties with a pre-trained CGCNN model

In directory `FpGNN`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar root_dir
```

## How to cite

Please cite the following work if you want to use FpCNN.

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
