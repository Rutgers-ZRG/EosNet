#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
import random
import warnings
import csv
from math import comb
from functools import reduce, lru_cache

import numpy as np
import fplib
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read as ase_read
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

def get_train_val_test_loader(dataset, classification=False,
                              collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1,
                              return_test=False, num_workers=1,
                              pin_memory=False, persistent_workers=True,
                              multiprocessing_context=None,
                              shuffle=False, drop_last=True,
                              **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    classification: bool
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    shuffle: bool
    drop_last: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    if classification:
        train_indices = indices[:train_size]
        train_targets = [np.array(dataset[i][1], dtype=np.int32).item() for i in train_indices]
        class_weights = compute_class_weight(
            class_weight = 'balanced',
            classes = np.unique(train_targets),
            y = train_targets)
        class_weights = class_weights/np.linalg.norm(class_weights)
        loss_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights_dict = {class_idx: weight for class_idx,
                              weight in zip(np.unique(train_targets), class_weights)}
        class_weights_tensor = torch.tensor([class_weights_dict[class_idx]
                                             for class_idx in train_targets], dtype=torch.float32)
        train_sampler = WeightedRandomSampler(
            weights = class_weights_tensor[train_indices],
            num_samples = train_size,
            replacement = False)
    else:
        train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=max(0, num_workers),
                              collate_fn=collate_fn,
                              shuffle=shuffle,
                              drop_last=drop_last,
                              persistent_workers=persistent_workers and num_workers > 0,
                              multiprocessing_context=multiprocessing_context,
                              pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=max(0, num_workers),
                            collate_fn=collate_fn,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            persistent_workers=persistent_workers and num_workers > 0,
                            multiprocessing_context=multiprocessing_context,
                            pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=max(0, num_workers),
                                 collate_fn=collate_fn,
                                 shuffle=shuffle,
                                 drop_last=drop_last,
                                 persistent_workers=persistent_workers and num_workers > 0,
                                 multiprocessing_context=multiprocessing_context,
                                 pin_memory=pin_memory)
    if classification:
        if return_test:
            return class_weights, train_loader, val_loader, test_loader
        else:
            return class_weights, train_loader, val_loader
    else:
        if return_test:
            return None, train_loader, val_loader, test_loader
        else:
            return None, train_loader, val_loader

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal properties.
    
    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      For IdTargetData: (struct_id, target)
      For StructData: ((atom_fea, nbr_fea, nbr_fea_idx, angle_fea), target, struct_id)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      angle_fea: torch.Tensor shape (n_i, num_angles, angle_fea_len)
      target: torch.Tensor shape (1, )
      struct_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    For IdTargetData:
      batch_target: torch.Tensor shape (N, 1)
      batch_struct_ids: list

    For StructData:
      batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
      batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom's M neighbors
      batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
      crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
      batch_target: torch.Tensor shape (N, 1)
        Target value for prediction
      batch_struct_ids: list
    """
    # IdTargetData
    if isinstance(dataset_list[0], tuple) and len(dataset_list[0]) == 2:
        batch_target, batch_struct_ids = [], []
        for struct_id, target in dataset_list:
            batch_target.append(torch.tensor([float(target)], dtype=torch.float))
            batch_struct_ids.append(struct_id)
        return torch.cat(batch_target, dim=0), batch_struct_ids
    
    # StructData
    elif isinstance(dataset_list[0], tuple) and len(dataset_list[0]) == 3:
        batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
        crystal_atom_idx = []
        batch_target = []
        batch_struct_ids = []
        
        base_idx = 0
        for (atom_fea, nbr_fea, nbr_fea_idx), target, struct_id in dataset_list:
            n_i = atom_fea.shape[0]
            
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
            
            crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
            if not isinstance(target, torch.Tensor):
                target = torch.tensor([float(target)], dtype=torch.float)
            batch_target.append(target)
            batch_struct_ids.append(struct_id)
            
            base_idx += n_i
        
        # Stack features
        batch_atom_fea = torch.cat(batch_atom_fea, dim=0)
        batch_nbr_fea = torch.cat(batch_nbr_fea, dim=0)
        batch_nbr_fea_idx = torch.cat(batch_nbr_fea_idx, dim=0)
            
        # Stack targets
        batch_target = torch.stack(batch_target)
        
        return (batch_atom_fea,
                batch_nbr_fea,
                batch_nbr_fea_idx,
                crystal_atom_idx), \
                batch_target, \
                batch_struct_ids
    else:
        raise ValueError("Unsupported dataset type")

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = min(2.0*step, (dmax-dmin)/3.0)
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class IdTargetData(Dataset):
    """ 
    A simple dataset to load just the struct_id and target from the dataset's id_prop.csv.
    This is used for sampling targets without loading the full crystal structure data.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    random_seed: int
        Random seed for shuffling the dataset.

    Returns
    -------

    struct_id: str or int
    target: torch.Tensor shape (1, )
    """
    def __init__(self, root_dir, random_seed=42):
        self.root_dir = root_dir
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.isfile(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f, delimiter=',')
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        struct_id, target = self.id_prop_data[idx]
        return struct_id, float(target)

def get_neighbor_info(atoms, radius, max_num_nbr, struct_id=None):
    """
    Get neighbor information using pymatgen's get_all_neighbors.
    
    Args:
        atoms: ASE Atoms object
        radius: Cutoff radius for neighbor search
        max_num_nbr: Maximum number of neighbors to consider
        struct_id: Optional structure identifier for warnings
        
    Returns:
        nbr_indices: Array of neighbor indices [n_atoms, max_num_nbr]
        nbr_distances: Array of distances [n_atoms, max_num_nbr]
        displacement_vectors: Array of displacement vectors [n_atoms, max_num_nbr, 3]
    """
    # Convert to pymatgen Structure
    structure = AseAtomsAdaptor.get_structure(atoms)
    
    # Get all neighbors using pymatgen
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    
    nbr_indices, nbr_distances, displacement_vectors = [], [], []
    for nbr in all_nbrs:
        n_nbr = len(nbr)
        
        if n_nbr < max_num_nbr:
            if struct_id:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(struct_id))
            
            # Get the last valid neighbor for padding
            last_idx = nbr[-1][2] if nbr else 0
            last_dis = nbr[-1][1] if nbr else radius + 1.0
            last_disp = nbr[-1][0].coords - nbr[-1][0].frac_coords if nbr else np.zeros(3)
            
            # Pad with the last valid neighbor
            local_indices = list(map(lambda x: x[2], nbr)) + [last_idx] * (max_num_nbr - n_nbr)
            local_distances = list(map(lambda x: x[1], nbr)) + [last_dis] * (max_num_nbr - n_nbr)
            local_disps = list(map(lambda x: x[0].coords - x[0].frac_coords, nbr)) + [last_disp] * (max_num_nbr - n_nbr)
        else:
            # Take only max_num_nbr neighbors
            local_indices = list(map(lambda x: x[2], nbr[:max_num_nbr]))
            local_distances = list(map(lambda x: x[1], nbr[:max_num_nbr]))
            local_disps = list(map(lambda x: x[0].coords - x[0].frac_coords, nbr[:max_num_nbr]))
        
        nbr_indices.append(local_indices)
        nbr_distances.append(local_distances)
        displacement_vectors.append(local_disps)
    
    return (np.array(nbr_indices), 
            np.array(nbr_distances), 
            np.array(displacement_vectors))

class StructData(Dataset):
    """
    The StructData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of POSCAR files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── struct_id_0.vasp
    ├── struct_id_0.vasp
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal (struct_id), and the second column recodes
    the value of target property.

    struct_id_X.vasp: a POSCAR file that recodes the crystal structure, where
    struct_id_X is the unique ID for the crystal structure X.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors, unit in Angstroms
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    nx: int
        Maximum number of neighbors to construct the Gaussian overlap matrix for atomic Fingerprint
    lmax: int
        Integer to control whether using s orbitals only or both s and p orbitals for
        calculating the Guassian overlap matrix (0 for s orbitals only, other integers
        will indicate that using both s and p orbitals)
    num_workers: int
        Number of workers to parallelize thed process of struct data loading
    save_to_disk: bool
        Whether to save the processed dataset to disk for faster future loading.

    Returns
    -------

    atom_fea: torch.Tensor shape (nat, atom_fp_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    struct_id: str or int
    """
    def __init__(self,
                 id_prop_data,
                 root_dir,
                 max_num_nbr=12,
                 radius=8.0,
                 dmin=0.5,
                 step=0.1,
                 var=1.0,
                 nx=256,
                 lmax=0,
                 batch_size=64,
                 drop_last=False,
                 save_to_disk=False):
        self.root_dir = root_dir
        self.id_prop_data = id_prop_data
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.nx = nx
        self.lmax = lmax
        self.save_to_disk = save_to_disk
        assert lmax == 0, 'p-orbitals is not supported at this time!'
        assert nx >= comb(max_num_nbr, 2), 'nx is too small for the given max_num_nbr!'
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.total_size = len(self.id_prop_data)
        
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step, var=var)
        
        # Create processed directory if it doesn't exist
        self.process_dir = os.path.join(self.root_dir, 'saved_npz_files')
        
        self.processed_data = None
        
        if self.drop_last:
            self.num_batches = self.total_size // self.batch_size
        else:
            self.num_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        
        # Check if all batch files exist
        all_batches_exist = all(
            os.path.exists(os.path.join(self.process_dir, f'processed_data-{i+1}.npz'))
            for i in range(self.num_batches)
        )
        
        if all_batches_exist and not save_to_disk:
            self.load_dataset()
        elif save_to_disk:
            os.makedirs(self.process_dir, exist_ok=True)
            self.save_dataset()
            self.load_dataset()
        else:
            self.processed_data = None

    def __len__(self):
        return len(self.id_prop_data)
    
    def read_types(self, cell_file):
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

    # @lru_cache(maxsize=None)
    def get_fp_mat(self, cell_file):
        atoms = ase_read(cell_file)
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        chem_nums = list(atoms.numbers)
        znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
        ntyp = len(znucl_list)
        znucl = np.array(znucl_list, int)
        types = self.read_types(cell_file)
        cell = (lat, rxyz, types, znucl)
        contract = False
        natx = int(self.nx)
        lmax = int(self.lmax)
        cutoff = np.float64(int(np.sqrt(self.radius))*3) # Shorter cutoff for GOM

        if lmax == 0:
            lseg = 1
            orbital = 's'
        else:
            lseg = 4
            orbital = 'sp'

        if len(rxyz) != len(types) or len(set(types)) != len(znucl):
            print("Structure file: " +
                  str(cell_file.split('/')[-1]) +
                  " is erroneous, please double check!")
            if contract:
                fp = np.zeros((len(rxyz), 20), dtype=np.float64)
            else:
                fp = np.zeros((len(rxyz), lseg*natx), dtype=np.float64)
        else:
            if contract:
                fp = fplib.get_sfp(cell,
                                   cutoff=cutoff,
                                   natx=natx,
                                   log=False,
                                   orbital=orbital) # Contracted FP
                tmp_fp = []
                for i in range(len(fp)):
                    if len(fp[i]) < 20:
                        tmp_fp_at = fp[i].tolist() + [0.0]*(20 - len(fp[i]))
                        tmp_fp.append(tmp_fp_at)
                fp = np.array(tmp_fp, dtype=np.float64)
            else:
                fp = fplib.get_lfp(cell,
                                   cutoff=cutoff,
                                   natx=natx,
                                   log=False,
                                   orbital=orbital) # Long FP
        return fp

    def process_structure(self, crystal, struct_id):
        """
        Convert structure into features.
        
        Args:
            crystal: Pymatgen Structure object
            struct_id: Structure identifier for warnings
            
        Returns:
            atom_fea: Atom features [n_atoms, atom_fea_len]
            nbr_fea: Bond features [n_atoms, max_num_nbr, bond_fea_len]
            nbr_fea_idx: Neighbor indices [n_atoms, max_num_nbr]
        """
        # Convert to ASE atoms
        atoms = crystal.to_ase_atoms()
        
        # Get neighbor information
        nbr_indices, nbr_distances, _ = get_neighbor_info(
            atoms, self.radius, self.max_num_nbr, struct_id=struct_id
        )
        
        # One-hot encoding
        chem_nums = list(atoms.numbers)
        max_atomic_number = max(max(chem_nums), 112)
        one_hot_encodings = []
        for atom in atoms:
            encoding = [0] * (max_atomic_number + 1)
            encoding[atom.number] = 1
            one_hot_encodings.append(encoding)
        one_hot_encodings = np.array(one_hot_encodings, dtype=np.int32)
        
        # Fingerprint features
        comb_n_nbr = comb(self.max_num_nbr, 2)
        cell_file = os.path.join(self.root_dir, struct_id + '.vasp')
        fp_mat = self.get_fp_mat(cell_file)
        fp_mat = fp_mat[:, :comb_n_nbr]
        fp_mat[np.abs(fp_mat) < 1.0e-10] = 0.0
        fp_mat = fp_mat / np.linalg.norm(fp_mat, axis=-1, keepdims=True)
        atom_fea = np.hstack((one_hot_encodings, fp_mat))
        # atom_fea = fp_mat / np.linalg.norm(fp_mat, axis=-1, keepdims=True)
        # atom_fea = one_hot_encodings
        
        # Process bond features using GaussianDistance
        nbr_fea = self.gdf.expand(nbr_distances)
        
        return atom_fea, nbr_fea, nbr_indices

    def __getitem__(self, idx):
        """Get a single data point"""
        struct_id, target = self.id_prop_data[idx]
        
        # If data is already processed
        if hasattr(self, 'processed_data') and self.processed_data is not None:
            return self.processed_data[idx]
        
        # Otherwise process on-the-fly
        crystal = self.read_structure(struct_id)
        atom_fea, nbr_fea, nbr_fea_idx = self.process_structure(crystal, struct_id)
        
        # Convert to tensors
        processed_features = (
            torch.FloatTensor(atom_fea),
            torch.FloatTensor(nbr_fea),
            torch.LongTensor(nbr_fea_idx)
        )
        return (processed_features, target, struct_id)

    def read_structure(self, struct_id):
        """Read structure from file"""
        cell_file = os.path.join(self.root_dir, struct_id + '.vasp')
        crystal = Structure.from_file(cell_file)
        return crystal

    def process_item(self, idx):
        struct_id, target = self.id_prop_data[idx]
        crystal = self.read_structure(struct_id)
        atom_fea, nbr_fea, nbr_fea_idx = self.process_structure(crystal, struct_id)

        if self.save_to_disk:
            return (atom_fea, nbr_fea, nbr_fea_idx), target, struct_id
        else:
            return (
                (torch.FloatTensor(atom_fea),
                 torch.FloatTensor(nbr_fea),
                 torch.LongTensor(nbr_fea_idx)),
                 target, struct_id)

    def load_dataset(self):
        """
        Loads only the required batch data
        """
        self.processed_data = []
        
        # Get the struct_ids we need
        batch_struct_ids = set(sid for sid, _ in self.id_prop_data)
        
        # If loading full dataset
        if len(batch_struct_ids) == self.total_size:
            for batch_idx in range(1, self.num_batches + 1):
                batch_file = os.path.join(self.process_dir, f'processed_data-{batch_idx}.npz')
                with np.load(batch_file, allow_pickle=True) as loader:
                    batch_data = loader['data']
                    for item in batch_data:
                        features_tuple, target, struct_id = item
                        processed_features = (
                            torch.FloatTensor(features_tuple[0]),
                            torch.FloatTensor(features_tuple[1]),
                            torch.LongTensor(features_tuple[2])
                        )
                        self.processed_data.append((processed_features, target, struct_id))
        else:
            # Create mapping from struct_id to batch number
            struct_to_batch = {}
            batch_idx = 1
            for i in range(0, self.total_size, self.batch_size):
                end_index = min(i + self.batch_size, self.total_size)
                for j in range(i, end_index):
                    struct_id, _ = self.id_prop_data[j]
                    struct_to_batch[struct_id] = batch_idx
                batch_idx += 1
            
            # Load only needed batches
            needed_batches = set(struct_to_batch[sid] for sid in batch_struct_ids)
            for batch_idx in needed_batches:
                batch_file = os.path.join(self.process_dir, f'processed_data-{batch_idx}.npz')
                with np.load(batch_file, allow_pickle=True) as loader:
                    batch_data = loader['data']
                    for item in batch_data:
                        features_tuple, target, struct_id = item
                        if struct_id in batch_struct_ids:
                            processed_features = (
                                torch.FloatTensor(features_tuple[0]),
                                torch.FloatTensor(features_tuple[1]),
                                torch.LongTensor(features_tuple[2])
                            )
                            self.processed_data.append((processed_features, target, struct_id))

    def save_dataset(self):
        """
        Saves processed data in batch-sized shards
        """
        print("Processing and saving dataset in batches...")
        
        for batch_idx in range(1, self.num_batches + 1):
            start_idx = (batch_idx - 1) * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.total_size)
            
            # Process batch
            batch_data = []
            for j in range(start_idx, end_idx):
                struct_id, target = self.id_prop_data[j]
                crystal = self.read_structure(struct_id)
                atom_fea, nbr_fea, nbr_fea_idx = self.process_structure(crystal, struct_id)
                batch_data.append(((atom_fea, nbr_fea, nbr_fea_idx), target, struct_id))
            
            # Save batch
            save_path = os.path.join(self.process_dir, f'processed_data-{batch_idx}.npz')
            np.savez_compressed(save_path, data=np.array(batch_data, dtype=object), allow_pickle=True)
        
        print(f"Saved {len(self.id_prop_data)} data points in {self.num_batches} batches")

    def __iter__(self):
        """Reset iterator index"""
        self.current_index = 0
        return self

    def clear_cache(self):
        """Clear any cached data"""
        if hasattr(self, 'processed_data'):
            self.processed_data = None