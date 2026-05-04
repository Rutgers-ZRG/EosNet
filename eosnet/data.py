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
import eosnet.fp as torch_fplib
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list
from ase import Atoms
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atoms_to_cell(atoms):
    """Convert ASE Atoms to (lat, rxyz, types, znucl) tuple for torch_fplib.

    types are 1-indexed (first-occurrence ordering, same as POSCAR convention).
    """
    lat = np.array(atoms.cell)
    rxyz = atoms.get_positions()
    znucl_list = []
    for z in atoms.numbers:
        if z not in znucl_list:
            znucl_list.append(z)
    znucl = np.array(znucl_list, dtype=int)
    z_to_type = {z: i + 1 for i, z in enumerate(znucl_list)}
    types = np.array([z_to_type[z] for z in atoms.numbers], dtype=int)
    return (lat, rxyz, types, znucl)


def _detect_data_format(root_dir):
    """Auto-detect dataset format: 'extxyz' or 'vasp'."""
    if os.path.isfile(os.path.join(root_dir, 'data.extxyz')):
        return 'extxyz'
    for f in os.listdir(root_dir):
        if f.endswith('.vasp'):
            return 'vasp'
    raise FileNotFoundError(
        f"No data.extxyz or .vasp files found in {root_dir}")


def _load_extxyz(root_dir):
    """Load all structures from data.extxyz, return {struct_id: Atoms} dict."""
    extxyz_path = os.path.join(root_dir, 'data.extxyz')
    all_atoms = ase_read(extxyz_path, index=':')
    atoms_dict = {}
    for i, atoms in enumerate(all_atoms):
        sid = str(atoms.info.get('struct_id',
                  atoms.info.get('config_type', f'struct_{i}')))
        atoms_dict[sid] = atoms
    return atoms_dict


# ---------------------------------------------------------------------------
# Data loaders / collate functions (unchanged)
# ---------------------------------------------------------------------------

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


def collate_pool_e3nn(dataset_list):
    """
    Collate for e3nn model format.

    For StructData with model_type='e3nn':
      ((atom_fea, edge_index, edge_vec), target, struct_id)

    Returns
    -------
    batch_atom_fea: torch.Tensor (N, orig_atom_fea_len)
    batch_edge_index: torch.LongTensor (2, E)
    batch_edge_vec: torch.Tensor (E, 3)
    crystal_atom_idx: list of LongTensor
    batch_target: torch.Tensor
    batch_struct_ids: list
    """
    # IdTargetData path
    if isinstance(dataset_list[0], tuple) and len(dataset_list[0]) == 2:
        batch_target, batch_struct_ids = [], []
        for struct_id, target in dataset_list:
            batch_target.append(torch.tensor([float(target)], dtype=torch.float))
            batch_struct_ids.append(struct_id)
        return torch.cat(batch_target, dim=0), batch_struct_ids

    # StructData e3nn path
    batch_atom_fea = []
    batch_edge_index = []
    batch_edge_vec = []
    crystal_atom_idx = []
    batch_target = []
    batch_struct_ids = []

    base_idx = 0
    for (atom_fea, edge_index, edge_vec), target, struct_id in dataset_list:
        n_i = atom_fea.shape[0]

        batch_atom_fea.append(atom_fea)
        batch_edge_index.append(edge_index + base_idx)
        batch_edge_vec.append(edge_vec)

        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        if not isinstance(target, torch.Tensor):
            target = torch.tensor([float(target)], dtype=torch.float)
        batch_target.append(target)
        batch_struct_ids.append(struct_id)

        base_idx += n_i

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_edge_index, dim=1),
            torch.cat(batch_edge_vec, dim=0),
            crystal_atom_idx), \
            torch.stack(batch_target), \
            batch_struct_ids


# ---------------------------------------------------------------------------
# Graph construction (now using ASE neighbor_list — no pymatgen)
# ---------------------------------------------------------------------------

def get_edge_data(atoms, radius):
    """Build edge list with displacement vectors for e3nn.

    Uses ASE's C-implemented neighbor_list with PBC support.

    Parameters
    ----------
    atoms: ASE Atoms object
    radius: float, cutoff radius

    Returns
    -------
    edge_index: np.array (2, num_edges) — [source, destination]
    edge_vec: np.array (num_edges, 3) — displacement vectors
    """
    i_idx, j_idx, D_vec = neighbor_list('ijD', atoms, radius)
    # Sort by (source atom, distance) for deterministic ordering
    distances = np.linalg.norm(D_vec, axis=1)
    sort_order = np.lexsort((distances, i_idx))
    edge_index = np.array([i_idx[sort_order], j_idx[sort_order]], dtype=np.int64)
    edge_vec = D_vec[sort_order].astype(np.float64)
    return edge_index, edge_vec


def get_neighbor_info(atoms, radius, max_num_nbr, struct_id=None, raw_nbr_data=None):
    """Get neighbor information using ASE's C neighbor list.

    Args:
        atoms: ASE Atoms object
        radius: Cutoff radius for neighbor search
        max_num_nbr: Maximum number of neighbors per atom
        struct_id: Optional structure identifier for warnings
        raw_nbr_data: optional (i_idx, j_idx, d_scalar, D_vec) to skip recomputation

    Returns:
        nbr_indices: np.array [n_atoms, max_num_nbr]
        nbr_distances: np.array [n_atoms, max_num_nbr]
        displacement_vectors: np.array [n_atoms, max_num_nbr, 3]
    """
    if raw_nbr_data is not None:
        i_idx, j_idx, d_scalar, D_vec = raw_nbr_data
    else:
        i_idx, j_idx, d_scalar, D_vec = neighbor_list('ijdD', atoms, radius)

    n_atoms = len(atoms)
    nbr_indices = np.zeros((n_atoms, max_num_nbr), dtype=np.int64)
    nbr_distances = np.full((n_atoms, max_num_nbr), radius + 1.0)
    displacement_vectors = np.zeros((n_atoms, max_num_nbr, 3))

    for atom_i in range(n_atoms):
        mask = (i_idx == atom_i)
        local_j = j_idx[mask]
        local_d = d_scalar[mask]
        local_D = D_vec[mask]

        # Sort by distance
        order = np.argsort(local_d)
        local_j = local_j[order]
        local_d = local_d[order]
        local_D = local_D[order]

        n_nbr = len(local_j)
        if n_nbr < max_num_nbr:
            if struct_id:
                warnings.warn(f'{struct_id} not find enough neighbors to '
                              f'build graph. If it happens frequently, '
                              f'consider increase radius.')
            nbr_indices[atom_i, :n_nbr] = local_j
            nbr_distances[atom_i, :n_nbr] = local_d
            displacement_vectors[atom_i, :n_nbr] = local_D
            # Pad with last valid neighbor
            if n_nbr > 0:
                nbr_indices[atom_i, n_nbr:] = local_j[-1]
                nbr_distances[atom_i, n_nbr:] = local_d[-1]
                displacement_vectors[atom_i, n_nbr:] = local_D[-1]
        else:
            nbr_indices[atom_i] = local_j[:max_num_nbr]
            nbr_distances[atom_i] = local_d[:max_num_nbr]
            displacement_vectors[atom_i] = local_D[:max_num_nbr]

    return nbr_indices, nbr_distances, displacement_vectors


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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class IdTargetData(Dataset):
    """
    A simple dataset to load just the struct_id and target.
    Supports both id_prop.csv (vasp) and data.extxyz formats.
    """
    def __init__(self, root_dir, random_seed=42, data_format='auto'):
        self.root_dir = root_dir

        if data_format == 'auto':
            data_format = _detect_data_format(root_dir)
        self.data_format = data_format

        if data_format == 'extxyz':
            atoms_dict = _load_extxyz(root_dir)
            self.id_prop_data = []
            for sid, atoms in atoms_dict.items():
                target = atoms.info.get('target', 0.0)
                self.id_prop_data.append([sid, str(target)])
        else:
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


class StructData(Dataset):
    """
    Dataset for crystal structures. Supports two formats:

    1. POSCAR (.vasp) — original CGCNN format:
       root_dir/
       ├── id_prop.csv
       ├── struct_id_0.vasp
       └── ...

    2. Extended XYZ — single-file format:
       root_dir/
       └── data.extxyz   (struct_id and target in atoms.info)

    Parameters
    ----------
    id_prop_data: list of [struct_id, target] pairs
    root_dir: str
    max_num_nbr: int
    radius: float (Angstroms)
    nx: int — fingerprint dimension
    lmax: int — 0 for s-orbital only
    model_type: 'cgcnn' or 'e3nn'
    data_format: 'auto', 'extxyz', or 'vasp'
    atoms_dict: optional pre-loaded {struct_id: Atoms} dict
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
                 save_to_disk=False,
                 model_type='cgcnn',
                 data_format='auto',
                 atoms_dict=None,
                 no_gom=False):
        self.root_dir = root_dir
        self.id_prop_data = id_prop_data
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.model_type = model_type
        self.nx = nx
        self.lmax = lmax
        self.no_gom = no_gom
        self.save_to_disk = save_to_disk
        if save_to_disk and model_type == 'e3nn':
            raise NotImplementedError(
                'save_to_disk is not yet supported for e3nn model_type. '
                'Use in-memory caching (save_to_disk=False) instead.')
        # lmax=0 for s-only, lmax>0 for s+p orbitals
        assert nx >= comb(max_num_nbr, 2), 'nx is too small for the given max_num_nbr!'
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.total_size = len(self.id_prop_data)

        # Data format and pre-loaded atoms
        if data_format == 'auto':
            data_format = _detect_data_format(root_dir)
        self.data_format = data_format

        if atoms_dict is not None:
            self._atoms_dict = atoms_dict
        elif data_format == 'extxyz':
            self._atoms_dict = _load_extxyz(root_dir)
        else:
            self._atoms_dict = None  # read .vasp on demand

        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step, var=var)

        # Create processed directory if it doesn't exist
        self.process_dir = os.path.join(self.root_dir, 'saved_npz_files')

        self.processed_data = None

        if self.drop_last:
            self.num_batches = self.total_size // self.batch_size
        else:
            self.num_batches = (self.total_size + self.batch_size - 1) // self.batch_size

        # Check if all batch files exist (vasp/cgcnn only)
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

        # Fingerprint cache (populated lazily or via precompute)
        self._fp_cache = {}
        # Full processed-result cache: struct_id → (processed_features, target, struct_id)
        self._processed_cache = {}
        # O(1) lookup: struct_id → index in id_prop_data
        self._sid_to_idx = {sid: i for i, (sid, _) in enumerate(self.id_prop_data)}

    def _load_atoms(self, struct_id):
        """Load ASE Atoms for a structure (from dict or .vasp file)."""
        if self._atoms_dict is not None:
            return self._atoms_dict[struct_id]
        cell_file = os.path.join(self.root_dir, struct_id + '.vasp')
        return ase_read(cell_file)

    def _precompute_fingerprints(self):
        """Batch-compute all GOM fingerprints."""
        import time
        cutoff = np.float64(int(np.sqrt(self.radius)) * 3)
        natx = int(self.nx)

        struct_ids = list(dict.fromkeys(sid for sid, _ in self.id_prop_data))
        cells = []
        valid_ids = []
        for sid in struct_ids:
            try:
                atoms = self._load_atoms(sid)
                cells.append(_atoms_to_cell(atoms))
                valid_ids.append(sid)
            except Exception as e:
                print(f"Warning: could not load {sid}: {e}")

        t0 = time.time()
        fps = torch_fplib.get_lfp_fast_batch(
            cells, cutoff=cutoff, natx=natx, device='cpu')
        dt = time.time() - t0
        print(f"Precomputed {len(fps)} fingerprints in {dt:.1f}s "
              f"({dt/len(fps)*1000:.1f}ms/struct)")

        for sid, fp in zip(valid_ids, fps):
            self._fp_cache[sid] = fp.numpy()

    def __len__(self):
        return len(self.id_prop_data)

    def get_fp_mat(self, atoms_or_file, struct_id=None, ase_neighbors=None):
        """Compute GOM fingerprint matrix.

        Args:
            atoms_or_file: ASE Atoms object (preferred) or path to .vasp file
            struct_id: cache key (derived from filename if not given)
            ase_neighbors: optional (i_idx, j_idx, D_vec) from ASE neighbor_list
                           to avoid redundant neighbor search
        """
        # Determine cache key
        if struct_id is not None:
            cache_key = struct_id
        elif isinstance(atoms_or_file, str):
            cache_key = os.path.splitext(os.path.basename(atoms_or_file))[0]
        else:
            cache_key = None

        if cache_key and cache_key in self._fp_cache:
            return self._fp_cache[cache_key]

        # Get cell tuple
        if isinstance(atoms_or_file, Atoms):
            cell = _atoms_to_cell(atoms_or_file)
        else:
            cell = _atoms_to_cell(ase_read(atoms_or_file))

        lat, rxyz, types, znucl = cell
        natx = int(self.nx)
        lmax = int(self.lmax)
        cutoff = np.float64(int(np.sqrt(self.radius)) * 3)

        if lmax == 0:
            lseg = 1
            orbital = 's'
        else:
            lseg = 4
            orbital = 'sp'

        if len(rxyz) != len(types) or len(set(types)) != len(znucl):
            print(f"Structure {struct_id or 'unknown'} is erroneous!")
            fp = np.zeros((len(rxyz), lseg * natx), dtype=np.float64)
        else:
            if (ase_neighbors is not None and cutoff <= self.radius):
                # Fast path: reuse ASE neighbor list, skip redundant search
                i_idx, j_idx, D_vec = ase_neighbors
                fp = torch_fplib.get_lfp_from_ase_neighbors(
                    rxyz, atoms_or_file.numbers, i_idx, j_idx, D_vec,
                    cutoff=cutoff, natx=natx, device='cpu').numpy()
            else:
                fp = torch_fplib.get_lfp_fast(
                    cell, cutoff=cutoff, natx=natx,
                    orbital=orbital, device='cpu').numpy()

        if cache_key:
            self._fp_cache[cache_key] = fp
        return fp

    def _build_atom_features(self, atoms, struct_id, ase_neighbors=None):
        """Build one-hot + GOM fingerprint atom features."""
        chem_nums = list(atoms.numbers)
        max_atomic_number = max(max(chem_nums), 112)
        one_hot = np.zeros((len(atoms), max_atomic_number + 1), dtype=np.int32)
        for i, z in enumerate(chem_nums):
            one_hot[i, z] = 1

        if self.no_gom:
            return one_hot.astype(np.float32)

        comb_n_nbr = comb(self.max_num_nbr, 2)
        lseg = 4 if self.lmax > 0 else 1
        fp_mat = self.get_fp_mat(atoms, struct_id=struct_id,
                                 ase_neighbors=ase_neighbors)
        fp_mat = fp_mat[:, :lseg * comb_n_nbr]
        fp_mat[np.abs(fp_mat) < 1.0e-10] = 0.0
        norms = np.linalg.norm(fp_mat, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-30, 1.0, norms)
        fp_mat = fp_mat / norms
        return np.hstack((one_hot, fp_mat))

    def process_structure(self, atoms, struct_id):
        """Convert ASE Atoms into features for CGCNN model.

        Args:
            atoms: ASE Atoms object
            struct_id: Structure identifier
        """
        # Single ASE neighbor search for both graph edges and fingerprints
        i_idx, j_idx, d_scalar, D_vec = neighbor_list('ijdD', atoms, self.radius)
        nbr_indices, nbr_distances, _ = get_neighbor_info(
            atoms, self.radius, self.max_num_nbr, struct_id=struct_id,
            raw_nbr_data=(i_idx, j_idx, d_scalar, D_vec)
        )
        atom_fea = self._build_atom_features(atoms, struct_id,
                                              ase_neighbors=(i_idx, j_idx, D_vec))
        nbr_fea = self.gdf.expand(nbr_distances)
        return atom_fea, nbr_fea, nbr_indices

    def process_structure_e3nn(self, atoms, struct_id):
        """Build graph data in e3nn format from ASE Atoms.

        Returns:
            atom_fea, edge_index, edge_vec
        """
        # Single ASE neighbor search for both edges and fingerprints
        i_idx, j_idx, D_vec = neighbor_list('ijD', atoms, self.radius)
        distances = np.linalg.norm(D_vec, axis=1)
        sort_order = np.lexsort((distances, i_idx))
        edge_index = np.array([i_idx[sort_order], j_idx[sort_order]], dtype=np.int64)
        edge_vec = D_vec[sort_order].astype(np.float64)
        atom_fea = self._build_atom_features(atoms, struct_id,
                                              ase_neighbors=(i_idx, j_idx, D_vec))
        return atom_fea, edge_index, edge_vec

    def __getitem__(self, idx):
        """Get a single data point"""
        struct_id, target = self.id_prop_data[idx]

        # Check in-memory cache first (persists across epochs)
        if struct_id in self._processed_cache:
            return self._processed_cache[struct_id]

        # If data is already processed (cgcnn format only)
        if hasattr(self, 'processed_data') and self.processed_data is not None:
            return self.processed_data[idx]

        # Load ASE Atoms (no pymatgen)
        atoms = self._load_atoms(struct_id)

        if self.model_type == 'e3nn':
            atom_fea, edge_index, edge_vec = \
                self.process_structure_e3nn(atoms, struct_id)
            processed_features = (
                torch.FloatTensor(atom_fea),
                torch.LongTensor(edge_index),
                torch.FloatTensor(edge_vec.astype(np.float32))
            )
        else:
            atom_fea, nbr_fea, nbr_fea_idx = \
                self.process_structure(atoms, struct_id)
            processed_features = (
                torch.FloatTensor(atom_fea),
                torch.FloatTensor(nbr_fea),
                torch.LongTensor(nbr_fea_idx)
            )
        result = (processed_features, target, struct_id)
        self._processed_cache[struct_id] = result
        return result

    def process_item(self, idx):
        struct_id, target = self.id_prop_data[idx]
        atoms = self._load_atoms(struct_id)
        atom_fea, nbr_fea, nbr_fea_idx = self.process_structure(atoms, struct_id)

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
                atoms = self._load_atoms(struct_id)
                atom_fea, nbr_fea, nbr_fea_idx = self.process_structure(atoms, struct_id)
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
        """Clear all cached data to free memory."""
        if hasattr(self, 'processed_data'):
            self.processed_data = None
        if hasattr(self, '_fp_cache'):
            self._fp_cache.clear()
        if hasattr(self, '_processed_cache'):
            self._processed_cache.clear()
