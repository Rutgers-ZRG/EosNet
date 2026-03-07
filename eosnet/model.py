#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn


class AtomConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(AtomConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([AtomConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


# ============================================================
# EOSNet v2: e3nn equivariant backbone + GOM fingerprint features
# ============================================================

try:
    from e3nn import o3
    from e3nn.nn import Gate
    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False


def scatter_sum(src, index, dim_size):
    """
    Scatter-add src into output of size dim_size along dim 0.

    Parameters
    ----------
    src: torch.Tensor shape (E, D)
    index: torch.LongTensor shape (E,)
    dim_size: int

    Returns
    -------
    out: torch.Tensor shape (dim_size, D)
    """
    out = src.new_zeros(dim_size, src.shape[1])
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    return out


class RadialBasis(nn.Module):
    """Gaussian radial basis functions with smooth polynomial cutoff."""

    def __init__(self, num_basis, cutoff):
        super(RadialBasis, self).__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer('centers', centers)
        self.width = 0.5 / ((centers[1] - centers[0]) ** 2) if num_basis > 1 else 1.0

    def forward(self, dist):
        """
        dist: (E,) distances → (E, num_basis) radial features
        """
        basis = torch.exp(-self.width * (dist.unsqueeze(-1) - self.centers) ** 2)
        u = (dist / self.cutoff).clamp(max=1.0)
        envelope = (1 - u ** 2) ** 3
        return basis * envelope.unsqueeze(-1)


class InteractionBlock(nn.Module):
    """e3nn tensor product interaction with gated nonlinearity."""

    def __init__(self, irreps_in, irreps_sh, irreps_out, num_radial,
                 fc_hidden=64):
        super(InteractionBlock, self).__init__()

        # Parse output irreps for Gate activation
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out
                                     if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out
                                    if ir.l > 0])
        num_gates = sum(mul for mul, ir in irreps_gated)
        irreps_gates = o3.Irreps(f"{num_gates}x0e") if num_gates > 0 \
            else o3.Irreps("")

        act_scalars = [torch.nn.functional.silu] * len(irreps_scalars)
        act_gates = [torch.sigmoid] * len(irreps_gates)

        self.gate = Gate(
            irreps_scalars, act_scalars,
            irreps_gates, act_gates,
            irreps_gated
        )

        # Tensor product: node_fea ⊗ edge_sh → gate input
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_sh, self.gate.irreps_in,
            shared_weights=False
        )

        # MLP: radial basis → tensor product weights
        self.fc = nn.Sequential(
            nn.Linear(num_radial, fc_hidden),
            nn.SiLU(),
            nn.Linear(fc_hidden, self.tp.weight_numel)
        )

        # Self-interaction (residual path)
        self.self_interaction = o3.Linear(irreps_in, irreps_out)

    def forward(self, node_fea, edge_sh, edge_radial,
                edge_src, edge_dst, num_nodes):
        # Self-interaction (residual)
        self_fea = self.self_interaction(node_fea)

        # Compute TP weights from radial features
        tp_weights = self.fc(edge_radial)

        # Message: tensor product of neighbor features with edge SH
        messages = self.tp(node_fea[edge_src], edge_sh, tp_weights)

        # Aggregate messages to destination nodes
        aggregated = scatter_sum(messages, edge_dst, num_nodes)

        # Apply gated activation
        out = self.gate(aggregated)

        # Residual connection
        out = out + self_fea

        return out


class EOSNetV2(nn.Module):
    """
    EOSNet v2: e3nn equivariant backbone + GOM fingerprint node features.

    Replaces CGCNN message passing with tensor product interactions
    while keeping GOM eigenvalues as input node features (l=0 scalars).
    """

    def __init__(self, orig_atom_fea_len,
                 irreps_hidden="32x0e+16x1o+8x2e",
                 max_ell=2, n_conv=3, num_radial_basis=16,
                 radial_cutoff=8.0, fc_hidden=64,
                 h_fea_len=128, n_h=1, classification=False):
        super(EOSNetV2, self).__init__()
        assert E3NN_AVAILABLE, \
            "e3nn is required for EOSNetV2. Install with: pip install e3nn"

        self.classification = classification
        self.radial_cutoff = radial_cutoff
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_ell)

        # Count scalar channels in hidden irreps
        num_scalars = sum(mul for mul, ir in self.irreps_hidden
                          if ir.l == 0)

        # Node embedding: scalar input features → hidden scalar channels
        self.embedding = nn.Sequential(
            nn.Linear(orig_atom_fea_len, num_scalars),
            nn.SiLU(),
            nn.Linear(num_scalars, num_scalars)
        )

        # Radial basis
        self.radial_basis = RadialBasis(num_radial_basis, radial_cutoff)

        # Interaction blocks
        irreps_input = o3.Irreps(f"{num_scalars}x0e")
        self.convs = nn.ModuleList()
        irreps_in = irreps_input
        for _ in range(n_conv):
            self.convs.append(InteractionBlock(
                irreps_in=irreps_in,
                irreps_sh=self.irreps_sh,
                irreps_out=self.irreps_hidden,
                num_radial=num_radial_basis,
                fc_hidden=fc_hidden,
            ))
            irreps_in = self.irreps_hidden

        # Readout: extract l=0 (scalar) features by irreps slicing
        self._scalar_indices = []
        offset = 0
        for mul, ir in self.irreps_hidden:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 0:
                self._scalar_indices.extend(range(offset, offset + dim))
            offset += dim
        self._scalar_indices = torch.LongTensor(self._scalar_indices)
        self._num_output_scalars = len(self._scalar_indices)

        # FC layers for prediction
        self.conv_to_fc = nn.Linear(self._num_output_scalars, h_fea_len)
        self.conv_to_fc_act = nn.SiLU()

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.acts = nn.ModuleList(
                [nn.SiLU() for _ in range(n_h - 1)])

        if classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, edge_index, edge_vec, crystal_atom_idx):
        """
        Forward pass

        Parameters
        ----------

        atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Node features (one-hot Z + GOM eigenvalues)
        edge_index: torch.LongTensor shape (2, E)
          [source, destination] edge indices
        edge_vec: torch.Tensor shape (E, 3)
          Displacement vectors (neighbor_image - center position)
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from crystal idx to atom idx
        """
        num_nodes = atom_fea.shape[0]
        edge_src, edge_dst = edge_index

        # Edge features: spherical harmonics + radial basis
        edge_length = edge_vec.norm(dim=-1)
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec,
            normalize=True, normalization='component'
        )
        edge_radial = self.radial_basis(edge_length)

        # Node embedding (all-scalar input)
        node_fea = self.embedding(atom_fea)

        # Interaction blocks
        for conv in self.convs:
            node_fea = conv(node_fea, edge_sh, edge_radial,
                            edge_src, edge_dst, num_nodes)

        # Extract l=0 (scalar) features for readout (robust to irreps ordering)
        scalar_fea = node_fea[:, self._scalar_indices.to(node_fea.device)]

        # Crystal-level mean pooling
        crys_fea = self.pooling(scalar_fea, crystal_atom_idx)

        # FC layers
        crys_fea = self.conv_to_fc_act(self.conv_to_fc(crys_fea))

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'acts'):
            for fc, act in zip(self.fcs, self.acts):
                crys_fea = act(fc(crys_fea))

        out = self.fc_out(crys_fea)

        if self.classification:
            out = self.logsoftmax(out)

        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """Mean pooling over atoms in each crystal."""
        assert sum(len(idx) for idx in crystal_atom_idx) == \
            atom_fea.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)