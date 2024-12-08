#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
from math import comb


class AtomConvLayer(nn.Module):
    """Convolutional operation on atoms"""
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
        
        # Node update components
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len, 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, bond_weights_i, bond_weights_j):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Maximum number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
            Atom features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Indices of M neighbors of each atom
        bond_weights_i: Variable(torch.Tensor) shape (N, M)
            Bond weights for center atoms
        bond_weights_j: Variable(torch.Tensor) shape (N, M)
            Bond weights for neighbor atoms

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
            Atom features after convolution
        """
        # Get neighbor features
        N, M = nbr_fea_idx.shape
        assert bond_weights_i.shape == (N, M), f"bond_weights_i should be shape ({N}, {M}), got {bond_weights_i.shape}"
        assert bond_weights_j.shape == (N, M), f"bond_weights_j should be shape ({N}, {M}), got {bond_weights_j.shape}"
        
        # Get neighbor features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx.view(-1)].view(N, M, -1)
        
        # Concatenate atom features and bond features
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea,
             nbr_fea], dim=2)
        
        # Apply pair-wise bond weights
        total_gated_fea = total_nbr_fea * bond_weights_i.unsqueeze(-1) * bond_weights_j.unsqueeze(-1)
        
        # Sum over neighbors
        total_gated_fea = total_gated_fea.sum(dim=1)
        
        # Update features
        total_gated_fea = self.bn1(self.fc_full(total_gated_fea))
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        atom_nbr_fea = nbr_filter * nbr_core
        atom_out_fea = self.bn2(atom_nbr_fea)
        atom_out_fea = self.softplus2(atom_out_fea)
        
        return atom_out_fea


class BondConvLayer(nn.Module):
    """Bond convolution layer that updates bond features using atom information"""
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
        super(BondConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        # Bond update network
        self.bond_update = nn.Sequential(
            nn.Linear(2*atom_fea_len + nbr_fea_len, 2*nbr_fea_len),
            nn.SiLU(),
            nn.Linear(2*nbr_fea_len, nbr_fea_len),
            nn.SiLU(),
            nn.Linear(nbr_fea_len, nbr_fea_len)
        )

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, bond_weights_i, bond_weights_j):
        """
        Forward pass

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Neighbor indices
        bond_weights_i: torch.Tensor shape (N, M)
            Bond weights for center atoms
        bond_weights_j: torch.Tensor shape (N, M)
            Bond weights for neighbor atoms

        Returns
        -------
        new_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Updated bond features
        """
        # Get neighbor features
        N, M = nbr_fea_idx.shape
        nbr_atom_fea = atom_fea[nbr_fea_idx.view(-1)].view(N, M, -1)
        
        # Combine atom and bond features
        atom_pair_fea = torch.cat([
            atom_fea.unsqueeze(1).expand(N, M, -1),
            nbr_atom_fea
        ], dim=2)
        
        total_fea = torch.cat([atom_pair_fea, nbr_fea], dim=2)
        
        # Update features
        new_nbr_fea = self.bond_update(total_fea)
        
        # Apply bond weights
        new_nbr_fea = new_nbr_fea * bond_weights_i.unsqueeze(-1)
        
        return new_nbr_fea


class ConvBlock(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, update_bond=False, resnet=True):
        super(ConvBlock, self).__init__()
        
        self.resnet = resnet
        
        # Atom convolution
        self.atom_conv = AtomConvLayer(atom_fea_len, nbr_fea_len)
        self.atom_norm = nn.LayerNorm(atom_fea_len)
        
        # Bond convolution
        self.update_bond = update_bond
        if update_bond:
            self.bond_conv = BondConvLayer(atom_fea_len, nbr_fea_len)
            self.bond_norm = nn.LayerNorm(nbr_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx,
                bond_weights_ag_i, bond_weights_ag_j,
                bond_weights_bg_i=None, bond_weights_bg_j=None):
        """
        Forward pass through both atom and bond convolutions
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Neighbor indices
        bond_weights_ag_i: torch.Tensor shape (N, M)
            Bond weights for atom graph center atoms
        bond_weights_ag_j: torch.Tensor shape (N, M)
            Bond weights for atom graph neighbor atoms
        bond_weights_bg_i: torch.Tensor shape (N, M)
            Bond weights for bond graph center atoms
        bond_weights_bg_j: torch.Tensor shape (N, M)
            Bond weights for bond graph neighbor atoms
            
        Returns
        -------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Updated atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Updated bond features
        """
        # Add shape assertions
        assert atom_fea.dim() == 2, f"atom_fea should be 2D, got shape {atom_fea.shape}"
        assert nbr_fea.dim() == 3, f"nbr_fea should be 3D, got shape {nbr_fea.shape}"
        assert nbr_fea_idx.dim() == 2, f"nbr_fea_idx should be 2D, got shape {nbr_fea_idx.shape}"
        
        # Local skip connection for atom features
        atom_identity = atom_fea
        atom_fea = self.atom_conv(atom_fea, nbr_fea, nbr_fea_idx, 
                                 bond_weights_ag_i, bond_weights_ag_j)
        if self.resnet:
            atom_fea = self.atom_norm(atom_fea + atom_identity)
        else:
            atom_fea = self.atom_norm(atom_fea)
        
        # Local skip connection for bond features
        if self.update_bond:
            bond_identity = nbr_fea
            nbr_fea = self.bond_conv(atom_fea, nbr_fea, nbr_fea_idx,
                                    bond_weights_bg_i, bond_weights_bg_j)
            if self.resnet:
                nbr_fea = self.bond_norm(nbr_fea + bond_identity)
            else:
                nbr_fea = self.bond_norm(nbr_fea)
        
        return atom_fea, nbr_fea


class GraphConvNet(nn.Module):
    """Graph convolution network for updating atom and bond features"""
    def __init__(self, atom_fea_len, nbr_fea_len, n_conv, update_bond=False, resnet=True):
        """
        Initialize GraphConvNet.
        
        Parameters
        ----------
        atom_fea_len: int
            Number of atom features
        nbr_fea_len: int
            Number of bond features
        n_conv: int
            Number of convolution layers
        update_bond: bool
            Whether to update bond features
        """
        super(GraphConvNet, self).__init__()
        self.update_bond = update_bond
        self.resnet = resnet
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                atom_fea_len=atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                update_bond=update_bond,
                resnet=resnet
            ) for _ in range(n_conv)
        ])
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx,
                bond_weights_ag_i, bond_weights_ag_j,
                bond_weights_bg_i=None, bond_weights_bg_j=None):
        """
        Forward pass
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Neighbor indices
        bond_weights_ag_i: torch.Tensor shape (N, M)
            Bond weights for atom graph center atoms
        bond_weights_ag_j: torch.Tensor shape (N, M)
            Bond weights for atom graph neighbor atoms
        bond_weights_bg_i: torch.Tensor shape (N, M), optional
            Bond weights for bond graph center atoms
        bond_weights_bg_j: torch.Tensor shape (N, M), optional
            Bond weights for bond graph neighbor atoms
            
        Returns
        -------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Updated atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Updated bond features
        """
        for conv_block in self.conv_blocks:
            atom_fea, nbr_fea = conv_block(
                atom_fea, nbr_fea, nbr_fea_idx,
                bond_weights_ag_i, bond_weights_ag_j,
                bond_weights_bg_i, bond_weights_bg_j
            )
        return atom_fea, nbr_fea


class AttentionReadout(nn.Module):
    """Multi-head attention readout layer"""
    def __init__(self, atom_fea_len, hidden_dim=32, num_heads=None, dropout=0.1):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        
        if num_heads is None:
            self.num_heads = max(1, atom_fea_len // hidden_dim)
        else:
            self.num_heads = num_heads
            
        self.key = nn.Sequential(
            nn.Linear(atom_fea_len, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, self.num_heads)
        )
        
        self.project = nn.Sequential(
            nn.Linear(self.num_heads * atom_fea_len, atom_fea_len),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, atom_fea, crystal_atom_idx):
        """
        Aggregate atom features to crystal features using attention
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom features after convolution
        crystal_atom_idx: list of torch.LongTensor
            Indices of atoms in each crystal
            
        Returns
        -------
        torch.Tensor shape (batch_size, atom_fea_len)
            Crystal features
        """
        weights = self.key(atom_fea)
        
        crystal_feas = []
        for idx_map in crystal_atom_idx:
            # Get features for this crystal
            crystal_atom_fea = atom_fea[idx_map]
            crystal_weights = self.softmax(weights[idx_map])
            
            # Weighted sum for each head separately
            weighted_sum = torch.zeros(self.num_heads, self.atom_fea_len, 
                                    device=atom_fea.device)
            for h in range(self.num_heads):
                head_weights = crystal_weights[:, h].unsqueeze(1)
                weighted_sum[h] = (crystal_atom_fea * head_weights).sum(0)
            
            # Concatenate results from all heads and project back
            crystal_fea = weighted_sum.view(-1)
            crystal_fea = self.project(crystal_fea)
            crystal_feas.append(crystal_fea)
            
        return torch.stack(crystal_feas, dim=0)


class EosNet(nn.Module):
    """Neural network for predicting equation of state parameters"""
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64,
                 h_fea_len=128, n_conv=3, n_h=1, update_bond=False, 
                 classification=False, attention=False, resnet=True,
                 hidden_dim=32, num_heads=None, dropout=0.1):
        """
        Initialize EosNet.
        
        Parameters
        ----------
        orig_atom_fea_len: int
            Number of input atom features
        nbr_fea_len: int
            Number of input bond features
        atom_fea_len: int
            Number of hidden atom features
        h_fea_len: int
            Number of hidden features in FC layers
        n_conv: int
            Number of convolution layers
        n_h: int
            Number of hidden layers after pooling
        update_bond: bool
            Whether to update bond features
        classification: bool
            Whether to do classification
        attention: bool
            Whether to use attention readout
        resnet: bool
            Whether to use resnet style normalization
        dropout: float
            Dropout rate for classification task
        """
        super(EosNet, self).__init__()
        
        self.classification = classification
        self.update_bond = update_bond
        self.attention = attention
        
        # Initial embedding
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        # Bond weights for atom graph
        self.bond_weights_ag = nn.Sequential(
            nn.Linear(nbr_fea_len, nbr_fea_len),
            nn.SiLU(),
            nn.Linear(nbr_fea_len, 1, bias=False)
        )
        
        # Bond weights for bond graph (only if update_bond is True)
        if update_bond:
            self.bond_weights_bg = nn.Sequential(
                nn.Linear(nbr_fea_len, nbr_fea_len),
                nn.SiLU(),
                nn.Linear(nbr_fea_len, 1, bias=False)
            )
        
        # Graph convolution network
        self.graph_net = GraphConvNet(
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            n_conv=n_conv,
            update_bond=update_bond,
            resnet=resnet
        )
        
        # Attention readout
        if attention:
            self.attention_readout = AttentionReadout(
                atom_fea_len,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Transition from conv to fc
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        # FC layers with residual connections
        if n_h > 1:
            self.fcs = nn.ModuleList()
            self.fc_projections = nn.ModuleList()
            self.softpluses = nn.ModuleList()
            self.fc_dropouts = nn.ModuleList()
            
            for _ in range(n_h-1):
                self.fcs.append(nn.Linear(h_fea_len, h_fea_len))
                self.fc_projections.append(nn.Linear(h_fea_len, h_fea_len))
                self.softpluses.append(nn.Softplus())
                self.fc_dropouts.append(nn.Dropout(p=dropout))
        
        # Output layer
        if classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
            
        # Transition dropout
        self.conv_to_fc_dropout = nn.Dropout(p=dropout)
        
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
            
        Returns
        -------
        torch.Tensor shape (N0, atom_fea_len)
            Pooled crystal features
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
            Original atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Bond features
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Neighbor indices
        crystal_atom_idx: list of torch.LongTensor of length N0
            Mapping from crystal idx to atom idx
            
        Returns
        -------
        out: torch.Tensor shape (N_crystals, 1) or (N_crystals, 2)
            Predicted properties
        """
        # Generate bond weights
        N, M = nbr_fea_idx.shape
        nbr_fea_flat = nbr_fea.view(-1, nbr_fea.shape[-1])
        
        # Weights for atom graph
        bond_weights_ag = self.bond_weights_ag(nbr_fea_flat).view(N, M)
        bond_weights_ag_i = torch.sigmoid(bond_weights_ag)
        bond_weights_ag_j = torch.gather(bond_weights_ag_i, 0, nbr_fea_idx)
        
        # Weights for bond graph (if updating bonds)
        if self.update_bond:
            bond_weights_bg = self.bond_weights_bg(nbr_fea_flat).view(N, M)
            bond_weights_bg_i = torch.sigmoid(bond_weights_bg)
            bond_weights_bg_j = torch.gather(bond_weights_bg_i, 0, nbr_fea_idx)
        else:
            bond_weights_bg_i = bond_weights_bg_j = None
        
        # Initial atom embedding
        atom_fea = self.embedding(atom_fea)
        
        # Update features through graph convolutions
        atom_fea, nbr_fea = self.graph_net(
            atom_fea, nbr_fea, nbr_fea_idx,
            bond_weights_ag_i, bond_weights_ag_j,
            bond_weights_bg_i, bond_weights_bg_j
        )
        
        # Choose readout method
        if self.attention:
            crys_fea = self.attention_readout(atom_fea, crystal_atom_idx)
        else:
            crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        # Apply dropout after pooling/attention
        crys_fea = self.conv_to_fc_dropout(crys_fea)
        
        # Transition to fc layers
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        # Apply dropout for classification
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        
        # FC layers with residual connections
        if hasattr(self, 'fcs'):
            for fc, softplus, projection, dropout in zip(
                self.fcs, self.softpluses, self.fc_projections, self.fc_dropouts):
                identity = crys_fea
                crys_fea = softplus(fc(crys_fea))
                crys_fea = dropout(crys_fea)
                identity = projection(identity)
                crys_fea = crys_fea + identity
        
        # Output
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
            
        return out