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

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, bond_weights):
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
        bond_weights: Variable(torch.Tensor) shape (N, M)
          Weights for scaling messages

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        N, M = nbr_fea_idx.shape
        
        # Node update
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([
            atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea, nbr_fea
        ], dim=2)
        
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        # Scale messages with bond weights
        nbr_filter = nbr_filter * bond_weights.unsqueeze(-1)
        nbr_core = nbr_core * bond_weights.unsqueeze(-1)
        
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        atom_out_fea = self.softplus2(atom_in_fea + nbr_sumed)
        
        return atom_out_fea


class BondConvLayer(nn.Module):
    """
    Bond convolution layer that updates bond features using atom and angle information
    """
    def __init__(self, atom_fea_len, nbr_fea_len, angle_fea_len, max_num_nbr):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        angle_fea_len: int
            Number of angle features
        max_num_nbr: int
            The maximum number of neighbors
        """
        super(BondConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.angle_fea_len = angle_fea_len
        self.max_num_nbr = max_num_nbr
        
        total_fea_len = (2 * self.atom_fea_len +
                        2 * self.nbr_fea_len +
                        comb(self.max_num_nbr, 2) * self.angle_fea_len)
        
        # Bond update network
        self.bond_update = nn.Sequential(
            nn.Linear(total_fea_len, 2 * self.nbr_fea_len),
            nn.BatchNorm1d(2 * self.nbr_fea_len),
            nn.Softplus(),
            nn.Linear(2 * self.nbr_fea_len, self.nbr_fea_len),
            nn.BatchNorm1d(self.nbr_fea_len),
            nn.Softplus()
        )
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, angle_fea, bond_weights):
        """
        Update bond features using atom and angle information
        
        Args:
            atom_fea: torch.Tensor shape (N, atom_fea_len)
                Atom features
            nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
                Bond features
            nbr_fea_idx: torch.LongTensor shape (N, M)
                Indices of M neighbors of each atom
            angle_fea: torch.Tensor shape (N, A, angle_fea_len)
                Angle features
            bond_weights: torch.Tensor shape (N, M)
                Weights for scaling messages
                
        Returns:
            nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
                Updated bond features
        """
        N, M = nbr_fea_idx.shape
        
        # Shape assertions
        assert atom_fea.shape[0] == N, f"Inconsistent batch size in atom_fea: {atom_fea.shape[0]} vs {N}"
        assert nbr_fea.shape[:2] == (N, M), f"Inconsistent neighbor dimensions in nbr_fea: {nbr_fea.shape[:2]} vs ({N}, {M})"
        assert bond_weights.shape == (N, M), f"Incorrect bond_weights shape: {bond_weights.shape} vs ({N}, {M})"
        
        # Process angle features
        if angle_fea is not None:
            angle_fea_flat = angle_fea.view(N, -1)
            angle_fea_flat = angle_fea_flat.unsqueeze(1)
            angle_fea_flat = angle_fea_flat.expand(-1, M, -1)
        else:
            angle_fea_flat = torch.zeros((N, M,
                                          comb(self.max_num_nbr, 2) * self.angle_fea_len),
                                          device=nbr_fea.device)
        
        # Get center atom features
        center_atom_fea = atom_fea.unsqueeze(1).expand(-1, M, -1)
        nbr_atom_fea = atom_fea[nbr_fea_idx]
        
        # Combine all features
        total_fea = torch.cat([
            center_atom_fea,
            nbr_atom_fea,
            nbr_fea,
            torch.roll(nbr_fea, shifts=-1, dims=1),
            angle_fea_flat
        ], dim=2)
        
        # Update bond features
        new_nbr_fea = self.bond_update(total_fea.view(N * M, -1)).view(N, M, -1)
        
        # Scale with bond weights (with proper broadcasting)
        new_nbr_fea = new_nbr_fea * bond_weights.unsqueeze(-1)
        
        return new_nbr_fea


class ConvBlock(nn.Module):
    """Combined convolution block for both atom and bond updates"""
    def __init__(self, atom_fea_len, nbr_fea_len, update_bond=False, angle_fea_len=None, max_num_nbr=None):
        """
        Initialize ConvBlock.
        
        Parameters
        ----------
        atom_fea_len: int
            Number of atom features
        nbr_fea_len: int
            Number of bond features
        angle_fea_len: int, optional
            Number of angle features
        max_num_nbr: int, optional
            Maximum number of neighbors
        """
        super(ConvBlock, self).__init__()
        self.atom_conv = AtomConvLayer(atom_fea_len, nbr_fea_len)
        self.update_bond = update_bond
        
        if update_bond and angle_fea_len is not None and max_num_nbr is not None:
            self.bond_conv = BondConvLayer(atom_fea_len, nbr_fea_len, angle_fea_len, max_num_nbr)
        else:
            self.bond_conv = None
        
        # Projections for residual connections
        self.atom_projection = nn.Linear(atom_fea_len, atom_fea_len)
        self.bond_projection = nn.Linear(nbr_fea_len, nbr_fea_len) if self.update_bond else None
            
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, angle_fea=None, bond_weights_ag=None, bond_weights_bg=None):
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
        angle_fea: torch.Tensor shape (N, A, angle_fea_len), optional
            Angle features
        bond_weights_ag: torch.Tensor shape (N, M), optional
            Bond weights for atom graph
        bond_weights_bg: torch.Tensor shape (N, M), optional
            Bond weights for bond graph
            
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
        if angle_fea is not None:
            assert angle_fea.dim() == 3, f"angle_fea should be 3D, got shape {angle_fea.shape}"
        
        # Save identities for residual connections
        atom_identity = self.atom_projection(atom_fea)
        bond_identity = self.bond_projection(nbr_fea) if self.update_bond else nbr_fea
        
        # Update features
        atom_fea = self.atom_conv(atom_fea, nbr_fea, nbr_fea_idx, bond_weights_ag)
        if self.update_bond and self.bond_conv is not None:
            nbr_fea = self.bond_conv(atom_fea, nbr_fea, nbr_fea_idx, angle_fea, bond_weights_bg)
        
        # Add residual connections
        atom_fea = atom_fea + atom_identity
        nbr_fea = nbr_fea + bond_identity
            
        return atom_fea, nbr_fea


class GraphConvNet(nn.Module):
    """Graph convolution network for updating atom and bond features"""
    def __init__(self, atom_fea_len, nbr_fea_len, n_conv, update_bond=False,
                 angle_fea_len=None, max_num_nbr=None):
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
        angle_fea_len: int, optional
            Number of angle features
        max_num_nbr: int, optional
            Maximum number of neighbors
        """
        super(GraphConvNet, self).__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                atom_fea_len=atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                update_bond=update_bond,
                angle_fea_len=angle_fea_len,
                max_num_nbr=max_num_nbr
            ) for _ in range(n_conv)
        ])
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, angle_fea=None,
                bond_weights_ag=None, bond_weights_bg=None):
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
        angle_fea: torch.Tensor shape (N, A, angle_fea_len), optional
            Angle features
        bond_weights_ag: torch.Tensor shape (N, M)
            Bond weights for atom graph
        bond_weights_bg: torch.Tensor shape (N, M)
            Bond weights for bond graph
            
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
                angle_fea, bond_weights_ag, bond_weights_bg
            )
        return atom_fea, nbr_fea


class EosNet(nn.Module):
    """Neural network for predicting equation of state parameters"""
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64,
                 h_fea_len=128, n_conv=3, n_h=1, angle_fea_len=None,
                 max_num_nbr=None, update_bond=False, classification=False):
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
        angle_fea_len: int, optional
            Number of angle features
        max_num_nbr: int, optional
            Maximum number of neighbors
        update_bond: bool
            Whether to update bond features
        classification: bool
            Whether to do classification
        """
        super(EosNet, self).__init__()
        
        self.classification = classification
        
        # Initial embedding and skip connection
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.skip_projection = nn.Linear(atom_fea_len, atom_fea_len)
        
        # Bond weights for atom and bond graphs
        self.bond_weights_ag = nn.Linear(nbr_fea_len, 1, bias=False)
        self.bond_weights_bg = nn.Linear(nbr_fea_len, 1, bias=False)
        
        # Graph convolution network
        self.graph_net = GraphConvNet(
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            n_conv=n_conv,
            update_bond=update_bond,
            angle_fea_len=angle_fea_len if update_bond else None,
            max_num_nbr=max_num_nbr if update_bond else None
        )
        
        # Transition from conv to fc
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        # FC layers with residual connections
        if n_h > 1:
            self.fcs = nn.ModuleList()
            self.fc_projections = nn.ModuleList()
            self.softpluses = nn.ModuleList()
            
            for _ in range(n_h-1):
                self.fcs.append(nn.Linear(h_fea_len, h_fea_len))
                self.fc_projections.append(nn.Linear(h_fea_len, h_fea_len))
                self.softpluses.append(nn.Softplus())
        
        # Output layer
        if classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
            
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

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, angle_fea=None):
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
        angle_fea: torch.Tensor shape (N, A, angle_fea_len), optional
            Angle features
            
        Returns
        -------
        out: torch.Tensor shape (N_crystals, 1) or (N_crystals, 2)
            Predicted properties
        """
        # Generate bond weights
        N, M = nbr_fea_idx.shape
        nbr_fea_flat = nbr_fea.view(-1, nbr_fea.shape[-1])
        bond_weights_ag = self.bond_weights_ag(nbr_fea_flat).view(N, M)
        bond_weights_bg = self.bond_weights_bg(nbr_fea_flat).view(N, M)
        
        # Initial atom embedding
        atom_fea = self.embedding(atom_fea)
        embedding_skip = self.skip_projection(atom_fea)
        
        # Update features through graph convolutions
        atom_fea, nbr_fea = self.graph_net(
            atom_fea, nbr_fea, nbr_fea_idx,
            angle_fea, bond_weights_ag, bond_weights_bg
        )
        
        # Add global skip connection
        atom_fea = atom_fea + embedding_skip
        
        # Pooling
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        # Transition to fc layers
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        # Apply dropout for classification
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        
        # FC layers with residual connections
        if hasattr(self, 'fcs'):
            for fc, softplus, projection in zip(self.fcs, self.softpluses, self.fc_projections):
                identity = crys_fea
                crys_fea = softplus(fc(crys_fea))
                identity = projection(identity)
                crys_fea = crys_fea + identity
        
        # Output
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
            
        return out