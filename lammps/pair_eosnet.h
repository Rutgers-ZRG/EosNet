/* -*- c++ -*- ----------------------------------------------------------
   pair_eosnet - LAMMPS pair_style for EOSNet MLIP

   EOSNet: e3nn equivariant backbone + GOM (Gaussian Overlap Matrix)
   fingerprints for per-atom energy prediction. Forces via autograd.

   Based on pair_nequip by Anders Johansson (Harvard).
   Extended with GOM neighbor topology construction.

   Contributing authors: Li Zhu (Rutgers), with AI assistance
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(eosnet, PairEOSNet)

#else

#ifndef LMP_PAIR_EOSNET_H
#define LMP_PAIR_EOSNET_H

#include "pair.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairEOSNet : public Pair {
 public:
  PairEOSNet(class LAMMPS *);
  ~PairEOSNet() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void allocate();

 protected:
  // Model
  torch::jit::Module model;
  torch::Device device = torch::kCPU;
  std::string model_path;

  // Cutoffs
  double e3nn_cutoff;    // Edge cutoff for e3nn message passing (default 5.0)
  double gom_cutoff;     // GOM neighbor cutoff (default 6.0)
  int natx;              // Max GOM neighbors per atom (default 32)

  // Type mapping: LAMMPS type (1-based) -> model atomic number
  std::vector<int> type_to_z;
  // Covalent radii indexed by atomic number
  std::vector<double> rcov_table;

  int debug_mode = 0;

  // Shared tag -> model index mapping (built in compute, used by build_edges/build_gom_data)
  std::vector<int> tag2i_map;

  // Build edge graph from LAMMPS neighbor list
  // Returns: edge_index [2, nedges], edge_cell_shift [nedges, 3]
  void build_edges(
      int inum, int *ilist, int *numneigh, int **firstneigh,
      double **x, int *type, tagint *tag,
      torch::Tensor &edge_index_out,
      torch::Tensor &edge_cell_shift_out,
      int &nedges_out);

  // Build GOM neighbor topology from LAMMPS neighbor list
  // Returns: nbr_idx [nat, natx], nbr_shifts [nat, natx, 3],
  //          nbr_rcov [nat, natx], n_sphere [nat], self_rcov [nat]
  void build_gom_data(
      int inum, int *ilist, int *numneigh, int **firstneigh,
      double **x, int *type, tagint *tag,
      torch::Tensor &nbr_idx_out,
      torch::Tensor &nbr_shifts_out,
      torch::Tensor &nbr_rcov_out,
      torch::Tensor &n_sphere_out,
      torch::Tensor &self_rcov_out);

  // Get simulation cell as tensor
  torch::Tensor get_cell();

  // Build tag -> local index mapping
  void get_tag2i(std::vector<int> &tag2i, int inum, int *ilist, tagint *tag);

  // Initialize covalent radii table
  void init_rcov_table();
};

}    // namespace LAMMPS_NS

#endif
#endif
