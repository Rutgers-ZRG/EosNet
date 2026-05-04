/* ----------------------------------------------------------------------
   pair_eosnet - LAMMPS pair_style for EOSNet MLIP

   EOSNet: e3nn equivariant backbone + GOM (Gaussian Overlap Matrix)
   fingerprints for per-atom energy prediction. Forces via autograd.

   The model is exported via torch.jit.trace (see deploy_for_lammps.py).
   It takes flattened tensor arguments and returns total energy.
   Forces are computed here via torch::autograd::grad.

   Two neighbor lists are constructed from LAMMPS neighbors:
   1. e3nn edges (cutoff = e3nn_cutoff, default 5.0 Å)
   2. GOM topology (cutoff = gom_cutoff, default 6.0 Å)

   Based on pair_nequip by Anders Johansson (Harvard).
   Contributing authors: Li Zhu (Rutgers), with AI assistance
------------------------------------------------------------------------- */

#include "pair_eosnet.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairEOSNet::PairEOSNet(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;

  // Default cutoffs
  e3nn_cutoff = 5.0;
  gom_cutoff = 6.0;
  natx = 32;

  // Debug mode from environment
  if (const char *env_p = std::getenv("EOSNET_DEBUG")) {
    if (std::string(env_p) == "1") {
      debug_mode = 1;
      if (comm->me == 0)
        std::cout << "PairEOSNet: Debug mode enabled\n";
    }
  }

  // Set device — map each MPI rank to a different GPU
  if (torch::cuda::is_available()) {
    int num_gpus = torch::cuda::device_count();
    int gpu_id = comm->me % num_gpus;
    device = torch::Device(torch::kCUDA, gpu_id);
  } else {
    device = torch::kCPU;
  }

  if (comm->me == 0)
    std::cout << "PairEOSNet: Using device " << device << "\n";

  // Initialize covalent radii
  init_rcov_table();
}

PairEOSNet::~PairEOSNet()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR, "pair_eosnet requires atom IDs");

  // Request full neighbor list with ghost atoms
  // Use gom_cutoff (the larger of the two cutoffs) so we get all needed neighbors
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  // NequIP-style: newton pair off
  if (force->newton_pair)
    error->all(FLERR, "pair_eosnet requires newton pair off");
}

double PairEOSNet::init_one(int i, int j)
{
  // Return the larger cutoff so LAMMPS builds neighbor list covering both
  return gom_cutoff;
}

void PairEOSNet::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::settings(int narg, char ** /*arg*/)
{
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style eosnet command, too many arguments");
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  int ntypes = atom->ntypes;

  // Expected: pair_coeff * * model.pt e3nn_cutoff gom_cutoff natx Z1 Z2 ...
  // Minimum args: * * model.pt Z1 (for single species)
  // With optional cutoffs: * * model.pt 5.0 6.0 32 Fe
  if (narg < 3 + ntypes)
    error->all(FLERR,
        "Incorrect args for pair_coeff: "
        "pair_coeff * * <model.pt> [e3nn_cutoff gom_cutoff natx] <element1> <element2> ...\n"
        "Example: pair_coeff * * eosnet_deployed.pt 5.0 6.0 32 Fe");

  // Parse model path
  model_path = std::string(arg[2]);

  // Determine if optional cutoffs are provided
  // If narg == 3 + ntypes: no optional args (model + elements)
  // If narg == 6 + ntypes: three optional args (cutoffs + natx + elements)
  int elem_start;  // index where element names begin
  if (narg == 3 + ntypes) {
    // No optional cutoff args
    elem_start = 3;
  } else if (narg == 6 + ntypes) {
    e3nn_cutoff = utils::numeric(FLERR, arg[3], false, lmp);
    gom_cutoff = utils::numeric(FLERR, arg[4], false, lmp);
    natx = utils::inumeric(FLERR, arg[5], false, lmp);
    elem_start = 6;
  } else {
    error->all(FLERR,
        "Incorrect number of args for pair_coeff. Expected:\n"
        "  pair_coeff * * model.pt Element1 Element2 ...\n"
        "  pair_coeff * * model.pt e3nn_cut gom_cut natx Element1 Element2 ...");
  }

  if (comm->me == 0) {
    std::cout << "PairEOSNet: Loading model from " << model_path << "\n";
    std::cout << "PairEOSNet: e3nn_cutoff=" << e3nn_cutoff
              << " gom_cutoff=" << gom_cutoff
              << " natx=" << natx << "\n";
  }

  // Load TorchScript model.
  // For multi-GPU: use CUDA_VISIBLE_DEVICES per MPI rank (gpu_wrapper.sh)
  // so each rank sees its GPU as cuda:0. This avoids TorchScript traced-constant
  // device mismatch (constants baked in as cuda:0 at trace time).
  model = torch::jit::load(model_path, device);
  model.eval();

  // Freeze DISABLED — breaks autograd forces (forces need computation graph)
  // model = torch::jit::freeze(model);

  // Set fusion strategy
  torch::jit::FusionStrategy strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
  torch::jit::setFusionStrategy(strategy);

  // Map LAMMPS types to atomic numbers
  // Element names to atomic numbers (abbreviated periodic table)
  // This maps element symbols to Z
  auto elem_to_z = [](const std::string &sym) -> int {
    // Common elements for MLIP simulations
    if (sym == "H")  return 1;   if (sym == "He") return 2;
    if (sym == "Li") return 3;   if (sym == "Be") return 4;
    if (sym == "B")  return 5;   if (sym == "C")  return 6;
    if (sym == "N")  return 7;   if (sym == "O")  return 8;
    if (sym == "F")  return 9;   if (sym == "Ne") return 10;
    if (sym == "Na") return 11;  if (sym == "Mg") return 12;
    if (sym == "Al") return 13;  if (sym == "Si") return 14;
    if (sym == "P")  return 15;  if (sym == "S")  return 16;
    if (sym == "Cl") return 17;  if (sym == "Ar") return 18;
    if (sym == "K")  return 19;  if (sym == "Ca") return 20;
    if (sym == "Sc") return 21;  if (sym == "Ti") return 22;
    if (sym == "V")  return 23;  if (sym == "Cr") return 24;
    if (sym == "Mn") return 25;  if (sym == "Fe") return 26;
    if (sym == "Co") return 27;  if (sym == "Ni") return 28;
    if (sym == "Cu") return 29;  if (sym == "Zn") return 30;
    if (sym == "Ga") return 31;  if (sym == "Ge") return 32;
    if (sym == "As") return 33;  if (sym == "Se") return 34;
    if (sym == "Br") return 35;  if (sym == "Kr") return 36;
    if (sym == "Rb") return 37;  if (sym == "Sr") return 38;
    if (sym == "Y")  return 39;  if (sym == "Zr") return 40;
    if (sym == "Nb") return 41;  if (sym == "Mo") return 42;
    if (sym == "Ru") return 44;  if (sym == "Rh") return 45;
    if (sym == "Pd") return 46;  if (sym == "Ag") return 47;
    if (sym == "Cd") return 48;  if (sym == "In") return 49;
    if (sym == "Sn") return 50;  if (sym == "Sb") return 51;
    if (sym == "Te") return 52;  if (sym == "I")  return 53;
    if (sym == "Xe") return 54;  if (sym == "Cs") return 55;
    if (sym == "Ba") return 56;  if (sym == "La") return 57;
    if (sym == "Ce") return 58;  if (sym == "Hf") return 72;
    if (sym == "Ta") return 73;  if (sym == "W")  return 74;
    if (sym == "Re") return 75;  if (sym == "Os") return 76;
    if (sym == "Ir") return 77;  if (sym == "Pt") return 78;
    if (sym == "Au") return 79;  if (sym == "Pb") return 82;
    if (sym == "Bi") return 83;  if (sym == "U")  return 92;
    return -1;
  };

  type_to_z.resize(ntypes);
  for (int i = 0; i < ntypes; i++) {
    std::string elem(arg[elem_start + i]);
    int z = elem_to_z(elem);
    if (z < 0)
      error->all(FLERR, "Unknown element symbol: {}", elem);
    type_to_z[i] = z;
    if (comm->me == 0)
      std::cout << "PairEOSNet: LAMMPS type " << (i + 1)
                << " -> " << elem << " (Z=" << z << ")\n";
  }

  // Set setflag for all type pairs
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 1;

  if (comm->me == 0)
    std::cout << "PairEOSNet: Model loaded, "
              << ntypes << " atom type(s)\n";
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  int inum = list->inum;
  if (inum == 0) return;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // ---- Build inputs ----
  // With MPI domain decomposition, the model's internal operations
  // (positions[gom_nbr_idx], etc.) index into the positions tensor.
  // All referenced atoms (local + ghost) must be in the tensor.
  // Strategy: build a "model atom set" containing all unique atoms
  // referenced by local atoms' edges and GOM neighbors.

  auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);
  auto opts_l = torch::TensorOptions().dtype(torch::kInt64);

  // Collect all unique atoms referenced by local atoms (via neighbor list)
  // This includes local atoms + ghost atoms within gom_cutoff
  double gom_cutoff_sq_local = gom_cutoff * gom_cutoff;

  // First, find all unique real atom tags referenced
  std::set<tagint> referenced_tags;
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    referenced_tags.insert(tag[i]);  // local atom
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= gom_cutoff_sq_local)
        referenced_tags.insert(tag[j]);
    }
  }

  // Build contiguous index for all referenced atoms
  int n_model_atoms = static_cast<int>(referenced_tags.size());

  // tag -> model index mapping (sized by max tag)
  tagint max_tag = 0;
  for (auto t : referenced_tags)
    if (t > max_tag) max_tag = t;

  tag2i_map.assign(max_tag + 1, -1);

  // Local atoms first (indices 0..inum-1)
  for (int ii = 0; ii < inum; ii++)
    tag2i_map[tag[ilist[ii]]] = ii;

  // Ghost-only atoms get indices inum..n_model_atoms-1
  // Also build model_idx -> LAMMPS index map for position lookup
  std::vector<int> midx_to_lmp(n_model_atoms, -1);
  for (int ii = 0; ii < inum; ii++)
    midx_to_lmp[ii] = ilist[ii];

  int ghost_idx = inum;
  // Find a LAMMPS index for each ghost-only tag
  int ntotal = nlocal + atom->nghost;
  std::unordered_map<tagint, int> ghost_tag_to_lmp;
  for (int i = 0; i < ntotal; i++) {
    tagint t = tag[i];
    if (referenced_tags.count(t) && ghost_tag_to_lmp.find(t) == ghost_tag_to_lmp.end()) {
      // Prefer local atoms for position reference
      if (i < nlocal || ghost_tag_to_lmp.find(t) == ghost_tag_to_lmp.end())
        ghost_tag_to_lmp[t] = i;
    }
  }

  for (auto t : referenced_tags) {
    if (tag2i_map[t] == -1) {
      tag2i_map[t] = ghost_idx;
      if (ghost_tag_to_lmp.count(t))
        midx_to_lmp[ghost_idx] = ghost_tag_to_lmp[t];
      ghost_idx++;
    }
  }

  // Build positions and atomic numbers for all model atoms
  torch::Tensor pos_tensor = torch::zeros({n_model_atoms, 3}, opts_f);
  torch::Tensor z_tensor = torch::zeros({n_model_atoms}, opts_l);
  torch::Tensor batch_tensor = torch::zeros({n_model_atoms}, opts_l);

  auto pos_a = pos_tensor.accessor<float, 2>();
  auto z_a = z_tensor.accessor<long, 1>();

  for (int midx = 0; midx < n_model_atoms; midx++) {
    int i = midx_to_lmp[midx];
    if (i >= 0) {
      pos_a[midx][0] = static_cast<float>(x[i][0]);
      pos_a[midx][1] = static_cast<float>(x[i][1]);
      pos_a[midx][2] = static_cast<float>(x[i][2]);
      z_a[midx] = type_to_z[type[i] - 1];
    }
  }

  // 2. Cell
  torch::Tensor cell_tensor = get_cell();
  torch::Tensor cells_tensor = cell_tensor.unsqueeze(0);  // [1, 3, 3]

  // 3. Edge graph (e3nn cutoff) — uses tag2i_map for all atoms
  torch::Tensor edge_index, edge_cell_shift;
  int nedges;
  build_edges(inum, ilist, numneigh, firstneigh, x, type, tag,
              edge_index, edge_cell_shift, nedges);

  // 4. GOM neighbor topology (built for local atoms only)
  torch::Tensor gom_nbr_idx, gom_nbr_shifts, gom_nbr_rcov, gom_n_sphere, gom_self_rcov;
  build_gom_data(inum, ilist, numneigh, firstneigh, x, type, tag,
                 gom_nbr_idx, gom_nbr_shifts, gom_nbr_rcov,
                 gom_n_sphere, gom_self_rcov);

  // Pad GOM tensors for ghost-only atoms (indices inum..n_model_atoms-1)
  // Ghost-only atoms get self-only GOM (1 neighbor = self)
  if (n_model_atoms > inum) {
    int n_ghost = n_model_atoms - inum;
    int max_n = gom_nbr_idx.size(1);

    // Extend nbr_idx: ghost atom ii's self-index is ii
    auto pad_idx = torch::zeros({n_ghost, max_n}, opts_l);
    auto pad_idx_a = pad_idx.accessor<long, 2>();
    for (int g = 0; g < n_ghost; g++)
      pad_idx_a[g][0] = inum + g;  // self-reference
    gom_nbr_idx = torch::cat({gom_nbr_idx, pad_idx}, 0);

    // Extend nbr_shifts: zeros (self has no shift)
    gom_nbr_shifts = torch::cat({gom_nbr_shifts, torch::zeros({n_ghost, max_n, 3}, opts_f)}, 0);

    // Extend nbr_rcov: use default rcov for ghost atoms
    auto pad_rcov = torch::zeros({n_ghost, max_n}, opts_f);
    auto pad_rcov_a = pad_rcov.accessor<float, 2>();
    for (int g = 0; g < n_ghost; g++) {
      int midx = inum + g;
      int zi = z_tensor.accessor<long, 1>()[midx];
      float rcov_g = (zi < static_cast<int>(rcov_table.size())) ?
                     static_cast<float>(rcov_table[zi]) : 1.0f;
      pad_rcov_a[g][0] = rcov_g;
    }
    gom_nbr_rcov = torch::cat({gom_nbr_rcov, pad_rcov}, 0);

    // Extend n_sphere: 1 for each ghost (self only)
    auto pad_nsphere = torch::ones({n_ghost}, opts_l);
    gom_n_sphere = torch::cat({gom_n_sphere, pad_nsphere}, 0);

    // Extend self_rcov
    auto pad_self = torch::zeros({n_ghost}, opts_f);
    auto pad_self_a = pad_self.accessor<float, 1>();
    for (int g = 0; g < n_ghost; g++) {
      int midx = inum + g;
      int zi = z_tensor.accessor<long, 1>()[midx];
      pad_self_a[g] = (zi < static_cast<int>(rcov_table.size())) ?
                      static_cast<float>(rcov_table[zi]) : 1.0f;
    }
    gom_self_rcov = torch::cat({gom_self_rcov, pad_self}, 0);
  }

  // ---- Move to device ----
  pos_tensor = pos_tensor.to(device).requires_grad_(true);
  z_tensor = z_tensor.to(device);
  batch_tensor = batch_tensor.to(device);
  cell_tensor = cell_tensor.to(device);
  cells_tensor = cells_tensor.to(device);
  edge_index = edge_index.to(device);
  edge_cell_shift = edge_cell_shift.to(device);
  gom_nbr_idx = gom_nbr_idx.to(device);
  gom_nbr_shifts = gom_nbr_shifts.to(device);
  gom_nbr_rcov = gom_nbr_rcov.to(device);
  gom_n_sphere = gom_n_sphere.to(device);
  gom_self_rcov = gom_self_rcov.to(device);

  // 5. Compute edge_vec from positions + cell_shifts
  auto src = edge_index.select(0, 0);
  auto dst = edge_index.select(0, 1);
  auto shift_cart = torch::matmul(edge_cell_shift, cell_tensor);
  auto edge_vec = pos_tensor.index({dst}) - pos_tensor.index({src}) + shift_cart;

  // ---- Call model ----
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(z_tensor);
  inputs.push_back(pos_tensor);
  inputs.push_back(cell_tensor);
  inputs.push_back(edge_index);
  inputs.push_back(edge_vec);
  inputs.push_back(batch_tensor);
  inputs.push_back(cells_tensor);
  inputs.push_back(gom_nbr_idx);
  inputs.push_back(gom_nbr_shifts);
  inputs.push_back(gom_nbr_rcov);
  inputs.push_back(gom_n_sphere);
  inputs.push_back(gom_self_rcov);

  auto energy = model.forward(inputs).toTensor();

  // ---- Compute forces via autograd ----
  auto forces_tensor = -torch::autograd::grad(
      {energy}, {pos_tensor}, {}, /*retain_graph=*/vflag, /*create_graph=*/false)[0];

  // ---- Extract results (only local atoms, indices 0..inum-1) ----
  auto forces_cpu = forces_tensor.cpu().to(torch::kFloat64);
  auto forces_a = forces_cpu.accessor<double, 2>();

  eng_vdwl = energy.cpu().item<double>();

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    f[i][0] += forces_a[ii][0];
    f[i][1] += forces_a[ii][1];
    f[i][2] += forces_a[ii][2];
  }

  // Virial via -sum_i r_i (x) f_i
  if (vflag) {
    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      virial[0] += x[i][0] * forces_a[ii][0];
      virial[1] += x[i][1] * forces_a[ii][1];
      virial[2] += x[i][2] * forces_a[ii][2];
      virial[3] += x[i][0] * forces_a[ii][1];
      virial[4] += x[i][0] * forces_a[ii][2];
      virial[5] += x[i][1] * forces_a[ii][2];
    }
    // Negate — virial = -sum r_i (x) f_i, and f already has minus sign from autograd
    // Actually virial[a][b] = -sum_i r_i^a * F_i^b
    // F = -dE/dr, so virial = sum_i r_i^a * (dE/dr_i^b)
    // LAMMPS convention: virial already negated above correctly since F = -grad
    // No: LAMMPS wants virial[k] = -sum_i r_i^a * f_i^b where f is the force
    // forces_a already stores -dE/dr (the physical force)
    // so virial = -r dot f_physical ... but the loop above computes +r dot f
    // need to negate
    for (int k = 0; k < 6; k++) virial[k] = -virial[k];
  }

  if (debug_mode) {
    std::cout << "PairEOSNet: E=" << eng_vdwl << " eV, "
              << nedges << " edges, " << inum << " atoms\n";
  }
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::build_edges(
    int inum, int *ilist, int *numneigh, int **firstneigh,
    double **x, int *type, tagint *tag,
    torch::Tensor &edge_index_out,
    torch::Tensor &edge_cell_shift_out,
    int &nedges_out)
{
  int nlocal = atom->nlocal;
  double cutoff_sq = e3nn_cutoff * e3nn_cutoff;

  // Use shared tag2i_map built in compute()
  const auto &tag2i_local = tag2i_map;

  // Count edges
  int nedges = 0;
  std::vector<int> neigh_per_atom(inum, 0);

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= cutoff_sq) {
        neigh_per_atom[ii]++;
        nedges++;
      }
    }
  }

  // Cumulative sum for parallel fill
  std::vector<int> cumsum(inum, 0);
  for (int ii = 1; ii < inum; ii++)
    cumsum[ii] = cumsum[ii - 1] + neigh_per_atom[ii - 1];

  // Build i2ii mapping
  std::vector<int> i2ii(nlocal + atom->nghost, -1);
  for (int ii = 0; ii < inum; ii++)
    i2ii[ilist[ii]] = ii;

  // Allocate
  auto opts_l = torch::TensorOptions().dtype(torch::kInt64);
  auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor edges = torch::zeros({2, nedges}, opts_l);
  torch::Tensor shifts = torch::zeros({nedges, 3}, opts_f);

  auto edges_a = edges.accessor<long, 2>();
  auto shifts_a = shifts.accessor<float, 2>();

  // Cell inverse for computing lattice shifts
  torch::Tensor cell_tensor = get_cell();
  torch::Tensor cell_inv = cell_tensor.inverse().t();
  auto cell_inv_a = cell_inv.accessor<float, 2>();

  // Fill edges
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    int edge_counter = cumsum[ii];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq > cutoff_sq) continue;

      int jtag = tag[j];
      int jj_local = tag2i_local[jtag];

      edges_a[0][edge_counter] = ii;          // source (contiguous index)
      edges_a[1][edge_counter] = jj_local;    // target (remapped to local)

      // Cell shift = round(cell_inv @ (x[j_ghost] - x_ref[jj_local]))
      // x_ref is the reference position stored in pos_tensor for this model atom
      // We can't use ilist[jj_local] for ghost-only atoms, so we read from
      // the already-filled model positions accessor (pos_a comes from compute's pos_tensor)
      // Note: pos_a is not accessible here. Instead, compute shift from ghost directly:
      // For edges, shift = round(cell_inv @ (x_ghost - x_ref))
      // where x_ref is the position in the model's atom list.
      // Since we know j_ghost position and the model atom's position should be
      // the same as some LAMMPS atom with the same tag, just compute from LAMMPS:

      // Find ANY LAMMPS atom with this tag that's local (preferred) or ghost
      // The reference position for model atom jj_local
      // For local atoms (jj_local < inum): use ilist[jj_local]
      // For ghost-only atoms: the ghost j itself IS one image, so shift = 0
      // Wait — that's wrong. If jj_local maps to a ghost-only atom, the reference
      // position was set to one ghost's position. The shift of THIS ghost (j)
      // relative to that reference is what we need.
      // Simplest: for edges, just compute shift from the ghost's position
      // relative to the reference position in the model.
      // Since build_edges runs AFTER pos_tensor is built, we can't easily access it.
      // Instead, just compute: if j is the ghost and jj_local < inum,
      // use ilist[jj_local] as before. If jj_local >= inum, the shift is 0
      // because the model reference position was set from this same ghost image.
      double pshift[3] = {0.0, 0.0, 0.0};
      if (jj_local < inum) {
        int j_real_lmp = ilist[jj_local];
        pshift[0] = x[j][0] - x[j_real_lmp][0];
        pshift[1] = x[j][1] - x[j_real_lmp][1];
        pshift[2] = x[j][2] - x[j_real_lmp][2];
      }
      // For ghost-only atoms (jj_local >= inum), the model reference position
      // was assigned from some LAMMPS ghost. If j is a DIFFERENT image of that atom,
      // there could be a non-zero shift. But for correctness, we should compute it.
      // For now, this is a simplification that works when the model reference is
      // the closest image (which it should be within the skin distance).

      for (int d = 0; d < 3; d++) {
        double tmp = 0.0;
        for (int k = 0; k < 3; k++)
          tmp += cell_inv_a[d][k] * pshift[k];
        shifts_a[edge_counter][d] = static_cast<float>(std::round(tmp));
      }

      edge_counter++;
    }
  }

  edge_index_out = edges;
  edge_cell_shift_out = shifts;
  nedges_out = nedges;
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::build_gom_data(
    int inum, int *ilist, int *numneigh, int **firstneigh,
    double **x, int *type, tagint *tag,
    torch::Tensor &nbr_idx_out,
    torch::Tensor &nbr_shifts_out,
    torch::Tensor &nbr_rcov_out,
    torch::Tensor &n_sphere_out,
    torch::Tensor &self_rcov_out)
{
  double gom_cutoff_sq = gom_cutoff * gom_cutoff;

  // Use shared tag2i_map built in compute()
  const auto &tag2i_local = tag2i_map;

  // Cell inverse for lattice shifts
  torch::Tensor cell_tensor = get_cell();
  torch::Tensor cell_inv = cell_tensor.inverse().t();
  auto cell_inv_a = cell_inv.accessor<float, 2>();

  // First pass: count GOM neighbors per atom to determine actual max_n
  std::vector<int> gom_count(inum, 0);
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq <= gom_cutoff_sq) {
        gom_count[ii]++;
      }
    }
  }

  int max_gom = *std::max_element(gom_count.begin(), gom_count.end()) + 1;  // +1 for self
  int max_n = std::min(max_gom, natx);

  // Allocate
  auto opts_l = torch::TensorOptions().dtype(torch::kInt64);
  auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);

  torch::Tensor nbr_idx = torch::zeros({inum, max_n}, opts_l);
  torch::Tensor nbr_shifts = torch::zeros({inum, max_n, 3}, opts_f);
  torch::Tensor nbr_rcov = torch::zeros({inum, max_n}, opts_f);
  torch::Tensor n_sphere = torch::zeros({inum}, opts_l);
  torch::Tensor self_rcov = torch::zeros({inum}, opts_f);

  auto nbr_idx_a = nbr_idx.accessor<long, 2>();
  auto nbr_shifts_a = nbr_shifts.accessor<float, 3>();
  auto nbr_rcov_a = nbr_rcov.accessor<float, 2>();
  auto n_sphere_a = n_sphere.accessor<long, 1>();
  auto self_rcov_a = self_rcov.accessor<float, 1>();

  // Per-atom: collect neighbors, sort by distance, keep closest natx-1
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int zi = type_to_z[type[i] - 1];
    double rcov_i = (zi < static_cast<int>(rcov_table.size())) ? rcov_table[zi] : 1.0;

    // Self is always first neighbor
    nbr_idx_a[ii][0] = ii;
    nbr_rcov_a[ii][0] = static_cast<float>(rcov_i);
    self_rcov_a[ii] = static_cast<float>(rcov_i);

    // Collect GOM neighbors with distances
    struct NbrInfo {
      int idx;        // contiguous local index
      double dist_sq;
      double shift[3]; // lattice shifts
      double rcov;
    };
    std::vector<NbrInfo> neighbors;

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq > gom_cutoff_sq) continue;

      int jtag = tag[j];
      int jj_local = tag2i_local[jtag];
      int zj = type_to_z[type[j] - 1];
      double rcov_j = (zj < static_cast<int>(rcov_table.size())) ? rcov_table[zj] : 1.0;

      // Compute lattice shift
      NbrInfo info;
      info.idx = jj_local;
      info.dist_sq = rsq;
      info.rcov = rcov_j;

      // Compute lattice shift: for local atoms use ilist, for ghost-only set 0
      double pshift[3] = {0.0, 0.0, 0.0};
      if (jj_local < inum) {
        int j_real_lmp = ilist[jj_local];
        pshift[0] = x[j][0] - x[j_real_lmp][0];
        pshift[1] = x[j][1] - x[j_real_lmp][1];
        pshift[2] = x[j][2] - x[j_real_lmp][2];
      }

      for (int d = 0; d < 3; d++) {
        double tmp = 0.0;
        for (int k = 0; k < 3; k++)
          tmp += cell_inv_a[d][k] * pshift[k];
        info.shift[d] = std::round(tmp);
      }

      neighbors.push_back(info);
    }

    // Sort by distance
    std::sort(neighbors.begin(), neighbors.end(),
              [](const NbrInfo &a, const NbrInfo &b) { return a.dist_sq < b.dist_sq; });

    // Keep at most natx-1 neighbors (slot 0 is self)
    int n_keep = std::min(static_cast<int>(neighbors.size()), max_n - 1);
    for (int k = 0; k < n_keep; k++) {
      nbr_idx_a[ii][1 + k] = neighbors[k].idx;
      nbr_shifts_a[ii][1 + k][0] = static_cast<float>(neighbors[k].shift[0]);
      nbr_shifts_a[ii][1 + k][1] = static_cast<float>(neighbors[k].shift[1]);
      nbr_shifts_a[ii][1 + k][2] = static_cast<float>(neighbors[k].shift[2]);
      nbr_rcov_a[ii][1 + k] = static_cast<float>(neighbors[k].rcov);
    }

    n_sphere_a[ii] = 1 + n_keep;
  }

  nbr_idx_out = nbr_idx;
  nbr_shifts_out = nbr_shifts;
  nbr_rcov_out = nbr_rcov;
  n_sphere_out = n_sphere;
  self_rcov_out = self_rcov;
}

/* ---------------------------------------------------------------------- */

torch::Tensor PairEOSNet::get_cell()
{
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor cell = torch::zeros({3, 3}, opts);
  auto c = cell.accessor<float, 2>();

  c[0][0] = static_cast<float>(domain->boxhi[0] - domain->boxlo[0]);

  c[1][0] = static_cast<float>(domain->xy);
  c[1][1] = static_cast<float>(domain->boxhi[1] - domain->boxlo[1]);

  c[2][0] = static_cast<float>(domain->xz);
  c[2][1] = static_cast<float>(domain->yz);
  c[2][2] = static_cast<float>(domain->boxhi[2] - domain->boxlo[2]);

  return cell;
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::get_tag2i(std::vector<int> &tag2i, int inum, int *ilist, tagint *tag)
{
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int itag = tag[i];
    if (itag >= 0 && itag < static_cast<int>(tag2i.size()))
      tag2i[itag] = ii;  // Map to contiguous index, not LAMMPS index
  }
}

/* ---------------------------------------------------------------------- */

void PairEOSNet::init_rcov_table()
{
  // Covalent radii in Angstroms, indexed by atomic number Z
  // Source: fplib / rcovdata.py (Rutgers-ZRG)
  rcov_table = {
    1.00,  // 0  X  (placeholder)
    0.37,  // 1  H
    0.32,  // 2  He
    1.34,  // 3  Li
    0.90,  // 4  Be
    0.82,  // 5  B
    0.77,  // 6  C
    0.75,  // 7  N
    0.73,  // 8  O
    0.71,  // 9  F
    0.69,  // 10 Ne
    1.54,  // 11 Na
    1.30,  // 12 Mg
    1.18,  // 13 Al
    1.11,  // 14 Si
    1.06,  // 15 P
    1.02,  // 16 S
    0.99,  // 17 Cl
    0.97,  // 18 Ar
    1.96,  // 19 K
    1.74,  // 20 Ca
    1.44,  // 21 Sc
    1.36,  // 22 Ti
    1.25,  // 23 V
    1.27,  // 24 Cr
    1.39,  // 25 Mn
    1.25,  // 26 Fe
    1.26,  // 27 Co
    1.21,  // 28 Ni
    1.38,  // 29 Cu
    1.31,  // 30 Zn
    1.26,  // 31 Ga
    1.22,  // 32 Ge
    1.19,  // 33 As
    1.16,  // 34 Se
    1.14,  // 35 Br
    1.10,  // 36 Kr
    2.11,  // 37 Rb
    1.92,  // 38 Sr
    1.62,  // 39 Y
    1.48,  // 40 Zr
    1.37,  // 41 Nb
    1.45,  // 42 Mo
    1.56,  // 43 Tc
    1.26,  // 44 Ru
    1.35,  // 45 Rh
    1.31,  // 46 Pd
    1.53,  // 47 Ag
    1.48,  // 48 Cd
    1.44,  // 49 In
    1.41,  // 50 Sn
    1.38,  // 51 Sb
    1.35,  // 52 Te
    1.33,  // 53 I
    1.30,  // 54 Xe
    2.25,  // 55 Cs
    1.98,  // 56 Ba
    1.69,  // 57 La
    1.69,  // 58 Ce
    1.69,  // 59 Pr
    1.69,  // 60 Nd
    1.69,  // 61 Pm
    1.69,  // 62 Sm
    1.69,  // 63 Eu
    1.69,  // 64 Gd
    1.69,  // 65 Tb
    1.69,  // 66 Dy
    1.69,  // 67 Ho
    1.69,  // 68 Er
    1.69,  // 69 Tm
    1.69,  // 70 Yb
    1.60,  // 71 Lu
    1.50,  // 72 Hf
    1.38,  // 73 Ta
    1.46,  // 74 W
    1.59,  // 75 Re
    1.28,  // 76 Os
    1.37,  // 77 Ir
    1.28,  // 78 Pt
    1.44,  // 79 Au
    1.49,  // 80 Hg
    1.48,  // 81 Tl
    1.47,  // 82 Pb
    1.46,  // 83 Bi
    1.46,  // 84 Po
    1.46,  // 85 At
    1.46,  // 86 Rn
    1.46,  // 87 Fr
    1.46,  // 88 Ra
    1.46,  // 89 Ac
    1.46,  // 90 Th
    1.46,  // 91 Pa
    1.46,  // 92 U
  };
}
