#!/bin/bash
#SBATCH --job-name=eosnet_md
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=lammps_test_%j.out
#SBATCH --error=lammps_test_%j.err

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate nequip

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

WORKDIR=/scratch/lz432/eosnet_lammps
cd $WORKDIR

LMP=$WORKDIR/lammps/build/lmp

echo "=== LAMMPS binary ==="
ls -la $LMP

echo ""
echo "=== Files ==="
ls -la eosnet_deployed.pt fe_bcc.data

echo ""
echo "=== Test: 10-step NVE ==="

cat > in.test << 'EOSEOF'
# EOSNet LAMMPS test: 27-atom BCC Fe
units           metal
atom_style      atomic
boundary        p p p
newton          off

read_data       fe_bcc.data

pair_style      eosnet
pair_coeff      * * eosnet_deployed.pt 5.0 6.0 64 Fe

mass            1 55.845

neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        0.001

thermo          1
thermo_style    custom step temp pe ke etotal press

velocity        all create 300.0 12345
fix             1 all nve

run             10
EOSEOF

echo "--- Running LAMMPS ---"
export EOSNET_DEBUG=1
$LMP -in in.test 2>&1

echo ""
echo "DONE"
