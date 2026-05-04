#!/bin/bash
#SBATCH --job-name=build_lammps
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=build_lammps_%j.out
#SBATCH --error=build_lammps_%j.err

set -e

source /home/lz432/miniconda3/etc/profile.d/conda.sh
conda activate nequip

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

WORKDIR=/scratch/lz432/eosnet_lammps
cd $WORKDIR

# Step 1: LAMMPS source should already be cloned
if [ ! -d "lammps/src" ]; then
    echo "ERROR: LAMMPS source not found at $WORKDIR/lammps/"
    exit 1
fi
echo "=== LAMMPS source found ==="

# Step 2: Copy pair_eosnet into LAMMPS source
echo "=== Installing pair_eosnet into LAMMPS ==="
cp $WORKDIR/pair_eosnet.h lammps/src/
cp $WORKDIR/pair_eosnet.cpp lammps/src/

# Step 3: Patch LAMMPS CMakeLists.txt to find Torch and link pair_eosnet
# We append to the end of the main CMakeLists.txt
CMAKELISTS=lammps/cmake/CMakeLists.txt
if ! grep -q "pair_eosnet" $CMAKELISTS 2>/dev/null; then
    echo "=== Patching CMakeLists.txt for Torch ==="
    cat >> $CMAKELISTS << 'EOSCMAKE'

# --- pair_eosnet: EOSNet MLIP with PyTorch ---
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
message(STATUS "pair_eosnet: Torch found at ${TORCH_INSTALL_PREFIX}")
message(STATUS "pair_eosnet: CUDA libraries: ${TORCH_CUDA_LIBRARIES}")
EOSCMAKE
fi

# Step 4: Build LAMMPS
echo "=== Building LAMMPS ==="
TORCH_PREFIX=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
echo "PyTorch cmake prefix: $TORCH_PREFIX"

# Clean previous build
rm -rf lammps/build
mkdir -p lammps/build
cd lammps/build

# CUDA toolkit from conda
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc

# Remove stale cuda 13.2 headers from targets/ that conflict with 12.1
# (conda cuda-toolkit 13.2 installed then downgraded to 12.1, but targets/ kept old headers)
CUDA_TARGETS=$CONDA_PREFIX/targets/x86_64-linux
if [ -f "$CUDA_TARGETS/include/cuda_runtime_api.h" ]; then
    TARGET_VER=$(grep CUDART_VERSION "$CUDA_TARGETS/include/cuda_runtime_api.h" | head -1 | awk '{print $3}')
    CONDA_VER=$(grep CUDART_VERSION "$CONDA_PREFIX/include/cuda_runtime_api.h" | head -1 | awk '{print $3}')
    echo "CUDA target headers version: $TARGET_VER, conda headers: $CONDA_VER"
    if [ "$TARGET_VER" != "$CONDA_VER" ]; then
        echo "Fixing version mismatch: replacing targets headers with conda 12.1 headers"
        # Backup and replace key headers
        for hdr in cuda_runtime.h cuda_runtime_api.h cuda.h; do
            if [ -f "$CUDA_TARGETS/include/$hdr" ] && [ -f "$CONDA_PREFIX/include/$hdr" ]; then
                cp "$CONDA_PREFIX/include/$hdr" "$CUDA_TARGETS/include/$hdr"
            fi
        done
    fi
fi

echo "CUDA nvcc: $(which nvcc) ($(nvcc --version 2>&1 | tail -1))"
echo "CUDA_HOME: $CUDA_HOME"
echo "Header version: $(grep CUDART_VERSION $CONDA_PREFIX/include/cuda_runtime_api.h | head -1)"

# Workaround: PyTorch 2.5 cmake references CUDA::nvToolsExt which doesn't exist
# in CUDAToolkit 12.x. Patch the Caffe2 cmake to create the target if missing.
CUDA_CMAKE="$CONDA_PREFIX/lib/python3.10/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake"
if ! grep -q "nvToolsExt_WORKAROUND" "$CUDA_CMAKE" 2>/dev/null; then
    echo "Patching cuda.cmake for nvToolsExt workaround..."
    # Add a fallback target before the set_property that fails
    sed -i '/set_property.*torch::nvtoolsext.*CUDA::nvToolsExt/i \
# nvToolsExt_WORKAROUND: create CUDA::nvToolsExt if not found (removed in CUDA 12)\
if(NOT TARGET CUDA::nvToolsExt)\
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)\
endif()' "$CUDA_CMAKE"
fi

cmake ../cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_PREFIX_PATH="$TORCH_PREFIX" \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=off \
    -D BUILD_OMP=on \
    -D PKG_MANYBODY=on \
    -D LAMMPS_EXCEPTIONS=on \
    -D BUILD_SHARED_LIBS=off \
    -D JPEG_INCLUDE_DIR="" \
    -D CUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX" \
    2>&1

echo ""
echo "=== Compiling (this may take a few minutes) ==="
make -j8 2>&1 | tail -40

echo ""
echo "=== Build result ==="
ls -la lmp 2>/dev/null && echo "SUCCESS: lmp binary built" || echo "FAILED: no lmp binary"

# Step 5: Quick test
if [ -f lmp ]; then
    echo ""
    echo "=== Quick smoke test ==="
    ./lmp -h 2>&1 | grep -i eosnet && echo "pair_eosnet registered!" || echo "pair_eosnet NOT found in styles"
fi

echo ""
echo "DONE"
