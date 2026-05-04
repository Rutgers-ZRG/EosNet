# CMakeLists.txt for pair_eosnet LAMMPS plugin
#
# Build instructions:
#   1. Build LAMMPS with CMake:
#      cd lammps/build
#      cmake ../cmake -D PKG_ML-SNAP=yes \
#                     -D CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
#                     -D PAIR_EOSNET_DIR=/path/to/EosNet-dev/lammps
#      make -j
#
#   2. Or as an external plugin (preferred):
#      mkdir build && cd build
#      cmake .. -D LAMMPS_SOURCE_DIR=/path/to/lammps/src \
#               -D CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
#      make -j
#
# Usage in LAMMPS input:
#   pair_style eosnet
#   pair_coeff * * eosnet_deployed.pt 5.0 6.0 32 Fe
#   # Or with default cutoffs:
#   pair_coeff * * eosnet_deployed.pt Fe

cmake_minimum_required(VERSION 3.18)
project(pair_eosnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch (libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- Option A: Build as LAMMPS plugin (standalone) ---
if(LAMMPS_SOURCE_DIR)
  message(STATUS "Building pair_eosnet as standalone plugin")
  message(STATUS "LAMMPS source: ${LAMMPS_SOURCE_DIR}")

  # Include LAMMPS headers
  include_directories(${LAMMPS_SOURCE_DIR})

  add_library(pair_eosnet SHARED
    pair_eosnet.cpp
  )
  target_link_libraries(pair_eosnet "${TORCH_LIBRARIES}")
  target_include_directories(pair_eosnet PRIVATE ${LAMMPS_SOURCE_DIR})

  install(TARGETS pair_eosnet DESTINATION lib)
  install(FILES pair_eosnet.h DESTINATION include)

# --- Option B: Build within LAMMPS cmake (add as extra pair style) ---
else()
  message(STATUS "Building pair_eosnet within LAMMPS")

  # When included from LAMMPS CMake, just add the source files
  target_sources(lammps PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pair_eosnet.cpp
  )
  target_include_directories(lammps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(lammps "${TORCH_LIBRARIES}")
endif()

message(STATUS "PyTorch: ${TORCH_LIBRARIES}")
message(STATUS "CUDA: ${TORCH_CUDA_LIBRARIES}")
