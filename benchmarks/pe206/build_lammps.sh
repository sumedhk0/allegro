#!/usr/bin/env bash
# Build LAMMPS (develop) with pair_nequip_allegro, AOTI model loading, and Kokkos CUDA
# for the RTX 4080 (Ada, sm_89). Run inside WSL Ubuntu after setup_venv.sh and
# setup_lammps_src.sh have finished.
set -euxo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
source ~/.venvs/allegro/bin/activate

TORCH_CMAKE=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
echo "torch cmake prefix: $TORCH_CMAKE"

cd ~/lammps
rm -rf build && mkdir build && cd build
cmake ../cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
  -DNEQUIP_AOT_COMPILE=ON \
  -DMKL_INCLUDE_DIR=/tmp \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DBUILD_MPI=ON \
  -DBUILD_OMP=OFF \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_ADA89=ON \
  -DPKG_MOLECULE=OFF \
  2>&1 | tail -30
ninja -j 12 2>&1 | tail -5
ls -la ~/lammps/build/lmp
~/lammps/build/lmp -h | grep -iE "allegro|kokkos" | head
echo BUILD_DONE
