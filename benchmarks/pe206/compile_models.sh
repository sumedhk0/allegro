#!/usr/bin/env bash
# Compile the run2 checkpoint into the AOTInductor variants used by the benchmark.
# Run inside WSL Ubuntu with the allegro venv.
set -euxo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
source ~/.venvs/allegro/bin/activate
# WSL keeps the driver's libcuda.so outside the linker search path; AOTInductor links -lcuda
export LIBRARY_PATH=/usr/lib/wsl/lib:${LIBRARY_PATH:-}

CKPT=$HOME/pe206/run2_best.ckpt
OUT=$HOME/pe206/models
mkdir -p "$OUT"

# nequip-compile rebuilds the training datamodule from the checkpoint to get
# example data with realistic shapes. The checkpoint stores the Windows dataset
# path "C:/Users/sumedh/Downloads/pe_206_dft.xyz", which on Linux is a relative
# path, so we make it resolve from the working directory with a symlink.
WORK=$HOME/pe206/compile_work
mkdir -p "$WORK/C:/Users/sumedh/Downloads"
ln -sf "$HOME/pe206/pe_206_dft.xyz" "$WORK/C:/Users/sumedh/Downloads/pe_206_dft.xyz"
cd "$WORK"

compile () {
  local name=$1; shift
  local out="$OUT/$name.nequip.pt2"
  if [ -f "$out" ]; then echo "skip $name (exists)"; return; fi
  /usr/bin/time -f "compile $name: %e s" \
    nequip-compile "$CKPT" "$out" --device cuda --mode aotinductor --target pair_allegro "$@" \
    2>&1 | grep -vE "^\s*$" | tail -4
}

compile base
# TF32 changes forces by ~5e-3 eV/A on this model, above nequip's default 2e-3
# check tolerance; relax the check for the benchmark variant only.
NEQUIP_TF32_MODEL_TOL=0.01 compile tf32 --tf32
compile cueq   --modifiers enable_CuEquivarianceContracter
compile triton --modifiers enable_TritonContracter
# ASE-target variant for the Python reference measurement
if [ ! -f "$OUT/base_ase.nequip.pt2" ]; then
  nequip-compile "$CKPT" "$OUT/base_ase.nequip.pt2" --device cuda --mode aotinductor --target ase 2>&1 | tail -2
fi
ls -la "$OUT"
echo COMPILE_DONE
