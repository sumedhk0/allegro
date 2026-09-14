#!/usr/bin/env bash
# Full LAMMPS benchmark matrix for the pe206 Allegro model (run inside WSL).
# Stage 1: system-size sweep, base model, Kokkos.
# Stage 2: compile variants (tf32 / cuEq / Triton) at 1x and 3x.
# Stage 3: runtime knobs at 3x: Kokkos off, skin 2.0, neighbor check every 10.
# Results accumulate in $BENCH_DIR/results.csv (one row per case).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
source ~/.venvs/allegro/bin/activate
export OMP_NUM_THREADS=1

export BENCH_DIR=$HOME/pe206/bench
export LMP=$HOME/lammps/build/lmp
MODELS=$HOME/pe206/models
SRC="/mnt/c/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/allegro/benchmarks/pe206"
mkdir -p "$BENCH_DIR"
cp "$SRC/in.pe206" "$SRC/run_bench.sh" "$SRC/ase_bench.py" "$BENCH_DIR/"
chmod +x "$BENCH_DIR/run_bench.sh"
# structure files are generated from the private dataset, never committed
DATASET=${DATASET:-$HOME/pe206/pe_206_dft.xyz}
[ -f "$BENCH_DIR/pe206.data" ] || python "$SRC/make_inputs.py" "$DATASET" --out "$BENCH_DIR"
RB="$BENCH_DIR/run_bench.sh"

STEPS=${STEPS:-200}

echo "=== stage 1: size sweep (base, kokkos)"
for n in 1 2 3 4; do "$RB" base "$MODELS/base.nequip.pt2" $n 1 $STEPS; done

echo "=== stage 2: compile variants"
for n in 1 3; do
  for v in tf32 cueq triton; do "$RB" $v "$MODELS/$v.nequip.pt2" $n 1 $STEPS; done
done

echo "=== stage 3: runtime knobs at 3x"
"$RB" base_nokk   "$MODELS/base.nequip.pt2" 3 0 $STEPS
"$RB" base_skin2  "$MODELS/base.nequip.pt2" 3 1 $STEPS 2.0 1
"$RB" base_nev10  "$MODELS/base.nequip.pt2" 3 1 $STEPS 1.0 10
NP=2 "$RB" base_2rank "$MODELS/base.nequip.pt2" 3 1 $STEPS 1.0 1

echo "=== repeat base 3x for noise estimate"
"$RB" base_rep    "$MODELS/base.nequip.pt2" 3 1 $STEPS

echo MATRIX_DONE
column -s, -t "$BENCH_DIR/results.csv"
