#!/usr/bin/env bash
# Ensemble cost comparison: NVE vs NVT (Nose-Hoover) vs NPT (Nose-Hoover, isotropic),
# base model, Kokkos, 1x and 3x cells, 3 interleaved repeats. Run inside WSL.
set -uo pipefail
export PATH=/usr/local/cuda-13.3/bin:$HOME/.local/bin:$PATH
source ~/.venvs/allegro/bin/activate
export OMP_NUM_THREADS=1
export BENCH_DIR=$HOME/pe206/bench
export LMP=$HOME/lammps/build/lmp
export CSV=$BENCH_DIR/results_ensembles.csv
MODELS=$HOME/pe206/models
SRC="/mnt/c/Users/sumedh/OneDrive - Georgia Institute of Technology/Python/allegro/benchmarks/pe206"
cp "$SRC/in.pe206" "$SRC/run_bench.sh" "$BENCH_DIR/" && sed -i 's/\r$//' "$BENCH_DIR/in.pe206" "$BENCH_DIR/run_bench.sh" && chmod +x "$BENCH_DIR/run_bench.sh"
DATASET=${DATASET:-$HOME/pe206/pe_206_dft.xyz}
[ -f "$BENCH_DIR/pe206.data" ] || python "$SRC/make_inputs.py" "$DATASET" --out "$BENCH_DIR"
STEPS=${STEPS:-200}
rm -f "$CSV"
for r in 1 2 3; do
  for n in 1 3; do
    for e in nve nvt npt; do
      ENSEMBLE=$e "$BENCH_DIR/run_bench.sh" ${e}_r$r "$MODELS/base.nequip.pt2" $n 1 $STEPS > /dev/null
    done
  done
done
echo ENSEMBLES_DONE
column -s, -t "$CSV"
