# pe206 Allegro inference benchmarks (2026-09-13)

Model: `configs/pe206_allegro.yaml` at commit cd36d20, run2 checkpoint
(r_max 4.5 Å, l_max 1, 1 layer, 64 scalar / 32 tensor features, float32, no ZBL).
Test-set accuracy of that checkpoint: 7.9 meV/atom, 58 meV/Å force MAE.

Hardware: RTX 4080 Laptop (12 GB, Ada), i7-13800H, Windows 11 → WSL2 Ubuntu 26.04.
Software: torch 2.14.0+cu130, nequip 0.19.1, allegro 0.8.3, CUDA 13.3 toolkit,
LAMMPS develop "2 Sep 2026" + pair_nequip_allegro (AOTInductor, Kokkos 5.2.1 CUDA).
MD: NVE, 0.5 fs, 300 K, 20 warm-up steps then 200 timed steps. ns/day assumes 0.5 fs.

## 1. LAMMPS pair_allegro/kk, base model: system-size sweep

| cell | atoms | ms/step | µs per atom-step | ns/day |
|---|---|---|---|---|
| 1×1×1 | 206 | 2.2–2.4 | 11 | 18.9 |
| 2×2×2 | 1648 | 7.3 | 4.4 | 5.9 |
| 3×3×3 | 5562 | 23–25 | 4.4 | 1.8 |
| 4×4×4 | 13184 | 56.0 | 4.2 | 0.77 |

The GPU saturates at roughly 1.5k atoms: above that, cost is ~4.3 µs per atom-step
(~230k atom-steps/s) and ms/step is linear in atom count. The 206-atom cell runs at
less than half efficiency because kernel launch overhead dominates.

## 2. Compile variants (3 interleaved repeats, mean ± sd)

| variant | 206 atoms ms/step | 5562 atoms ms/step | NVE drift 206 / 5562 (eV over 200 steps) |
|---|---|---|---|
| base (fp32 AOTI) | 2.30 ± 0.16 | 24.6 ± 0.9 | -0.0017 / -0.21, identical every run |
| `--tf32` | 2.18 ± 0.15 | 23.0 ± 0.05 | -0.017, +0.009, -0.005 / -0.24, -0.32, -0.29 |
| `--modifiers enable_TritonContracter` | 1.97 ± 0.10 | 24.3 ± 0.9 | same as base |
| `--modifiers enable_CuEquivarianceContracter` | fails to load | fails to load | n/a |

- TF32 buys ~6 % at 5562 atoms and nothing measurable at 206 atoms, at the cost of a
  10× larger and non-deterministic energy drift. Not worth it for this model.
- Triton: `enable_TritonContracter` only swaps in the kernel when the tensor product
  has non-diagonal Clebsch-Gordan blocks and more than one path
  (`allegro/nn/_strided/_contract.py`). At 1 layer the only tensor product outputs
  scalars, so its blocks are diagonal and nothing is replaced. The 206-atom edge is
  most likely inductor codegen variance between two compiles of the same graph; at
  5562 atoms the two are identical within noise.
- cuEquivariance cannot be used from pair_allegro: its op is registered through
  `torch.library.custom_op`, which AOTInductor cannot call without a Python interpreter
  (`results/cueq_pair_allegro_error.txt`). It works from ASE, where it is within noise
  of base for this model.

## 3. Runtime knobs, base model, 5562 atoms (single runs)

| setting | ms/step | vs Kokkos default |
|---|---|---|
| Kokkos (`-k on g 1 -sf kk`, newton on, neigh half) | 23.2–25.1 | reference |
| no Kokkos (plain `pair_allegro`) | 28.4 | +18 % |
| skin 2.0 Å instead of 1.0 | 24.0 | no change |
| neighbor check every 10 instead of 1 | 23.9 | no change |
| 2 MPI ranks on 1 GPU | fails | plugin requires one GPU per rank |

## 4. ASE / Python path (compiled model, `--target ase`), same GPU

| model | 206 atoms ms/call | 5562 atoms ms/call |
|---|---|---|
| base | 4.2 | 59 |
| tf32 | 4.3 | 61 |
| cuEq | 4.0 | 60 |
| triton | 3.5 | 54 |

LAMMPS+Kokkos is 1.8× faster than ASE at 206 atoms and 2.4× at 5562 atoms. The ASE
path rebuilds neighbor lists on the CPU every call, so it is also sensitive to CPU
load (with a parallel compile running, the 206-atom call took 8.9 ms).

## Conclusions for MD on this machine

1. Use LAMMPS with `pair_allegro` under Kokkos. Everything else on the list is noise
   for a 1-layer, l_max 1 model.
2. Expect ~4.3 µs per atom-step once the box exceeds ~1.5k atoms: 5.9 ns/day at 1.6k
   atoms, 1.8 ns/day at 5.6k, 0.77 ns/day at 13k, all at 0.5 fs.
3. Keep fp32. TF32's 6 % is not worth the drift.
4. Further speed must come from the model (fewer edges via smaller r_max, narrower
   MLPs) rather than from the runtime; those knobs were fixed for this study.
5. Multi-GPU scaling is the only untested lever; the plugin needs one GPU per rank.

## Files

- `results/results.csv` — matrix single runs (note: `natoms` column shows the
  pre-replicate count for that first pass; `results_repeats.csv` is correct).
- `results/results_repeats.csv` — 3× interleaved repeats with NVE drift.
- `results/ase_*.csv` — ASE reference timings.
- `results/logs/` — LAMMPS logs per case.
- Scripts: `build_lammps.sh`, `fix_compute_header.py`, `compile_models.sh`,
  `run_bench.sh`, `run_matrix.sh`, `ase_bench.py`, `in.pe206`, `pe206.data`.
