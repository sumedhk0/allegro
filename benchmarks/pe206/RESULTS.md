# How fast can this Allegro model run MD on an RTX 4080 laptop?

Benchmark date: 2026-09-13. Branch `pe206-training`.

## TL;DR

- **Fastest setup: LAMMPS + `pair_allegro` + Kokkos (GPU-resident), fp32 AOTInductor model.**
- **Throughput ceiling: about 4.3 µs per atom per MD step** once the box has more than
  ~1500 atoms. That is ~230,000 atom-steps per second.
- In wall-clock terms at a 0.5 fs timestep: **19 ns/day for 206 atoms, 5.9 ns/day for
  1,648 atoms, 1.8 ns/day for 5,562 atoms, 0.77 ns/day for 13,184 atoms.**
- Nothing else on the runtime side moves the needle: TF32 gives 6% and hurts energy
  conservation; the Triton and cuEquivariance kernels do nothing for a 1-layer model
  (and cuEquivariance cannot even be loaded by LAMMPS); neighbor-list settings are
  irrelevant; running without Kokkos costs 18%.
- Any further speedup has to come from the model (smaller cutoff, narrower MLPs) or
  from more GPUs.

## What was benchmarked

**The model.** Allegro trained from scratch on the pe_206 polyethylene DFT dataset
(1,002 frames of a 206-atom periodic cell, C68H138, with energies, forces and stresses).
Hyperparameters, as chosen for this study and not swept:

| setting | value |
|---|---|
| cutoff radius | 4.5 Å (~36 neighbors per atom) |
| layers / l_max | 1 / 1 |
| scalar / tensor features | 64 / 32 |
| precision | float32 |
| pair potential (ZBL) | none |
| training | 100 epochs, Adam 1e-3, batch 4, EMA |

Test-set accuracy of the checkpoint used: 7.9 meV/atom energy MAE, 58 meV/Å force MAE.
Config: `configs/pe206_allegro.yaml`. Note the dataset's stress labels use the opposite
sign convention from nequip/ASE; the config flips them on load.

**The machine.** NVIDIA RTX 4080 Laptop GPU (12 GB), Intel i7-13800H, Windows 11.
All inference ran inside WSL2 (Ubuntu 26.04) because compiled inference is not possible
on native Windows here (no MSVC; cuEquivariance ships Linux-only wheels).
Stack: torch 2.14.0+cu130, nequip 0.19.1, allegro 0.8.3, CUDA 13.3 toolkit,
LAMMPS develop ("2 Sep 2026") with `pair_nequip_allegro`, Kokkos 5.2.1.

**The MD protocol.** NVE, 0.5 fs timestep, velocities initialised at 300 K, 20 warm-up
steps then 200 timed steps. Cells are the 206-atom frame replicated 1×, 2×, 3×, 4× per
axis. "ns/day" always assumes the 0.5 fs step.

## Terms used below

- **AOTInductor**: PyTorch's ahead-of-time compiler. `nequip-compile` turns the
  checkpoint into a `.nequip.pt2` file that LAMMPS loads without Python.
- **Kokkos**: LAMMPS' GPU backend. With it, the neighbor list and atom data live on the
  GPU; without it, they are copied host↔device every step.
- **TF32**: reduced-precision matrix multiplies on NVIDIA tensor cores, enabled at
  compile time with `--tf32`.
- **Triton / cuEquivariance contracters**: replacement GPU kernels for Allegro's tensor
  product, enabled with `--modifiers` at compile time.

## Results

### 1. Size scaling (base model, Kokkos)

| cell | atoms | ms per step | µs per atom-step | ns/day |
|---|---|---|---|---|
| 1×1×1 | 206 | 2.2 – 2.4 | 11 | 19 |
| 2×2×2 | 1,648 | 7.3 | 4.4 | 5.9 |
| 3×3×3 | 5,562 | 23 – 25 | 4.4 | 1.8 |
| 4×4×4 | 13,184 | 56 | 4.2 | 0.77 |

The 206-atom cell cannot fill the GPU, so it runs at less than half efficiency. From
~1,500 atoms up, cost is linear in atom count at ~4.3 µs per atom-step.

### 2. Compiled-model variants (3 interleaved repeats, mean ± std)

| variant | 206 atoms, ms/step | 5,562 atoms, ms/step | energy drift over 200 steps (eV), 206 / 5,562 atoms |
|---|---|---|---|
| base (fp32) | 2.30 ± 0.16 | 24.6 ± 0.9 | −0.0017 / −0.21, identical in every run |
| TF32 | 2.18 ± 0.15 | 23.0 ± 0.05 | −0.017, +0.009, −0.005 / −0.24, −0.32, −0.29 |
| Triton kernel | 1.97 ± 0.10 | 24.3 ± 0.9 | same as base |
| cuEquivariance kernel | fails to load | fails to load | — |

What this means:

- **TF32**: ~6% faster on the large cell, nothing measurable on the small one. The
  price is an energy drift that is 10× larger and varies from run to run. Not
  recommended for production MD.
- **Triton**: the modifier only replaces tensor products that produce non-scalar
  outputs from more than one path. A 1-layer Allegro model has a single, scalar-output
  tensor product, so nothing is replaced. The apparent edge on the 206-atom cell is
  compile-to-compile variance in generated code; at 5,562 atoms the two are identical.
  This kernel would matter for models with 2+ layers or l_max ≥ 2.
- **cuEquivariance**: its operator is registered via `torch.library.custom_op`, which an
  AOTInductor binary cannot call outside Python. LAMMPS refuses the model
  (`results/cueq_pair_allegro_error.txt`). It does load from Python/ASE, where it runs
  at the same speed as base for this model.

### 3. LAMMPS runtime settings (base model, 5,562 atoms)

| setting | ms/step | relative |
|---|---|---|
| Kokkos, `newton on`, `neigh half` (default used everywhere else) | 23 – 25 | reference |
| plain `pair_allegro`, no Kokkos | 28.4 | +18% slower |
| neighbor skin 2.0 Å instead of 1.0 Å | 24.0 | no change |
| neighbor list check every 10 steps instead of every step | 23.9 | no change |
| 2 MPI ranks sharing the GPU | fails | plugin requires one GPU per rank |

### 4. Python (ASE) path, for comparison

Same compiled model loaded through `NequIPCalculator`, one force call per measurement.

| model | 206 atoms, ms/call | 5,562 atoms, ms/call |
|---|---|---|
| base | 4.2 | 59 |
| TF32 | 4.3 | 61 |
| cuEquivariance | 4.0 | 60 |
| Triton | 3.5 | 54 |

LAMMPS with Kokkos is 1.8× faster than ASE at 206 atoms and 2.4× faster at 5,562 atoms.
The ASE path builds its neighbor list on the CPU every call, so it also slows down when
the CPU is busy (8.9 ms instead of 4.2 ms for 206 atoms while a compile ran alongside).

## Take-aways

1. Run production MD in LAMMPS with `pair_allegro` under Kokkos, fp32, no modifiers.
   Command shape:
   `mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.pe206`
2. Size the simulation box to at least ~1,500 atoms if you care about GPU efficiency;
   below that, per-atom cost more than doubles.
3. Plan on ~4.3 µs per atom-step. For a 10,000-atom box at 0.5 fs that is ~1 ns/day.
4. To go faster, change the model, not the runtime: a smaller cutoff (4.0 Å is ~24
   neighbors instead of 36) and narrower MLPs are the levers; each needs retraining.
5. A second GPU would be the only other lever; the plugin needs one GPU per MPI rank.

## Reproducing

All scripts live in this directory and were run inside WSL2 Ubuntu.

1. `build_lammps.sh` — builds LAMMPS develop with `pair_nequip_allegro`, AOTInductor
   loading and Kokkos CUDA (sm_89). It first runs `fix_compute_header.py`, which patches
   a plugin header that no longer compiles against current LAMMPS develop.
2. `compile_models.sh` — compiles the checkpoint into the base, TF32, cuEquivariance and
   Triton variants with `nequip-compile`.
3. `make_inputs.py <dataset.xyz>` — writes the LAMMPS data file and an extxyz frame from
   the dataset. These files are derived from private DFT data and are not committed.
4. `run_matrix.sh` — the full LAMMPS matrix; `run_bench.sh` runs one case and appends a
   CSV row (ms/step, µs/atom-step, ns/day, energy drift).
5. `ase_bench.py` — the Python/ASE reference timings.

Raw outputs: `results/results.csv` (matrix, single runs; its `natoms` column reflects the
pre-replicate count for that pass), `results/results_repeats.csv` (3× repeats with
drift), `results/ase_*.csv`, and per-case LAMMPS logs in `results/logs/`.
