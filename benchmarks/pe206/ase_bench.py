"""Python/ASE inference reference for the pe206 Allegro model (run inside WSL).

Times single-frame force+energy calls through NequIPCalculator on a compiled
(AOTInductor, target ase) model at several replication factors, and a short
VelocityVerlet MD to get an ASE-side ms/step. Complements the LAMMPS numbers.

Usage: python ase_bench.py <model.nequip.pt2> [ncalls] [nrep ...]
"""

import sys
import time

import ase.io
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from nequip.integrations.ase import NequIPCalculator

model_path = sys.argv[1]
ncalls = int(sys.argv[2]) if len(sys.argv) > 2 else 50
nreps = [int(x) for x in sys.argv[3:]] or [1, 2, 3]

frame = ase.io.read("pe206_frame500.xyz")
calc = NequIPCalculator.from_compiled_model(model_path, device="cuda")

print("model", model_path)
print("nrep,natoms,ms_per_call,ms_per_md_step")
for n in nreps:
    atoms = frame.repeat((n, n, n))
    atoms.calc = calc
    rng = np.random.default_rng(0)
    base = atoms.get_positions()
    # warm-up
    for _ in range(5):
        atoms.set_positions(base + 0.01 * rng.standard_normal(base.shape))
        atoms.get_forces()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ncalls):
        atoms.set_positions(base + 0.01 * rng.standard_normal(base.shape))
        atoms.get_forces()
    torch.cuda.synchronize()
    ms_call = 1000 * (time.perf_counter() - t0) / ncalls

    atoms.set_positions(base)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(1))
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    dyn.run(5)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dyn.run(ncalls)
    torch.cuda.synchronize()
    ms_step = 1000 * (time.perf_counter() - t0) / ncalls
    print(f"{n},{len(atoms)},{ms_call:.3f},{ms_step:.3f}", flush=True)
