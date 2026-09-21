"""Run MD with a trained Allegro checkpoint through ASE. No compilation, no LAMMPS.

Loads the checkpoint in eager mode (works on Windows and Linux, GPU or CPU), reads
cutoff and atom types from the checkpoint itself, and runs NVE (Velocity Verlet),
NVT (Langevin) or NPT (Berendsen, isotropic). Writes an extxyz trajectory plus a CSV
log of time, temperature, energies, pressure and volume.

Examples
    python examples/md_ase.py --ckpt outputs/.../best.ckpt --structure pe_206_dft.xyz --frame 500
    python examples/md_ase.py --ckpt best.ckpt --structure cell.xyz --repeat 2 --temp 400 --steps 20000

Speed on an RTX 4080 laptop, eager fp32: roughly 4 ms/step for 206 atoms. For boxes
above ~2000 atoms or millions of steps, use the LAMMPS route in benchmarks/pe206/.
"""

import argparse
import csv
import time

import ase.io
import numpy as np
import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from nequip.utils.global_state import set_global_state

set_global_state()
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform  # noqa: E402
from nequip.integrations.ase import NequIPCalculator  # noqa: E402
from nequip.model import ModelFromCheckpoint  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", required=True, help="nequip checkpoint (best.ckpt / last.ckpt)")
ap.add_argument("--structure", required=True, help="any ASE-readable structure file")
ap.add_argument("--frame", type=int, default=0, help="frame index if the file holds several")
ap.add_argument("--repeat", type=int, default=1, help="replicate the cell n x n x n")
ap.add_argument("--ensemble", choices=["nvt", "nve", "npt"], default="nvt")
ap.add_argument("--temp", type=float, default=300.0, help="K")
ap.add_argument("--friction", type=float, default=0.01, help="Langevin friction, 1/fs (nvt)")
ap.add_argument("--press", type=float, default=1.0, help="bar (npt)")
ap.add_argument("--taut", type=float, default=100.0, help="Berendsen T coupling time, fs (npt)")
ap.add_argument("--taup", type=float, default=1000.0, help="Berendsen P coupling time, fs (npt)")
ap.add_argument("--compressibility", type=float, default=3.3e-5, help="1/bar (npt); ~1/3 GPa for polyethylene")
ap.add_argument("--dt", type=float, default=0.5, help="fs")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--every", type=int, default=50, help="log/dump interval in steps")
ap.add_argument("--out", default="md", help="output prefix -> <out>.xyz, <out>.csv")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

# --- model -> calculator, configured from the checkpoint's own metadata
loaded = ModelFromCheckpoint(args.ckpt)
model = (list(loaded.values())[0] if isinstance(loaded, torch.nn.ModuleDict) else loaded).eval()
r_max = float(model.metadata["r_max"])
type_names = model.metadata["type_names"].split()
calc = NequIPCalculator(
    model,
    device=args.device,
    transforms=[
        ChemicalSpeciesToAtomTypeMapper(model_type_names=type_names),
        NeighborListTransform(r_max=r_max),
    ],
)
print(f"model: r_max={r_max} A, types={type_names}, device={args.device}")

# --- structure
atoms = ase.io.read(args.structure, index=args.frame)
if args.repeat > 1:
    atoms = atoms.repeat((args.repeat,) * 3)
atoms.calc = calc
print(f"system: {len(atoms)} atoms, cell {np.round(atoms.cell.lengths(), 2)} A, pbc {atoms.pbc}")

rng = np.random.default_rng(args.seed)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, rng=rng)
Stationary(atoms)

if args.ensemble == "nvt":
    dyn = Langevin(atoms, args.dt * units.fs, temperature_K=args.temp, friction=args.friction / units.fs, rng=rng)
elif args.ensemble == "npt":
    # Berendsen barostat: works with triclinic cells (ase.md.npt.NPT needs an upper-triangular cell)
    dyn = NPTBerendsen(
        atoms, args.dt * units.fs, temperature_K=args.temp, pressure_au=args.press * units.bar,
        taut=args.taut * units.fs, taup=args.taup * units.fs, compressibility_au=args.compressibility / units.bar,
    )
else:
    dyn = VelocityVerlet(atoms, args.dt * units.fs)

# --- logging
traj_file = f"{args.out}.xyz"
csv_file = f"{args.out}.csv"
open(traj_file, "w").close()
log = open(csv_file, "w", newline="")
writer = csv.writer(log)
writer.writerow(["step", "time_ps", "T_K", "Epot_eV", "Ekin_eV", "Etot_eV", "P_bar", "V_A3", "ms_per_step"])
t_last = time.perf_counter()


def snapshot():
    global t_last
    step = dyn.nsteps
    now = time.perf_counter()
    ms = 1000 * (now - t_last) / max(args.every, 1)
    t_last = now
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    T = atoms.get_temperature()
    # pressure from the model stress (ASE sign: negative trace = compressed); virial part only
    P = -atoms.get_stress(voigt=True)[:3].mean() / units.bar
    V = atoms.get_volume()
    writer.writerow([step, step * args.dt / 1000, f"{T:.1f}", f"{epot:.4f}", f"{ekin:.4f}", f"{epot + ekin:.4f}", f"{P:.1f}", f"{V:.2f}", f"{ms:.2f}"])
    log.flush()
    ase.io.write(traj_file, atoms, format="extxyz", append=True)
    print(f"step {step:7d}  t={step * args.dt / 1000:7.3f} ps  T={T:6.1f} K  Epot={epot:11.4f} eV  P={P:8.1f} bar  V={V:9.2f} A3  {ms:6.2f} ms/step", flush=True)


dyn.attach(snapshot, interval=args.every)  # ASE also fires this at step 0
t0 = time.perf_counter()
dyn.run(args.steps)
elapsed = time.perf_counter() - t0
ms_step = 1000 * elapsed / args.steps
ns_day = args.dt * 1e-6 * 86400 / (elapsed / args.steps)
print(f"done: {args.steps} steps in {elapsed:.1f} s = {ms_step:.2f} ms/step = {ns_day:.2f} ns/day at {args.dt} fs")
print(f"trajectory: {traj_file}   log: {csv_file}")
log.close()
