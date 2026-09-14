"""Generate the benchmark structure files from the (private) pe_206 dataset.

Writes into --out (default: current directory):
  pe206.data          LAMMPS data file (atom_style atomic, types 1=C 2=H, triclinic)
  pe206_frame500.xyz  the same frame as extxyz, used by ase_bench.py

The DFT dataset itself is not part of the repository; point this script at your
copy of pe_206_dft.xyz. Frame 500 (mid-trajectory) is the default.

Usage: python make_inputs.py /path/to/pe_206_dft.xyz [--frame 500] [--out DIR]
"""

import argparse
import pathlib

import ase.io
from ase.io.lammpsdata import write_lammps_data

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--frame", type=int, default=500)
ap.add_argument("--out", default=".")
args = ap.parse_args()

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
atoms = ase.io.read(args.dataset, index=args.frame)
write_lammps_data(
    out / "pe206.data", atoms, specorder=["C", "H"], atom_style="atomic", masses=True, force_skew=True
)
ase.io.write(out / f"pe206_frame{args.frame}.xyz", atoms, format="extxyz")
print(f"wrote {out / 'pe206.data'} and {out / f'pe206_frame{args.frame}.xyz'} ({len(atoms)} atoms)")
