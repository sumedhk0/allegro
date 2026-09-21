"""Convert any ASE-readable structure to a LAMMPS data file for pair_allegro.

Usage: python to_lammps_data.py <structure> <out.data> [--frame N] [--types "C H"]

Atom types are numbered in the order given by --types, which must match the
pair_coeff line (pair_coeff * * model.nequip.pt2 C H). Masses are written.
"""

import argparse

import ase.io
from ase.io.lammpsdata import write_lammps_data

ap = argparse.ArgumentParser()
ap.add_argument("structure")
ap.add_argument("out")
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--types", default="C H")
args = ap.parse_args()

atoms = ase.io.read(args.structure, index=args.frame)
types = args.types.split()
missing = sorted(set(atoms.get_chemical_symbols()) - set(types))
if missing:
    raise SystemExit(f"structure contains species {missing} not in --types {types}")
write_lammps_data(args.out, atoms, specorder=types, atom_style="atomic", masses=True, force_skew=True)
print(f"wrote {args.out}: {len(atoms)} atoms, types {types}, cell {atoms.cell.cellpar().round(3)}")
