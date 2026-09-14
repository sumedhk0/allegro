"""Make pair_nequip_allegro's compute header build against LAMMPS develop (>= Sep 2026).

LAMMPS' newer style registry generates `styles/style_compute.cpp` from the
`ComputeStyle(name, Class)` lines of every compute header, but no longer pastes
the surrounding `#ifdef COMPUTE_CLASS` block verbatim. The plugin declares its
four typedefs (ComputeAllegro, ...) inside that block, so the generated file
references names that no longer exist and the build fails with
`identifier "ComputeAllegro" is undefined`.

This moves the typedefs into the header's namespace body, where any includer
sees them. Idempotent; run after patch_lammps.sh and before cmake.
"""

import pathlib
import sys

lammps_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path.home() / "lammps")
p = lammps_dir / "src" / "compute_nequip_allegro.h"
s = p.read_text()

typedefs = """typedef ComputeNequIPAllegro<0,0> ComputeAllegro;
typedef ComputeNequIPAllegro<0,1> ComputeAllegroPerAtom;
typedef ComputeNequIPAllegro<1,0> ComputeNequIP;
typedef ComputeNequIPAllegro<1,1> ComputeNequIPPerAtom;
"""
marker = "  std::string compute_name;\n};\n\n}\n"

if s.count(typedefs) == 1 and s.index(typedefs) > s.index("namespace LAMMPS_NS"):
    print("already patched", p)
    sys.exit(0)

assert typedefs in s, "typedef block not found in " + str(p)
assert marker in s, "end-of-class marker not found in " + str(p)
s = s.replace(typedefs + "\n", "", 1) if (typedefs + "\n") in s else s.replace(typedefs, "", 1)
s = s.replace(marker, "  std::string compute_name;\n};\n\n" + typedefs + "\n}\n", 1)
p.write_text(s)
print("patched", p)
