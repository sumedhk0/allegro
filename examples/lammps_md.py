#!/usr/bin/env python3
"""Allegro + LAMMPS in one file. Run inside WSL Ubuntu with the allegro venv active.

    python lammps_md.py build                       # one-time: LAMMPS + pair_allegro + Kokkos CUDA
    python lammps_md.py compile CKPT [--variant V]  # checkpoint -> compiled .nequip.pt2
    python lammps_md.py run --structure FILE ...    # one MD run (nve / nvt / npt)
    python lammps_md.py bench ...                   # throughput matrix -> CSV

Defaults assume the layout used on this machine (override with flags or env vars):
    LMP        ~/lammps/build/lmp          the LAMMPS binary
    MODEL      ~/pe206/models/base.nequip.pt2
    CUDA_HOME  /usr/local/cuda-13.3
    DATASET    ~/pe206/pe_206_dft.xyz      only needed by `compile` (example data)

Sections below, in order: configuration, small helpers, structure conversion, LAMMPS
input generation, running LAMMPS and reading its log, then one function per subcommand.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ----------------------------------------------------------------------------- configuration
HOME = Path.home()
LMP = Path(os.environ.get("LMP", HOME / "lammps/build/lmp"))
MODEL = Path(os.environ.get("MODEL", HOME / "pe206/models/base.nequip.pt2"))
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.3"))
DATASET = Path(os.environ.get("DATASET", HOME / "pe206/pe_206_dft.xyz"))
LAMMPS_SRC = HOME / "lammps"
PLUGIN_SRC = HOME / "pair_nequip_allegro"
KOKKOS_ARGS = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"]

# nequip-compile variants: name -> extra flags. TF32 fails nequip's post-compile numerics
# check on this model by ~5e-3 eV/A (forces) and ~1.4e-2 eV (virial), so its check
# tolerance is relaxed to 0.05 for the benchmark variant only.
VARIANTS = {
    "base": [],
    "tf32": ["--tf32"],
    "triton": ["--modifiers", "enable_TritonContracter"],
    "cueq": ["--modifiers", "enable_CuEquivarianceContracter"],  # loads in ASE only, not in pair_allegro
}


# ----------------------------------------------------------------------------- helpers
def sh(cmd: list[str], cwd: Path | None = None, env: dict | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a command, echo it, raise on failure."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [str(c) for c in cmd], cwd=cwd, env=full_env, check=True,
        stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None,
    )


def unix_path(p: str | Path) -> Path:
    """Accept Windows paths (C:\\... or C:/...) when running under WSL."""
    s = str(p)
    if re.match(r"^[A-Za-z]:[\\/]", s) and shutil.which("wslpath"):
        s = subprocess.run(["wslpath", "-u", s.replace("\\", "/")], capture_output=True, text=True, check=True).stdout.strip()
    return Path(s).expanduser()


def venv_env() -> dict:
    """PATH with CUDA and the venv first; LIBRARY_PATH so AOTInductor can link -lcuda under WSL."""
    return {
        "PATH": f"{CUDA_HOME / 'bin'}:{Path(sys.executable).parent}:{HOME / '.local/bin'}:{os.environ.get('PATH', '')}",
        "LIBRARY_PATH": f"/usr/lib/wsl/lib:{os.environ.get('LIBRARY_PATH', '')}",
        "OMP_NUM_THREADS": "1",
    }


# ----------------------------------------------------------------------------- structure -> LAMMPS data
def write_data_file(structure: Path, out: Path, types: list[str], frame: int = 0, nrep: int = 1):
    """Convert any ASE-readable structure to a LAMMPS data file (atom_style atomic).

    Types are numbered in the order of `types`, which must match the pair_coeff line.
    Replication is done here rather than with LAMMPS' `replicate` so `natoms` in the log
    and in the returned dict are unambiguous.
    """
    import ase.io
    from ase.io.lammpsdata import write_lammps_data

    atoms = ase.io.read(structure, index=frame)
    if nrep > 1:
        atoms = atoms.repeat((nrep,) * 3)
    missing = sorted(set(atoms.get_chemical_symbols()) - set(types))
    if missing:
        raise SystemExit(f"structure has species {missing} not covered by --types {types}")
    write_lammps_data(out, atoms, specorder=types, atom_style="atomic", masses=True, force_skew=True)
    return len(atoms)


# ----------------------------------------------------------------------------- LAMMPS input
def lammps_input(*, model: Path, types: list[str], ensemble: str, temp: float, press: float, pcouple: str,
                 dt_fs: float, steps: int, warmup: int, dump_every: int, thermo: int, skin: float,
                 nevery: int, seed: int) -> str:
    """Return a complete LAMMPS input script as text (metal units: eV, Angstrom, ps, bar)."""
    dt_ps = dt_fs / 1000
    fix = {
        "nve": "fix 1 all nve",
        "nvt": f"fix 1 all nvt temp {temp} {temp} {100 * dt_ps:.6f}",
        "npt": f"fix 1 all npt temp {temp} {temp} {100 * dt_ps:.6f} {pcouple} {press} {press} {1000 * dt_ps:.6f}",
    }[ensemble]
    dump = f"dump 1 all custom {dump_every} traj.lammpstrj id type x y z vx vy vz\ndump_modify 1 sort id" if dump_every else "# no dump"
    warm = f"run {warmup}\n" if warmup else ""
    return textwrap.dedent(f"""\
        # generated by lammps_md.py
        units         metal
        atom_style    atomic
        boundary      p p p
        newton        on
        read_data     structure.data
        pair_style    allegro
        pair_coeff    * * {model} {' '.join(types)}
        neighbor      {skin} bin
        neigh_modify  every {nevery} delay 0 check yes
        velocity      all create {temp} {seed} mom yes rot yes dist gaussian
        timestep      {dt_ps}
        {fix}
        thermo        {thermo}
        thermo_style  custom step time temp pe ke etotal press vol cpu
        thermo_modify flush yes
        {dump}
        {warm}run {steps}
        """)


# ----------------------------------------------------------------------------- run + parse
def run_lammps(outdir: Path, kokkos: bool = True, np_ranks: int = 1) -> dict:
    """Run LAMMPS on outdir/in.md, return timing parsed from the last `run` block of the log."""
    cmd = ["mpirun", "-np", str(np_ranks), LMP, *(KOKKOS_ARGS if kokkos else []), "-in", "in.md", "-log", "log.lammps", "-screen", "none"]
    t0 = time.perf_counter()
    with open(outdir / "stdout.txt", "w") as fh:  # pair_allegro's own banner and any errors land here
        proc = subprocess.run([str(c) for c in cmd], cwd=outdir, env={**os.environ, **venv_env()}, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    log = (outdir / "log.lammps").read_text(errors="replace") if (outdir / "log.lammps").exists() else ""
    return {"status": proc.returncode, "wall_s": round(wall, 2), **parse_log(log)}


def parse_log(log: str) -> dict:
    """ms/step, ns/day, atoms and total-energy drift for the LAST run block in a LAMMPS log."""
    out = {"ms_per_step": float("nan"), "us_per_atom_step": float("nan"), "ns_per_day": float("nan"), "natoms": 0, "etot_drift_eV": float("nan")}
    loops = re.findall(r"Loop time of ([\d.]+) on \d+ procs for (\d+) steps with (\d+) atoms", log)
    if not loops:
        return out
    secs, steps, natoms = float(loops[-1][0]), int(loops[-1][1]), int(loops[-1][2])
    out.update(ms_per_step=round(1000 * secs / steps, 3), us_per_atom_step=round(1e6 * secs / steps / natoms, 3), natoms=natoms)
    perf = re.findall(r"Performance: ([\d.]+) ns/day", log)
    if perf:
        out["ns_per_day"] = float(perf[-1])
    # thermo table of the last block: header line starts with "Step", TotEng column by name
    blocks = log.split("Loop time")[-2] if len(loops) > 0 else log
    lines = blocks.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].split()[:1] == ["Step"]:
            cols = lines[i].split()
            rows = [ln.split() for ln in lines[i + 1:] if ln.strip() and ln.split()[0].isdigit()]
            if "TotEng" in cols and len(rows) > 1:
                k = cols.index("TotEng")
                out["etot_drift_eV"] = round(float(rows[-1][k]) - float(rows[0][k]), 5)
            break
    return out


# ----------------------------------------------------------------------------- subcommand: run
def cmd_run(a):
    model, structure = unix_path(a.model), unix_path(a.structure)
    outdir = unix_path(a.out) if a.out else HOME / "pe206/md" / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.ensemble}_{a.temp:g}K"
    outdir.mkdir(parents=True, exist_ok=True)
    natoms = write_data_file(structure, outdir / "structure.data", a.types.split(), a.frame, a.nrep)
    (outdir / "in.md").write_text(lammps_input(
        model=model, types=a.types.split(), ensemble=a.ensemble, temp=a.temp, press=a.press, pcouple=a.pcouple,
        dt_fs=a.dt, steps=a.steps, warmup=0, dump_every=a.dump_every, thermo=a.thermo, skin=a.skin, nevery=a.nevery, seed=a.seed))
    print(f"{natoms} atoms, {a.ensemble} at {a.temp} K, {a.steps} steps of {a.dt} fs -> {outdir}")
    r = run_lammps(outdir, kokkos=not a.no_kokkos)
    if r["status"] != 0:
        raise SystemExit(f"LAMMPS failed (exit {r['status']}); see {outdir / 'stdout.txt'}")
    print(f"done: {r['ms_per_step']} ms/step, {r['ns_per_day']} ns/day, drift {r['etot_drift_eV']} eV over the run")
    print(f"trajectory: {outdir / 'traj.lammpstrj'}   log: {outdir / 'log.lammps'}")
    if shutil.which("wslpath"):
        print("windows path:", subprocess.run(["wslpath", "-w", str(outdir)], capture_output=True, text=True).stdout.strip())


# ----------------------------------------------------------------------------- subcommand: bench
def cmd_bench(a):
    """Throughput matrix: every combination of model x nrep x ensemble x kokkos, `repeats` times,
    interleaved so slow GPU clock drift hits all variants equally. Appends to a CSV."""
    structure, types = unix_path(a.structure), a.types.split()
    bench = unix_path(a.out)
    bench.mkdir(parents=True, exist_ok=True)
    csv_path = bench / "results.csv"
    fields = ["label", "model", "nrep", "natoms", "ensemble", "kokkos", "repeat", "steps", "status", "ms_per_step", "us_per_atom_step", "ns_per_day", "etot_drift_eV", "wall_s"]
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for rep in range(1, a.repeats + 1):
            for nrep in a.nrep:
                for ensemble in a.ensembles:
                    for model in a.models:
                        for kokkos in a.kokkos:
                            model_p = unix_path(model)
                            label = f"{model_p.name.split('.')[0]}_{ensemble}_n{nrep}_kk{kokkos}_r{rep}"
                            d = bench / label
                            d.mkdir(exist_ok=True)
                            natoms = write_data_file(structure, d / "structure.data", types, a.frame, nrep)
                            (d / "in.md").write_text(lammps_input(
                                model=model_p, types=types, ensemble=ensemble, temp=a.temp, press=a.press, pcouple="iso",
                                dt_fs=a.dt, steps=a.steps, warmup=a.warmup, dump_every=0, thermo=50, skin=a.skin, nevery=a.nevery, seed=12345))
                            r = run_lammps(d, kokkos=bool(kokkos))
                            row = {"label": label, "model": model_p.name, "nrep": nrep, "natoms": natoms or r["natoms"], "ensemble": ensemble,
                                   "kokkos": kokkos, "repeat": rep, "steps": a.steps, **{k: r[k] for k in ("status", "ms_per_step", "us_per_atom_step", "ns_per_day", "etot_drift_eV", "wall_s")}}
                            w.writerow(row)
                            f.flush()
                            print(f"{label:40s} {r['ms_per_step']:8.3f} ms/step  {r['ns_per_day']:7.3f} ns/day  status {r['status']}", flush=True)
    summarize(csv_path)


def summarize(csv_path: Path):
    """Mean +- std of ms/step per (model, ensemble, nrep, kokkos) across repeats."""
    rows = [r for r in csv.DictReader(open(csv_path)) if r["status"] == "0"]
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["ensemble"], int(r["nrep"]), r["kokkos"], r["natoms"]), []).append(float(r["ms_per_step"]))
    print(f"\n{'model':22s} {'ens':4s} {'nrep':>4s} {'kk':>2s} {'atoms':>6s}   ms/step (mean +- sd, n)")
    for (m, e, n, kk, na), v in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{m:22s} {e:4s} {n:4d} {kk:>2s} {na:>6s}   {statistics.mean(v):8.3f} +- {sd:5.3f}  (n={len(v)})")


# ----------------------------------------------------------------------------- subcommand: compile
def cmd_compile(a):
    """nequip-compile a checkpoint into an AOTInductor .nequip.pt2 for pair_allegro.

    nequip-compile rebuilds the training datamodule for example data; the checkpoint may
    store a Windows dataset path (C:/Users/...), which is a *relative* path on Linux, so a
    symlink under a scratch working directory makes it resolve without touching the ckpt.
    """
    ckpt = unix_path(a.ckpt)
    out = unix_path(a.out) if a.out else ckpt.parent / f"{a.variant}.nequip.pt2"
    work = HOME / "pe206/compile_work"
    if a.dataset_path_in_ckpt:
        link = work / a.dataset_path_in_ckpt
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(unix_path(a.dataset))
    work.mkdir(parents=True, exist_ok=True)
    env = venv_env()
    if a.variant == "tf32":
        env["NEQUIP_TF32_MODEL_TOL"] = "0.05"
    sh(["nequip-compile", ckpt, out, "--device", "cuda", "--mode", "aotinductor", "--target", a.target, *VARIANTS[a.variant]], cwd=work, env=env)
    print("compiled:", out)


# ----------------------------------------------------------------------------- subcommand: build
def cmd_build(a):
    """Build LAMMPS develop + pair_nequip_allegro with AOTInductor loading and Kokkos CUDA.

    Two fixes learned the hard way: (1) CUDA 13.0's headers clash with Ubuntu 26.04's glibc,
    so nvcc comes from CUDA_HOME (13.3); (2) current LAMMPS develop's style registry drops the
    typedefs that pair_nequip_allegro keeps inside `#ifdef COMPUTE_CLASS`, so they are moved
    into the header's namespace before cmake runs.
    """
    import torch  # only needed here, for the libtorch cmake prefix

    if not LAMMPS_SRC.exists():
        sh(["bash", "-c", f"wget -qO /tmp/lammps.tar.gz https://github.com/lammps/lammps/archive/refs/heads/develop.tar.gz && mkdir -p {LAMMPS_SRC} && tar -xzf /tmp/lammps.tar.gz -C {LAMMPS_SRC} --strip-components=1"])
    if not PLUGIN_SRC.exists():
        sh(["git", "clone", "--depth=1", "https://github.com/mir-group/pair_nequip_allegro", PLUGIN_SRC])
        sh(["./patch_lammps.sh", LAMMPS_SRC], cwd=PLUGIN_SRC)
    fix_compute_header(LAMMPS_SRC / "src/compute_nequip_allegro.h")
    build = LAMMPS_SRC / "build"
    build.mkdir(exist_ok=True)
    sh(["cmake", "../cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}", "-DNEQUIP_AOT_COMPILE=ON", "-DMKL_INCLUDE_DIR=/tmp",
        f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}", f"-DCMAKE_CUDA_COMPILER={CUDA_HOME / 'bin/nvcc'}",
        "-DBUILD_MPI=ON", "-DBUILD_OMP=OFF", "-DPKG_KOKKOS=ON", "-DKokkos_ENABLE_SERIAL=ON", "-DKokkos_ENABLE_CUDA=ON",
        f"-DKokkos_ARCH_{a.arch}=ON"], cwd=build, env=venv_env())
    sh(["ninja", f"-j{a.jobs}"], cwd=build, env=venv_env())
    print("built:", build / "lmp")


def fix_compute_header(header: Path):
    s = header.read_text()
    typedefs = ("typedef ComputeNequIPAllegro<0,0> ComputeAllegro;\ntypedef ComputeNequIPAllegro<0,1> ComputeAllegroPerAtom;\n"
                "typedef ComputeNequIPAllegro<1,0> ComputeNequIP;\ntypedef ComputeNequIPAllegro<1,1> ComputeNequIPPerAtom;\n")
    marker = "  std::string compute_name;\n};\n\n}\n"
    if typedefs in s and s.index(typedefs) > s.index("namespace LAMMPS_NS"):
        return  # already patched
    s = s.replace(typedefs + "\n", "", 1).replace(marker, "  std::string compute_name;\n};\n\n" + typedefs + "\n}\n", 1)
    header.write_text(s)
    print("patched", header)


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--structure", required=True, help="any ASE-readable file (Windows paths accepted)")
    common.add_argument("--frame", type=int, default=0)
    common.add_argument("--types", default="C H", help="model type names in pair_coeff order")
    common.add_argument("--temp", type=float, default=300.0, help="K")
    common.add_argument("--press", type=float, default=1.0, help="bar (npt)")
    common.add_argument("--dt", type=float, default=0.5, help="fs")
    common.add_argument("--skin", type=float, default=1.0, help="neighbor skin, Angstrom")
    common.add_argument("--nevery", type=int, default=1, help="neighbor check interval")

    r = sub.add_parser("run", parents=[common], help="one MD run")
    r.add_argument("--model", default=str(MODEL), help="compiled .nequip.pt2")
    r.add_argument("--ensemble", choices=["nve", "nvt", "npt"], default="nvt")
    r.add_argument("--pcouple", choices=["iso", "aniso", "tri"], default="iso")
    r.add_argument("--nrep", type=int, default=1, help="replicate n x n x n")
    r.add_argument("--steps", type=int, default=10000)
    r.add_argument("--dump-every", type=int, default=100, help="0 disables the trajectory dump")
    r.add_argument("--thermo", type=int, default=50)
    r.add_argument("--seed", type=int, default=12345)
    r.add_argument("--out", help="output directory (default ~/pe206/md/<timestamp>)")
    r.add_argument("--no-kokkos", action="store_true")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("bench", parents=[common], help="throughput matrix")
    b.add_argument("--models", nargs="+", default=[str(MODEL)])
    b.add_argument("--nrep", nargs="+", type=int, default=[1, 3])
    b.add_argument("--ensembles", nargs="+", default=["nve"], choices=["nve", "nvt", "npt"])
    b.add_argument("--kokkos", nargs="+", type=int, default=[1], choices=[0, 1])
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--steps", type=int, default=200)
    b.add_argument("--warmup", type=int, default=20)
    b.add_argument("--out", default=str(HOME / "pe206/bench_py"))
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("compile", help="checkpoint -> .nequip.pt2")
    c.add_argument("ckpt")
    c.add_argument("--variant", choices=list(VARIANTS), default="base")
    c.add_argument("--target", default="pair_allegro", help="pair_allegro (LAMMPS) or ase")
    c.add_argument("--out", help="output path (default <ckpt dir>/<variant>.nequip.pt2)")
    c.add_argument("--dataset", default=str(DATASET), help="dataset file for example data")
    c.add_argument("--dataset-path-in-ckpt", default="C:/Users/sumedh/Downloads/pe_206_dft.xyz",
                   help="dataset path as stored in the checkpoint config, made resolvable via symlink")
    c.set_defaults(func=cmd_compile)

    d = sub.add_parser("build", help="build LAMMPS with pair_allegro + Kokkos CUDA")
    d.add_argument("--arch", default="ADA89", help="Kokkos GPU arch (ADA89 = RTX 40xx)")
    d.add_argument("--jobs", type=int, default=12)
    d.set_defaults(func=cmd_build)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
