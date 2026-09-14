#!/usr/bin/env bash
# Run one LAMMPS benchmark case and append a CSV row.
# Usage: run_bench.sh <label> <model.pt2> <nrep> <kokkos:0|1> [nsteps] [skin] [nevery] [extra lmp args...]
# Parses the timed `run` block: Loop time -> ms/step, plus LAMMPS' own ns/day.
set -uo pipefail
LABEL=$1; MODEL=$2; NREP=$3; KK=$4; NSTEPS=${5:-200}; SKIN=${6:-1.0}; NEVERY=${7:-1}; shift 7 2>/dev/null || shift $#
EXTRA="$*"
BENCH=${BENCH_DIR:-$HOME/pe206/bench}
LMP=${LMP:-$HOME/lammps/build/lmp}
CSV=${CSV:-$BENCH/results.csv}
mkdir -p "$BENCH/logs"
LOG="$BENCH/logs/${LABEL}_n${NREP}_kk${KK}.log"

if [ "$KK" = "1" ]; then
  KKARGS="-k on g 1 -sf kk -pk kokkos newton on neigh half"
else
  KKARGS=""
fi

cd "$BENCH"
NP=${NP:-1}
/usr/bin/time -f "%e %M" -o "$LOG.time" \
  mpirun -np "$NP" "$LMP" -in in.pe206 -log "$LOG" -screen none \
    -var model "$MODEL" -var nrep "$NREP" -var nsteps "$NSTEPS" -var skin "$SKIN" -var nevery "$NEVERY" \
    $KKARGS $EXTRA
STATUS=$?

# last Loop time line = timed run; it also carries the post-replicate atom count
LOOP=$(grep "Loop time" "$LOG" | tail -1)
SECS=$(echo "$LOOP" | awk '{print $4}')
STEPS=$(echo "$LOOP" | awk '{print $9}')
NATOMS=$(echo "$LOOP" | awk '{print $12}')
NSDAY=$(grep -A2 "Loop time" "$LOG" | tail -3 | grep -m1 "Performance:" | awk '{print $2}')
MSSTEP=$(awk -v s="$SECS" -v n="$STEPS" 'BEGIN{ if (n>0) printf "%.3f", 1000*s/n; else print "nan"}')
USPERATOMSTEP=$(awk -v s="$SECS" -v n="$STEPS" -v a="$NATOMS" 'BEGIN{ if (n>0 && a>0) printf "%.3f", 1e6*s/(n*a); else print "nan"}')
# /usr/bin/time prepends "Command exited with non-zero status N" on failure; take the last line
WALL=$(tail -1 "$LOG.time" | awk '{print $1}'); RSS=$(tail -1 "$LOG.time" | awk '{print $2}')
# NVE energy drift over the timed run (eV), from the thermo table of the last run block
DRIFT=$(awk '/^ *Step /{hdr=1; c=0; next} hdr && /^ *[0-9]+ /{c++; e[c]=$5} /^Loop time/{hdr=0; if (c>1) d=e[c]-e[1]} END{if (d!="") printf "%.5f", d; else print "nan"}' "$LOG")
[ -f "$CSV" ] || echo "label,model,nrep,natoms,kokkos,np,nsteps,skin,nevery,status,ms_per_step,us_per_atom_step,ns_per_day,etot_drift_eV,wall_s,max_rss_kb" > "$CSV"
echo "$LABEL,$(basename "$MODEL"),$NREP,$NATOMS,$KK,$NP,$NSTEPS,$SKIN,$NEVERY,$STATUS,$MSSTEP,$USPERATOMSTEP,$NSDAY,$DRIFT,$WALL,$RSS" | tee -a "$CSV"
