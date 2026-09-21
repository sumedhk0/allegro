@echo off
rem One-command LAMMPS MD from Windows. Forwards to lammps_md.sh inside WSL Ubuntu.
rem Example:
rem   lammps_md.cmd --structure C:\Users\sumedh\Downloads\pe_206_dft.xyz --frame 500 --ensemble npt --temp 300 --steps 20000
rem Options: see lammps_md.sh (--model --nrep --ensemble --temp --press --steps --dt --dump-every --out ...)
rem Backslashes do not survive the trip into WSL, so Windows paths are passed with forward slashes;
rem lammps_md.sh converts anything that starts with a drive letter via wslpath.
setlocal EnableDelayedExpansion
for /f "usebackq delims=" %%p in (`wsl -d Ubuntu -- wslpath -u "%~dp0lammps_md.sh"`) do set "SCRIPT=%%p"
set "ARGS=%*"
if defined ARGS set "ARGS=!ARGS:\=/!"
wsl -d Ubuntu -- bash "%SCRIPT%" !ARGS!
endlocal
