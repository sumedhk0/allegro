@echo off
rem Windows entry point for examples\lammps_md.py (runs inside WSL Ubuntu with the allegro venv).
rem   lammps_md.cmd run --structure C:\Users\sumedh\Downloads\pe_206_dft.xyz --frame 500 --ensemble npt --steps 20000
rem   lammps_md.cmd bench --structure C:\Users\sumedh\Downloads\pe_206_dft.xyz --frame 500 --ensembles nve npt
rem Backslashes do not survive the trip into WSL, so Windows paths are passed with forward slashes;
rem the Python script converts anything that starts with a drive letter via wslpath.
setlocal EnableDelayedExpansion
for /f "usebackq delims=" %%p in (`wsl -d Ubuntu -- wslpath -u "%~dp0..\..\examples\lammps_md.py"`) do set "SCRIPT=%%p"
set "ARGS=%*"
if defined ARGS set "ARGS=!ARGS:\=/!"
wsl -d Ubuntu -- bash -c "source ~/.venvs/allegro/bin/activate && python '%SCRIPT%' !ARGS!"
endlocal
