@echo off
REM Build custom NMODL catalogue (ampasyn, nmdasyn, hhnotemp, ...).
REM Rebuild after changing files in toric_spines_sim\mechanisms\my_catalogue\
cd /d "%~dp0..\toric_spines_sim\mechanisms"
uv run arbor-build-catalogue custom my_catalogue
