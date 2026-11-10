@echo off
rem Runs the CGAR experiments on the UCI Student Performance data (Windows).
rem Double-click this file. It installs the needed packages into a local .venv folder.
cd /d "%~dp0code"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from python.org and tick "Add python.exe to PATH", then run this file again.
  pause
  exit /b 1
)

if not exist "..\.venv" python -m venv ..\.venv
call ..\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib xgboost
if errorlevel 1 (
  echo Package installation failed. Copy the error above and send it to Claude.
  pause
  exit /b 1
)

python run_experiments.py --uci-dir ..\data\uci --out ..\paper\results --n-seeds 10
if errorlevel 1 (
  echo The run failed. Copy the error above and send it to Claude.
  pause
  exit /b 1
)

echo.
echo Finished. Tables and figures are in the paper\results folder.
pause
