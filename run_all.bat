@echo off
rem Runs ALL CGAR experiments (UCI + OULAD [+ xAPI if present]) on Windows. Double-click.
rem FULL run: tuned models, beam search, subgroup report, early-warning cutoffs, plan export.
rem It can take SEVERAL HOURS (OULAD is large). Keep the laptop plugged in and awake.
cd /d "%~dp0code"

if not exist "..\data\oulad\studentInfo.csv" (
  echo OULAD files not found in data\oulad. Extract the OULAD zip there first.
  pause
  exit /b 1
)

if not exist "..\.venv" python -m venv ..\.venv
call ..\.venv\Scripts\activate.bat
pip install numpy pandas scipy scikit-learn matplotlib xgboost

set XAPI=
if exist "..\data\xapi\xAPI-Edu-Data.csv" set XAPI=--xapi-csv ..\data\xapi\xAPI-Edu-Data.csv

python run_experiments.py --uci-dir ..\data\uci --oulad-dir ..\data\oulad %XAPI% --out ..\paper\results --n-seeds 20 --oulad-seeds 10 --tune --beam 3 --cutoffs 30,60,90 --cutoff-seeds 3 --export-plans 30
if errorlevel 1 (
  echo The run failed. Copy the error above and send it to Claude.
  pause
  exit /b 1
)

echo.
echo Finished. Tables and figures are in the paper\results folder.
pause
