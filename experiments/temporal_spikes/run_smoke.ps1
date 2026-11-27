# Whole temporal-spike pipeline at toy scale with RANDOM weights (no checkpoint needed).
# Outputs go to cache/smoke, results/smoke, logs/smoke and never touch pilot/full results.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "..\..\.venv\Scripts\python.exe"
& $py -m pytest tests -q
& $py extract.py --stage smoke --T 4 --random-init
& $py extract.py --stage smoke --T 8 --random-init
& $py run_interventions.py --stage smoke --T 4 --random-init
& $py analyze_capacity.py --stage smoke
& $py analyze_interventions.py --stage smoke
& $py analyze_detection.py --stage smoke
& $py make_figures.py --stage smoke
