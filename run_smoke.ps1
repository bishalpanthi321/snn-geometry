# Whole pipeline at toy scale (2 epochs, 500 test images, 2 corruptions x 2 severities).
# Outputs go to smoke_run/ and never touch the real results.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest -q
& $py train.py --model ann --smoke
& $py train.py --model snn --smoke --micro-batch 32
& $py generate_corruptions.py --smoke
& $py extract_features.py --model ann --smoke
& $py extract_features.py --model snn --smoke
& $py extract_features.py --model snn --smoke --T 8
& $py analyze.py --smoke
& $py make_figures.py --smoke
