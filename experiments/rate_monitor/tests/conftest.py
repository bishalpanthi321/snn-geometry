import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
for p in (str(EXP), str(EXP.parent.parent), str(EXP.parent / "temporal_spikes")):
    if p not in sys.path:
        sys.path.insert(0, p)
