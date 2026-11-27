import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
for p in (str(EXP), str(EXP.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
