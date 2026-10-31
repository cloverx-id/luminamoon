import os
import sys
from pathlib import Path

_base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = _base.parent if _base.name == "eval" else _base
for _path in [str(_root), str(_base), str(_root / "python"), str(_root / "optimizer" / "reference")]:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.append(_path)

try:
    from eval.run_all import run_all
except ImportError:
    from run_all import run_all


if __name__ == "__main__":
    if not run_all():
        sys.exit(1)
