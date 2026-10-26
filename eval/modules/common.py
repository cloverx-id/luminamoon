import os
import subprocess
import sys
from pathlib import Path

for _pkg in ["z3", "sympy", "scipy", "torch", "numpy"]:
    try:
        __import__(_pkg)
    except ImportError:
        _name = "z3-solver" if _pkg == "z3" else _pkg
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _name])

_base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = _base.parent.parent if _base.name == "modules" else (_base.parent if _base.name == "eval" else _base)

for _p in [str(_root / "python"), str(_root / "optimizer" / "reference"), str(_root)]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.append(_p)

try:
    from luminamoon import LuminaV
except ImportError:
    try:
        from luminamoon.optimizer import LuminaV
    except ImportError:
        from luminav import LuminaV

# Pade [13/6] rational polynomial approximation matching LuminaV math core
def tanh_pade_f32(x_in: float) -> float:
    x = max(min(float(x_in), 7.9988117), -7.9988117)
    x2 = x * x
    p = -2.7607684e-16 * x2 + 2.000188e-13
    q = 1.1982584e-6 * x2 + 1.1853471e-4
    p = p * x2 - 8.604672e-11
    q = q * x2 + 2.2684346e-3
    p = p * x2 + 5.122297e-8
    q = q * x2 + 4.8935246e-3
    p = p * x2 + 1.4857224e-5
    p = p * x2 + 6.3726193e-4
    p = p * x2 + 4.8935246e-3
    return (x * p) / q
