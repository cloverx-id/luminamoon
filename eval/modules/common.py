import math
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

import numpy as np

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


def tanh_pade_f32(x_in: float) -> float:
    val = float(x_in)
    if math.isnan(val):
        return float("nan")
    x = max(min(val, 7.9988117), -7.9988117)
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


def _quantize_bf16_rne(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    bf16_u32 = (u32 + rounding_bias) & np.uint32(0xFFFF0000)
    return bf16_u32.view(np.float32).astype(np.float64)


def _step_optimizer(opt_name, w, g, m, v, t, lr, wd=0.0, beta1=0.9, beta2=0.999, lam=0.3, tau=0.8, eps=1e-8):
    if opt_name == "Legacy_BCLM":
        w1 = beta1**2
        w2 = 1.0 - w1
        clm_m = w1 * m + w2 * g
        clm_env = w1 * np.abs(m) + w2 * np.abs(g) + eps
        u = clm_m / clm_env
        m = beta1 * m + (1.0 - beta1) * g
        w = w * (1.0 - lr * wd) - lr * u

    elif opt_name == "LuminaV_Buffer1_TDCLM":
        w1 = beta1**2
        w2 = 1.0 - w1
        clm_m = w1 * m + w2 * g
        clm_defect = lam * np.abs(g - m)
        clm_env = w1 * np.abs(m) + w2 * np.abs(g) + clm_defect + eps
        u = clm_m / clm_env
        m = beta1 * m + (1.0 - beta1) * g
        w = w * (1.0 - lr * wd) - lr * u

    elif opt_name == "LuminaV_Buffer2_Dual":
        m_new = beta1 * m + (1.0 - beta1) * g
        clm_m = beta1 * m_new + (1.0 - beta1) * g
        diff = g - m_new
        v = beta2 * v + (1.0 - beta2) * (diff**2)

        bc1 = 1.0 - beta1 ** (t + 1)
        bc2 = max(1.0 - beta2**t, 1e-15)
        c1 = (bc1 * tau) / np.sqrt(bc2)
        c2 = eps * bc1 * tau
        sigma = np.sqrt(np.maximum(v, 0.0)) * c1 + c2
        u = np.tanh(clm_m / np.maximum(sigma, 1e-12))

        m = m_new
        w = w * (1.0 - lr * wd) - lr * u

    elif opt_name == "AdamW_FP32_Master":
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        u = m_hat / (np.sqrt(v_hat) + eps)
        w = w * (1.0 - lr * wd) - lr * u

    elif opt_name == "AdamW_Native_BF16":
        g_bf16 = _quantize_bf16_rne(g)
        m = _quantize_bf16_rne(beta1 * m + (1.0 - beta1) * g_bf16)
        v = _quantize_bf16_rne(beta2 * v + (1.0 - beta2) * (g_bf16**2))
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        u = m_hat / (np.sqrt(v_hat) + 1e-6)
        w = _quantize_bf16_rne(w * (1.0 - lr * wd) - lr * u)

    return w, m, v
