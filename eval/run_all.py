import os
import sys
from pathlib import Path

_base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = _base.parent if _base.name == "eval" else _base
for _p in [str(_root), str(_base), str(_root / "python"), str(_root / "optimizer" / "reference")]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.append(_p)

try:
    from eval.modules import (
        run_smt_suite,
        run_math_verification,
        run_benchmarks_2d,
        run_stochastic_scenarios,
        run_defect_factor_sweep,
        run_hyperparameter_sweeps,
        run_ablation_sweep,
        run_parity_audit,
        run_checkpoint_suite,
    )
except ImportError:
    from modules import (
        run_smt_suite,
        run_math_verification,
        run_benchmarks_2d,
        run_stochastic_scenarios,
        run_defect_factor_sweep,
        run_hyperparameter_sweeps,
        run_ablation_sweep,
        run_parity_audit,
        run_checkpoint_suite,
    )


def run_all() -> bool:
    smt_ok = run_smt_suite()
    math_ok = run_math_verification()
    stage1_ok = bool(smt_ok and math_ok)
    print(f"\nLSO-Eval Stage I Overall Status     : {'PASSED' if stage1_ok else 'FAILED'}\n")

    s2_1a = run_benchmarks_2d()
    s2_1b = run_stochastic_scenarios()
    s2_2a = run_defect_factor_sweep()
    s2_2b = run_hyperparameter_sweeps()
    s2_3a = run_ablation_sweep()
    s2_3b = run_parity_audit()
    s2_4 = run_checkpoint_suite()
    stage2_ok = all([s2_1a, s2_1b, s2_2a, s2_2b, s2_3a, s2_3b, s2_4])
    print(f"\nLSO-Eval Stage II Overall Status    : {'PASSED' if stage2_ok else 'FAILED'}")

    return stage1_ok and stage2_ok


if __name__ == "__main__":
    if not run_all():
        sys.exit(1)
