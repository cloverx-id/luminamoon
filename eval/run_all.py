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
        run_benchmarks_2d,
        run_stochastic_scenarios,
        run_math_verification,
        run_defect_factor_sweep,
        run_hyperparameter_sweeps,
        run_ablation_sweep,
        run_parity_audit,
        run_parameter_audit,
        run_fair_lr_grid,
        run_checkpoint_suite,
    )
except ImportError:
    from modules import (
        run_smt_suite,
        run_benchmarks_2d,
        run_stochastic_scenarios,
        run_math_verification,
        run_defect_factor_sweep,
        run_hyperparameter_sweeps,
        run_ablation_sweep,
        run_parity_audit,
        run_parameter_audit,
        run_fair_lr_grid,
        run_checkpoint_suite,
    )

def run_all():
    print("================================================================================")
    print("          LuminaMoon v1.0.8.post1 (LuminaV v1.5.5) Complete LSO-Eval Test Suite              ")
    print("================================================================================")

    run_smt_suite()
    run_benchmarks_2d()
    run_stochastic_scenarios()
    run_math_verification()
    run_defect_factor_sweep()
    run_hyperparameter_sweeps()
    run_ablation_sweep()
    run_parity_audit()
    run_parameter_audit()
    run_fair_lr_grid()
    run_checkpoint_suite()

    print("\nAll evaluation suites completed successfully.")

if __name__ == "__main__":
    run_all()
