from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from ._version import __version__, get_version

_SEP = "___"


def _get_math_version() -> str:
    ver_file = Path(__file__).resolve().parent / "optimizer" / "version.py"
    if ver_file.is_file():
        for line in ver_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("MATH_VERSION") and "=" in line:
                return line.split("=", 1)[1].strip().strip("\"'")
    from .optimizer.version import MATH_VERSION

    return MATH_VERSION

VALID_TOPICS = ("all", "optimizer", "luminav", "nn", "data", "migration", "hardware")

_CLI_HEADER = """\
LUMINAMOON v{version} — API, PARAMETER & CLI REFERENCE (LuminaV MATH_VERSION {math_version})
___

CLI Usage (Terminal; prefix with `!` in Jupyter / Google Colab):
  luminamoon --help
  luminamoon --topic=optimizer --output=luminav_help.txt
  luminamoon --topic=nn --no-hardware
  luminamoon --topic=data --no-hardware
  luminamoon --output=output.txt --quiet --no-hardware
  luminamoon --version
  luminamoon doctor [--benchmark] [--steps 7] [--measure-last 2]
  luminamoon setup triton [--dry-run] [--yes]
  python -m luminamoon --help --output=output.txt

CLI Flags & Subcommands:
  -h, --help          Show API, parameter, and default-value documentation.
  -t, --topic TOPIC   Section: all, optimizer, luminav, nn, data, migration, hardware.
  -o, --output FILE   Write documentation output to a UTF-8 file.
  -q, --quiet         Suppress stdout when --output is specified.
  --no-hardware       Skip live hardware capability probing.
  -V, --version       Print package version and LuminaV MATH_VERSION.
  doctor              Run live micro-probes (and optional --benchmark) across all backends.
  setup TARGET        Opt-in installer helper for 'triton', 'gpu', or 'cubecl' (OS-aware).

Python Interactive Usage:
  import luminamoon as lm
  lm.help()                                      # Print full documentation
  lm.help("optimizer", output="output.txt")      # Print & save optimizer docs
  lm.LuminaV.help(output="luminav_params.txt")   # Optimizer-focused help
"""

_OPTIMIZER_SECTION = """\
___
1. OPTIMIZER MODULE (`luminamoon.LuminaV` / `luminamoon.optimizer`)

Import:
  from luminamoon import AutoTune, LuminaV
  from luminamoon.optimizer import (
      AutoTune, LuminaV, BufferMode, BoundType, MomentumType, VarianceType,
      MasterWeights, ExecutionMode, MATH_VERSION, get_math_version,
  )

Constructor Signature:
  LuminaV(
      params,
      lr=3e-4,
      betas=(0.9, 0.999),
      eps=1e-8,
      weight_decay=8e-2,
      cautious_weight_decay=False,
      variance_type="standard",
      variance_temper=3.0,
      momentum_type="standard",
      tau=0.8,
      lam=0.3,
      cautious=True,
      cautious_clamp_min=0.5,
      buffer=2,
      stochastic_rounding=True,
      bound=True,
      bound_type="coordinate",
      bound_ratio=0.05,
      bound_floor=1e-2,
      master_weights="none",
      execution="auto",
      fused_single_pass=True,
      fuse_tensors=False,
      bucket_cap_mb=500.0,
      prewarm_jit=True,
      ema_decay="auto",
      fsdp_sync=False,
      seed=1337,
      tuning=None,
  )

Parameter Reference:
  Parameter               Type                  Default        Valid Domain / Choices                           Description
  params                  Iterable              (required)     Non-empty Tensor/ndarray list or param_groups    Model parameters (torch.Tensor, np.ndarray, or list of group dicts).
  lr                      float                 3.0e-4         Finite, >= 0.0                                   Learning rate (eta).
  betas                   Tuple[float, float]   (0.9, 0.999)   beta1, beta2 in [0.0, 1.0)                       Momentum (beta1) and variance (beta2) exponential decay rates.
  eps                     float                 1.0e-8         Finite, > 0.0                                    Denominator numerical stability constant (epsilon).
  weight_decay            float                 8.0e-2         Finite, >= 0.0                                   Decoupled weight decay coefficient (wd).
  cautious_weight_decay   bool                  False          True | False                                     Apply weight decay only where step_val * param >= 0.0 (CWD).
  variance_type           VarianceType          "standard"     "standard" | "rtv"                               2-buffer variance update: standard L2 or Rational-Tempered (RTV).
  variance_temper         float                 3.0            Finite, > 0.0                                    RTV tempering ratio vs max(sigma_prev, |m_prev|) when variance_type="rtv".
  momentum_type           MomentumType          "standard"     "standard" | "rpm"                               Momentum update: standard EMA or pairwise Cayley rotation (RPM).
  tau                     float                 0.8            Finite, > 0.0                                    CLM temperature scale factor (active when buffer=2).
  lam                     float                 0.3            Finite, >= 0.0                                   Triangle-defect penalty weight lambda (active on buffer=1, "t1", "t2").
  cautious                bool                  True           True | False                                     Enable sign-aligned cautious masking and rescaling.
  cautious_clamp_min      float                 0.5            (0.0, 1.0]                                       Lower clamp bound for mean mask ratio m_bar (caps cautious_scale <= 1/c).
  buffer                  BufferMode            2              2 | 1 | "t1" | "t2" (case-insensitive strings)   2=Dual CLM, 1=Single TD-CLM, "t1"=1-Buf Triangle, "t2"=2-Buf Triangle.
  stochastic_rounding     bool                  True           True | False                                     Enable MurmurHash3 + 3D Weyl-Kronecker SR for BF16/FP16.
  bound                   bool                  True           True | False                                     Enable trust-region step bounding.
  bound_type              BoundType             "coordinate"   "coordinate" | "radial"                          Element-wise coordinate bound or tensor-norm radial bound.
  bound_ratio             float                 0.05           Finite, > 0.0                                    Maximum step ratio relative to weight magnitude/norm (rho).
  bound_floor             float                 1.0e-2         Finite, > 0.0                                    Coordinate bound floor radius f: rho*(|p|+f) or rho*sqrt(p^2+f^2).
  master_weights          MasterWeights         "none"         "none" | "semi" | "full"                         FP32 master copy mode for BF16/FP16 parameters (see matrix below).
  execution               ExecutionMode         "auto"         "auto"|"triton"|"cubecl"|"simd"|"eager"|"scalar" Backend route ("auto" benchmarks tail steps to select fastest backend).
  fused_single_pass       bool                  True           True | False                                     1-pass kernel + post-step EMA on step > 1; False uses 2-pass mode.
  fuse_tensors            bool                  False          True | False                                     Coalesce homogeneous PyTorch tensors into 1D contiguous buckets.
  bucket_cap_mb           float                 500.0          Finite, > 0.0                                    Maximum memory size (MB) per contiguous bucket when fuse_tensors=True.
  prewarm_jit             bool                  True           True | False                                     Pre-compile Triton GPU kernels on 64-elem dummy buffers at init.
  ema_decay               float | str | None    "auto"         "auto" | None | float in (0.0, 1.0)              Post-step EMA decay ("auto" -> 0.95 for 1-buffer, 0.80 for 2-buffer).
  fsdp_sync               bool                  False          True | False                                     All-reduce scratch statistics across FSDP ranks via torch.distributed.
  seed                    int                   1337           int in [0, 4294967295]                           Base deterministic seed for 3D Weyl-Kronecker PRNG.
  tuning                  AutoTune | dict|None  None           AutoTune(model[, loss_fn=...]) or override dict  Overrides defaults (lr, wd, buffer, etc.); saved in state_dict()['luminav_tuning'].

Master Weights Matrix (for BF16 / FP16 parameters):
  - "none": No FP32 master copy; states stored in 16-bit (FP16 uses In-Container Bitwise-BF16); SR/RNE on param & state.
  - "semi": FP32 master_param copy + 16-bit states (FP16 uses In-Container Bitwise-BF16); RNE on param, SR/RNE on state.
  - "full": FP32 master_param copy + FP32 states.

LuminaV Methods & Attributes:
  - opt.step(closure=None, grads=None) -> Optional[float]
  - opt.zero_grad(set_to_none=True) -> None
  - opt.add_param_group(param_group: dict) -> None
  - opt.state_dict() -> dict / opt.load_state_dict(state_dict: dict) -> None
  - opt.last_backend -> Optional[str] / opt.resolved_execution -> Optional[str] / opt.tuning_info -> Optional[dict]
  - AutoTune(model, *, loss_fn=None, optimizer_cls=None, tune_params=None) -> tune.for_optimizer(cls) / tune.build(cls, params)
  - LuminaV.detect_hardware() -> dict / LuminaV.probe_backends(*, force=False) -> dict
  - LuminaV.benchmark_backends(params, *, steps=7, measure_last=2, max_memory_mb=128.0, use_cache=False) -> dict
  - LuminaV.get_math_version() -> str  (returns "2.0.0"; also available as LuminaV.MATH_VERSION)
  - LuminaV.help(topic="optimizer", *, output=None, print_out=True, include_hardware=True) -> str
"""

from ._cli_sections import _DATA_SECTION, _MIGRATION_SECTION, _NN_SECTION


def _format_hardware_section() -> str:
    lines = [
        _SEP,
        "5. DETECTED HARDWARE & RUNTIME CAPABILITIES",
        "",
    ]
    try:
        from .optimizer import LuminaV

        hw: Dict[str, Any] = LuminaV.detect_hardware()
        for k, v in hw.items():
            lines.append(f"  {k:<26}: {v}")
    except Exception as exc:
        lines.append(f"  optimizer hardware probe  : unavailable ({exc})")
    try:
        from .data import data_cpu_isa

        lines.append(f"  {'data_cpu_isa':<26}: {data_cpu_isa()}")
    except Exception as exc:
        lines.append(f"  data_cpu_isa              : unavailable ({exc})")
    return "\n".join(lines) + "\n"


def format_help(topic: str = "all", *, include_hardware: bool = True) -> str:
    norm = topic.strip().lower()
    if norm not in VALID_TOPICS:
        valid_str = ", ".join(f"'{t}'" for t in VALID_TOPICS)
        matches = difflib.get_close_matches(norm, VALID_TOPICS, n=1, cutoff=0.5)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        raise ValueError(f"Invalid help topic: '{topic}'. Expected one of {valid_str}.{hint}")

    parts: List[str] = [_CLI_HEADER.format(version=get_version(), math_version=_get_math_version())]
    if norm in ("all", "optimizer", "luminav"):
        parts.append(_OPTIMIZER_SECTION)
    if norm in ("all", "nn"):
        parts.append(_NN_SECTION)
    if norm in ("all", "data"):
        parts.append(_DATA_SECTION)
    if norm in ("all", "migration"):
        parts.append(_MIGRATION_SECTION)
    if norm == "hardware" or (norm == "all" and include_hardware):
        parts.append(_format_hardware_section())
    return "\n".join(parts)


def help(
    topic: str = "all",
    *,
    output: Optional[Union[str, Path]] = None,
    print_out: bool = True,
    include_hardware: bool = True,
) -> str:
    """Display and optionally save LuminaMoon API, parameter, and default-value documentation."""
    text = format_help(topic=topic, include_hardware=include_hardware)
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    if print_out:
        print(text)
    return text


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="luminamoon",
        add_help=False,
        description="LuminaMoon v2.0.0 command-line API reference and hardware inspector.",
    )
    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show full API, parameter, and default-value documentation.",
    )
    parser.add_argument(
        "-t", "--topic",
        default="all",
        choices=VALID_TOPICS,
        help="Documentation section to display (default: all).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="FILE",
        help="Write the documentation output to a UTF-8 file (e.g., --output=output.txt).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress stdout printing when --output is specified.",
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="Skip live hardware capability probing.",
    )
    parser.add_argument(
        "-V", "--version",
        action="store_true",
        help="Print package version and LuminaV MATH_VERSION.",
    )
    return parser


def _run_doctor_cli(rest: Sequence[str]) -> int:
    from ._cli_backends import format_doctor_report

    p = argparse.ArgumentParser(prog="luminamoon doctor")
    p.add_argument("--benchmark", action="store_true", help="Run backend benchmark (timing only the last few steps).")
    p.add_argument("--steps", type=int, default=7, help="Total benchmark steps (default: 7).")
    p.add_argument("--measure-last", type=int, default=2, help="Number of final steps timed (default: 2).")
    ns = p.parse_args(list(rest))
    print(format_doctor_report(force=True, benchmark=ns.benchmark, steps=ns.steps, measure_last=ns.measure_last))
    return 0


def _run_setup_cli(rest: Sequence[str]) -> int:
    from ._cli_backends import run_setup

    p = argparse.ArgumentParser(prog="luminamoon setup")
    p.add_argument("target", choices=("triton", "gpu", "cubecl"), help="Backend target to install or inspect.")
    p.add_argument("-y", "--yes", action="store_true", help="Skip interactive confirmation prompt.")
    p.add_argument("--dry-run", action="store_true", help="Print resolved command without modifying packages.")
    ns = p.parse_args(list(rest))
    return run_setup(ns.target, yes=ns.yes, dry_run=ns.dry_run)


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in ("doctor", "--doctor"):
        return _run_doctor_cli(raw[1:])
    if raw and raw[0] == "setup":
        return _run_setup_cli(raw[1:])

    parser = _build_arg_parser()
    args = parser.parse_args(raw)

    if args.version and not args.help and args.output is None:
        print(f"luminamoon {__version__} (LuminaV MATH_VERSION {_get_math_version()})")
        return 0

    should_print = not (args.quiet and args.output is not None)
    help(
        topic=args.topic,
        output=args.output,
        print_out=should_print,
        include_hardware=not args.no_hardware,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
