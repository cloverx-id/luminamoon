from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from ._version import __version__, get_version

VALID_TOPICS = ("all", "optimizer", "luminav", "nn", "data", "migration", "hardware")

_CLI_HEADER = """\
LUMINAMOON v{version} — API, PARAMETER & CLI REFERENCE (LuminaV MATH_VERSION 2.0.0)
___

CLI Usage (Terminal / Google Colab / Jupyter):
  !luminamoon --help
  !luminamoon --output=output.txt
  !luminamoon --topic=optimizer --output=luminav_help.txt
  !python -m luminamoon --help --output=output.txt

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
  from luminamoon import LuminaV
  from luminamoon.optimizer import (
      LuminaV, BufferMode, BoundType, VarianceType, MasterWeights,
      ExecutionMode, MATH_VERSION, get_math_version,
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
  tau                     float                 0.8            Finite, > 0.0                                    CLM temperature scale factor (active when buffer=2).
  lam                     float                 0.3            Finite, >= 0.0                                   Triangle-defect penalty weight lambda (active on buffer=1, "t1", "t2").
  cautious                bool                  True           True | False                                     Enable sign-aligned cautious masking and rescaling.
  cautious_clamp_min      float                 0.5            (0.0, 1.0]                                       Lower clamp bound for mean mask ratio m_bar (caps cautious_scale <= 1/c).
  buffer                  BufferMode            2              2 | 1 | "t1" | "t2" (case-insensitive strings)   2=Dual CLM, 1=Single TD-CLM, "t1"=1-Buf Triangle, "t2"=2-Buf Triangle.
  stochastic_rounding     bool                  True           True | False                                     Enable 3D Weyl-Kronecker quasi-Monte Carlo SR for BF16/FP16.
  bound                   bool                  True           True | False                                     Enable trust-region step bounding.
  bound_type              BoundType             "coordinate"   "coordinate" | "radial"                          Element-wise coordinate bound or tensor-norm radial bound.
  bound_ratio             float                 0.05           Finite, > 0.0                                    Maximum step ratio relative to weight magnitude/norm (rho).
  bound_floor             float                 1.0e-2         Finite, > 0.0                                    Coordinate bound floor radius f: rho*(|p|+f) or rho*sqrt(p^2+f^2).
  master_weights          MasterWeights         "none"         "none" | "semi" | "full"                         FP32 master copy mode for BF16/FP16 parameters (see matrix below).
  execution               ExecutionMode         "auto"         "auto"|"triton"|"cubecl"|"vulkan"|"simd"|"eager"|"scalar"  Backend dispatch route.
  fused_single_pass       bool                  True           True | False                                     1-pass kernel + post-step EMA on step > 1; False forces 2-pass exact.
  fuse_tensors            bool                  False          True | False                                     Coalesce homogeneous PyTorch tensors into 1D contiguous buckets.
  bucket_cap_mb           float                 500.0          Finite, > 0.0                                    Maximum memory size (MB) per contiguous bucket when fuse_tensors=True.
  prewarm_jit             bool                  True           True | False                                     Pre-compile Triton GPU kernels on 64-elem dummy buffers at init.
  ema_decay               float | str | None    "auto"         "auto" | None | float in (0.0, 1.0)              Post-step EMA decay ("auto" -> 0.95 for 1-buffer, 0.80 for 2-buffer).
  fsdp_sync               bool                  False          True | False                                     All-reduce scratch statistics across FSDP ranks via torch.distributed.
  seed                    int                   1337           int in [0, 4294967295]                           Base deterministic seed for 3D Weyl-Kronecker PRNG.

Master Weights Matrix (for BF16 / FP16 parameters):
  - "none": No FP32 master copy; states stored in 16-bit (FP16 uses In-Container Bitwise-BF16); SR/RNE on param & state.
  - "semi": FP32 master_param copy + 16-bit states (FP16 uses In-Container Bitwise-BF16); RNE on param, SR/RNE on state.
  - "full": FP32 master_param copy + FP32 states.

LuminaV Methods & Attributes:
  - opt.step(closure=None, grads=None) -> Optional[float]
  - opt.zero_grad(set_to_none=True) -> None
  - opt.add_param_group(param_group: dict) -> None
  - opt.state_dict() -> dict / opt.load_state_dict(state_dict: dict) -> None
  - LuminaV.detect_hardware() -> dict
  - LuminaV.get_math_version() -> str  (returns "2.0.0"; also available as LuminaV.MATH_VERSION)
  - LuminaV.help(topic="optimizer", output=None, print_out=True) -> str
"""

_NN_SECTION = """\
___
2. NEURAL NETWORK MODULE (`luminamoon.nn`)

Import:
  from luminamoon import (
      RaNorm, RaBatchNorm2d, RaTanh, RaGELU, RaSwiGLU, RaGLU,
      RaAttention, LinearCrossEntropy, RaMoE,
      ra_tanh, ra_gelu, pos_ra_gate, ra_swiglu, ra_glu,
      ra_attention, ra_softmax, linear_cross_entropy, permute_tokens_by_expert,
  )

Modules & Functional APIs:
  Component / Function                                                       Default Arguments & Description
  RaNorm(dim, eps=1e-6, elementwise_affine=True, bias=True,                  Zero-sqrt L1 rational normalization: x / (mean(|x|) + eps) (Gaussian var ~ pi/2).
         device=None, dtype=None)
  RaBatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True,           2D zero-sqrt L1 rational batch normalization with running L1 mean/dev.
                track_running_stats=True, device=None, dtype=None)
  RaTanh(b=1.0) / ra_tanh(x, b=1.0)                                          Normalized rational tanh: b * (b + |x|) * x / (b * (b + |x|) + x^2) on (-b, b).
  RaGELU() / ra_gelu(x)                                                      Cancellation-free rational GELU: x * pos_ra_gate(sqrt(2 / pi) * x, b=1.0).
  pos_ra_gate(z, b=1.0)                                                      Strictly positive rational gate on (0, 1) without per-element exp() or erf().
  RaSwiGLU(in_features, hidden_features=None, out_features=None,             Rational SwiGLU FFN block (fused Triton / SIMD / CubeCL, normalized up to FLT_MAX).
           bias=False, b=1.0)
  ra_swiglu(gate, up, b=1.0)                                                 Functional rational SwiGLU: (gate * pos_ra_gate(gate, b=b)) * up.
  RaGLU(in_features, hidden_features=None, out_features=None,                Hard-bounded Rational GLU FFN block: RaTanh(gate; b) * up.
        bias=False, b=1.0) / ra_glu(gate, up, b=1.0)
  RaAttention(dim, num_heads=8, num_kv_heads=None, qkv_bias=False,           MHA / GQA attention with compact-support ra_softmax (0 per-element exp/sqrt/tanh).
              qk_norm=False, attn_drop=0.0, proj_drop=0.0, is_causal=False,
              scale=None, ra_scale=0.0625, q_chunk_size=None)
  ra_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,      Functional tiled Flash Rational Attention (MHA & GQA).
               scale=None, ra_scale=0.0625, q_chunk_size=None)
  ra_softmax(scores, dim=-1, scale=0.0625)                                   Cayley-Pade [1/1]^8 rational softmax with compact support (Delta < 1 / scale).
  LinearCrossEntropy(in_features, num_classes, bias=False,                   Chunked SRAM-fused projection + CE loss (~256-400 MiB VRAM, FP16 dZ pre-scaled).
                     ignore_index=-100, reduction="mean",
                     label_smoothing=0.0, logit_scale=1.0, chunk_size=None)
  linear_cross_entropy(x, weight, target, bias=None, ignore_index=-100,      Functional chunked fused linear cross-entropy with FP16 underflow protection.
                       reduction="mean", label_smoothing=0.0,
                       logit_scale=1.0, chunk_size=None)
  RaMoE(in_features, hidden_features=None, out_features=None, num_experts=8, Rational Mixture-of-Experts with packed 3D weights, grouped GEMM, and optional
        top_k=2, *, num_shared_experts=0, expert_type="ra_swiglu", b=1.0,    router_logit_bound (via ra_tanh) to prevent compact-support dead experts.
        ra_scale=0.0625, router_logit_bound=None, bias=False,
        aux_loss_coef=0.01, return_aux_loss=False)
  permute_tokens_by_expert(x_2d, top_indices, top_weights, num_experts)      Permute tokens into an expert-grouped buffer with on-device int64 counts.
"""

_DATA_SECTION = """\
___
3. DATA PIPELINE MODULE (`luminamoon.data`)

Import:
  from luminamoon import (
      FusedPipeline, TextCleaner, QualityFilter, PIIMasker,
      MinHashDeduplicator, StreamDeduplicator, StringTensor,
      TokenRingBuffer, ResumablePipeline, StateTracker, StreamCoordinate,
      clean_text, clean_batch, filter_quality, is_high_quality, mask_pii,
      hash_record, jaccard_similarity, fused_process_batch,
      postprocess_and_dedup_records, create_resumable_collate,
      serialize_to_arrow, serialize_to_parquet, serialize_to_jsonl,
      data_cpu_isa, suggest_closest_matches,
  )

Core Classes & Functions:
  Component / Function                                                       Default Arguments & Description
  FusedPipeline(min_length=50, max_length=100_000, min_alpha_ratio=0.6,      Single-pass Rust SIMD text cleaning, quality filtering, PII masking,
                max_symbol_ratio=0.1, min_entropy=2.5, max_repetition=0.3,   and exact xxHash64 / approximate MinHash LSH deduplication.
                mask_pii=True, exact_dedup=True, minhash_dedup=False,
                minhash_threshold=0.8)
  TextCleaner(normalize_whitespace=True, strip_control=True,                 SIMD-accelerated UTF-8 whitespace & control-character normalizer.
              strip_zero_width=True)
  QualityFilter(min_length=50, max_length=100_000, min_alpha_ratio=0.6,      Heuristic document quality filter (length, alpha, symbols, Shannon entropy).
                max_symbol_ratio=0.1, min_entropy=2.5, max_repetition=0.3)
  PIIMasker(mask_email=True, mask_phone=True, mask_ip=True,                  Fast regex-free finite-automaton PII redactor ([EMAIL], [PHONE], [IP], etc.).
            mask_credit_card=True, mask_ssn=True)
  StreamDeduplicator(capacity=1_000_000)                                     Transactional xxHash64 exact deduplicator with commit/rollback & WAL file I/O.
  MinHashDeduplicator(num_perm=128, threshold=0.8, num_bands=16, seed=1337)  128-permutation LSH near-duplicate detector.
  StringTensor.from_list(strings)                                            German-style 16-byte string views (4B len + 4B prefix + 8B inline/offset).
  TokenRingBuffer(capacity, pad_token_id=0, eos_token_id=2)                  Lock-free ring buffer packing variable-length token sequences into fixed ctx.
  ResumablePipeline(dataset, batch_size=32, seed=1337)                       Deterministic mid-epoch checkpointable data loader with StateTracker.
  serialize_to_arrow / serialize_to_parquet / serialize_to_jsonl             Zero-copy tabular export utilities for processed text batches.
  data_cpu_isa() -> str                                                      Return active SIMD instruction set for data engine (e.g., AVX512, AVX2, NEON).
"""

_MIGRATION_SECTION = """\
___
4. MIGRATION GUIDE (v1.x -> v2.0.0)

  Legacy v1.x Call / Value                     v2.0.0 Replacement
  LuminaV(..., version="1.5.5" / spec="1.6.0") Removed; v2.0.0 uses a single specification (check LuminaV.MATH_VERSION).
  LuminaV(..., centered_defect=True)           Removed; defect is always coordinate-local since v1.5.4.
  LuminaV(..., alpha_ss=0.5)                   Removed; configure `lam` (default 0.3) instead.
  LuminaV(..., deterministic=True)             Removed; pass `seed=...` and enable `torch.use_deterministic_algorithms(True)`.
  buffer="dual" | "td-clm" | "td-t1" | "td-t2" Use canonical literals: `buffer=2`, `buffer=1`, `buffer="t1"`, `buffer="t2"`.
  master_weights=False | True | "fp32"|"hybrid"Use `master_weights="none"`, `"semi"`, or `"full"`.
  bound_type="none" | 1 | 2                    Use `bound=False` to disable, or `bound_type="coordinate"` / `"radial"`.
  execution="foreach" | "single"|"scalar_core" Use `execution="auto"`, `"eager"`, or `"scalar"`.
  LuminaV.get_version() / set_version()        Use `LuminaV.get_math_version()` or `luminamoon.get_version()`.
  LuminaV.get_capabilities()                   Use `LuminaV.detect_hardware()`.
"""


def _format_hardware_section() -> str:
    lines = [
        "___",
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
        raise ValueError(f"Invalid help topic: '{topic}'. Expected one of {valid_str}.")

    parts: List[str] = [_CLI_HEADER.format(version=get_version())]
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.version and not args.help and args.output is None:
        print(f"luminamoon {__version__} (LuminaV MATH_VERSION 2.0.0)")
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
