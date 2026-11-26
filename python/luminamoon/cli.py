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
  from luminamoon.nn import (
      QRoPE, qrope, DistanceDecayBias, Sinusoidal3DEmbedding,
      LatentKVAttention, SlotCrossAttention, use_attention_backend, get_attention_backend,
      ReversibleDKDBlock, ReversibleDKDStack, resolve_selective_reversible,
      ResidualRollout, OrthogonalHub, DirectiveGate, TwoTierMemory,
  )

Modules & Functional APIs:
  Component / Function                                                       Default Arguments & Description
  RaNorm(normalized_shape, eps=1e-6, elementwise_affine=True, bias=False,    Zero-sqrt L1 rational normalization: x / (mean(|x|) + eps) (Gaussian var ~ pi/2).
         device=None, dtype=None)
  RaBatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True,           2D zero-sqrt L1 rational batch normalization with running L1 mean/dev.
                track_running_stats=True, device=None, dtype=None)
  RaTanh(b=1.0) / ra_tanh(x, b=1.0)                                          Normalized rational tanh: b * (b + |x|) * x / (b * (b + |x|) + x^2) on (-b, b).
  RaGELU() / ra_gelu(x)                                                      Cancellation-free rational GELU: x * pos_ra_gate(sqrt(2 / pi) * x, b=1.0).
  pos_ra_gate(z, b=1.0)                                                      Strictly positive rational gate on (0, 1) without per-element exp() or erf().
  RaSwiGLU(in_features, hidden_features=None, out_features=None,             Rational SwiGLU FFN block (fused Triton / SIMD / CubeCL, normalized up to FLT_MAX).
           bias=False, b=1.0, device=None, dtype=None)
  ra_swiglu(gate, up, b=1.0)                                                 Functional rational SwiGLU: (gate * pos_ra_gate(gate, b=b)) * up.
  RaGLU(in_features, hidden_features=None, out_features=None,                Hard-bounded Rational GLU FFN block: RaTanh(gate; b) * up.
        bias=False, b=1.0, device=None, dtype=None) / ra_glu(gate, up, b=1.0)
  RaAttention(dim, num_heads=8, num_kv_heads=None, qkv_bias=False,           MHA / GQA attention with compact-support ra_softmax (0 per-element exp/sqrt/tanh);
              qk_norm=False, attn_drop=0.0, proj_drop=0.0, is_causal=False,  `forward(x, attn_mask=None, doc_ids=None)` supports `doc_ids` for packed sequences.
              scale=None, ra_scale=0.0625, q_chunk_size=None,
              device=None, dtype=None)
  ra_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,      Functional tiled rational attention (MHA & GQA; supports `doc_ids`).
               scale=None, ra_scale=0.0625, q_chunk_size=None, doc_ids=None)
  ra_softmax(scores, dim=-1, scale=0.0625)                                   Cayley-Pade [1/1]^8 rational softmax with compact support (Delta < 1 / scale).
  LinearCrossEntropy(in_features=None, vocab_size=None, *, bias=False,       Chunked fused linear projection + CE loss (~400 MiB VRAM, FP16 dZ pre-scaled).
                     ignore_index=-100, reduction="mean",
                     label_smoothing=0.0, logit_scale=1.0, chunk_size=None,
                     save_logits=False, device=None, dtype=None)
  linear_cross_entropy(x, weight, target, bias=None, *, ignore_index=-100,   Functional chunked fused linear cross-entropy with FP16 underflow protection.
                       reduction="mean", label_smoothing=0.0,
                       logit_scale=1.0, chunk_size=None, save_logits=False)
  RaMoE(in_features, hidden_features=None, out_features=None, num_experts=8, Rational Mixture-of-Experts with packed 3D weights, grouped GEMM, and optional
        top_k=2, *, num_shared_experts=0, expert_type="ra_swiglu", b=1.0,    router_logit_bound (via ra_tanh) to prevent compact-support dead experts.
        ra_scale=0.0625, router_logit_bound=None, bias=False,
        aux_loss_coef=0.01, return_aux_loss=False, device=None, dtype=None)
  permute_tokens_by_expert(x_2d, top_indices, top_weights, num_experts)      Permute tokens into an expert-grouped buffer with on-device int64 counts.

PyTorch-Only Position, Latent Attention, Reversible Blocks & Memory (`luminamoon.nn`):
  QRoPE(head_dim=32, max_seq_len=4096, base=10000.0,                         Spin(4) double-quaternion rotary embedding (`qrope` + Triton CUDA).
        freq_ratio=0.618...)
  DistanceDecayBias(num_heads=8, hub_size=0, ..., mode="linear"|"rational",  Per-head distance-decay bias (clamped [-10, 0]) + hub damping.
                    learnable=False, ...)
  Sinusoidal3DEmbedding(dim, theta=10000.0)                                  3D coordinate `(z, y, x)` sinusoidal embedding (even `dim >= 6`).
  LatentKVAttention(dim, num_heads=8, ..., kv_latent_dim=None, hub_size=0,   Low-rank KV attention with QK-RaNorm, QRoPE, DistanceDecayBias,
                    ..., bias_mode="linear", attention_backend="auto",       `cache_mode="latent"|"kv"`, and `auto|sdpa|ra|flash|cudnn|fa3|flex`.
                    attention_fn=None, cache_mode="latent", ...)
  SlotCrossAttention(dim, num_heads=8, num_terminals=16,                     Multi-head slot cross-attention with orthogonal slot queries,
                     slots_per_terminal=4, attention_backend="auto",         FiLM modulation from `terminal_base`, and backend dispatch.
                     attention_fn=None, ...)
  use_attention_backend(backend) / get_attention_backend()                   ContextVar manager and getter for `"auto"` attention backend.
  ReversibleDKDBlock(dim, num_heads=8, ..., dt=0.5, ..., reversible=False,   Volume-preserving Drift-Kick-Drift block with selective reversibility,
                     ..., selective_type="none", every=2, ...,               in-place `Module.compile`, and custom attention/FFN/norm layers.
                     compile_block=False, ..., kick_attn=None,
                     drift_ffn=None, norm_cls=None, ...)
  ReversibleDKDStack(num_layers, dim, ..., selective_type="none", every=2,   Multi-layer DKD stack with selective reversibility, in-place
                     ..., compile_blocks=False, ..., kick_attn_fn=None,      `Module.compile`, and per-layer custom attention/FFN factories.
                     drift_ffn_fn=None, ...)
  resolve_selective_reversible(layer_idx=None, num_layers=None, *, ...)      Resolve per-layer boolean reversibility for a selective policy.
  ResidualRollout(dim, num_steps=2, activation="ra_tanh"|"tanh", ...)        Weight-tied residual recurrence using `ra_tanh` by default.
  OrthogonalHub(hub_size=128, dim=512, use_context_gate=False, ...)          SO(H) prefix hub via float32 Cayley transform with eval caching.
  DirectiveGate(dim, init_logit_scale=4.0, ...)                              Pre-norm cosine gating and FiLM modulation over chunk summaries.
  TwoTierMemory(dim, max_l1=16, max_l2=32, num_heads=4, ...)                 Differentiable L1 leading slots, multi-head L2 selector, `encode_keys`.
"""

_DATA_SECTION = """\
___
3. DATA PIPELINE MODULE (`luminamoon.data`)

Import:
  from luminamoon.data import (
      DataPipeline, TokenBatch, SequencePacker,
      FusedPipeline, ResumablePipeline, TokenRingBuffer, create_resumable_collate,
      TextCleaner, QualityFilter, PIIMasker,
      StreamDeduplicator, MinHashDeduplicator, StringTensor,
      StateTracker, StreamCoordinate,
      clean_text, clean_batch, filter_quality, is_high_quality, mask_pii,
      hash_record, jaccard_similarity, fused_process_batch,
      serialize_to_arrow, serialize_to_parquet, serialize_to_jsonl,
      data_cpu_isa, suggest_closest_matches,
  )

Core Classes & Functions:
  Component / Function                                                       Default Arguments & Description
  DataPipeline(source, *, tokenizer=None, batch_size=4, seq_len=2048,        Streaming data pipeline: multi-source mixing, rank/worker sharding,
               packing="best_fit", pack_window=64, eos_token_id=None,        clean/quality/dedup/PII/custom filter, document-isolated packing,
               pad_token_id=None, ignore_index=-100, drop_last=False,        optional `prefetch_batches`, `commit()`, and PyTorch `state_dict()`.
               text_key="text", clean=True, quality=None, dedup=None,
               mask_pii=False, filter_fn=None, rank=None, world_size=None,
               seed=None, shuffle_buffer=0, checkpoint=None, device=None,
               ..., prefetch_batches=0)
  TokenBatch(input_ids, labels, doc_ids, position_ids, cu_seqlens,           Packed LM batch (`doc_ids` for `RaAttention`, `causal_doc_mask()` for dense
             attention_mask, source_refs)                                    attention, `as_xy()`, `.to(device)`).
  SequencePacker(*, batch_size=4, seq_len=2048, mode="best_fit",             Low-level document packer (`"best_fit"`, `"concat"`, `"pad"`).
                 pack_window=64, pad_token_id=0, ignore_index=-100)
  FusedPipeline(*, trim_whitespace=True, normalize_newlines=True,            Single-pass Rust Rayon text cleaning, quality filtering, and
                min_byte_len=None, max_byte_len=None,                        128-bit exact deduplication returning `StringTensor`.
                truncate_over_max=False, json_extract_key=None,
                filter_quality=None, min_chars=None, max_chars=None,
                min_words=None, min_entropy=None, max_repetition_ratio=None,
                max_symbol_ratio=None, deduplicator=None, commit_dedup=False)
  TextCleaner(trim_whitespace=True, normalize_newlines=True,                 SIMD-accelerated UTF-8 newline normalizer, trimmer, byte-bound
              min_byte_len=None, max_byte_len=None, truncate_over_max=False, truncator, and dotted-path JSON extractor.
              json_extract_key=None)
  QualityFilter(min_chars=None, max_chars=None, min_words=None,              1-pass heuristic document filter (entropy, trigram repetition,
                min_entropy=None, max_repetition_ratio=None,                 symbol ratio). Use `QualityFilter.for_pretraining()` preset.
                max_symbol_ratio=None)
  PIIMasker(mask_emails=True, mask_ips=True, mask_api_keys=True,             PII masker with boundary checks and Luhn card validation
            mask_cards=True, mask_phones=False, *, validate_luhn=True)       (`[EMAIL]`, `[IP]`, `[API_KEY]`, `[CARD]`, `[PHONE]`).
  StreamDeduplicator(wal_path=None, initial_hashes=None, max_capacity=None)  128-bit XXH3 two-stage deduplicator with O(delta) append-only WAL.
  MinHashDeduplicator(threshold=0.85, k_shingle=5, num_perm=64)              Single-pass XXH3-128 double-hashed MinHash LSH + Rayon `filter_batch`.
  StringTensor(items=None)                                                   Arrow `Utf8View`-aligned 16-byte string views + `__arrow_c_array__` PyCapsule.
  StateTracker(meta_path)                                                    256B dual-slot LMDL v2 CRC32 checkpoint (`commit_up_to`, `consumed_tokens`).
  ResumablePipeline(source, checkpoint_path=None, ..., batch_size=64,        Low-level resumable streaming text pipeline with `commit()` & sharding.
                    prefetch_batches=0, rank=0, world_size=1)
  TokenRingBuffer(source=None, *, capacity_tokens=1_000_000,                 Circular token ring buffer (`token_dtype="int64"|"int32"`), batched
                  auto_prefill_tokens=500_000, batch_size=4, seq_len=2048,   background tokenization (`tokenizer_batch_size`), pinned CUDA DMA.
                  ..., tokenizer_batch_size=256, token_dtype="int64", ...)
  create_resumable_collate(cleaner=None, quality_filter=None, ...,           PyTorch DataLoader `collate_fn` factory.
                           tokenizer=None, text_key="text")
  serialize_to_parquet / serialize_to_arrow / serialize_to_jsonl             Atomic Rust 64-bit columnar export utilities (ZSTD Parquet / IPC / JSONL).
  data_cpu_isa() -> str                                                      Return active SIMD ISA for data engine (`AVX-512`, `AVX2`, `NEON`, `SCALAR`).
"""

_MIGRATION_SECTION = """\
___
4. MIGRATION GUIDE (v1.x -> v2.0.0)

  Legacy v1.x Call / Value                            v2.0.0 Replacement
  LuminaV(..., version="1.5.5" / spec="1.6.0")        Removed; v2.0.0 uses a single specification (check LuminaV.MATH_VERSION).
  LuminaV(..., centered_defect=True)                  Removed; defect is always coordinate-local since v1.5.4.
  LuminaV(..., alpha_ss=0.5)                          Removed; configure `lam` (default 0.3) instead.
  LuminaV(..., deterministic=True)                    Removed; pass `seed=...` and enable `torch.use_deterministic_algorithms(True)`.
  buffer="dual" | "td-clm" | "td-t1" | "td-t2"        Use standard literals: `buffer=2`, `buffer=1`, `buffer="t1"`, `buffer="t2"`.
  master_weights=False | True | "fp32" | "hybrid"     Use `master_weights="none"`, `"semi"`, or `"full"`.
  bound_type="none" | 1 | 2                           Use `bound=False` to disable, or `bound_type="coordinate"` / `"radial"`.
  execution="foreach"|"single"|"vulkan"|"scalar_core" Use `execution="auto"`, `"eager"`, `"cubecl"`, or `"scalar"`.
  LuminaV.get_version() / set_version()               Use `LuminaV.get_math_version()` or `luminamoon.get_version()`.
  LuminaV.get_capabilities()                          Use `LuminaV.detect_hardware()`.
  nn: PosRaGLU / pos_ra_glu                           Use `RaSwiGLU` / `ra_swiglu` (or `RaGLU` / `ra_glu` for hard-bounded gates).
  nn: TriGLU / RaEMA / ra_ema                         Removed in v2.0.0; use `RaSwiGLU` / `RaGLU` and `RaAttention`.
  nn: L1Norm / RationalMoE / FusedLinearCE            Use `RaNorm`, `RaMoE`, and `LinearCrossEntropy`.
"""


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
