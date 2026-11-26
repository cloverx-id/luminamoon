"""CLI documentation section templates for luminamoon.cli."""

from __future__ import annotations

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

__all__ = ["_NN_SECTION", "_DATA_SECTION", "_MIGRATION_SECTION"]
