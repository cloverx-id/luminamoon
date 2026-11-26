from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ._version import __version__, get_version

if TYPE_CHECKING:
    from . import cli, data, nn, optimizer
    from .data import (
        DataPipeline,
        FusedPipeline,
        MinHashDeduplicator,
        PIIMasker,
        QualityFilter,
        ResumablePipeline,
        SequencePacker,
        StateTracker,
        StreamCoordinate,
        StreamDeduplicator,
        StringTensor,
        TextCleaner,
        TokenBatch,
        TokenRingBuffer,
        clean_batch,
        clean_text,
        create_resumable_collate,
        data_cpu_isa,
        filter_quality,
        fused_process_batch,
        hash_record,
        is_high_quality,
        jaccard_similarity,
        mask_pii,
        serialize_to_arrow,
        serialize_to_jsonl,
        serialize_to_parquet,
        suggest_closest_matches,
    )
    from .nn import (
        LinearCrossEntropy,
        RaAttention,
        RaBatchNorm2d,
        RaGELU,
        RaGLU,
        RaMoE,
        RaNorm,
        RaSwiGLU,
        RaTanh,
        linear_cross_entropy,
        permute_tokens_by_expert,
        pos_ra_gate,
        ra_attention,
        ra_gelu,
        ra_glu,
        ra_softmax,
        ra_swiglu,
        ra_tanh,
    )
    from .optimizer import AutoTune, LuminaV
    from .cli import format_help, help

_SUBMODULES = frozenset({"cli", "data", "nn", "optimizer"})

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AutoTune": (".optimizer", "AutoTune"),
    "LuminaV": (".optimizer", "LuminaV"),
    "help": (".cli", "help"),
    "format_help": (".cli", "format_help"),
    # Neural network components
    "RaNorm": (".nn", "RaNorm"),
    "RaBatchNorm2d": (".nn", "RaBatchNorm2d"),
    "RaTanh": (".nn", "RaTanh"),
    "ra_tanh": (".nn", "ra_tanh"),
    "RaGELU": (".nn", "RaGELU"),
    "ra_gelu": (".nn", "ra_gelu"),
    "pos_ra_gate": (".nn", "pos_ra_gate"),
    "ra_swiglu": (".nn", "ra_swiglu"),
    "RaSwiGLU": (".nn", "RaSwiGLU"),
    "ra_glu": (".nn", "ra_glu"),
    "RaGLU": (".nn", "RaGLU"),
    "RaAttention": (".nn", "RaAttention"),
    "ra_attention": (".nn", "ra_attention"),
    "ra_softmax": (".nn", "ra_softmax"),
    "LinearCrossEntropy": (".nn", "LinearCrossEntropy"),
    "linear_cross_entropy": (".nn", "linear_cross_entropy"),
    "RaMoE": (".nn", "RaMoE"),
    "permute_tokens_by_expert": (".nn", "permute_tokens_by_expert"),
    # Data pipeline components
    "DataPipeline": (".data", "DataPipeline"),
    "FusedPipeline": (".data", "FusedPipeline"),
    "MinHashDeduplicator": (".data", "MinHashDeduplicator"),
    "PIIMasker": (".data", "PIIMasker"),
    "QualityFilter": (".data", "QualityFilter"),
    "ResumablePipeline": (".data", "ResumablePipeline"),
    "SequencePacker": (".data", "SequencePacker"),
    "StateTracker": (".data", "StateTracker"),
    "StreamCoordinate": (".data", "StreamCoordinate"),
    "StreamDeduplicator": (".data", "StreamDeduplicator"),
    "StringTensor": (".data", "StringTensor"),
    "TextCleaner": (".data", "TextCleaner"),
    "TokenBatch": (".data", "TokenBatch"),
    "TokenRingBuffer": (".data", "TokenRingBuffer"),
    "clean_batch": (".data", "clean_batch"),
    "clean_text": (".data", "clean_text"),
    "create_resumable_collate": (".data", "create_resumable_collate"),
    "data_cpu_isa": (".data", "data_cpu_isa"),
    "filter_quality": (".data", "filter_quality"),
    "fused_process_batch": (".data", "fused_process_batch"),
    "hash_record": (".data", "hash_record"),
    "is_high_quality": (".data", "is_high_quality"),
    "jaccard_similarity": (".data", "jaccard_similarity"),
    "mask_pii": (".data", "mask_pii"),
    "serialize_to_arrow": (".data", "serialize_to_arrow"),
    "serialize_to_jsonl": (".data", "serialize_to_jsonl"),
    "serialize_to_parquet": (".data", "serialize_to_parquet"),
    "suggest_closest_matches": (".data", "suggest_closest_matches"),
}

__all__ = [
    "__version__",
    "get_version",
    "help",
    "format_help",
    "cli",
    "data",
    "nn",
    "optimizer",
    "AutoTune",
    "LuminaV",
    "RaNorm",
    "RaBatchNorm2d",
    "RaTanh",
    "ra_tanh",
    "RaGELU",
    "ra_gelu",
    "pos_ra_gate",
    "ra_swiglu",
    "RaSwiGLU",
    "ra_glu",
    "RaGLU",
    "RaAttention",
    "ra_attention",
    "ra_softmax",
    "LinearCrossEntropy",
    "linear_cross_entropy",
    "RaMoE",
    "permute_tokens_by_expert",
    "DataPipeline",
    "FusedPipeline",
    "MinHashDeduplicator",
    "PIIMasker",
    "QualityFilter",
    "ResumablePipeline",
    "SequencePacker",
    "StateTracker",
    "StreamCoordinate",
    "StreamDeduplicator",
    "StringTensor",
    "TextCleaner",
    "TokenBatch",
    "TokenRingBuffer",
    "clean_batch",
    "clean_text",
    "create_resumable_collate",
    "data_cpu_isa",
    "filter_quality",
    "fused_process_batch",
    "hash_record",
    "is_high_quality",
    "jaccard_similarity",
    "mask_pii",
    "serialize_to_arrow",
    "serialize_to_jsonl",
    "serialize_to_parquet",
    "suggest_closest_matches",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        mod_path, attr_name = target
        mod = importlib.import_module(mod_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    from .optimizer.migration import removed_top_level_attr_message

    msg = removed_top_level_attr_message(name)
    if msg is not None:
        raise AttributeError(msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
