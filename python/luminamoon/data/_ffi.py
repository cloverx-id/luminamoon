from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from ._stubs import (
        MinHashDeduplicator,
        QualityFilter,
        StateTracker,
        StreamCoordinate,
        StreamDeduplicator,
        StringTensor,
        clean_batch_records,
        clean_single_record,
        dataloader_cpu_isa,
        fused_pipeline_process_batch,
        postprocess_and_dedup_records,
        serialize_table_to_bytes,
        suggest_closest_matches,
    )
else:
    def _load_rs_module() -> Any:
        try:
            from luminamoon.optimizer import _luminamoon_optimizer_rs as mod
            return mod
        except ImportError:
            import importlib.machinery
            import importlib.util

            suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
            for entry in sys.path:
                if not entry:
                    continue
                opt_dir = Path(entry) / "luminamoon" / "optimizer"
                if not opt_dir.is_dir():
                    continue
                for candidate in opt_dir.iterdir():
                    if candidate.name.startswith("_luminamoon_optimizer_rs") and candidate.name.endswith(suffixes):
                        spec = importlib.util.spec_from_file_location(
                            "luminamoon.optimizer._luminamoon_optimizer_rs",
                            candidate,
                        )
                        if spec is not None and spec.loader is not None:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[spec.name] = module
                            spec.loader.exec_module(module)
                            return module
            raise

    try:
        _rs_mod = _load_rs_module()
        StreamCoordinate = _rs_mod.StreamCoordinate
        StreamDeduplicator = _rs_mod.StreamDeduplicator
        StateTracker = _rs_mod.StateTracker
        QualityFilter = _rs_mod.QualityFilter
        MinHashDeduplicator = _rs_mod.MinHashDeduplicator
        StringTensor = _rs_mod.StringTensor
        dataloader_cpu_isa = _rs_mod.dataloader_cpu_isa
        suggest_closest_matches = _rs_mod.suggest_closest_matches
        serialize_table_to_bytes = _rs_mod.serialize_table_to_bytes
        clean_single_record = _rs_mod.clean_single_record
        clean_batch_records = _rs_mod.clean_batch_records
        postprocess_and_dedup_records = _rs_mod.postprocess_and_dedup_records
        fused_pipeline_process_batch = _rs_mod.fused_pipeline_process_batch
    except Exception:
        from .fallbacks import (
            MinHashDeduplicator,
            PIIMasker,
            QualityFilter,
            RustStringTensor,
            StateTracker,
            StreamCoordinate,
            StreamDeduplicator,
            StringTensor,
            TextCleaner,
            clean_batch_records,
            clean_single_record,
            dataloader_cpu_isa,
            fused_pipeline_process_batch,
            postprocess_and_dedup_records,
            serialize_table_to_bytes,
            serialize_to_arrow_ipc,
            serialize_to_csv,
            serialize_to_jsonl,
            serialize_to_parquet,
            suggest_closest_matches,
        )

__all__ = [
    "MinHashDeduplicator",
    "QualityFilter",
    "StateTracker",
    "StreamCoordinate",
    "StreamDeduplicator",
    "StringTensor",
    "clean_batch_records",
    "clean_single_record",
    "dataloader_cpu_isa",
    "fused_pipeline_process_batch",
    "postprocess_and_dedup_records",
    "serialize_table_to_bytes",
    "suggest_closest_matches",
]
