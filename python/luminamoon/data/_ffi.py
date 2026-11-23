from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

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
        fused_pipeline_process_batch_indexed,
        serialize_table_to_bytes,
        suggest_closest_matches,
    )
else:
    def _load_rs_module() -> Any:
        mod_name = "luminamoon.optimizer._luminamoon_optimizer_rs"
        existing = sys.modules.get(mod_name)
        if existing is not None:
            return existing

        import importlib.machinery
        import importlib.util

        # Load the native extension directly from the sibling optimizer directory so
        # importing luminamoon.data does not trigger luminamoon.optimizer/torch imports
        # and does not pick up binaries from unrelated sys.path environments.
        opt_dir = Path(__file__).resolve().parents[1] / "optimizer"
        if opt_dir.is_dir():
            suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
            for candidate in sorted(opt_dir.iterdir()):
                if candidate.name.startswith("_luminamoon_optimizer_rs") and candidate.name.endswith(suffixes):
                    spec = importlib.util.spec_from_file_location(mod_name, candidate)
                    if spec is not None and spec.loader is not None:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[mod_name] = module
                        try:
                            spec.loader.exec_module(module)
                        except Exception:
                            sys.modules.pop(mod_name, None)
                            raise
                        return module
        raise ImportError(f"Native extension '_luminamoon_optimizer_rs' not found in {opt_dir}")

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
        fused_pipeline_process_batch = _rs_mod.fused_pipeline_process_batch
        fused_pipeline_process_batch_indexed = _rs_mod.fused_pipeline_process_batch_indexed
    except Exception as exc:
        import logging

        logging.getLogger("luminamoon.data").warning(
            "Native Rust extension '_luminamoon_optimizer_rs' could not be loaded (%s); falling back to pure-Python data engine.",
            exc,
        )
        from .fallbacks import (
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
            fused_pipeline_process_batch_indexed,
            serialize_table_to_bytes,
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
    "fused_pipeline_process_batch_indexed",
    "serialize_table_to_bytes",
    "suggest_closest_matches",
]
