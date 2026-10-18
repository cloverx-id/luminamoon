# `python/luminamoon/`: Python Engine, Triton JIT & FSDP Synchronization

The Python layer (`python/luminamoon/optimizer/`) provides clean integration with deep learning ecosystems (PyTorch, NumPy), Triton JIT GPU execution, and distributed collective synchronization.

[Versi Bahasa Indonesia (README.id.md)](README.id.md)

---

## 1. Class Architecture & Inheritance

```mermaid
classDiagram
    class StandaloneOptimizer {
        +state: ParamStateDict
        +param_groups: list
        +zero_grad(set_to_none)
        +state_dict()
        +load_state_dict()
    }

    class TorchOptimizer {
        <<PyTorch Base>>
    }

    class LuminaVGpuMixin {
        #_step_gpu_group()
        #_dispatch_triton()
        #_sync_fsdp_scratch_batch()
    }

    class LuminaV {
        +step(closure, grads)
        +detect_hardware()
        -_step_numpy_group()
        -_validate_and_normalize_group()
    }

    TorchOptimizer <|-- LuminaV : When PyTorch available
    StandaloneOptimizer <|-- LuminaV : When PyTorch absent
    LuminaVGpuMixin <|-- LuminaV
```

---

## 2. Distributed FSDP Collective Synchronization (`gpu_backend.py`)

When training across distributed GPU clusters using PyTorch Fully Sharded Data Parallel (FSDP):
1. **Pass 1 Reduction Coalescing**:
   Partial cautious masks $\sum \mathcal{M}_i$ and gradient divergence values are coalesced into a single flat scratch tensor per device.
2. **Collective All-Reduce**:
   Executes `torch.distributed.all_reduce(..., op=ReduceOp.SUM)` once per step across all GPU ranks, eliminating per-tensor communication barriers.
3. **Pass 2 Normalized Mutation**:
   Each rank normalizes the global mask ratio and computes parameter updates using identical global bounding scales.

---

## 3. Checkpoint Cleanliness & State Serialization

LuminaV automatically strips transient GPU cache pointers and FFI metadata before serialization:

```python
def state_dict(self) -> Dict[str, Any]:
    # Filters out runtime transient keys:
    # '_plan', 'p_contig', 'exp_avg_contig', 'exp_avg_sq_contig'
    # Retains only persistent math states: 'step', 'exp_avg', 'scratch'
```

This ensures serialized checkpoints saved with `torch.save()` or `pickle` remain clean, compact, and forward-compatible.
