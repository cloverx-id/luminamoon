# User Guide

Comprehensive guide on advanced features of LuminaV, selecting buffer modes, mixed-precision training, boundary dynamics, state checkpointing, and distributed training.

## 1. Selecting Buffer Modes: Dual-Buffer vs Single-Buffer

LuminaV offers two primary optimization paradigms controlled via the `buffer` argument:

```python
# Mode 1: Dual-Buffer CLM (Default)
optimizer = LuminaV(model.parameters(), buffer=2)

# Mode 2: Single-Buffer TD-CLM (Memory-Efficient)
optimizer = LuminaV(model.parameters(), buffer=1)
```

### Technical Comparison

| Characteristic | Dual-Buffer CLM (`buffer=2`) | Single-Buffer TD-CLM (`buffer=1`) |
| :--- | :--- | :--- |
| **State Buffers** | Maintains both `exp_avg` ($m$) and `exp_avg_sq` ($v$). | Maintains only `exp_avg` ($m$). |
| **Memory Footprint** | Identical to AdamW ($2 \times$ parameter size). | **50% state memory savings** ($1 \times$ parameter size). |
| **Scaling Mechanism** | Computes $\sigma_t = \sqrt{v_t} \cdot c_1 + c_2$ with dynamic Nesterov bias correction. | Computes Temporal-Difference Defect Envelope $\text{env} = \beta_1^2 |m| + (1-\beta_1^2)|g| + \lambda |\Delta| + \epsilon$. |
| **Optimal Use Case** | Pre-training Transformers, large foundation models, Vision models (ViT/CNN) where gradient variance stability is paramount. | Large-scale LLM fine-tuning (LoRA / Full), memory-constrained devices, edge hardware, and high-throughput training. |

---

## 2. Mixed-Precision & Stochastic Rounding

LuminaV natively supports low-precision training (`torch.float16` and `torch.bfloat16`) with active gradient underflow protection.

### Master Weights Configuration

The `master_weights` argument controls how the optimizer manages FP32 parameter copies:

- `"none"` or `0`: No master weights. Low-precision parameters are updated directly using *Stochastic Rounding*.
- `"fp32"` or `1`: Maintains FP32 master weights for FP16 and BF16 parameters. Accumulator state follows the original parameter dtype to conserve memory.
- `"pure_fp32"` or `2`: Master weights and internal state accumulators (`exp_avg`, `exp_avg_sq`) are maintained in full FP32.

```python
# Optimal configuration for FP16 / BF16 models
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    master_weights="fp32",
    stochastic_rounding=True,
)
```

### Stochastic Rounding (SR)

When parameters or momentum accumulators are stored in low precision, standard deterministic rounding (Round-to-Nearest-Even / RNE) can cause small gradient updates to vanish (*gradient stagnation*).

LuminaV integrates a hardware-independent XORShift/PCG PRNG that performs *Stochastic Rounding* deterministically on each step:
- The probability of rounding up is exactly proportional to the sub-representation numerical fraction.
- Eliminates parameter stagnation without CPU/GPU synchronization latency.

---

## 3. Boundary Dynamics & Bounding

To prevent gradient explosion and divergent loss spikes, LuminaV incorporates dynamic parameter bounding:

```python
optimizer = LuminaV(
    model.parameters(),
    bound=True,
    bound_type="coordinate",  # Options: 'coordinate' or 'radial'
    bound_ratio=0.05,         # Maximum update step is 5% of parameter norm
)
```

### Bounding Modes: Coordinate vs Radial

1. **Coordinate Bounding (`bound_type="coordinate"` or `2`)**:
   - Bounds each element independently using a hyperbolic tangent envelope:
     $$B_i = \text{bound\_ratio} \cdot \sqrt{\theta_i^2 + 10^{-4}}$$
     $$\Delta\theta_i = B_i \cdot \tanh\left(\frac{\Delta\theta_i}{B_i}\right)$$
   - Exceptionally robust against sparse outliers and extreme gradients in embedding layers.

2. **Radial Bounding (`bound_type="radial"` or `1`)**:
   - Bounds the entire update vector within a spherical hyper-ball:
     $$R = \text{bound\_ratio} \cdot \max(\|\theta\|_2, 1.0)$$
     $$r = \frac{\eta \|\Delta\theta\|_2}{R}, \quad \text{scale} = \frac{\tanh(r)}{\max(r, 10^{-8})}$$
   - Preserves the exact direction of the update vector without angular distortion.

---

## 4. Cautious Learning Momentum

When `cautious=True` is enabled, the optimizer only updates parameters whose direction aligns with the current gradient:

$$\text{mask}_i = \mathbb{I}(u_i \cdot g_i > 0)$$
$$\bar{m} = \max\left(\text{clamp\_min}, \frac{1}{N}\sum_{i=1}^N \text{mask}_i\right)$$
$$\Delta\theta_i = u_i \cdot \text{mask}_i \cdot \frac{1}{\bar{m}}$$

Dynamic scaling $\frac{1}{\bar{m}}$ maintains effective update magnitude even when a fraction of parameters is masked out.

---

## 5. Centered Defect Compensation

Enabling `centered_defect=True` centers the temporal gradient deviation $\Delta_t = g_t - m_t$ across the parameter tensor or distributed shard:

$$\text{diff} = (g_t - m_t) - \mathbb{E}[g_t - m_t]$$

This systematically eliminates drift bias in distributed setups or training runs with high batch-to-batch statistical variance.

---

## 6. Fused Single-Pass Execution

By default, `fused_single_pass=True` is enabled:
- Starting from step 2, cautious mask reduction and radial scale tracking are updated asynchronously via Exponential Moving Average (EMA).
- Eliminates the need for two-pass global reductions, cutting memory bandwidth consumption by 50% and maximizing kernel throughput.

---

## 7. State Checkpointing (`state_dict` & `load_state_dict`)

LuminaV provides full serialization support for PyTorch and standalone NumPy:

```python
# Saving checkpoint
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

# Loading checkpoint
loaded_checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
```

> [!NOTE]
> Transient scratch buffers (such as `p_contig`, `_plan`, and `_contig_step`) are automatically stripped during `state_dict()` serialization to produce clean, compact checkpoints.

---

## 8. Distributed Training & FSDP / DDP

In distributed environments such as Fully Sharded Data Parallel (FSDP) or Distributed Data Parallel (DDP):

```python
optimizer = LuminaV(
    model.parameters(),
    lr=1e-3,
    fsdp_sync=True,  # Activates collective synchronization across GPU ranks
    cautious=True,
    bound=True,
    bound_type="radial",
)
```

When `fsdp_sync=True`:
- Global reductions for `mask.sum()`, $\|u\|_2^2$, and $\|\theta\|_2^2$ are batched into a single collective `all_reduce` call.
- Empty shards across ranks are handled deterministically without stalling FSDP collective synchronization barriers.
