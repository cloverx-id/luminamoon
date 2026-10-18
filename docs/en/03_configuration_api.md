# Configuration & API Reference

Complete class signature, hyperparameter reference table, public methods, validation rules, and error handling for `LuminaV`.

## Constructor Signature

```python
from luminamoon import LuminaV

optimizer = LuminaV(
    params,
    lr: float = 8.0e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    weight_decay: float = 8.0e-2,
    tau: float = 0.8,
    lam: float = 0.3,
    cautious: bool = True,
    cautious_clamp_min: float = 0.5,
    buffer: Union[int, str] = 2,
    centered_defect: bool = False,
    stochastic_rounding: bool = True,
    bound: bool = True,
    bound_type: Union[int, str] = "coordinate",
    bound_ratio: float = 0.05,
    master_weights: Union[bool, int, str] = "none",
    execution: Union[int, str] = "auto",
    fused_single_pass: bool = True,
    ema_decay: Optional[Union[float, str]] = "auto",
    fsdp_sync: bool = False,
    seed: int = 1337,
)
```

---

## Full Parameter Reference Table

| Parameter | Type | Default | Allowed Values | Technical Description |
| :--- | :--- | :--- | :--- | :--- |
| `params` | `Iterable` | *(Required)* | Tensor list, NumPy array list, or parameter group dicts | Model parameter weights to optimize. |
| `lr` | `float` | `8.0e-4` | $> 0.0$ | Primary learning rate. |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | $0.0 \le \beta < 1.0$ | First-order ($\beta_1$) and second-order ($\beta_2$) momentum decay coefficients. |
| `eps` | `float` | `1.0e-8` | $> 0.0$ | Numerical stability constant preventing division by zero. |
| `weight_decay` | `float` | `8.0e-2` | $\ge 0.0$ | Decoupled weight decay coefficient (AdamW style). |
| `tau` | `float` | `0.8` | $> 0.0$ | Temperature parameter for CLM scaling (Dual-Buffer). |
| `lam` | `float` | `0.3` | $\ge 0.0$ | Temporal-difference defect penalty coefficient for Single-Buffer TD-CLM. |
| `cautious` | `bool` | `True` | `True`, `False` | Enables cautious masking (updates only applied when $u \cdot g > 0$). |
| `cautious_clamp_min` | `float` | `0.5` | $0.0 < x \le 1.0$ | Floor clamp for cautious mask ratio to prevent over-amplification. |
| `buffer` | `int \| str` | `2` | `1`, `2`, `"single"`, `"dual"`, `"clm"`, `"tdclm"` | Momentum buffer mode: `2` (Dual-Buffer CLM) or `1` (Single-Buffer TD-CLM). |
| `centered_defect` | `bool` | `False` | `True`, `False` | Centers temporal gradient difference $(g - m)$ across batch/tensor. |
| `stochastic_rounding` | `bool` | `True` | `True`, `False` | Enables hardware-independent PRNG stochastic rounding for FP16/BF16. |
| `bound` | `bool` | `True` | `True`, `False` | Enables gradient step bounding. |
| `bound_type` | `int \| str` | `"coordinate"` | `1`, `2`, `"coordinate"`, `"radial"` | Bounding mode: `"coordinate"` (per-element) or `"radial"` (hyperspherical norm). |
| `bound_ratio` | `float` | `0.05` | $\ge 0.0$ | Maximum update step ratio relative to parameter norm. `0.0` disables bounding. |
| `master_weights` | `bool \| int \| str` | `"none"` | `"none"` / `0`, `"fp32"` / `1`, `"pure_fp32"` / `2` | FP32 master weight strategy when training in FP16/BF16. |
| `execution` | `int \| str` | `"auto"` | `"auto"`, `"vulkan"`, `"simd"`, `"scalar"`, `"triton"`, `"single"` | Target hardware backend execution preference. |
| `fused_single_pass` | `bool` | `True` | `True`, `False` | Blends parameter update and EMA tracking into a single kernel pass. |
| `ema_decay` | `float \| str` | `"auto"` | `"auto"` or float $[0.0, 1.0)$ | EMA tracking coefficient for mask and bounding. `"auto"` sets `0.95` (buffer 1) or `0.80` (buffer 2). |
| `fsdp_sync` | `bool` | `False` | `True`, `False` | Enables global collective synchronization across distributed GPU ranks. |
| `seed` | `int` | `1337` | Integer | Base seed for per-parameter stochastic rounding PRNG generators. |

---

## Public Methods

### 1. `step(closure=None, grads=None)`
Performs a single optimization step.

- **Arguments**:
  - `closure` (*Optional[Callable]*): Optional closure function that re-evaluates the model and returns the loss (supports L-BFGS style evaluators).
  - `grads` (*Optional[List[Any] | Dict[Any, Any]]*): Explicit gradient list or dictionary. Required when operating on standalone NumPy arrays.
- **Returns**: `Optional[float]` (loss value if closure was supplied).

### 2. `zero_grad(set_to_none=True)`
Resets the gradients of all optimized parameters.

- **Arguments**:
  - `set_to_none` (*bool*): If `True`, sets `p.grad = None` (reduces memory consumption and improves PyTorch memory allocation speed). If `False`, performs `p.grad.zero_()`.

### 3. `state_dict()`
Returns a clean, serializable dictionary representing the optimizer state (`state` and `param_groups`). Transient hardware buffers (`_plan`, `p_contig`, etc.) are automatically stripped.

### 4. `load_state_dict(state_dict)`
Restores optimizer state from a previously saved checkpoint. Supports both PyTorch Tensors and NumPy arrays, properly restoring param groups and internal accumulator states.

### 5. `detect_hardware()`
Static method that detects host CPU SIMD capabilities, Vulkan UMA support, and GPU dispatch thresholds.

---

## Parameter Validation & Typo Suggestions

LuminaV enforces strict parameter validation to eliminate runtime typos. If an unrecognized string option is passed, the optimizer calculates the Levenshtein distance and provides an actionable correction:

```python
# Misspelled parameter option
LuminaV(model.parameters(), bound_type="radal")
# Raises:
# ValueError: Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?

LuminaV(model.parameters(), execution="vulkn")
# Raises:
# ValueError: Invalid execution mode: 'vulkn'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean 'vulkan'?
```
