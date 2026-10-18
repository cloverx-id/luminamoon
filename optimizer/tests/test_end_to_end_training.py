import torch
import torch.nn as nn
from luminamoon import LuminaV


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_training_test(buffer_mode: int = 2, execution: str = "auto") -> None:
    mode_name = "Dual-Buffer CLM" if buffer_mode == 2 else "Single-Buffer TD-CLM"
    print(f"[Test] {mode_name} (buffer={buffer_mode}, execution={execution})")

    hw = LuminaV.detect_hardware()
    print(
        f"[Hardware] simd={hw.get('simd_backend')} vulkan={hw.get('vulkan_available')} device={hw.get('gpu_device_name')} wave32={hw.get('wave32_supported')}"
    )

    torch.manual_seed(42)

    model = SimpleMLP()
    criterion = nn.MSELoss()
    optimizer = LuminaV(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        buffer=buffer_mode,
        execution=execution,
        bound=True,
        cautious=True,
    )

    x = torch.randn(32, 64)
    target = torch.randn(32, 10)

    initial_weights = [p.clone().detach() for p in model.parameters()]
    losses = []

    for step in range(1, 11):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"Step {step:02d} | Loss: {loss_val:.6f}")

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
    )

    for i, (init_w, cur_w) in enumerate(zip(initial_weights, model.parameters())):
        weight_diff = (cur_w - init_w).abs().max().item()
        assert weight_diff > 0.0, f"Parameter {i} was not updated (diff = 0.0)"

    pct_drop = ((losses[0] - losses[-1]) / losses[0]) * 100
    print(
        f"[Result] PASS (Loss: {losses[0]:.6f} to {losses[-1]:.6f}, Reduction: {pct_drop:.2f}%)"
    )


def run_error_validation_tests() -> None:
    p = [nn.Parameter(torch.randn(8, 8))]

    try:
        LuminaV(p, buffer=3)
        raise AssertionError("Expected ValueError for buffer=3")
    except ValueError as e:
        expected = "Invalid buffer count: 3. LuminaV only supports buffer=1 (Single-Buffer TD-CLM) or buffer=2 (Dual-Buffer CLM)."
        assert str(e) == expected, f"Unexpected buffer error: {e}"

    try:
        LuminaV(p, bound_type="radal")
        raise AssertionError("Expected ValueError for bound_type='radal'")
    except ValueError as e:
        expected = "Invalid bound_type: 'radal'. Expected 'coordinate' or 'radial'. Did you mean 'radial'?"
        assert str(e) == expected, f"Unexpected bound_type error: {e}"

    try:
        LuminaV(p, execution="vulkn")
        raise AssertionError("Expected ValueError for execution='vulkn'")
    except ValueError as e:
        expected = "Invalid execution mode: 'vulkn'. Expected 'auto', 'vulkan', 'simd', or 'scalar'. Did you mean 'vulkan'?"
        assert str(e) == expected, f"Unexpected execution error: {e}"

    try:
        LuminaV(p, master_weights="invalid_mw")
        raise AssertionError("Expected ValueError for master_weights='invalid_mw'")
    except ValueError as e:
        assert "Expected False, 'none', 'semi', or 'full'" in str(e), (
            f"Unexpected master_weights error: {e}"
        )

    try:
        LuminaV(p, lr=-0.01)
        raise AssertionError("Expected ValueError for lr=-0.01")
    except ValueError as e:
        assert "-0.01" in str(e) and "[0.0, inf)" in str(e), f"Unexpected lr error: {e}"

    try:
        LuminaV(p, betas=(1.2, 0.999))
        raise AssertionError("Expected ValueError for beta1=1.2")
    except ValueError as e:
        assert "1.2" in str(e) and "[0.0, 1.0)" in str(e), f"Unexpected beta1 error: {e}"

    try:
        LuminaV(p, eps=0.0)
        raise AssertionError("Expected ValueError for eps=0.0")
    except ValueError as e:
        assert "0.0" in str(e) and "(0.0, inf)" in str(e), f"Unexpected eps error: {e}"

    non_contig_p = nn.Parameter(torch.randn(8, 8).t())
    non_contig_p.grad = torch.randn(8, 8)
    opt_nc = LuminaV([non_contig_p])
    try:
        opt_nc.step()
        raise AssertionError("Expected ValueError for non-contiguous tensor")
    except ValueError as e:
        assert ".contiguous()" in str(e), f"Unexpected contiguity error: {e}"

    int_p = nn.Parameter(torch.zeros(8, 8, dtype=torch.float64), requires_grad=False)
    int_p.grad = torch.zeros(8, 8, dtype=torch.float64)
    opt_dt = LuminaV([int_p])
    try:
        opt_dt.step()
        raise AssertionError("Expected ValueError for float64 tensor")
    except ValueError as e:
        assert (
            "torch.float32" in str(e)
            and "torch.bfloat16" in str(e)
            and "torch.float16" in str(e)
        ), f"Unexpected dtype error: {e}"

    print("[Result] Error validation and typo suggestion tests PASS")


def run_mixed_precision_and_checkpoint_tests() -> None:
    torch.manual_seed(123)
    p_fp32 = nn.Parameter(torch.randn(16, 16, dtype=torch.float32))
    p_bf16 = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
    p_fp16 = nn.Parameter(torch.randn(16, 16, dtype=torch.float16))
    p_empty = nn.Parameter(torch.empty(0, dtype=torch.float32))

    p_fp32.grad = torch.randn_like(p_fp32)
    p_bf16.grad = torch.randn_like(p_bf16)
    p_fp16.grad = torch.randn_like(p_fp16)
    p_empty.grad = torch.empty_like(p_empty)

    opt = LuminaV(
        [p_fp32, p_bf16, p_fp16, p_empty],
        lr=1e-2,
        master_weights=True,
        stochastic_rounding=False,
        execution="simd",
    )
    opt.step()

    assert "master_param" not in opt.state[p_fp32]
    assert "master_param" in opt.state[p_bf16]
    assert "master_param" in opt.state[p_fp16]

    saved = opt.state_dict()
    opt2 = LuminaV(
        [p_fp32, p_bf16, p_fp16, p_empty],
        lr=1e-2,
        master_weights="full",
        stochastic_rounding=True,
        execution="simd",
    )
    opt2.load_state_dict(saved)
    opt2.step()
    assert opt2.state[p_fp32]["step"] == 2
    assert opt2.state[p_bf16]["step"] == 2
    assert opt2.state[p_fp16]["step"] == 2
    print("[Result] Mixed-precision master_weights, FP16 RNE/SR, and state_dict tests PASS")


if __name__ == "__main__":
    run_training_test(buffer_mode=2, execution="auto")
    run_training_test(buffer_mode=1, execution="auto")
    run_error_validation_tests()
    run_mixed_precision_and_checkpoint_tests()
    print("[Summary] All end-to-end training and validation tests completed.")
