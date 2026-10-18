import numpy as np
from luminamoon import LuminaV


def run_numpy_training_test(buffer_mode: int = 2, execution: str = "auto") -> None:
    mode_name = "Dual-Buffer CLM" if buffer_mode == 2 else "Single-Buffer TD-CLM"
    print(f"[NumPy Test] {mode_name} (buffer={buffer_mode}, execution={execution})")

    rng = np.random.default_rng(42)

    # Synthetic linear dataset: Y = X @ W_true + b_true + noise
    n_samples, in_dim, out_dim = 128, 16, 4
    x = rng.standard_normal((n_samples, in_dim), dtype=np.float32)
    w_true = rng.standard_normal((in_dim, out_dim), dtype=np.float32)
    b_true = rng.standard_normal((out_dim,), dtype=np.float32)
    y = x @ w_true + b_true + 0.01 * rng.standard_normal((n_samples, out_dim), dtype=np.float32)

    # Model parameters
    w = rng.standard_normal((in_dim, out_dim), dtype=np.float32)
    b = np.zeros((out_dim,), dtype=np.float32)

    initial_w = w.copy()
    initial_b = b.copy()

    optimizer = LuminaV(
        [w, b],
        lr=0.05,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        buffer=buffer_mode,
        execution=execution,
        bound=True,
        cautious=True,
    )

    losses = []
    for step in range(1, 21):
        # Forward pass: MSE loss
        pred = x @ w + b
        diff = pred - y
        loss = float(np.mean(diff ** 2))
        losses.append(loss)

        # Backward pass: analytical gradients
        grad_w = (2.0 / n_samples) * (x.T @ diff)
        grad_b = (2.0 / n_samples) * np.sum(diff, axis=0)

        # Optimizer step with explicit gradients
        optimizer.step(grads=[grad_w, grad_b])

        if step % 5 == 0 or step == 1:
            print(f"Step {step:02d} | Loss: {loss:.6f}")

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
    )

    diff_w = float(np.max(np.abs(w - initial_w)))
    diff_b = float(np.max(np.abs(b - initial_b)))
    assert diff_w > 0.0, "Weight W was not updated"
    assert diff_b > 0.0, "Bias b was not updated"

    pct_drop = ((losses[0] - losses[-1]) / losses[0]) * 100
    print(
        f"[Result] PASS (Loss: {losses[0]:.6f} to {losses[-1]:.6f}, Reduction: {pct_drop:.2f}%)"
    )


def run_numpy_contiguity_and_state_test() -> None:
    print("[NumPy Test] Contiguity and state checkpointing")
    rng = np.random.default_rng(123)

    # Test non-contiguous parameter rejection
    non_contig = rng.standard_normal((8, 8), dtype=np.float32).T
    assert not non_contig.flags.c_contiguous
    opt_nc = LuminaV([non_contig])
    try:
        opt_nc.step(grads=[np.ones((8, 8), dtype=np.float32)])
        raise AssertionError("Expected ValueError for non-contiguous parameter array")
    except ValueError as e:
        assert "np.ascontiguousarray()" in str(e), f"Unexpected contiguity error: {e}"

    # Test state_dict and load_state_dict with NumPy
    p1 = rng.standard_normal((10,), dtype=np.float32)
    opt1 = LuminaV([p1], lr=0.01)
    opt1.step(grads=[np.ones((10,), dtype=np.float32)])
    assert opt1.state[p1]["step"] == 1

    sd = opt1.state_dict()
    assert "state" in sd and "param_groups" in sd

    opt2 = LuminaV([p1], lr=0.01)
    opt2.load_state_dict(sd)
    opt2.step(grads=[np.ones((10,), dtype=np.float32)])
    assert opt2.state[p1]["step"] == 2

    print("[Result] PASS (Contiguity validation and state checkpointing verified)")


if __name__ == "__main__":
    run_numpy_training_test(buffer_mode=2, execution="auto")
    run_numpy_training_test(buffer_mode=1, execution="auto")
    run_numpy_training_test(buffer_mode=2, execution="simd")
    run_numpy_contiguity_and_state_test()
    print("[Summary] All NumPy standalone tests completed successfully.")
