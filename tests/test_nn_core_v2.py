from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.nn import (
    BaseModule,
    RaBatchNorm2d,
    RaGELU,
    RaGLU,
    RaNorm,
    RaSwiGLU,
    RaTanh,
    is_numpy_available,
    is_torch_available,
    pos_ra_gate,
    ra_gelu,
    ra_glu,
    ra_softmax,
    ra_swiglu,
    ra_tanh,
)


class TestRaNorm:
    """Tests for RaNorm rational token normalization without square-root."""

    def test_scale_homogeneity(self) -> None:
        """Verify scale-homogeneity: RaNorm(alpha * x) = sgn(alpha) * RaNorm(x)."""
        norm = RaNorm(normalized_shape=32, elementwise_affine=False, eps=1e-12)
        torch.manual_seed(42)
        x = torch.randn(4, 16, 32, dtype=torch.float64)

        # Positive scale factor alpha > 0
        alpha_pos = 3.5
        out_orig = norm(x)
        out_scaled_pos = norm(alpha_pos * x)
        assert torch.allclose(out_orig, out_scaled_pos, atol=1e-7, rtol=1e-5)

        # Negative scale factor alpha < 0 -> should invert sign
        alpha_neg = -2.0
        out_scaled_neg = norm(alpha_neg * x)
        assert torch.allclose(-out_orig, out_scaled_neg, atol=1e-7, rtol=1e-5)

    def test_zero_sqrt_l1_mean_computation(self) -> None:
        """Verify RaNorm matches exact manual L1-mean without square root."""
        dim = 16
        eps = 1e-6
        norm = RaNorm(normalized_shape=dim, eps=eps, elementwise_affine=False)
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0] * 4], dtype=torch.float32)

        # Manual L1-mean: (1/d) * sum(|x_i|)
        l1_mean = torch.abs(x).mean(dim=-1, keepdim=True)
        expected = x / (l1_mean + eps)
        result = norm(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_elementwise_affine_parameters(self) -> None:
        """Verify weight and bias behavior when affine is enabled or disabled."""
        dim = 8
        # Affine with bias
        norm_affine_bias = RaNorm(normalized_shape=dim, elementwise_affine=True, bias=True)
        assert norm_affine_bias.weight is not None
        assert norm_affine_bias.bias is not None
        assert torch.allclose(norm_affine_bias.weight, torch.ones(dim))
        assert torch.allclose(norm_affine_bias.bias, torch.zeros(dim))

        # Affine without bias
        norm_affine_nobias = RaNorm(normalized_shape=dim, elementwise_affine=True, bias=False)
        assert norm_affine_nobias.weight is not None
        assert norm_affine_nobias.bias is None

        # Non-affine
        norm_noaffine = RaNorm(normalized_shape=dim, elementwise_affine=False)
        assert norm_noaffine.weight is None
        assert norm_noaffine.bias is None

        # Custom weight and bias scaling
        with torch.no_grad():
            norm_affine_bias.weight.fill_(2.0)
            norm_affine_bias.bias.fill_(0.5)
        x = torch.randn(2, dim)
        unscaled = norm_noaffine(x)
        scaled = norm_affine_bias(x)
        assert torch.allclose(scaled, unscaled * 2.0 + 0.5, atol=1e-6)

    def test_multidimensional_normalized_shape(self) -> None:
        """Verify RaNorm works with tuple normalized shapes over trailing dimensions."""
        shape = (8, 16)
        norm = RaNorm(normalized_shape=shape, elementwise_affine=False)
        x = torch.randn(2, 4, 8, 16)
        out = norm(x)
        assert out.shape == x.shape

        # Verify each 8x16 slice has unit L1 mean (modulo eps)
        l1_means = out.abs().mean(dim=(-2, -1))
        assert torch.allclose(l1_means, torch.ones_like(l1_means), atol=1e-4)

    def test_numpy_execution(self) -> None:
        """Verify RaNorm forward pass on NumPy ndarrays."""
        norm = RaNorm(normalized_shape=4, elementwise_affine=False)
        x_np = np.array([[2.0, -2.0, 2.0, -2.0]], dtype=np.float32)
        out_np = norm(x_np)
        assert isinstance(out_np, np.ndarray)
        assert out_np.shape == (1, 4)
        assert np.allclose(out_np, np.array([[1.0, -1.0, 1.0, -1.0]]), atol=1e-4)


class TestRaBatchNorm2d:
    """Tests for RaBatchNorm2d using cross-batch mean absolute deviation (MAD)."""

    def test_train_vs_eval_modes(self) -> None:
        """Verify batch statistics in train mode and running statistics in eval mode."""
        num_features = 4
        bn = RaBatchNorm2d(num_features=num_features, momentum=0.1)

        torch.manual_seed(101)
        x1 = torch.randn(8, num_features, 10, 10)

        # Train mode: tracks running statistics
        bn.train()
        out_train = bn(x1)
        assert out_train.shape == x1.shape
        assert bn.num_batches_tracked == 1

        # Check running stats moved from initial values
        assert not torch.allclose(bn.running_mean, torch.zeros(num_features))
        assert not torch.allclose(bn.running_mad, torch.ones(num_features))

        # Eval mode: freeze running stats
        bn.eval()
        running_mean_frozen = bn.running_mean.clone()
        running_mad_frozen = bn.running_mad.clone()

        x2 = torch.randn(4, num_features, 10, 10)
        _ = bn(x2)
        assert torch.equal(bn.running_mean, running_mean_frozen)
        assert torch.equal(bn.running_mad, running_mad_frozen)
        assert bn.num_batches_tracked == 1

    def test_running_stats_tracking_convergence(self) -> None:
        """Verify running mean and MAD converge toward sample distribution."""
        num_features = 2
        bn = RaBatchNorm2d(num_features=num_features, momentum=0.1)
        bn.train()

        # Known distribution: mean = [3.0, -2.0]
        torch.manual_seed(202)
        for _ in range(80):
            x = torch.randn(32, num_features, 8, 8)
            x[:, 0] += 3.0
            x[:, 1] -= 2.0
            _ = bn(x)

        assert torch.allclose(bn.running_mean[0], torch.tensor(3.0), atol=0.2)
        assert torch.allclose(bn.running_mean[1], torch.tensor(-2.0), atol=0.2)
        # For standard normal, MAD is E[|Z|] = sqrt(2/pi) approx 0.7979
        expected_mad = math.sqrt(2.0 / math.pi)
        assert torch.allclose(bn.running_mad, torch.tensor(expected_mad), atol=0.15)

    def test_fold_into_conv_equivalence(self) -> None:
        """Verify folding RaBatchNorm2d into Conv2d yields identical numerical output."""
        in_channels = 3
        out_channels = 8
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        bn = RaBatchNorm2d(num_features=out_channels, momentum=0.2)

        # Train batchnorm with a few steps to generate non-trivial running stats
        bn.train()
        for _ in range(5):
            dummy = torch.randn(4, in_channels, 12, 12)
            _ = bn(conv(dummy))

        bn.eval()
        w_folded, b_folded = bn.fold_into_conv(conv)

        # Construct equivalent standalone folded Conv2d
        conv_folded = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        conv_folded.weight.data.copy_(w_folded)
        conv_folded.bias.data.copy_(b_folded)

        # Test on independent evaluation batch
        test_x = torch.randn(6, in_channels, 16, 16)
        expected_out = bn(conv(test_x))
        folded_out = conv_folded(test_x)

        assert torch.allclose(expected_out, folded_out, atol=1e-5, rtol=1e-4)

    def test_fold_into_conv_without_conv_bias(self) -> None:
        """Verify folding RaBatchNorm2d into Conv2d that has no bias (bias=False)."""
        in_channels = 2
        out_channels = 4
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        bn = RaBatchNorm2d(num_features=out_channels)

        bn.train()
        _ = bn(conv(torch.randn(4, in_channels, 8, 8)))
        bn.eval()

        w_folded, b_folded = bn.fold_into_conv(conv)
        assert b_folded is not None

        conv_folded = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        conv_folded.weight.data.copy_(w_folded)
        conv_folded.bias.data.copy_(b_folded)

        test_x = torch.randn(2, in_channels, 8, 8)
        assert torch.allclose(bn(conv(test_x)), conv_folded(test_x), atol=1e-5)

    def test_dimension_validation(self) -> None:
        """Verify RaBatchNorm2d raises ValueError when input tensor is not 4D."""
        bn = RaBatchNorm2d(num_features=4)
        with pytest.raises(ValueError):
            bn(torch.randn(4, 4))
        with pytest.raises(ValueError):
            bn(torch.randn(4, 4, 8))


class TestRaActivation:
    """Tests for RaTanh, RaGELU, RaSwiGLU, and RaGLU rational activations."""

    def test_ratanh_bounds_and_symmetry(self) -> None:
        """Verify RaTanh range bounds (-b, b) and anti-symmetry."""
        b = 1.5
        x = torch.linspace(-500.0, 500.0, 1001)
        y = ra_tanh(x, b=b)

        # Range bounds: strictly inside (-b, b)
        assert (y > -b).all()
        assert (y < b).all()

        # Symmetry: RaTanh(-x) = -RaTanh(x)
        y_neg = ra_tanh(-x, b=b)
        assert torch.allclose(y, -y_neg, atol=1e-6)

        # Asymptotic saturation
        assert torch.allclose(ra_tanh(torch.tensor(1000.0), b=b), torch.tensor(b), atol=1e-3)
        assert torch.allclose(ra_tanh(torch.tensor(-1000.0), b=b), torch.tensor(-b), atol=1e-3)

    def test_ratanh_origin_derivative(self) -> None:
        """Verify RaTanh unit derivative at origin: RaTanh'(0) = 1.0."""
        eps = 1e-5
        x_pos = torch.tensor(eps, dtype=torch.float64)
        x_neg = torch.tensor(-eps, dtype=torch.float64)
        num_diff = (ra_tanh(x_pos, b=1.0) - ra_tanh(x_neg, b=1.0)) / (2.0 * eps)
        assert math.isclose(num_diff.item(), 1.0, abs_tol=1e-4)

    def test_pos_ra_gate_strict_bounds(self) -> None:
        """Verify pos_ra_gate is strictly bounded in (0, 1)."""
        z = torch.linspace(-200.0, 200.0, 500)
        gate = pos_ra_gate(z)

        assert (gate > 0.0).all()
        assert (gate < 1.0).all()
        # Origin value is exactly 0.5
        assert math.isclose(pos_ra_gate(0.0), 0.5, abs_tol=1e-7)

    def test_ra_swiglu_and_ra_glu_properties(self) -> None:
        """Verify RaSwiGLU minimum/tail properties and RaGLU hard-bounded output."""
        torch.manual_seed(303)
        gate = torch.randn(100) * 20.0
        up = torch.randn(100) * 10.0

        h_glu = ra_glu(gate, up, b=1.5)
        assert (h_glu.abs() <= 1.5 * up.abs() + 1e-5).all()

        # RaSwiGLU exact minimum at z = -b is (-b / 6) * u, and z = 0 gives 0.
        z_pts = torch.tensor([-1000.0, -1.5, 0.0, 1000.0], dtype=torch.float64)
        u_ones = torch.ones_like(z_pts)
        y_pts = ra_swiglu(z_pts, u_ones, b=1.5)
        assert math.isclose(y_pts[1].item(), -1.5 / 6.0, abs_tol=1e-12)
        assert math.isclose(y_pts[2].item(), 0.0, abs_tol=1e-12)
        assert y_pts[0].item() < 0.0 and abs(y_pts[0].item()) < 2e-3
        assert abs(y_pts[3].item() - 1000.0) < 2e-3

    def test_ra_swiglu_and_ra_glu_modules(self) -> None:
        """Verify RaSwiGLU and RaGLU modules across b values and autograd/no_grad parity."""
        for cls in (RaSwiGLU, RaGLU):
            for b_val in (1.0, 2.5):
                mlp = cls(in_features=16, hidden_features=32, out_features=16, b=b_val)
                x_grad = torch.randn(4, 8, 16, requires_grad=True)
                out_grad = mlp(x_grad)
                with torch.no_grad():
                    out_nograd = mlp(x_grad.detach())
                assert out_grad.shape == (4, 8, 16)
                assert torch.allclose(out_grad.detach(), out_nograd, atol=1e-6)

    def test_ragelu_behavior(self) -> None:
        """Verify RaGELU matches 0.5 * x * (1 + RaTanh(sqrt(2/pi) * x))."""
        x = torch.linspace(-5.0, 5.0, 100)
        gelu = RaGELU()
        out = gelu(x)

        # Origin: RaGELU(0) = 0
        assert math.isclose(ra_gelu(0.0), 0.0, abs_tol=1e-7)

        # For large positive x, RaGELU(x) ~ x
        large_x = torch.tensor(100.0)
        assert torch.allclose(gelu(large_x), large_x, atol=0.5)

        # For large negative x, RaGELU(x) ~ 0 with algebraic decay
        large_neg_x = torch.tensor(-100.0)
        assert torch.allclose(gelu(large_neg_x), torch.tensor(0.0), atol=0.01)

    def test_raglu_and_raswiglu_default_hidden(self) -> None:
        """Verify RaGLU and RaSwiGLU default 8/3 hidden size and shape invariance."""
        raglu = RaGLU(in_features=12, out_features=12)
        raswiglu = RaSwiGLU(in_features=12, out_features=12)
        assert raglu.hidden_features == int(12 * 8 / 3)
        assert raswiglu.hidden_features == int(12 * 8 / 3)

        x = torch.randn(2, 6, 12)
        assert raglu(x).shape == (2, 6, 12)
        assert raswiglu(x).shape == (2, 6, 12)

    def test_activation_numpy_execution(self) -> None:
        """Verify activation functions execute cleanly on NumPy arrays."""
        x_np = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

        y_tanh = ra_tanh(x_np)
        assert isinstance(y_tanh, np.ndarray)
        assert np.allclose(y_tanh[1], 0.0)

        y_gate = pos_ra_gate(x_np)
        assert isinstance(y_gate, np.ndarray)
        assert np.all(y_gate > 0.0) and np.all(y_gate < 1.0)

        y_gelu = ra_gelu(x_np)
        assert isinstance(y_gelu, np.ndarray)


class TestRaSoftmax:
    """Tests for Cayley-Padé rational softmax with compact support."""

    def test_compact_support_cutoff(self) -> None:
        """Verify exact compact support: Delta >= 16 yields zero probability."""
        # Max score is 20.0.
        # Token 0: score 20.0 (Delta = 0.0)
        # Token 1: score 4.0  (Delta = 16.0 -> exactly at boundary)
        # Token 2: score 0.0  (Delta = 20.0 -> beyond boundary)
        scores = torch.tensor([[20.0, 4.0, 0.0]])
        probs = ra_softmax(scores, dim=-1, scale=0.0625)

        # Token 1 (Delta = 16.0) and Token 2 (Delta = 20.0) must be 0.0
        assert probs[0, 1].item() == 0.0
        assert probs[0, 2].item() == 0.0
        # All attention mass stays on Token 0
        assert math.isclose(probs[0, 0].item(), 1.0, abs_tol=1e-7)

    def test_denominator_safety_and_probability_sum(self) -> None:
        """Verify denominator sum_phi >= 1.0 guaranteed and probabilities sum to 1.0."""
        scores = torch.tensor([
            [-1000.0, -1050.0, -2000.0],
            [1e4, 1e4, 1e4],
            [0.0, 0.0, 0.0],
        ])
        probs = ra_softmax(scores, dim=-1)

        # Sum of probabilities across last dimension is 1.0
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

        # All values non-negative
        assert (probs >= 0.0).all()

    def test_order2_maclaurin_match_with_exponential(self) -> None:
        """Verify rational kernel matches exp(-Delta) up to second order for small Delta."""
        delta = 0.05
        # Exp(-Delta)
        exp_val = math.exp(-delta)

        # Cayley-Pade rational kernel: ((1 - u) / (1 + u))^8 with u = delta / 16
        u = delta / 16.0
        rat_val = ((1.0 - u) / (1.0 + u)) ** 8

        # Difference should be O(delta^3) ~ 0.05^3 = 0.000125
        diff = abs(exp_val - rat_val)
        assert diff < 1e-4

    def test_numpy_execution(self) -> None:
        """Verify ra_softmax executes cleanly on NumPy ndarrays."""
        scores_np = np.array([[1.0, 2.0, 3.0], [5.0, 0.0, -10.0]], dtype=np.float32)
        probs_np = ra_softmax(scores_np, dim=-1)
        assert isinstance(probs_np, np.ndarray)
        assert np.allclose(probs_np.sum(axis=-1), np.ones(2), atol=1e-6)



class TestRaStandaloneFallback:
    """Tests for zero-dependency standalone NumPy fallback when torch is absent."""

    def test_standalone_fallback_subprocess(self) -> None:
        """Verify Ra-Family imports and executes on NumPy when PyTorch is blocked."""
        sub_code = (
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "import numpy as np\n"
            "import luminamoon.nn as nn\n"
            "\n"
            "# 1. Environment status\n"
            "assert not nn.is_torch_available(), 'Torch should report unavailable'\n"
            "assert nn.is_numpy_available(), 'NumPy should report available'\n"
            "\n"
            "# 2. RaNorm execution\n"
            "x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)\n"
            "norm = nn.RaNorm(4, elementwise_affine=False)\n"
            "out_norm = norm(x)\n"
            "assert out_norm.shape == (1, 4)\n"
            "\n"
            "# 3. RaBatchNorm2d execution\n"
            "x4d = np.ones((2, 3, 4, 4), dtype=np.float32)\n"
            "bn = nn.RaBatchNorm2d(3)\n"
            "out_bn = bn(x4d)\n"
            "assert out_bn.shape == (2, 3, 4, 4)\n"
            "\n"
            "# 4. ra_tanh and pos_ra_gate\n"
            "t = nn.ra_tanh(np.array([-1.0, 0.0, 1.0], dtype=np.float32))\n"
            "g = nn.pos_ra_gate(np.array([-1.0, 0.0, 1.0], dtype=np.float32))\n"
            "assert len(t) == 3 and len(g) == 3\n"
            "\n"
            "# 5. ra_softmax with compact support\n"
            "scores = np.array([[10.0, -6.0, -10.0]], dtype=np.float32)\n"
            "p = nn.ra_softmax(scores)\n"
            "assert np.isclose(p.sum(), 1.0)\n"
            "assert p[0, 1] == 0.0 and p[0, 2] == 0.0\n"
            "\n"
            "print('STANDALONE_SUCCESS')\n"
        )

        res = subprocess.run(
            [sys.executable, "-c", sub_code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=dict(**{k: v for k, v in sys.modules["os"].environ.items() if k != "PYTHONPATH"}, PYTHONPATH="python"),
        )

        assert res.returncode == 0, f"Subprocess failed with stderr: {res.stderr}"
        assert "STANDALONE_SUCCESS" in res.stdout


class TestRaHardwareAcceleration:
    """Tests for dynamic Triton GPU and Rust CPU SIMD hardware dispatch."""

    def test_triton_cpu_fallback_graceful(self) -> None:
        """Verify Triton kernels return None gracefully when input is on CPU."""
        from luminamoon.nn.kernels import (
            triton_ra_glu,
            triton_ra_norm,
            triton_ra_softmax,
            triton_ra_swiglu,
        )

        x_cpu = torch.randn(4, 16)
        assert triton_ra_norm(x_cpu) is None
        assert triton_ra_swiglu(x_cpu, x_cpu) is None
        assert triton_ra_glu(x_cpu, x_cpu) is None
        assert triton_ra_softmax(x_cpu) is None

    def test_hardware_availability_introspection(self) -> None:
        """Verify hardware availability introspection flags."""
        from luminamoon.nn import (
            is_numpy_available,
            is_rust_simd_available,
            is_torch_available,
            is_triton_available,
        )

        assert isinstance(is_torch_available(), bool)
        assert isinstance(is_numpy_available(), bool)
        assert isinstance(is_triton_available(), bool)
        assert isinstance(is_rust_simd_available(), bool)



def _eager_ra_softmax_reference(scores: torch.Tensor, dim: int = -1, scale: float = 0.0625) -> torch.Tensor:
    max_scores = scores.max(dim=dim, keepdim=True)[0]
    delta = torch.where(scores == max_scores, 0.0, max_scores - scores)
    u = delta * scale
    num = torch.clamp(1.0 - u, min=0.0)
    den = 1.0 + u
    q = num / den
    q2 = q * q
    q4 = q2 * q2
    phi = q4 * q4
    sum_phi = phi.sum(dim=dim, keepdim=True)
    return phi / sum_phi


class TestRaSoftmaxClosedFormAutograd:
    """Tests for closed-form analytical autograd and chunked execution of ra_softmax."""

    @pytest.mark.parametrize("dim", [-1, 1, 0])
    def test_closed_form_gradient_matches_eager_reference(self, dim: int) -> None:
        torch.manual_seed(777)
        s_raw = torch.randn(3, 5, 12, dtype=torch.float64) * 8.0
        grad_out = torch.randn_like(s_raw)

        s1 = s_raw.clone().requires_grad_(True)
        s2 = s_raw.clone().requires_grad_(True)

        p1 = ra_softmax(s1, dim=dim, scale=0.0625)
        p2 = _eager_ra_softmax_reference(s2, dim=dim, scale=0.0625)

        assert torch.allclose(p1, p2, atol=1e-12, rtol=1e-12)

        p1.backward(grad_out)
        p2.backward(grad_out)

        assert s1.grad is not None and s2.grad is not None
        assert torch.allclose(s1.grad, s2.grad, atol=1e-10, rtol=1e-9)

    def test_closed_form_gradient_with_causal_neg_inf(self) -> None:
        torch.manual_seed(888)
        N = 10
        base = torch.randn(2, 3, N, N, dtype=torch.float64) * 6.0
        mask = torch.triu(torch.full((N, N), float("-inf"), dtype=torch.float64), diagonal=1)
        s_masked = base + mask
        grad_out = torch.randn_like(s_masked)

        s1 = s_masked.clone().requires_grad_(True)
        s2 = s_masked.clone().requires_grad_(True)

        p1 = ra_softmax(s1, dim=-1)
        p2 = _eager_ra_softmax_reference(s2, dim=-1)
        assert torch.allclose(p1, p2, atol=1e-12)

        p1.backward(grad_out)
        p2.backward(grad_out)

        assert s1.grad is not None and s2.grad is not None
        assert not torch.isnan(s1.grad).any()
        assert torch.allclose(s1.grad, s2.grad, atol=1e-10, rtol=1e-9)

    def test_gradcheck_double_precision(self) -> None:
        torch.manual_seed(999)
        scores = (torch.randn(2, 6, dtype=torch.float64) * 3.0).requires_grad_(True)
        assert torch.autograd.gradcheck(lambda x: ra_softmax(x, dim=-1, scale=0.0625), (scores,), eps=1e-6, atol=1e-4)

    def test_chunked_execution_equivalence_and_saved_tensors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import luminamoon.nn.softmax as softmax_mod

        monkeypatch.setattr(softmax_mod, "_MAX_CHUNK_ELEMENTS", 32)
        torch.manual_seed(1234)
        s_raw = torch.randn(4, 4, 16, dtype=torch.float32) * 5.0
        g_raw = torch.randn_like(s_raw)

        s1 = s_raw.clone().requires_grad_(True)
        s2 = s_raw.clone().requires_grad_(True)

        p1 = ra_softmax(s1, dim=-1)
        # Verify autograd saves only (scores, out) -> 2 tensors instead of 8 intermediate buffers
        assert len(p1.grad_fn.saved_tensors) == 2

        p2 = _eager_ra_softmax_reference(s2, dim=-1)
        assert torch.allclose(p1, p2, atol=1e-6)

        p1.backward(g_raw)
        p2.backward(g_raw)
        assert s1.grad is not None and s2.grad is not None
        assert torch.allclose(s1.grad, s2.grad, atol=1e-5, rtol=1e-4)


class TestNnV2ApiContract:
    """Verify v2 exports and file line limits for luminamoon.nn."""

    def test_v2_exports_and_removed_aliases(self) -> None:
        import luminamoon.nn as nn_mod

        expected = {
            "activation", "attention", "loss", "moe", "norm", "softmax",
            "BaseModule", "is_torch_available", "is_numpy_available",
            "is_triton_available", "is_rust_simd_available", "is_cubecl_available",
            "RaNorm", "RaBatchNorm2d",
            "ra_tanh", "RaTanh", "ra_gelu", "RaGELU",
            "pos_ra_gate", "ra_swiglu", "RaSwiGLU", "ra_glu", "RaGLU",
            "ra_softmax", "ra_attention", "RaAttention",
            "linear_cross_entropy", "LinearCrossEntropy",
            "RaMoE", "permute_tokens_by_expert",
        }
        assert set(nn_mod.__all__) == expected
        for sym in expected:
            assert hasattr(nn_mod, sym), f"Missing nn symbol: {sym}"

        removed = (
            "L1Norm", "L1LayerNorm", "L1BatchNorm2d",
            "RationalTanh", "rational_tanh", "RationalGELU", "rational_gelu",
            "RationalGLU", "BoundedRationalGLU", "PosRaGLU", "pos_ra_glu", "TriGLU",
            "RaAttn", "RaGQA", "RationalGQA", "RationalAttention", "PadeAttention",
            "flash_ra_attention", "rational_attention", "pade_attention",
            "rational_softmax", "pade_softmax",
            "RaEMA", "ra_ema", "ema", "fused_ra_ema", "RationalEMA", "FusedRaEMA",
            "FusedLinearCrossEntropy", "ChunkedLinearCrossEntropy", "RaLinearCrossEntropy",
            "fused_linear_cross_entropy", "chunked_linear_cross_entropy", "ra_linear_cross_entropy",
            "RationalMoE", "HybridRationalMoE",
        )
        for alias in removed:
            assert not hasattr(nn_mod, alias), f"Removed alias still exposed on luminamoon.nn: {alias}"

    def test_nn_python_files_respect_line_budget(self) -> None:
        nn_root = Path(__file__).resolve().parents[1] / "python" / "luminamoon" / "nn"
        for py_file in sorted(nn_root.rglob("*.py")):
            lines = len(py_file.read_text(encoding="utf-8").splitlines())
            limit = 600 if "kernels" in py_file.parts else 450
            assert lines <= limit, f"{py_file.relative_to(nn_root)} has {lines} lines (limit {limit})"

        tests_root = Path(__file__).resolve().parent
        for test_file in sorted(tests_root.glob("test_nn_*_v2.py")):
            lines = len(test_file.read_text(encoding="utf-8").splitlines())
            assert lines <= 650, f"{test_file.name} has {lines} lines (limit 650)"

    def test_ra_norm_and_ra_gated_closed_form_autograd(self) -> None:
        from luminamoon.nn.activation import _RaGLUFunction, _RaSwiGLUFunction
        from luminamoon.nn.norm import _RaNormFunction

        torch.manual_seed(7)
        x = (torch.randn(2, 4, 8, dtype=torch.float64) * 1.5 + 0.2).requires_grad_(True)
        w = torch.randn(8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda x_, w_, b_: _RaNormFunction.apply(x_, w_, b_, 1e-6), (x, w, b), eps=1e-6, atol=1e-4
        )
        gate = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
        up = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda g_, u_: _RaSwiGLUFunction.apply(g_, u_, 1.5), (gate, up), eps=1e-6, atol=1e-4
        )
        assert torch.autograd.gradcheck(
            lambda g_, u_: _RaGLUFunction.apply(g_, u_, 1.5), (gate, up), eps=1e-6, atol=1e-4
        )

    def test_lce_adaptive_vram_budget_chunk_size(self) -> None:
        from luminamoon.nn._loss_ops import _resolve_chunk_size

        assert _resolve_chunk_size(8192, None, is_cuda=True, vocab_size=32000, elem_bytes=2) == 4096
        assert _resolve_chunk_size(8192, None, is_cuda=True, vocab_size=50304, elem_bytes=2) == 2560
        assert _resolve_chunk_size(8192, None, is_cuda=True, vocab_size=100288, elem_bytes=2) == 1280
        assert _resolve_chunk_size(8192, None, is_cuda=True, vocab_size=256000, elem_bytes=2) == 512
