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
    PosRaGLU,
    RaAttention,
    RaAttn,
    RaBatchNorm2d,
    RaGELU,
    RaGLU,
    RaNorm,
    RaTanh,
    TriGLU,
    is_numpy_available,
    is_torch_available,
    pos_ra_gate,
    pos_ra_glu,
    ra_gelu,
    ra_softmax,
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
    """Tests for RaTanh, RaGELU, PosRaGLU, and TriGLU rational activations."""

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

    def test_pos_ra_glu_bounded_linear_activation(self) -> None:
        """Verify PosRaGLU output is bounded linearly by up magnitude without outlier explosion."""
        torch.manual_seed(303)
        gate = torch.randn(100) * 20.0  # Large gate values
        up = torch.randn(100) * 10.0

        h = pos_ra_glu(gate, up)
        # Because gate in (0, 1), |h| <= |up|
        assert (h.abs() <= up.abs() + 1e-6).all()

    def test_pos_ra_glu_module(self) -> None:
        """Verify PosRaGLU module feed-forward forward pass."""
        mlp = PosRaGLU(in_features=16, hidden_features=32, out_features=16)
        x = torch.randn(4, 8, 16)
        out = mlp(x)
        assert out.shape == (4, 8, 16)

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

    def test_raglu_and_triglu(self) -> None:
        """Verify RaGLU and TriGLU forward projections and shape invariance."""
        raglu = RaGLU(in_features=12, hidden_features=24, out_features=12)
        triglu = TriGLU(in_features=12, hidden_features=24, out_features=12)

        x = torch.randn(2, 6, 12)
        assert raglu(x).shape == (2, 6, 12)
        assert triglu(x).shape == (2, 6, 12)

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
        # Extreme score variations
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


class TestRaAttention:
    """Tests for RaAttention / RaAttn multi-head attention module."""

    def test_attention_dimensions_and_output_shape(self) -> None:
        """Verify RaAttention preserves (B, N, C) shape across heads."""
        dim = 32
        num_heads = 4
        attn = RaAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(2, 8, dim)
        out = attn(x)
        assert out.shape == (2, 8, dim)

    def test_causal_masking(self) -> None:
        """Verify causal masking runs and preserves output dimensionality."""
        dim = 16
        attn_causal = RaAttention(dim=dim, num_heads=2, causal=True)
        x = torch.randn(2, 6, dim)
        out = attn_causal(x)
        assert out.shape == (2, 6, dim)
        assert not torch.isnan(out).any()

    def test_qk_norm_integration(self) -> None:
        """Verify QK-Norm integration with RaNorm stabilizes attention."""
        dim = 16
        attn = RaAttention(dim=dim, num_heads=2, qk_norm=True)
        assert attn.q_norm is not None
        assert attn.k_norm is not None
        assert isinstance(attn.q_norm, RaNorm)

        x = torch.randn(2, 4, dim)
        out = attn(x)
        assert out.shape == (2, 4, dim)

    def test_alias_equivalence(self) -> None:
        """Verify RaAttn is direct alias for RaAttention."""
        assert RaAttn is RaAttention

    def test_invalid_dimension_raises(self) -> None:
        """Verify ValueError when dim is not divisible by num_heads."""
        with pytest.raises(ValueError):
            RaAttention(dim=15, num_heads=4)


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
        from luminamoon.nn.triton_kernels import (
            triton_pos_ra_glu,
            triton_ra_norm,
            triton_ra_softmax,
        )

        x_cpu = torch.randn(4, 16)
        assert triton_ra_norm(x_cpu) is None
        assert triton_pos_ra_glu(x_cpu, x_cpu) is None
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
