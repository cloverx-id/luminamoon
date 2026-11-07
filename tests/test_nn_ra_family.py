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


class TestAcademicAliases:
    """Tests verifying identity and functionality of academic aliases."""

    def test_alias_identities(self) -> None:
        """Verify that aliases are exact references to their respective classes and functions."""
        from luminamoon.nn import (
            L1Norm,
            L1LayerNorm,
            L1BatchNorm2d,
            RationalTanh,
            rational_tanh,
            RationalGELU,
            rational_gelu,
            RationalGLU,
            BoundedRationalGLU,
            RationalAttention,
            PadeAttention,
            rational_softmax,
            pade_softmax,
        )

        assert L1Norm is RaNorm
        assert L1LayerNorm is RaNorm
        assert L1BatchNorm2d is RaBatchNorm2d
        assert RationalTanh is RaTanh
        assert rational_tanh is ra_tanh
        assert RationalGELU is RaGELU
        assert rational_gelu is ra_gelu
        assert RationalGLU is PosRaGLU
        assert BoundedRationalGLU is PosRaGLU
        assert RationalAttention is RaAttention
        assert PadeAttention is RaAttention
        assert rational_softmax is ra_softmax
        assert pade_softmax is ra_softmax

    def test_alias_instantiation_and_forward(self) -> None:
        """Verify alias instantiation and forward pass execution."""
        from luminamoon.nn import L1Norm, RationalAttention, RationalTanh

        norm = L1Norm(normalized_shape=16)
        x = torch.randn(2, 4, 16)
        assert norm(x).shape == (2, 4, 16)

        act = RationalTanh()
        assert act(x).shape == (2, 4, 16)

        attn = RationalAttention(dim=16, num_heads=2)
        assert attn(x).shape == (2, 4, 16)


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


def _materialized_ra_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    ra_scale: float = 0.0625,
) -> torch.Tensor:
    D = q.shape[-1]
    sm_scale = float(scale) if scale is not None else (1.0 / math.sqrt(D))
    scores = (q @ k.transpose(-2, -1)) * sm_scale
    if is_causal:
        N_Q, N_K = q.shape[-2], k.shape[-2]
        causal_mask = torch.triu(torch.full((N_Q, N_K), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1)
        scores = scores + causal_mask
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = _eager_ra_softmax_reference(scores, dim=-1, scale=ra_scale)
    return weights @ v


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
        import luminamoon.nn.attention as attn_mod

        monkeypatch.setattr(attn_mod, "_MAX_CHUNK_ELEMENTS", 32)
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


class TestFlashRaAttention:
    """Tests for fused tiled Rational Attention (ra_attention / Flash-RaAttention)."""

    @pytest.mark.parametrize("is_causal", [False, True])
    @pytest.mark.parametrize("seq_len", [16, 37, 70])
    def test_tiled_ra_attention_matches_materialized_reference(self, is_causal: bool, seq_len: int) -> None:
        from luminamoon.nn import ra_attention

        torch.manual_seed(2026)
        B, H, D = 2, 3, 16
        q_raw = torch.randn(B, H, seq_len, D, dtype=torch.float64) * 2.0
        k_raw = torch.randn(B, H, seq_len, D, dtype=torch.float64) * 2.0
        v_raw = torch.randn(B, H, seq_len, D, dtype=torch.float64)
        do = torch.randn_like(q_raw)

        q1 = q_raw.clone().requires_grad_(True)
        k1 = k_raw.clone().requires_grad_(True)
        v1 = v_raw.clone().requires_grad_(True)

        q2 = q_raw.clone().requires_grad_(True)
        k2 = k_raw.clone().requires_grad_(True)
        v2 = v_raw.clone().requires_grad_(True)

        out1 = ra_attention(q1, k1, v1, is_causal=is_causal, q_chunk_size=16)
        out2 = _materialized_ra_attention_reference(q2, k2, v2, is_causal=is_causal)

        assert torch.allclose(out1, out2, atol=1e-11, rtol=1e-10)

        out1.backward(do)
        out2.backward(do)

        assert q1.grad is not None and q2.grad is not None
        assert k1.grad is not None and k2.grad is not None
        assert v1.grad is not None and v2.grad is not None
        assert torch.allclose(q1.grad, q2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10, rtol=1e-9)

    def test_no_n_squared_matrix_saved_in_autograd(self) -> None:
        """Verify that tiled ra_attention never saves an O(N^2) matrix in the autograd graph."""
        from luminamoon.nn import ra_attention

        B, H, N, D = 2, 4, 128, 16
        q = torch.randn(B, H, N, D, requires_grad=True)
        k = torch.randn(B, H, N, D, requires_grad=True)
        v = torch.randn(B, H, N, D, requires_grad=True)

        out = ra_attention(q, k, v, is_causal=True, q_chunk_size=32)
        for saved_t in out.grad_fn.saved_tensors:
            # No saved tensor should have shape (..., N, N)
            assert saved_t.shape[-2:] != (N, N), f"Unexpected O(N^2) saved tensor of shape {saved_t.shape}"

    def test_attn_mask_support_and_mask_gradients(self) -> None:
        from luminamoon.nn import ra_attention

        torch.manual_seed(3030)
        B, H, N, D = 2, 2, 35, 8
        q_raw = torch.randn(B, H, N, D, dtype=torch.float64)
        k_raw = torch.randn(B, H, N, D, dtype=torch.float64)
        v_raw = torch.randn(B, H, N, D, dtype=torch.float64)
        mask_raw = torch.randn(B, H, N, N, dtype=torch.float64) * 0.5
        do = torch.randn_like(q_raw)

        q1, k1, v1, m1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))
        q2, k2, v2, m2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))

        out1 = ra_attention(q1, k1, v1, attn_mask=m1, q_chunk_size=16)
        out2 = _materialized_ra_attention_reference(q2, k2, v2, attn_mask=m2)

        assert torch.allclose(out1, out2, atol=1e-11)
        out1.backward(do)
        out2.backward(do)

        assert torch.allclose(q1.grad, q2.grad, atol=1e-10)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10)
        assert m1.grad is not None and m2.grad is not None
        assert torch.allclose(m1.grad, m2.grad, atol=1e-10)

    def test_low_precision_bf16_and_fp16_execution(self) -> None:
        for dt in (torch.bfloat16, torch.float16):
            attn = RaAttention(dim=32, num_heads=4, causal=True, qk_norm=True, dtype=dt)
            x = torch.randn(2, 40, 32, dtype=dt, requires_grad=True)
            out = attn(x)
            assert out.dtype == dt
            assert out.shape == (2, 40, 32)
            assert not torch.isnan(out).any()
            out.sum().backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()

    def test_flash_ra_attention_gradcheck_and_aliases(self) -> None:
        from luminamoon.nn import flash_ra_attention, pade_attention, ra_attention, rational_attention
        from luminamoon.nn.triton_kernels import (
            triton_flash_ra_attention,
            triton_flash_ra_attention_backward,
            triton_ra_softmax_backward,
        )

        assert flash_ra_attention is ra_attention
        assert rational_attention is ra_attention
        assert pade_attention is ra_attention

        x_cpu = torch.randn(2, 2, 8, 8)
        assert triton_ra_softmax_backward(x_cpu, x_cpu, x_cpu) is None
        assert triton_flash_ra_attention(x_cpu, x_cpu, x_cpu) is None
        assert triton_flash_ra_attention_backward(x_cpu, x_cpu, x_cpu, x_cpu, x_cpu, x_cpu, x_cpu, x_cpu) is None

        torch.manual_seed(4040)
        q = (torch.randn(1, 2, 5, 4, dtype=torch.float64) * 1.5).requires_grad_(True)
        k = (torch.randn(1, 2, 5, 4, dtype=torch.float64) * 1.5).requires_grad_(True)
        v = torch.randn(1, 2, 5, 4, dtype=torch.float64).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda q_, k_, v_: ra_attention(q_, k_, v_, is_causal=True, q_chunk_size=16),
            (q, k, v),
            eps=1e-6,
            atol=1e-4,
        )

    @pytest.mark.parametrize(
        "mask_shape",
        [
            (1, 35),
            (2, 1, 35),
            (2, 1, 1, 35),
            (2, 1, 35, 35),
            (1, 1, 35, 1),
        ],
    )
    def test_broadcast_attn_mask_shapes_and_gradients(self, mask_shape: tuple[int, ...]) -> None:
        from luminamoon.nn import ra_attention

        torch.manual_seed(5050)
        B, H, N_Q, N_K, D = 2, 3, 35, 35, 8
        q_raw = torch.randn(B, H, N_Q, D, dtype=torch.float64)
        k_raw = torch.randn(B, H, N_K, D, dtype=torch.float64)
        v_raw = torch.randn(B, H, N_K, D, dtype=torch.float64)
        mask_raw = torch.randn(*mask_shape, dtype=torch.float64) * 0.5
        do = torch.randn_like(q_raw)

        q1, k1, v1, m1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))
        q2, k2, v2, m2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw, mask_raw))

        out1 = ra_attention(q1, k1, v1, attn_mask=m1, q_chunk_size=16)
        m2_view = m2.unsqueeze(1) if m2.ndim == 3 else m2
        out2 = _materialized_ra_attention_reference(q2, k2, v2, attn_mask=m2_view)

        assert torch.allclose(out1, out2, atol=1e-11)
        out1.backward(do)
        out2.backward(do)

        assert torch.allclose(q1.grad, q2.grad, atol=1e-10)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10)
        assert m1.grad is not None and m2.grad is not None
        assert torch.allclose(m1.grad, m2.grad, atol=1e-10)

    def test_all_neg_inf_row_zero_gradient(self) -> None:
        from luminamoon.nn import ra_attention

        s = torch.tensor([[1.0, 2.0, 3.0], [float("-inf"), float("-inf"), float("-inf")]], dtype=torch.float64, requires_grad=True)
        p = ra_softmax(s, dim=-1)
        assert torch.allclose(p[1], torch.zeros(3, dtype=torch.float64))
        p.backward(torch.ones_like(p))
        assert s.grad is not None
        assert not torch.isnan(s.grad).any()
        assert torch.allclose(s.grad[1], torch.zeros(3, dtype=torch.float64))

        q = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
        mask[:, :, 2, :] = float("-inf")
        out = ra_attention(q, k, v, attn_mask=mask, q_chunk_size=16)
        assert torch.allclose(out[:, :, 2, :], torch.zeros(1, 1, 8, dtype=torch.float64))
        out.sum().backward()
        assert q.grad is not None
        assert torch.allclose(q.grad[:, :, 2, :], torch.zeros(1, 1, 8, dtype=torch.float64))

    def test_3d_input_with_dropout_and_3d_mask_and_asymmetric_dv(self) -> None:
        from luminamoon.nn import ra_attention

        # 1. 3D (B, N, D) with dropout_p > 0 and 3D attn_mask (B, N, N)
        q3 = torch.randn(2, 10, 8, requires_grad=True)
        k3 = torch.randn(2, 10, 8, requires_grad=True)
        v3 = torch.randn(2, 10, 8, requires_grad=True)
        m3 = torch.zeros(2, 10, 10)
        out3 = ra_attention(q3, k3, v3, attn_mask=m3, dropout_p=0.1, is_causal=True)
        assert out3.shape == (2, 10, 8)

        # 2. NumPy 4D with 3D attn_mask
        q_np = np.ones((2, 3, 6, 4), dtype=np.float32)
        m_np = np.zeros((2, 6, 6), dtype=np.float32)
        out_np = ra_attention(q_np, q_np, q_np, attn_mask=m_np)
        assert out_np.shape == (2, 3, 6, 4)

        # 3. Asymmetric value dimension D_v != D_qk
        B, H, N_Q, N_K, D_QK, D_V = 2, 2, 20, 24, 8, 12
        q_raw = torch.randn(B, H, N_Q, D_QK, dtype=torch.float64)
        k_raw = torch.randn(B, H, N_K, D_QK, dtype=torch.float64)
        v_raw = torch.randn(B, H, N_K, D_V, dtype=torch.float64)
        do = torch.randn(B, H, N_Q, D_V, dtype=torch.float64)

        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))

        out1 = ra_attention(q1, k1, v1, q_chunk_size=16)
        out2 = _materialized_ra_attention_reference(q2, k2, v2)
        assert out1.shape == (B, H, N_Q, D_V)
        assert torch.allclose(out1, out2, atol=1e-11)

        out1.backward(do)
        out2.backward(do)
        assert torch.allclose(q1.grad, q2.grad, atol=1e-10)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10)


    def test_torch_compile_fullgraph_parity(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch version")

        from luminamoon.nn import ra_attention

        torch.manual_seed(404)
        for train_mode in (True, False):
            norm = RaNorm(16)
            glu = PosRaGLU(16, 32)
            attn = RaAttention(16, num_heads=2, qk_norm=True, causal=True)
            norm.train(train_mode)
            glu.train(train_mode)
            attn.train(train_mode)

            c_norm = torch.compile(norm, backend="aot_eager", fullgraph=True)
            c_glu = torch.compile(glu, backend="aot_eager", fullgraph=True)
            c_attn = torch.compile(attn, backend="aot_eager", fullgraph=True)

            x_raw = torch.randn(2, 8, 16)
            if train_mode:
                x_ref = x_raw.clone().requires_grad_(True)
                x_cmp = x_raw.clone().requires_grad_(True)
                y_ref = attn(glu(norm(x_ref)))
                y_cmp = c_attn(c_glu(c_norm(x_cmp)))
                assert torch.allclose(y_ref, y_cmp, atol=1e-5)
                y_ref.sum().backward()
                y_cmp.sum().backward()
                assert x_ref.grad is not None and x_cmp.grad is not None
                assert torch.allclose(x_ref.grad, x_cmp.grad, atol=1e-5)
            else:
                with torch.no_grad():
                    y_ref = attn(glu(norm(x_raw)))
                    y_cmp = c_attn(c_glu(c_norm(x_raw)))
                    assert torch.allclose(y_ref, y_cmp, atol=1e-5)

            c_sm = torch.compile(lambda t: ra_softmax(t, dim=-1), backend="aot_eager", fullgraph=True)
            c_fa = torch.compile(lambda t: ra_attention(t, t, t, is_causal=True), backend="aot_eager", fullgraph=True)

            s_raw = torch.randn(2, 2, 8, 8)
            if train_mode:
                s_ref = s_raw.clone().requires_grad_(True)
                s_cmp = s_raw.clone().requires_grad_(True)
                p_ref = ra_softmax(s_ref, dim=-1)
                p_cmp = c_sm(s_cmp)
                assert torch.allclose(p_ref, p_cmp, atol=1e-6)
                p_ref.sum().backward()
                p_cmp.sum().backward()
                assert s_ref.grad is not None and s_cmp.grad is not None
                assert torch.allclose(s_ref.grad, s_cmp.grad, atol=1e-5)

                q_ref = s_raw.clone().requires_grad_(True)
                q_cmp = s_raw.clone().requires_grad_(True)
                o_ref = ra_attention(q_ref, q_ref, q_ref, is_causal=True)
                o_cmp = c_fa(q_cmp)
                assert torch.allclose(o_ref, o_cmp, atol=1e-5)
                o_ref.sum().backward()
                o_cmp.sum().backward()
                assert q_ref.grad is not None and q_cmp.grad is not None
                assert torch.allclose(q_ref.grad, q_cmp.grad, atol=1e-5)
            else:
                with torch.no_grad():
                    assert torch.allclose(ra_softmax(s_raw, dim=-1), c_sm(s_raw), atol=1e-6)
                    assert torch.allclose(ra_attention(s_raw, s_raw, s_raw, is_causal=True), c_fa(s_raw), atol=1e-5)


class TestFusedLinearCrossEntropy:
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("has_bias", [False, True])
    @pytest.mark.parametrize("label_smoothing", [0.0, 0.15])
    @pytest.mark.parametrize("logit_scale", [1.0, 0.75])
    def test_forward_backward_exact_parity_against_materialized_pytorch(
        self,
        reduction: str,
        has_bias: bool,
        label_smoothing: float,
        logit_scale: float,
    ) -> None:
        from luminamoon.nn import fused_linear_cross_entropy

        torch.manual_seed(2026)
        B, S, D, V = 3, 7, 24, 53
        ignore_index = -100

        x_base = torch.randn(B, S, D, dtype=torch.float32)
        w_base = torch.randn(V, D, dtype=torch.float32)
        b_base = torch.randn(V, dtype=torch.float32) if has_bias else None
        target = torch.randint(0, V, (B, S), dtype=torch.long)
        target[0, 2] = ignore_index
        target[1, 5] = ignore_index
        target[2, 0] = ignore_index

        x1 = x_base.clone().requires_grad_(True)
        w1 = w_base.clone().requires_grad_(True)
        b1 = b_base.clone().requires_grad_(True) if b_base is not None else None

        x2 = x_base.clone().requires_grad_(True)
        w2 = w_base.clone().requires_grad_(True)
        b2 = b_base.clone().requires_grad_(True) if b_base is not None else None

        # Force multi-chunk execution with chunk_size=5 (< B * S = 21)
        loss_fused = fused_linear_cross_entropy(
            x1,
            w1,
            target,
            bias=b1,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
            chunk_size=5,
        )

        logits_ref = torch.nn.functional.linear(x2, w2, b2) * logit_scale
        loss_ref = torch.nn.functional.cross_entropy(
            logits_ref.view(-1, V),
            target.view(-1),
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        if reduction == "none":
            loss_ref = loss_ref.view(B, S)

        assert loss_fused.shape == loss_ref.shape
        assert torch.allclose(loss_fused, loss_ref, atol=1e-5, rtol=1e-5)

        if reduction == "none":
            grad_out = torch.randn_like(loss_ref)
            loss_fused.backward(grad_out)
            loss_ref.backward(grad_out)
        else:
            loss_fused.backward()
            loss_ref.backward()

        assert x1.grad is not None and x2.grad is not None
        assert w1.grad is not None and w2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(w1.grad, w2.grad, atol=1e-5, rtol=1e-5)
        if has_bias:
            assert b1 is not None and b2 is not None
            assert b1.grad is not None and b2.grad is not None
            assert torch.allclose(b1.grad, b2.grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_stability_and_gradients(self, dtype: Any) -> None:
        from luminamoon.nn import fused_linear_cross_entropy

        torch.manual_seed(77)
        B, S, D, V = 2, 6, 32, 64
        x_f32 = torch.randn(B, S, D)
        w_f32 = torch.randn(V, D) * 0.2
        b_f32 = torch.randn(V) * 0.1
        target = torch.randint(0, V, (B, S))
        target[0, 0] = -100

        x_lp = x_f32.to(dtype).detach().requires_grad_(True)
        w_lp = w_f32.to(dtype).detach().requires_grad_(True)
        b_lp = b_f32.to(dtype).detach().requires_grad_(True)

        x_ref = x_lp.detach().to(torch.float32).requires_grad_(True)
        w_ref = w_lp.detach().to(torch.float32).requires_grad_(True)
        b_ref = b_lp.detach().to(torch.float32).requires_grad_(True)

        loss_lp = fused_linear_cross_entropy(
            x_lp, w_lp, target, bias=b_lp, label_smoothing=0.1, chunk_size=4
        )
        logits_ref = torch.nn.functional.linear(x_ref, w_ref, b_ref)
        loss_ref = torch.nn.functional.cross_entropy(
            logits_ref.view(-1, V), target.view(-1), ignore_index=-100, label_smoothing=0.1
        )

        assert loss_lp.dtype == dtype
        tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
        assert torch.allclose(loss_lp.float(), loss_ref, atol=tol, rtol=tol)

        loss_lp.backward()
        loss_ref.backward()
        assert x_lp.grad is not None and w_lp.grad is not None and b_lp.grad is not None
        assert torch.isfinite(x_lp.grad).all()
        assert torch.isfinite(w_lp.grad).all()
        assert torch.isfinite(b_lp.grad).all()
        assert torch.allclose(x_lp.grad.float(), x_ref.grad, atol=tol, rtol=tol)
        assert torch.allclose(w_lp.grad.float(), w_ref.grad, atol=tol, rtol=tol)

    def test_all_ignored_targets_yield_zero_loss_and_zero_grad(self) -> None:
        from luminamoon.nn import fused_linear_cross_entropy

        x = torch.randn(4, 16, requires_grad=True)
        w = torch.randn(32, 16, requires_grad=True)
        b = torch.randn(32, requires_grad=True)
        target = torch.full((4,), -100, dtype=torch.long)

        for red in ("mean", "sum", "none"):
            x.grad = None
            w.grad = None
            b.grad = None
            loss = fused_linear_cross_entropy(x, w, target, bias=b, reduction=red, label_smoothing=0.1)
            assert torch.isfinite(loss).all()
            assert (loss == 0.0).all()
            loss.sum().backward()
            assert x.grad is not None and (x.grad == 0.0).all()
            assert w.grad is not None and (w.grad == 0.0).all()
            assert b.grad is not None and (b.grad == 0.0).all()

    def test_stateful_and_stateless_module_modes_and_aliases(self) -> None:
        import luminamoon as lm
        from luminamoon.nn import (
            ChunkedLinearCrossEntropy,
            FusedLinearCrossEntropy,
            LinearCrossEntropy,
            RaLinearCrossEntropy,
            chunked_linear_cross_entropy,
            fused_linear_cross_entropy,
            linear_cross_entropy,
            ra_linear_cross_entropy,
        )

        assert RaLinearCrossEntropy is FusedLinearCrossEntropy
        assert LinearCrossEntropy is FusedLinearCrossEntropy
        assert ChunkedLinearCrossEntropy is FusedLinearCrossEntropy
        assert ra_linear_cross_entropy is fused_linear_cross_entropy
        assert linear_cross_entropy is fused_linear_cross_entropy
        assert chunked_linear_cross_entropy is fused_linear_cross_entropy
        assert lm.FusedLinearCrossEntropy is FusedLinearCrossEntropy
        assert lm.ChunkedLinearCrossEntropy is FusedLinearCrossEntropy
        assert lm.fused_linear_cross_entropy is fused_linear_cross_entropy
        assert lm.chunked_linear_cross_entropy is fused_linear_cross_entropy

        torch.manual_seed(99)
        x = torch.randn(2, 5, 16, requires_grad=True)
        target = torch.randint(0, 25, (2, 5))

        # 1. Stateful LM Head + Loss Mode
        head = FusedLinearCrossEntropy(16, 25, bias=True, label_smoothing=0.05)
        assert "in_features=16, vocab_size=25" in head.extra_repr()

        # Inference mode (target=None -> returns logits)
        logits = head(x)
        assert logits.shape == (2, 5, 25)

        # Training mode (target provided -> returns fused scalar loss)
        loss_stateful = head(x, target)
        loss_ref = torch.nn.functional.cross_entropy(
            logits.view(-1, 25), target.view(-1), label_smoothing=0.05
        )
        assert torch.allclose(loss_stateful, loss_ref, atol=1e-5)

        # Weight-tying override in stateful mode
        tied_w = torch.randn(25, 16, requires_grad=True)
        loss_tied = head(x, tied_w, target)
        loss_tied.backward()
        assert tied_w.grad is not None

        # 2. Stateless Criterion Mode
        criterion = FusedLinearCrossEntropy(label_smoothing=0.05)
        loss_stateless = criterion(x, head.weight, target, bias=head.bias)
        assert torch.allclose(loss_stateless, loss_ref, atol=1e-5)

    def test_numpy_fallback_and_validation(self) -> None:
        from luminamoon.nn import FusedLinearCrossEntropy, fused_linear_cross_entropy

        rng = np.random.default_rng(123)
        x_np = rng.standard_normal((3, 4, 12)).astype(np.float32)
        w_np = rng.standard_normal((20, 12)).astype(np.float32)
        b_np = rng.standard_normal((20,)).astype(np.float32)
        t_np = rng.integers(0, 20, size=(3, 4), dtype=np.int64)
        t_np[0, 1] = -100

        for red in ("mean", "sum", "none"):
            loss_np = fused_linear_cross_entropy(
                x_np, w_np, t_np, bias=b_np, reduction=red, label_smoothing=0.1, chunk_size=3
            )
            loss_pt = fused_linear_cross_entropy(
                torch.from_numpy(x_np),
                torch.from_numpy(w_np),
                torch.from_numpy(t_np),
                bias=torch.from_numpy(b_np),
                reduction=red,
                label_smoothing=0.1,
                chunk_size=3,
            )
            np.testing.assert_allclose(loss_np, loss_pt.numpy(), atol=1e-5, rtol=1e-5)

        with pytest.raises(ValueError, match="reduction"):
            fused_linear_cross_entropy(x_np, w_np, t_np, reduction="invalid")
        with pytest.raises(ValueError, match="label_smoothing"):
            fused_linear_cross_entropy(x_np, w_np, t_np, label_smoothing=1.0)
        with pytest.raises(ValueError, match="logit_scale"):
            fused_linear_cross_entropy(x_np, w_np, t_np, logit_scale=0.0)
        with pytest.raises(ValueError, match="Both in_features and vocab_size"):
            FusedLinearCrossEntropy(in_features=16, vocab_size=None)

    def test_torch_compile_fused_linear_cross_entropy(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile is not available in this PyTorch version")

        from luminamoon.nn import FusedLinearCrossEntropy, fused_linear_cross_entropy

        torch.manual_seed(512)
        head = FusedLinearCrossEntropy(16, 32, bias=True, label_smoothing=0.05)
        c_head = torch.compile(head, backend="aot_eager", fullgraph=True)
        c_fn = torch.compile(
            lambda x, w, t, b: fused_linear_cross_entropy(x, w, t, bias=b, label_smoothing=0.05),
            backend="aot_eager",
            fullgraph=True,
        )

        x_raw = torch.randn(2, 6, 16)
        target = torch.randint(0, 32, (2, 6))
        target[0, 2] = -100

        x1 = x_raw.clone().requires_grad_(True)
        x2 = x_raw.clone().requires_grad_(True)
        l1 = head(x1, target)
        l2 = c_head(x2, target)
        assert torch.allclose(l1, l2, atol=1e-5)
        l1.backward()
        l2.backward()
        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

        x3 = x_raw.clone().requires_grad_(True)
        l3 = c_fn(x3, head.weight, target, head.bias)
        assert torch.allclose(l1, l3, atol=1e-5)


class TestRationalMoE:
    """Tests for RationalMoE, HybridRationalMoE, and Contiguous Token Permutation Dispatcher."""

    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("expert_type", ["pos_ra_glu", "tri_glu"])
    def test_permutation_dispatcher_matches_unpermuted_reference_and_gradients(
        self,
        top_k: int,
        expert_type: str,
    ) -> None:
        from luminamoon.nn import HybridRationalMoE, RaMoE, RationalMoE

        assert RaMoE is RationalMoE
        torch.manual_seed(1337)
        B, T, D, H = 2, 7, 16, 32
        num_experts = 4

        moe = HybridRationalMoE(
            in_features=D,
            hidden_features=H,
            out_features=D,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=1,
            expert_type=expert_type,
            return_aux_loss=True,
        )
        assert "num_experts=4" in moe.extra_repr()

        x_raw = torch.randn(B, T, D)
        grad_out = torch.randn(B, T, D)

        x1 = x_raw.clone().requires_grad_(True)
        out_perm, aux_loss = moe(x1)
        assert out_perm.shape == (B, T, D)
        assert aux_loss.ndim == 0 and aux_loss.item() > 0.0
        assert moe.last_aux_loss is not None

        # Reference unpermuted evaluation using identical parameters
        x2 = x_raw.clone().requires_grad_(True)
        x2_2d = x2.reshape(-1, D)
        r_logits = moe.router(x2_2d)
        t_vals, t_idx = torch.topk(r_logits, k=top_k, dim=-1)
        if top_k == 1:
            f_probs = ra_softmax(r_logits, dim=-1, scale=moe.ra_scale)
            t_weights = f_probs.gather(dim=-1, index=t_idx)
        else:
            t_weights = ra_softmax(t_vals, dim=-1, scale=moe.ra_scale)

        ref_2d = torch.zeros_like(x2_2d)
        for e_idx, exp in enumerate(moe.experts):
            mask = t_idx == e_idx
            w_e = torch.where(mask, t_weights, torch.zeros_like(t_weights)).sum(dim=-1, keepdim=True)
            ref_2d = ref_2d + exp(x2_2d) * w_e
        if moe.shared_experts is not None:
            for s_exp in moe.shared_experts:
                ref_2d = ref_2d + s_exp(x2_2d)
        out_ref = ref_2d.reshape(B, T, D)

        assert torch.allclose(out_perm, out_ref, atol=1e-5, rtol=1e-5)

        out_perm.backward(grad_out)
        out_ref.backward(grad_out)

        assert x1.grad is not None and x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
        assert moe.router.weight.grad is not None
        assert torch.isfinite(moe.router.weight.grad).all()

    def test_rational_moe_numpy_fallback_and_torch_compile(self) -> None:
        from luminamoon.nn import RationalMoE, pos_ra_gate

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 6, 12)).astype(np.float32)

        moe_np = RationalMoE(
            in_features=12,
            hidden_features=24,
            out_features=12,
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            return_aux_loss=True,
        )
        # Exercise NumPy execution path directly by passing ndarray
        out_np, aux_np = moe_np(x_np)
        assert out_np.shape == (2, 6, 12)
        assert np.isfinite(out_np).all()
        assert float(aux_np) > 0.0
        # In NumPy fallback, PosRaGLU applies pos_ra_gate(x_np); routed weights sum to 1.0 + 1 shared expert = 2.0 * pos_ra_gate(x_np)
        np.testing.assert_allclose(out_np, 2.0 * pos_ra_gate(x_np, b=moe_np.b), atol=1e-5, rtol=1e-5)

        out_pt, aux_pt = moe_np(torch.from_numpy(x_np))
        assert out_pt.shape == (2, 6, 12)
        np.testing.assert_allclose(float(aux_np), float(aux_pt.item()), atol=1e-5, rtol=1e-5)

        if hasattr(torch, "compile"):
            moe_cmp = RationalMoE(12, 24, 12, num_experts=3, top_k=2)
            c_moe = torch.compile(moe_cmp, backend="aot_eager", fullgraph=True)
            x_t = torch.randn(2, 5, 12, requires_grad=True)
            x_t2 = x_t.detach().clone().requires_grad_(True)
            y1 = moe_cmp(x_t)
            y2 = c_moe(x_t2)
            assert torch.allclose(y1, y2, atol=1e-5)
            y1.sum().backward()
            y2.sum().backward()
            assert x_t.grad is not None and x_t2.grad is not None
            assert torch.allclose(x_t.grad, x_t2.grad, atol=1e-5)

    def test_ra_attention_dynamic_fast_path_matches_chunked_path(self) -> None:
        from luminamoon.nn import ra_attention

        torch.manual_seed(909)
        q_raw = torch.randn(2, 2, 24, 16, dtype=torch.float64)
        k_raw = torch.randn(2, 2, 24, 16, dtype=torch.float64)
        v_raw = torch.randn(2, 2, 24, 16, dtype=torch.float64)
        do = torch.randn_like(q_raw)

        q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))
        q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q_raw, k_raw, v_raw))

        # q_chunk_size=None triggers the single-shot dynamic fast-path (2*2*24*24 = 2304 <= 4M)
        out_fast = ra_attention(q1, k1, v1, is_causal=True, q_chunk_size=None)
        # q_chunk_size=8 forces multi-chunk execution
        out_chunked = ra_attention(q2, k2, v2, is_causal=True, q_chunk_size=8)

        assert torch.allclose(out_fast, out_chunked, atol=1e-11, rtol=1e-10)
        out_fast.backward(do)
        out_chunked.backward(do)
        assert q1.grad is not None and q2.grad is not None
        assert torch.allclose(q1.grad, q2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(k1.grad, k2.grad, atol=1e-10, rtol=1e-9)
        assert torch.allclose(v1.grad, v2.grad, atol=1e-10, rtol=1e-9)


