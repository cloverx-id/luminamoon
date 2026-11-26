from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    if any((ROOT_PYTHON / "luminamoon" / "optimizer").glob("_luminamoon_optimizer_rs*")):
        sys.path.insert(0, str(ROOT_PYTHON))
    else:
        sys.path.append(str(ROOT_PYTHON))

import numpy as np
import pytest
import torch
import torch.nn as nn

from luminamoon import AutoTune, LuminaV
from luminamoon.nn import RaMoE
from luminamoon.optimizer.backends import reset_autotune_cache


class _TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.ff = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.head(self.ff(x + a))


class _TinyCNNWithBNAndDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.25)
        self.fc = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn(self.conv(x)))
        h = self.drop(h).mean(dim=(2, 3))
        return self.fc(h)


class _TinyViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 64, kernel_size=4, stride=4)
        self.ln = nn.LayerNorm(64)
        self.attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch(x).flatten(2).transpose(1, 2)
        h = self.ln(t)
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.head((t + a).mean(dim=1))


def test_autotune_architecture_precision_and_memory_priors() -> None:
    reset_autotune_cache()

    # 1. FP32 CNN
    cnn = _TinyCNNWithBNAndDropout()
    t_cnn = AutoTune(cnn, tune_hardware=False, verbose=False)
    assert t_cnn.profile.architecture == "cnn"
    assert t_cnn.profile.has_batchnorm is True
    assert t_cnn.config["stochastic_rounding"] is False
    assert t_cnn.config["master_weights"] == "none"
    assert t_cnn.sources["lr"] == "prior"
    assert t_cnn.config["lr"] > 3e-4

    # 2. BF16 Transformer -> CWD + SR enabled, master_weights='none'
    lm = _TinyTransformer().to(dtype=torch.bfloat16)
    t_lm = AutoTune(lm, tune_hardware=False, verbose=False)
    assert t_lm.profile.architecture == "transformer"
    assert t_lm.profile.has_bf16 is True
    assert t_lm.config["cautious_weight_decay"] is True
    assert t_lm.config["stochastic_rounding"] is True
    assert t_lm.config["master_weights"] == "none"
    assert t_lm.config["weight_decay"] == pytest.approx(0.1)

    # 3. FP16 ViT -> master_weights='semi', tighter bound_ratio=0.03
    vit = _TinyViT().to(dtype=torch.float16)
    t_vit = AutoTune(vit, tune_hardware=False, verbose=False)
    assert t_vit.profile.architecture == "vit"
    assert t_vit.profile.has_fp16 is True
    assert t_vit.config["master_weights"] == "semi"
    assert t_vit.config["bound_ratio"] == pytest.approx(0.03)

    # 4. MoE -> enables RTV on dual buffer and tighter bound_ratio
    moe = RaMoE(in_features=32, hidden_features=64, num_experts=4, top_k=2)
    t_moe = AutoTune(moe, tune_hardware=False, verbose=False)
    assert t_moe.profile.has_moe is True
    assert t_moe.config["variance_type"] == "rtv"
    assert t_moe.config["bound_ratio"] == pytest.approx(0.03)

    # 5. Low-memory priority selects single-buffer Triangle 't1'
    t_low = AutoTune(cnn, memory_priority="low_memory", tune_hardware=False, verbose=False)
    assert t_low.config["buffer"] == "t1"


def test_autotune_empirical_loss_probe_bit_exact_four_part_reset_and_verbose_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reset_autotune_cache()
    torch.manual_seed(123)
    model = _TinyCNNWithBNAndDropout()
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 4, (8,))
    criterion = nn.CrossEntropyLoss()

    # Seed pre-existing grads & specific training mode to confirm complete 4-part restoration
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.125)
    model.train(True)

    orig_params = [p.detach().clone() for p in model.parameters()]
    orig_buffers = [b.detach().clone() for b in model.buffers()]
    orig_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    orig_rng = torch.random.get_rng_state().clone()

    tune = AutoTune(
        model,
        loss_fn=lambda m: criterion(m(x), y),
        lr_steps=4,
        steps=5,
        measure_last=1,
        strategy="full",
        use_cache=False,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "LuminaV AutoTune Summary" in out
    assert "Active Setup (Parameter -> Value [Source]):" in out
    assert "[empirical]" in out
    assert "[fastest]" in out

    assert tune.lr_probe_results is not None
    assert tune.lr_probe_results["status"] == "ok"
    assert tune.sources["lr"] == "empirical"
    assert tune.sources["variance_type"] == "empirical"
    assert len(tune.lr_probe_results["trials"]) == 5

    # CRITICAL: Bit-exact restoration of parameters, BatchNorm buffers, pre-existing grads, training mode, and RNG!
    assert model.training is True
    for p, w0 in zip(model.parameters(), orig_params):
        assert torch.equal(p.detach(), w0), "Parameters mutated by empirical AutoTune!"
    for b, b0 in zip(model.buffers(), orig_buffers):
        assert torch.equal(b.detach(), b0), "BatchNorm buffers mutated by empirical AutoTune!"
    for p, g0 in zip(model.parameters(), orig_grads):
        assert p.grad is not None and torch.equal(p.grad.detach(), g0), "Original gradients not restored!"
    assert torch.equal(torch.random.get_rng_state(), orig_rng), "PyTorch RNG state not restored!"


def test_autotune_empirical_probe_detects_gradient_spikes_and_enables_rtv() -> None:
    torch.manual_seed(9)
    # 128-element linear layer where one coordinate has 50x scale -> triggers heavy-tailed gradient spike (>15x RMS)
    w = nn.Parameter(torch.randn(128))
    scale = torch.ones(128)
    scale[0] = 60.0

    # Zero-argument loss_fn lambda also supported
    tune = AutoTune(
        [w],
        loss_fn=lambda: ((w * scale) ** 2).sum(),
        lr_steps=3,
        buffer=2,
        tune_hardware=False,
        verbose=False,
    )
    assert tune.lr_probe_results is not None
    assert tune.lr_probe_results["spike_detected"] is True
    assert tune.config["variance_type"] == "rtv"
    assert tune.sources["variance_type"] == "empirical"


def test_autotune_luminav_overrides_state_dict_weights_only_and_selective_params(tmp_path: Path) -> None:
    reset_autotune_cache()
    torch.manual_seed(42)
    model = _TinyCNNWithBNAndDropout()
    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    criterion = nn.CrossEntropyLoss()

    tune = AutoTune(
        model,
        loss_fn=lambda m: criterion(m(x), y),
        lr_steps=3,
        steps=5,
        measure_last=1,
        use_cache=False,
        verbose=False,
    )

    # 1. Full plug-in overrides defaults while keeping execution='auto' portable
    opt = LuminaV(model.parameters(), tuning=tune)
    grp = opt.param_groups[0]
    assert grp["lr"] == pytest.approx(tune.config["lr"])
    assert grp["weight_decay"] == pytest.approx(tune.config["weight_decay"])
    assert grp["execution"] == "auto"
    assert opt.resolved_execution == tune.execution

    opt.zero_grad(set_to_none=True)
    criterion(model(x), y).backward()
    opt.step()
    assert opt.last_backend is not None

    # 2. Save state_dict -> verify weights_only=True safety, JSON serialization, and full provenance persistence
    sd = opt.state_dict()
    assert "luminav_tuning" in sd
    meta = sd["luminav_tuning"]
    json.dumps(meta)  # Strictly JSON-serializable primitives
    assert meta["tuning_applied"] is True
    assert meta["profile"]["architecture"] == "cnn"
    assert meta["effective_config"]["lr"] == pytest.approx(tune.config["lr"])
    assert meta["hyperparameter_sources"]["lr"] == "empirical"
    assert meta["resolved_execution"] == tune.execution
    assert meta["last_backend"] == opt.last_backend

    ckpt_path = tmp_path / "opt_tuned.pt"
    torch.save(sd, ckpt_path)
    loaded_sd = torch.load(ckpt_path, weights_only=True)

    opt_restored = LuminaV(model.parameters())
    opt_restored.load_state_dict(loaded_sd)
    assert opt_restored.param_groups[0]["lr"] == pytest.approx(tune.config["lr"])
    assert opt_restored.tuning_info is not None
    assert opt_restored.tuning_info["profile"]["architecture"] == "cnn"
    assert opt_restored.tuning_info["hyperparameter_sources"]["lr"] == "empirical"
    assert opt_restored._resolved_auto_exec == {}  # Reset for cross-machine portability

    # 3. Explicit user override (even when equal to DEFAULT_LR=3e-4 or DEFAULT_BUFFER=2!) is preserved truthfully
    opt_user = LuminaV(model.parameters(), lr=3.0e-4, buffer="t1", tuning=tune)
    assert opt_user.param_groups[0]["lr"] == pytest.approx(3.0e-4)
    assert opt_user.param_groups[0]["buffer"] == "t1"
    sd_user = opt_user.state_dict()
    assert sd_user["luminav_tuning"]["effective_config"]["lr"] == pytest.approx(3.0e-4)
    assert sd_user["luminav_tuning"]["hyperparameter_sources"]["lr"] == "user_override"
    assert sd_user["luminav_tuning"]["hyperparameter_sources"]["buffer"] == "user_override"

    # 3b. Invalid loss_fn contracts fail fast while restoring parameters bit-exactly
    w_before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="scalar"):
        AutoTune(model, loss_fn=lambda m: m(x), lr_steps=2, tune_hardware=False, verbose=False)
    with pytest.raises(ValueError, match="requires_grad=False"):
        AutoTune(model, loss_fn=lambda m: torch.tensor(1.0), lr_steps=2, tune_hardware=False, verbose=False)
    for p, w0 in zip(model.parameters(), w_before):
        assert torch.equal(p.detach(), w0)

    # 4. Selective tune_params restricts which defaults AutoTune overrides
    tune_sel = AutoTune(model, tune_params=("lr",), tune_hardware=False, verbose=False)
    assert set(tune_sel.config.keys()) == {"lr"}
    opt_sel = LuminaV(model.parameters(), tuning=tune_sel)
    assert opt_sel.param_groups[0]["lr"] == pytest.approx(tune_sel.config["lr"])
    assert opt_sel.param_groups[0]["weight_decay"] == pytest.approx(8.0e-2)  # LuminaV default untouched

    # 5. Standalone NumPy state_dict also preserves luminav_tuning
    arr = np.ones(32, dtype=np.float32)
    t_np = AutoTune([arr], tune_hardware=False, verbose=False)
    opt_np = LuminaV([arr], tuning=t_np)
    opt_np.step(grads=[np.full_like(arr, 0.05)])
    sd_np = opt_np.state_dict()
    assert sd_np["luminav_tuning"]["tuning_applied"] is True
    opt_np2 = LuminaV([arr.copy()])
    opt_np2.load_state_dict(sd_np)
    assert opt_np2.tuning_info is not None and opt_np2.tuning_info["tuning_applied"] is True


def test_autotune_optimizer_agnostic_adamw_and_sgd() -> None:
    torch.manual_seed(77)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    x = torch.randn(8, 16)
    y = torch.randn(8, 4)

    # 1. Direct AdamW target skips LuminaV hardware bench and runs empirical probe with AdamW
    tune_adam = AutoTune(
        model,
        loss_fn=lambda m: nn.functional.mse_loss(m(x), y),
        optimizer_cls=torch.optim.AdamW,
        lr_steps=3,
        verbose=False,
    )
    assert tune_adam.optimizer_name == "AdamW"
    assert tune_adam.results is None
    assert tune_adam.sources["lr"] == "empirical"
    assert "buffer" not in tune_adam.config
    assert "execution" not in tune_adam.config

    opt_adam = tune_adam.build(torch.optim.AdamW, model.parameters())
    assert isinstance(opt_adam, torch.optim.AdamW)
    opt_adam.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt_adam.step()

    # 2. Direct SGD target maps momentum cleanly and builds a working SGD optimizer
    tune_sgd = AutoTune(model, optimizer_cls=torch.optim.SGD, verbose=False)
    sgd_kwargs = tune_sgd.for_optimizer(torch.optim.SGD)
    assert "lr" in sgd_kwargs and "momentum" in sgd_kwargs and "betas" not in sgd_kwargs
    opt_sgd = tune_sgd.build(torch.optim.SGD, model.parameters())
    assert isinstance(opt_sgd, torch.optim.SGD)

    # 3. Default LuminaV AutoTune can still export compatible kwargs for AdamW
    tune_default = AutoTune(model, tune_hardware=False, verbose=False)
    adam_from_lum = tune_default.for_optimizer(torch.optim.AdamW, eps=1e-6)
    assert set(adam_from_lum.keys()) <= {"lr", "betas", "weight_decay", "eps"}
    assert adam_from_lum["eps"] == pytest.approx(1e-6)
    opt_adam2 = torch.optim.AdamW(model.parameters(), **adam_from_lum)
    assert isinstance(opt_adam2, torch.optim.AdamW)


def test_execution_auto_benchmarks_without_autotune_and_saves_state_dict() -> None:
    reset_autotune_cache()
    p = nn.Parameter(torch.randn(64, 64))
    p.grad = torch.full_like(p, 0.05)
    opt = LuminaV([p], lr=1e-2)  # Default execution="auto", no AutoTune
    assert opt.param_groups[0]["execution"] == "auto"

    opt.step()
    assert opt.param_groups[0]["execution"] == "auto"
    assert opt.last_backend is not None
    sd = opt.state_dict()
    assert "luminav_tuning" in sd
    assert sd["luminav_tuning"]["last_backend"] == opt.last_backend
    assert sd["luminav_tuning"]["effective_config"]["execution"] == "auto"


def test_autotune_edge_cases_generators_neg_loss_var_kw_and_mismatched_preseed() -> None:
    reset_autotune_cache()
    torch.manual_seed(17)
    m = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(4, 16)
    y = torch.randn(4, 4)

    # 1. One-shot generators model.parameters() and param-group generators must not exhaust between stages
    t_gen = AutoTune(m.parameters(), steps=4, measure_last=1, use_cache=False, verbose=False)
    assert t_gen.execution is not None
    t_grp = AutoTune([{"params": m.parameters(), "lr": 1e-3}], steps=4, measure_last=1, use_cache=False, verbose=False)
    assert t_grp.execution is not None and t_grp.sampled_params == 4

    # 2. Negative strictly decreasing loss must not trigger false oscillation penalties
    w_neg = nn.Parameter(torch.zeros(16))
    t_neg = AutoTune(
        [w_neg],
        loss_fn=lambda: -100.0 + 0.5 * ((w_neg - 1.0) ** 2).sum(),
        lr_steps=4,
        tune_hardware=False,
        verbose=False,
    )
    assert t_neg.lr_probe_results is not None
    assert all(int(tr["oscillations"]) == 0 for tr in t_neg.lr_probe_results["trials"])

    # 3. External **kwargs-accepting optimizer works cleanly with loss_fn & summary()
    class _VarKwOpt(torch.optim.Optimizer):
        def __init__(self, params: Any, **kwargs: Any) -> None:
            super().__init__(params, {"lr": float(kwargs.get("lr", 1e-3))})

        @torch.no_grad()
        def step(self, closure: Any = None) -> Any:
            for grp in self.param_groups:
                for p in grp["params"]:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-grp["lr"])
            return None

    t_vkw = AutoTune(
        m, loss_fn=lambda net: nn.functional.mse_loss(net(x), y),
        optimizer_cls=_VarKwOpt, lr_steps=2, verbose=False,
    )
    assert "lr" in t_vkw.config and "Empirical Probe" in t_vkw.summary()

    # 4. Single-buffer spike probe must NOT claim variance_type='rtv' was enabled
    w_sp = nn.Parameter(torch.randn(128))
    sc = torch.ones(128)
    sc[0] = 60.0
    t_s1 = AutoTune([w_sp], loss_fn=lambda: ((w_sp * sc) ** 2).sum(), buffer="t1", lr_steps=3, tune_hardware=False, verbose=False)
    assert t_s1.config["variance_type"] == "standard"
    assert not any("enabled variance_type='rtv'" in n for n in t_s1.notes)

    # 5. Non-finite lr_multipliers & unknown LuminaV kwargs fail fast cleanly
    with pytest.raises(ValueError, match="lr_multipliers"):
        AutoTune([w_neg], loss_fn=lambda: (w_neg ** 2).sum(), lr_multipliers=(0.5, float("inf")), tune_hardware=False, verbose=False)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        AutoTune([w_neg], tune_hardware=False, verbose=False, bogus_kwarg=123)

    # 6. Pre-seeded unavailable backend (e.g. cubecl/triton from another host) re-benchmarks instead of crashing execution='auto'
    hw = LuminaV.detect_hardware()
    if not hw.get("cubecl_available", False):
        t_gen.execution = "cubecl"
        opt_safe = LuminaV(m.parameters(), tuning=t_gen)
        # Empty-grad step must not trigger auto-benchmark
        opt_safe.step()
        assert opt_safe._resolved_auto_exec == {}
        for p in m.parameters():
            p.grad = torch.full_like(p, 0.05)
        opt_safe.step()
        assert opt_safe.last_backend is not None and opt_safe.resolved_execution != "cubecl"


def test_autotune_new_modules_respect_line_budget() -> None:
    for rel, lim in (
        ("luminamoon/optimizer/_model_profile.py", 450),
        ("luminamoon/optimizer/_lr_probe.py", 450),
        ("luminamoon/optimizer/autotune.py", 450),
        ("luminamoon/optimizer/luminav.py", 450),
        ("luminamoon/optimizer/state.py", 450),
        ("luminamoon/optimizer/backends/autotune.py", 450),
        ("luminamoon/optimizer/backends/probe.py", 450),
        ("luminamoon/optimizer/backends/gpu.py", 450),
    ):
        n = len((ROOT_PYTHON / rel).read_text(encoding="utf-8").splitlines())
        assert n <= lim, f"{rel} has {n} lines (limit {lim})"
