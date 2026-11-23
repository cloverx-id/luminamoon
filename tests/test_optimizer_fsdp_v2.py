from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    if any((ROOT_PYTHON / "luminamoon" / "optimizer").glob("_luminamoon_optimizer_rs*")):
        sys.path.insert(0, str(ROOT_PYTHON))
    else:
        sys.path.append(str(ROOT_PYTHON))

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from luminamoon.optimizer import LuminaV
from luminamoon.optimizer.backends.fsdp import (
    fallback_post_step_ema,
    sync_empty_fsdp_shard,
    sync_fsdp_scratch,
    sync_fsdp_scratch_batch,
)
from luminamoon.optimizer.config import StepOptions


def _make_step_opts(
    *,
    buffer: Any = 2,
    bound_type: str = "radial",
    cautious: bool = True,
    fused_single_pass: bool = True,
    ema_decay: float = 0.80,
    fsdp_sync: bool = True,
    lr: float = 1e-2,
    bound_ratio: float = 0.05,
) -> StepOptions:
    return StepOptions(
        lr=lr,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        cautious_weight_decay=False,
        variance_type="standard",
        variance_temper=3.0,
        tau=0.8,
        lam=0.3,
        cautious=cautious,
        clamp_min=0.5,
        buffer=buffer,
        use_sr=False,
        bound=True,
        bound_type=bound_type,  # type: ignore[arg-type]
        bound_ratio=bound_ratio,
        bound_floor=1e-2,
        master_weights="none",
        execution="eager",
        fused_single_pass=fused_single_pass,
        ema_decay=ema_decay,
        fsdp_sync=fsdp_sync,
        seed=1337,
    )


def _fsdp_parity_worker(
    rank: int,
    world_size: int,
    store_path: str,
    out_dir: str,
    buffer_mode: Any,
    fused_single_pass: bool,
    full_params_data: List[torch.Tensor],
    step_grads_data: List[List[torch.Tensor]],
) -> None:
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        sharded_params: List[torch.Tensor] = []
        split_slices: List[Tuple[int, int]] = []
        for full_p in full_params_data:
            n = full_p.numel()
            half = n // world_size
            start = rank * half
            end = n if rank == world_size - 1 else (rank + 1) * half
            split_slices.append((start, end))
            sharded_params.append(full_p[start:end].clone().detach())

        opt = LuminaV(
            sharded_params,
            lr=1e-2,
            weight_decay=0.05,
            buffer=buffer_mode,
            cautious=True,
            bound=True,
            bound_type="radial",
            bound_ratio=0.05,
            stochastic_rounding=False,
            fused_single_pass=fused_single_pass,
            fsdp_sync=True,
            execution="eager",
            prewarm_jit=False,
            seed=1337,
        )
        assert opt.param_groups[0]["fsdp_sync"] is True

        for grads_for_step in step_grads_data:
            for p, full_g, (start, end) in zip(sharded_params, grads_for_step, split_slices):
                p.grad = full_g[start:end].clone()
            opt.step()

        scratches = [opt.state[p]["scratch"].clone() for p in sharded_params]
        torch.save(
            {"params": [p.detach().clone() for p in sharded_params], "scratches": scratches},
            Path(out_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _fsdp_empty_shard_worker(
    rank: int,
    world_size: int,
    store_path: str,
    out_dir: str,
    num_params: int,
    fused_single_pass: bool,
) -> None:
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        gen = torch.Generator().manual_seed(2026)
        params: List[torch.Tensor] = []
        grads_per_step: List[List[torch.Tensor]] = [[], [], []]
        for _ in range(num_params):
            full_p = torch.randn(64, generator=gen, dtype=torch.float32)
            full_g_steps = [torch.randn(64, generator=gen, dtype=torch.float32) for _ in range(3)]
            if rank == 0:
                p = full_p.clone().detach()
                for s_idx in range(3):
                    grads_per_step[s_idx].append(full_g_steps[s_idx].clone())
            else:
                # Simulate an empty FSDP shard on rank 1
                p = torch.empty((0,), dtype=torch.float32)
                for s_idx in range(3):
                    grads_per_step[s_idx].append(torch.empty((0,), dtype=torch.float32))
            params.append(p)

        opt = LuminaV(
            params,
            lr=1e-2,
            weight_decay=0.02,
            buffer="t2",
            cautious=True,
            bound=True,
            bound_type="radial",
            bound_ratio=0.05,
            stochastic_rounding=False,
            fused_single_pass=fused_single_pass,
            fsdp_sync=True,
            execution="eager",
            prewarm_jit=False,
            seed=1337,
        )
        for s_idx in range(3):
            for p, g in zip(params, grads_per_step[s_idx]):
                p.grad = g
            opt.step()

        scratches = [opt.state[p]["scratch"].clone() for p in params]
        torch.save({"scratches": scratches}, Path(out_dir) / f"empty_rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_fsdp_sync_disabled_when_distributed_uninitialized(caplog: pytest.LogCaptureFixture) -> None:
    p = torch.randn(32, dtype=torch.float32)
    with caplog.at_level("WARNING", logger="LuminaV"):
        opt = LuminaV([p], fsdp_sync=True, prewarm_jit=False)
    assert opt.param_groups[0]["fsdp_sync"] is False
    assert any("fsdp_sync=True" in rec.message for rec in caplog.records)


def test_fsdp_scratch_batch_coalesces_to_single_all_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Tuple[Tuple[int, ...], Any]] = []

    def fake_all_reduce(buf: torch.Tensor, op: Any = None) -> None:
        calls.append((tuple(buf.shape), op))
        # Simulate summing two identical ranks
        buf.mul_(2.0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    sc0 = torch.tensor([10.0, 2.5, 4.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    sc1 = torch.tensor([5.0, 1.0, 9.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    totals = sync_fsdp_scratch_batch([sc0, sc1], [20, 10])

    assert len(calls) == 1
    assert calls[0][0] == (2, 4)
    assert totals == [40.0, 20.0]
    assert torch.allclose(sc0[:3], torch.tensor([20.0, 5.0, 8.0]))
    assert torch.allclose(sc1[:3], torch.tensor([10.0, 2.0, 18.0]))
    assert sc0[3].item() == 1.0 and sc1[4].item() == 1.0


def test_fsdp_fallback_post_step_ema_fast_brake_and_empty_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    # Single-rank mock: add a virtual peer contribution inside all_reduce
    def fake_all_reduce(buf: torch.Tensor, op: Any = None) -> None:
        if buf.ndim == 1:
            buf[0] += 8.0
            buf[1] += 100.0
            buf[2] += 1.0
            buf[3] += 16.0

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    opts = _make_step_opts(buffer="t2", bound_type="radial", ema_decay=0.80, lr=0.5, bound_ratio=0.05)

    # 1. Fast-brake radial check: large u_sq_sum makes target_bound << old_bound (1.0),
    # so gamma_eff must clamp to min(0.80, 0.10) = 0.10.
    sc = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    fallback_post_step_ema(sc, 0, opts, scratch_synced=False)
    assert sc[:3].abs().max().item() == 0.0
    assert sc[5].item() == 0.0
    # m_bar = clamp(8 / 16, 0.5, 1.0) = 0.5 -> target_cautious = 2.0 -> 0.8*1 + 0.2*2 = 1.2
    assert math.isclose(sc[3].item(), 1.2, rel_tol=1e-6)
    r = (opts.lr * (math.sqrt(100.0) * 1.2)) / (opts.bound_ratio * max(math.sqrt(1.0), 1.0))
    target_bound = (1.0 + r) / (1.0 + r + r * r)
    expected_bound = 0.10 * 1.0 + 0.90 * target_bound
    assert math.isclose(sc[4].item(), expected_bound, rel_tol=1e-5)

    # 2. Empty shard on step=1 must use eff_decay=0.0 (exact overwrite)
    sc_empty = torch.tensor([99.0, 99.0, 99.0, 1.0, 1.0, 7.0], dtype=torch.float32)
    sync_empty_fsdp_shard(sc_empty, step=1, opts=opts, scratch_synced=False)
    assert sc_empty[:3].abs().max().item() == 0.0
    assert sc_empty[5].item() == 0.0
    assert math.isclose(sc_empty[3].item(), 2.0, rel_tol=1e-6)


@pytest.mark.parametrize("buffer_mode", [2, 1, "t1", "t2"])
@pytest.mark.parametrize("fused_single_pass", [True, False])
def test_fsdp_multirank_gloo_matches_unsharded_parity(
    tmp_path: Path,
    buffer_mode: Any,
    fused_single_pass: bool,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    gen = torch.Generator().manual_seed(42)
    full_params = [
        torch.randn(128, generator=gen, dtype=torch.float32),
        torch.randn(96, generator=gen, dtype=torch.float32),
    ]
    step_grads = [
        [torch.randn(128, generator=gen, dtype=torch.float32), torch.randn(96, generator=gen, dtype=torch.float32)]
        for _ in range(4)
    ]

    # Reference single-rank unsharded run
    ref_params = [p.clone().detach() for p in full_params]
    ref_opt = LuminaV(
        ref_params,
        lr=1e-2,
        weight_decay=0.05,
        buffer=buffer_mode,
        cautious=True,
        bound=True,
        bound_type="radial",
        bound_ratio=0.05,
        stochastic_rounding=False,
        fused_single_pass=fused_single_pass,
        fsdp_sync=False,
        execution="eager",
        prewarm_jit=False,
        seed=1337,
    )
    for grads_for_step in step_grads:
        for p, g in zip(ref_params, grads_for_step):
            p.grad = g.clone()
        ref_opt.step()
    ref_scratches = [ref_opt.state[p]["scratch"].clone() for p in ref_params]

    # 2-rank simulated FSDP run over gloo + FileStore
    world_size = 2
    store_path = str(tmp_path / f"gloo_store_{buffer_mode}_{fused_single_pass}")
    out_dir = tmp_path / f"out_{buffer_mode}_{fused_single_pass}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mp.spawn(
        _fsdp_parity_worker,
        args=(world_size, store_path, str(out_dir), buffer_mode, fused_single_pass, full_params, step_grads),
        nprocs=world_size,
        join=True,
    )

    r0: Dict[str, Any] = torch.load(out_dir / "rank_0.pt", weights_only=True)
    r1: Dict[str, Any] = torch.load(out_dir / "rank_1.pt", weights_only=True)

    for idx, ref_p in enumerate(ref_params):
        merged = torch.cat([r0["params"][idx], r1["params"][idx]], dim=0)
        assert torch.allclose(merged, ref_p, atol=1e-5, rtol=1e-5), (
            f"FSDP sharded mismatch for buffer={buffer_mode}, fused={fused_single_pass}, param={idx}"
        )
        assert torch.allclose(r0["scratches"][idx][3:5], r1["scratches"][idx][3:5], atol=1e-6, rtol=1e-6)
        assert torch.allclose(r0["scratches"][idx][3:5], ref_scratches[idx][3:5], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_params", [1, 3])
@pytest.mark.parametrize("fused_single_pass", [True, False])
def test_fsdp_multirank_empty_shard_no_deadlock_and_synced_scales(
    tmp_path: Path,
    num_params: int,
    fused_single_pass: bool,
) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    world_size = 2
    store_path = str(tmp_path / f"empty_store_{num_params}_{fused_single_pass}")
    out_dir = tmp_path / f"empty_out_{num_params}_{fused_single_pass}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mp.spawn(
        _fsdp_empty_shard_worker,
        args=(world_size, store_path, str(out_dir), num_params, fused_single_pass),
        nprocs=world_size,
        join=True,
    )

    r0 = torch.load(out_dir / "empty_rank_0.pt", weights_only=True)
    r1 = torch.load(out_dir / "empty_rank_1.pt", weights_only=True)
    for idx in range(num_params):
        assert torch.allclose(r0["scratches"][idx][3:5], r1["scratches"][idx][3:5], atol=1e-6, rtol=1e-6)
        assert r0["scratches"][idx][:3].abs().max().item() == 0.0
        assert r1["scratches"][idx][:3].abs().max().item() == 0.0


def _fsdp2_dtensor_worker(
    rank: int,
    world_size: int,
    store_path: str,
    out_dir: str,
    dtype_name: str,
    sr: bool,
    master_weights: str,
    hsdp_2d: bool,
) -> None:
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        mesh = (
            init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard"))
            if hsdp_2d
            else init_device_mesh("cpu", (world_size,))
        )
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16)).to(dtype=dtype)
        for m in model:
            if isinstance(m, nn.Linear):
                fully_shard(m, mesh=mesh)
        fully_shard(model, mesh=mesh)

        params = list(model.parameters())
        opt = LuminaV(
            params,
            lr=1e-2,
            buffer="t2",
            cautious=True,
            bound=True,
            bound_type="radial",
            stochastic_rounding=sr,
            master_weights=master_weights,  # type: ignore[arg-type]
            fsdp_sync=True,
            execution="eager",
            prewarm_jit=False,
            seed=1337,
        )
        torch.manual_seed(99)
        x = torch.randn(4, 32, dtype=dtype)
        for _ in range(3):
            opt.zero_grad()
            loss = model(x).float().pow(2).mean()
            loss.backward()
            opt.step()

        torch.save(
            {
                "full_params": [p.full_tensor().detach().float().clone() for p in params],
                "scratches": [opt.state[p]["scratch"].clone() for p in params],
                "all_state_dtensor": all(isinstance(opt.state[p]["exp_avg"], DTensor) for p in params),
            },
            Path(out_dir) / f"dt_rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _has_fsdp2() -> bool:
    try:
        from torch.distributed.device_mesh import init_device_mesh  # noqa: F401
        from torch.distributed.fsdp import fully_shard  # noqa: F401
        from torch.distributed.tensor import DTensor  # noqa: F401

        return dist.is_available()
    except Exception:
        return False


@pytest.mark.parametrize(
    ("dtype_name", "sr", "master_weights", "hsdp_2d", "world_size", "atol"),
    [
        ("fp32", False, "none", False, 2, 1e-5),
        ("bf16", False, "none", False, 2, 1e-5),
        ("bf16", True, "none", False, 2, 1e-2),
        ("fp16", True, "semi", False, 2, 1e-4),
        ("fp32", False, "none", True, 4, 1e-5),
    ],
)
def test_fsdp2_fully_shard_dtensor_and_hsdp_2d(
    tmp_path: Path,
    dtype_name: str,
    sr: bool,
    master_weights: str,
    hsdp_2d: bool,
    world_size: int,
    atol: float,
) -> None:
    if not _has_fsdp2():
        pytest.skip("PyTorch FSDP2 (fully_shard / DTensor) is not available")

    import torch.nn as nn

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    torch.manual_seed(42)
    ref_model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16)).to(dtype=dtype)
    ref_opt = LuminaV(
        ref_model.parameters(),
        lr=1e-2,
        buffer="t2",
        cautious=True,
        bound=True,
        bound_type="radial",
        stochastic_rounding=sr,
        master_weights=master_weights,  # type: ignore[arg-type]
        fsdp_sync=False,
        execution="eager",
        prewarm_jit=False,
        seed=1337,
    )
    torch.manual_seed(99)
    x = torch.randn(4, 32, dtype=dtype)
    for _ in range(3):
        ref_opt.zero_grad()
        loss = ref_model(x).float().pow(2).mean()
        loss.backward()
        ref_opt.step()
    ref_params = [p.detach().float().clone() for p in ref_model.parameters()]
    ref_scratches = [ref_opt.state[p]["scratch"].clone() for p in ref_model.parameters()]

    out_dir = tmp_path / f"fsdp2_{dtype_name}_sr{int(sr)}_{master_weights}_{int(hsdp_2d)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = str(out_dir / "store")
    mp.spawn(
        _fsdp2_dtensor_worker,
        args=(world_size, store_path, str(out_dir), dtype_name, sr, master_weights, hsdp_2d),
        nprocs=world_size,
        join=True,
    )
    r0 = torch.load(out_dir / "dt_rank_0.pt", weights_only=True)
    assert r0["all_state_dtensor"] is True
    for p_f, p_r, sc_f, sc_r in zip(r0["full_params"], ref_params, r0["scratches"], ref_scratches):
        assert torch.isfinite(p_f).all()
        assert torch.allclose(p_f, p_r, atol=atol, rtol=atol)
        if not sr:
            assert torch.allclose(sc_f[3:5], sc_r[3:5], atol=1e-5, rtol=1e-5)

