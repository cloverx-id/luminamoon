from __future__ import annotations

import threading
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_PYTHON = ROOT_DIR / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

import luminamoon as lm
from luminamoon.data import DataPipeline, StreamDeduplicator, StringTensor, TokenRingBuffer
from luminamoon.optimizer import AutoTune, LuminaV, _luminamoon_optimizer_rs as rs


def test_ffi_rejects_structured_readonly_misaligned_and_bf16_aliasing() -> None:
    if rs is None:
        pytest.skip("Rust extension not compiled")

    # Structured dtype spoofing must be rejected before writing.
    x = np.ones((4,), dtype=np.float32)
    spoof = np.zeros((4,), dtype=np.dtype([("float32", "u1")]))
    with pytest.raises(ValueError, match="Unsupported dtype"):
        rs.ra_tanh_f32_pyo3(x, spoof, 1.0)

    # Read-only mutable output buffer must be rejected.
    ro = np.zeros((4,), dtype=np.float32)
    ro.setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        rs.ra_tanh_f32_pyo3(x, ro, 1.0)

    # Misaligned float32 pointer must be rejected.
    raw = np.zeros((32,), dtype=np.uint8)
    misaligned = np.ndarray(shape=(4,), dtype=np.float32, buffer=raw[1:17])
    out = np.zeros((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="aligned"):
        rs.ra_tanh_f32_pyo3(misaligned, out, 1.0)

    # BF16 param + FP32 master_param sharing param address must still check overlap against grad.
    import torch

    buf = torch.zeros(12, dtype=torch.uint16)
    p = buf[:4].view(torch.bfloat16)
    g = buf[4:8].view(torch.bfloat16)
    m = torch.zeros(4, dtype=torch.bfloat16)
    mp = buf[:8].view(torch.float32)
    scratch = torch.zeros(6, dtype=torch.float32)
    cfg = {
        "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0,
        "buffer": 1, "master_weights": "semi", "execution": "scalar",
    }
    with pytest.raises(ValueError, match="aliasing"):
        rs.step_pyo3([p], [g], [m], [None], [mp], [scratch], [1], [0], cfg)


def test_string_tensor_double_consume_arrow_capsule_raises() -> None:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("pyarrow not installed")

    st = StringTensor(["alpha", "a string longer than 12 bytes", ""])
    arr = pa.array(st)
    capsule = arr.__arrow_c_array__()

    class _ReplayCapsule:
        def __arrow_c_array__(self, requested_schema=None):
            return capsule

    first = StringTensor.from_arrow(_ReplayCapsule())
    assert first.to_list() == ["alpha", "a string longer than 12 bytes", ""]
    with pytest.raises(ValueError):
        StringTensor.from_arrow(_ReplayCapsule())


def test_data_wal_crash_sync_torn_append_and_rank_shift(tmp_path: Path) -> None:
    wal = tmp_path / "torn.dedup"
    d = StreamDeduplicator(wal_path=wal)
    assert d.is_unique("doc-a", batch_id=1)
    assert d.is_unique("doc-b", batch_id=1)
    d.commit(batch_id=1)
    with open(wal, "ab") as f:
        f.write(b"\x01\x02\x03\x04\x05\x06\x07")
    d2 = StreamDeduplicator(wal_path=wal)
    assert d2.count == 2
    assert d2.is_seen("doc-a") and d2.is_seen("doc-b")

    # Simulate crash where WAL has extra hashes beyond committed state.json.
    ckpt = tmp_path / "ckpt"
    docs = [{"text": f"unique training sample number {i} with enough words"} for i in range(6)]
    p1 = DataPipeline(docs, tokenizer=lambda s: [1, 2, 3, 4], batch_size=1, seq_len=4, packing="pad", dedup="exact", checkpoint=ckpt)
    it = iter(p1)
    _ = next(it)
    p1.commit()
    assert p1.deduplicator is not None
    p1.deduplicator.is_unique(docs[1]["text"], batch_id=99)
    p1.deduplicator.commit(batch_id=99)

    p2 = DataPipeline(docs, tokenizer=lambda s: [1, 2, 3, 4], batch_size=1, seq_len=4, packing="pad", dedup="exact", checkpoint=ckpt)
    rem = list(p2)
    assert len(rem) >= 5


def test_ring_buffer_small_prefill_manual_mode_and_atomic_push() -> None:
    rb = TokenRingBuffer(source=None, capacity_tokens=64, auto_prefill_tokens=8, batch_size=2, seq_len=4)
    rb.push(list(range(16)))
    rb.mark_eof()
    b = rb.pop_batch()
    assert b is not None and b[0].shape == (2, 4)
    rb.close()

    rb2 = TokenRingBuffer(source=[[10, 11, 12, 13] for _ in range(4)], capacity_tokens=64, auto_prefill_tokens=8, batch_size=2, seq_len=4)
    assert rb2.wait_for_prefill(timeout=2.0)
    rb2.close()

    rb3 = TokenRingBuffer(source=None, capacity_tokens=64, auto_prefill_tokens=8, batch_size=2, seq_len=4)
    errs: list[Exception] = []

    def _writer(val: int) -> None:
        try:
            for _ in range(8):
                rb3.push([val] * 8)
        except Exception as exc:
            errs.append(exc)

    t1 = threading.Thread(target=_writer, args=(1,))
    t2 = threading.Thread(target=_writer, args=(2,))
    t1.start()
    t2.start()
    out: list[int] = []
    while len(out) < 128:
        out.extend(rb3.pop_tokens(8, timeout=2.0).tolist())
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    rb3.close()
    assert not errs
    for i in range(0, 128, 8):
        assert len(set(out[i : i + 8])) == 1


def test_fp16_norm_backward_and_optimizer_edges() -> None:
    import torch
    from luminamoon.nn import RaBatchNorm2d, RaNorm

    rn = RaNorm(8, eps=1e-6).half()
    x0 = torch.zeros(2, 4, 8, dtype=torch.float16, requires_grad=True)
    y0 = rn(x0)
    assert torch.isfinite(y0).all()
    y0.sum().backward()
    assert x0.grad is not None and torch.isfinite(x0.grad).all()

    bn = RaBatchNorm2d(4, eps=1e-5).half().eval()
    bn.running_mad.zero_()
    xb = torch.randn(2, 4, 2, 2, dtype=torch.float16, requires_grad=True)
    yb = bn(xb)
    assert torch.isfinite(yb).all()
    yb.sum().backward()
    assert xb.grad is not None and torch.isfinite(xb.grad).all()

    with pytest.raises(ValueError, match="out_channels"):
        bn.fold_into_conv(torch.nn.Conv2d(3, 5, 1))

    conv_fp16 = torch.nn.Conv2d(3, 4, 1).half()
    wf, bf = bn.fold_into_conv(conv_fp16)
    assert torch.isfinite(wf).all() and torch.isfinite(bf).all()

    # 0-D NumPy scalar parameter step
    s0 = np.array(1.5, dtype=np.float32)
    opt_np = LuminaV([s0], lr=1e-2, execution="scalar")
    opt_np.step(grads=[np.array(0.5, dtype=np.float32)])
    assert np.isfinite(s0) and float(s0) != 1.5

    # fuse_tensors=True with explicit grads when requires_grad=False
    p_ng = torch.nn.Parameter(torch.ones(16), requires_grad=False)
    opt_f = LuminaV([p_ng], lr=1e-2, fuse_tensors=True, execution="eager")
    before = p_ng.detach().clone()
    opt_f.step(grads=[torch.ones(16)])
    assert not torch.equal(p_ng.detach(), before)

    # AutoTune does not mutate non-contiguous parameters
    base = torch.randn(8, 8)
    nc = torch.nn.Parameter(base.t())
    snap = nc.detach().clone()
    _ = AutoTune([nc], verbose=False, steps=3, measure_last=1)
    assert not nc.is_contiguous() and torch.equal(nc.detach(), snap)


def test_public_dir_help_and_cli_dry_run_quoting() -> None:
    assert set(dir(lm)) == set(lm.__all__)
    assert set(dir(lm.optimizer)) == set(lm.optimizer.__all__)
    txt = LuminaV.help(include_hardware=False, print_out=False)
    assert "LuminaV" in txt and "DETECTED HARDWARE" not in txt
    assert "!luminamoon" not in txt
    with pytest.raises(ValueError, match="Did you mean 'optimizer'"):
        lm.format_help("optimzer")
