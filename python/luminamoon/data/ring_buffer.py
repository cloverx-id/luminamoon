from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import threading
from typing import Any

from .collate import _extract_text
from .string_tensor import StringTensor


class TokenRingBuffer:
    """Contiguous token ring buffer with background prefill for streaming LM training.

    Maintains a circular reservoir of token IDs in host RAM (`int64` by default, or `int32`
    to halve buffer size and host-to-device transfer volume) with batched background
    tokenization, optional double-buffered pinned host memory for CUDA transfers, and an initial
    `auto_prefill_tokens` warmup barrier.
    """

    def __init__(
        self,
        source: Iterable[Any] | None = None,
        *,
        capacity_tokens: int = 1_000_000,
        auto_prefill_tokens: int = 500_000,
        batch_size: int = 4,
        seq_len: int = 2048,
        tokenizer: Any | None = None,
        text_key: str = "text",
        eos_token_id: int | None = None,
        pin_memory: bool = True,
        device: Any | None = None,
        producer_chunk_tokens: int = 16_384,
        tokenizer_batch_size: int = 256,
        token_dtype: Any = "int64",
        prefill_timeout: float | None = 30.0,
    ) -> None:
        if capacity_tokens <= 0:
            raise ValueError(f"capacity_tokens must be > 0, got {capacity_tokens}.")
        if auto_prefill_tokens < 0:
            raise ValueError(f"auto_prefill_tokens must be >= 0, got {auto_prefill_tokens}.")
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError(f"batch_size ({batch_size}) and seq_len ({seq_len}) must be > 0.")
        if tokenizer_batch_size <= 0:
            raise ValueError(f"tokenizer_batch_size must be > 0, got {tokenizer_batch_size}.")

        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                'NumPy is required for TokenRingBuffer. Install it via `pip install "luminamoon[numpy]"` or `pip install numpy`.'
            ) from exc

        resolved_np_dtype = np.dtype(token_dtype)
        if resolved_np_dtype not in (np.dtype(np.int64), np.dtype(np.int32)):
            raise ValueError(f"token_dtype must be 'int64' or 'int32', got {token_dtype!r}.")

        self.source = source
        self.capacity_tokens = int(capacity_tokens)
        self.auto_prefill_tokens = min(int(auto_prefill_tokens), self.capacity_tokens)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.eos_token_id = eos_token_id
        self.pin_memory = bool(pin_memory)
        self.device = device
        self.producer_chunk_tokens = max(1, min(int(producer_chunk_tokens), self.capacity_tokens))
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.token_dtype = resolved_np_dtype.name
        self.prefill_timeout = prefill_timeout

        self._np = np
        self._ring_dtype = resolved_np_dtype
        self._torch: Any = None
        self._cuda_available = False
        try:
            import torch

            self._torch = torch
            self._cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            pass

        self._torch_ring_dtype: Any = None
        if self._torch is not None:
            self._torch_ring_dtype = (
                self._torch.int32 if self._ring_dtype == np.dtype(np.int32) else self._torch.int64
            )
        self._batch_tok_mode: str | None = None

        self._ring = np.empty((self.capacity_tokens,), dtype=self._ring_dtype)
        self._pinned_slots: list[tuple[Any, Any]] = [(None, None), (None, None)]
        self._pinned_slot_idx = 0
        self._dma_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._read_pos = 0
        self._write_pos = 0
        self._size = 0
        self._total_written = 0
        self._total_read = 0
        self._eof = False
        self._error: Exception | None = None

        self._cond = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if source is not None:
            self._start_producer()
            if self.auto_prefill_tokens > 0:
                try:
                    self.wait_for_prefill(
                        target_tokens=self.auto_prefill_tokens,
                        timeout=self.prefill_timeout,
                    )
                except BaseException:
                    self.close()
                    raise

    @property
    def available_tokens(self) -> int:
        """Return the number of unread tokens currently stored in the ring buffer."""
        with self._cond:
            return self._size

    @property
    def total_read_tokens(self) -> int:
        """Return cumulative tokens popped by consumers."""
        with self._cond:
            return self._total_read

    @property
    def is_eof(self) -> bool:
        """Return True if the source stream has been completely ingested."""
        with self._cond:
            return self._eof

    def _encode_text_to_tokens(self, text: str) -> list[int]:
        if not text:
            return []
        if self.tokenizer is not None:
            tok_fn: Any = self.tokenizer
            if hasattr(tok_fn, "encode"):
                try:
                    encoded: Any = tok_fn.encode(text, add_special_tokens=False)
                except TypeError:
                    encoded = tok_fn.encode(text)
            elif callable(tok_fn):
                try:
                    res: Any = tok_fn(text, add_special_tokens=False)
                except TypeError:
                    res = tok_fn(text)
                encoded = res["input_ids"] if isinstance(res, Mapping) and "input_ids" in res else res
            else:
                raise TypeError(f"Unsupported tokenizer object: {type(self.tokenizer)}")
            toks = [int(t) for t in encoded]
        else:
            toks = list(text.encode("utf-8", errors="replace"))
        if self.eos_token_id is not None and toks and toks[-1] != int(self.eos_token_id):
            toks.append(int(self.eos_token_id))
        return toks

    def _encode_text_batch(self, texts: list[str]) -> list[int]:
        non_empty = [t for t in texts if t]
        if not non_empty:
            return []
        eos = int(self.eos_token_id) if self.eos_token_id is not None else None
        if self.tokenizer is None:
            out: list[int] = []
            for t in non_empty:
                toks = list(t.encode("utf-8", errors="replace"))
                if toks:
                    out.extend(toks)
                    if eos is not None and out[-1] != eos:
                        out.append(eos)
            return out

        tok_fn: Any = self.tokenizer
        if callable(tok_fn) and len(non_empty) > 1 and self._batch_tok_mode != "single":
            res: Any = None
            if self._batch_tok_mode in (None, "batch_no_special"):
                try:
                    res = tok_fn(non_empty, add_special_tokens=False)
                    self._batch_tok_mode = "batch_no_special"
                except TypeError:
                    res = None
            if res is None and self._batch_tok_mode in (None, "batch_plain"):
                try:
                    res = tok_fn(non_empty)
                    self._batch_tok_mode = "batch_plain"
                except TypeError:
                    self._batch_tok_mode = "single"
                    res = None

            if res is not None:
                ids_batch = res["input_ids"] if isinstance(res, Mapping) and "input_ids" in res else res
                if (
                    isinstance(ids_batch, (list, tuple))
                    and len(ids_batch) == len(non_empty)
                    and (not ids_batch or isinstance(ids_batch[0], (list, tuple)))
                ):
                    out = []
                    for row in ids_batch:
                        if row:
                            out.extend(int(t) for t in row)
                            if eos is not None and out[-1] != eos:
                                out.append(eos)
                    return out
                self._batch_tok_mode = "single"

        out = []
        for t in non_empty:
            out.extend(self._encode_text_to_tokens(t))
        return out

    def _try_extract_raw_text(self, item: Any) -> str | None:
        if isinstance(item, str):
            return item
        if isinstance(item, Mapping) and "input_ids" not in item:
            val = item.get(self.text_key)
            if isinstance(val, str):
                return val
        return None

    def _extract_tokens_from_item(self, item: Any) -> list[int]:
        if isinstance(item, int) and not isinstance(item, bool):
            return [item]
        if isinstance(item, StringTensor):
            out: list[int] = []
            for text in item.to_list():
                out.extend(self._encode_text_to_tokens(text))
            return out
        if self._torch is not None and isinstance(item, self._torch.Tensor):
            t_item: Any = item
            if t_item.ndim == 2:
                out = []
                for row in t_item.detach().cpu().tolist():
                    if row:
                        out.extend(int(t) for t in row)
                        if self.eos_token_id is not None and out[-1] != int(self.eos_token_id):
                            out.append(int(self.eos_token_id))
                return out
            toks = [int(t) for t in t_item.detach().cpu().reshape(-1).tolist()]
            if self.eos_token_id is not None and toks and toks[-1] != int(self.eos_token_id):
                toks.append(int(self.eos_token_id))
            return toks
        if isinstance(item, self._np.ndarray):
            arr: Any = item
            if arr.ndim == 2:
                out = []
                for row in arr.tolist():
                    if row:
                        out.extend(int(t) for t in row)
                        if self.eos_token_id is not None and out[-1] != int(self.eos_token_id):
                            out.append(int(self.eos_token_id))
                return out
            toks = [int(t) for t in arr.reshape(-1).tolist()]
            if self.eos_token_id is not None and toks and toks[-1] != int(self.eos_token_id):
                toks.append(int(self.eos_token_id))
            return toks
        if isinstance(item, Mapping) and "input_ids" in item:
            raw_ids = item["input_ids"]
            if self._torch is not None and isinstance(raw_ids, self._torch.Tensor):
                return self._extract_tokens_from_item(raw_ids)
            if isinstance(raw_ids, self._np.ndarray):
                return self._extract_tokens_from_item(raw_ids)
            return self._extract_tokens_from_item(list(raw_ids))
        if isinstance(item, (list, tuple)):
            if not item:
                return []
            first = item[0]
            if isinstance(first, int) and not isinstance(first, bool):
                toks = [int(t) for t in item]
                if self.eos_token_id is not None and toks and toks[-1] != int(self.eos_token_id):
                    toks.append(int(self.eos_token_id))
                return toks
            # Expand batches of strings, dicts, or nested token lists without silently dropping item[1:].
            out = []
            for elem in item:
                out.extend(self._extract_tokens_from_item(elem))
            return out

        text = _extract_text(item, text_key=self.text_key)
        return self._encode_text_to_tokens(text) if text else []

    def push(self, tokens: Any) -> int:
        """Write a sequence or 1D array/tensor of token IDs into the ring buffer."""
        if self._torch is not None and isinstance(tokens, self._torch.Tensor):
            arr = tokens.detach().cpu().numpy().astype(self._ring_dtype, copy=False).reshape(-1)
        elif isinstance(tokens, self._np.ndarray):
            arr = tokens.astype(self._ring_dtype, copy=False).reshape(-1)
        else:
            arr = self._np.asarray(tokens, dtype=self._ring_dtype).reshape(-1)

        total = int(arr.size)
        offset = 0
        with self._write_lock:
            while offset < total and not self._stop_event.is_set():
                with self._cond:
                    while self._size >= self.capacity_tokens and not self._stop_event.is_set():
                        self._cond.wait(timeout=0.05)
                    if self._stop_event.is_set():
                        break

                    free_space = self.capacity_tokens - self._size
                    n_write = min(total - offset, free_space)
                    first_part = min(n_write, self.capacity_tokens - self._write_pos)
                    self._ring[self._write_pos : self._write_pos + first_part] = arr[
                        offset : offset + first_part
                    ]
                    second_part = n_write - first_part
                    if second_part > 0:
                        self._ring[:second_part] = arr[offset + first_part : offset + n_write]

                    self._write_pos = (self._write_pos + n_write) % self.capacity_tokens
                    self._size += n_write
                    self._total_written += n_write
                    offset += n_write
                    self._cond.notify_all()
        return offset

    def mark_eof(self) -> None:
        """Mark the manual producer stream as finished so waiting consumers exit."""
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def _start_producer(self) -> None:
        def _producer_loop() -> None:
            staging: list[int] = []
            text_batch: list[str] = []
            prefill_flush = min(
                self.producer_chunk_tokens,
                max(1024, self.auto_prefill_tokens // 4) if self.auto_prefill_tokens > 0 else self.producer_chunk_tokens,
            )

            def _flush_texts() -> None:
                if not text_batch:
                    return
                toks = self._encode_text_batch(text_batch)
                text_batch.clear()
                if toks:
                    staging.extend(toks)

            try:
                assert self.source is not None
                for item in self.source:
                    if self._stop_event.is_set():
                        break
                    raw_text = self._try_extract_raw_text(item)
                    if raw_text is not None:
                        text_batch.append(raw_text)
                        if len(text_batch) < self.tokenizer_batch_size:
                            continue
                        _flush_texts()
                    else:
                        _flush_texts()
                        toks = self._extract_tokens_from_item(item)
                        if toks:
                            staging.extend(toks)

                    flush_limit = (
                        prefill_flush
                        if self.available_tokens < self.auto_prefill_tokens
                        else self.producer_chunk_tokens
                    )
                    if len(staging) >= flush_limit:
                        self.push(staging)
                        staging.clear()
                _flush_texts()
                if staging and not self._stop_event.is_set():
                    self.push(staging)
                    staging.clear()
            except Exception as exc:
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()
            finally:
                with self._cond:
                    self._eof = True
                    self._cond.notify_all()

        self._thread = threading.Thread(target=_producer_loop, daemon=True)
        self._thread.start()

    def wait_for_prefill(
        self,
        target_tokens: int | None = None,
        timeout: float | None = 30.0,
    ) -> int:
        """Block until the buffer reaches `target_tokens` or the source hits EOF."""
        import time

        goal = (
            min(int(target_tokens), self.capacity_tokens)
            if target_tokens is not None
            else self.auto_prefill_tokens
        )
        if goal <= 0:
            return self.available_tokens

        deadline = (time.monotonic() + float(timeout)) if timeout is not None else None
        with self._cond:
            while (
                self._size < goal
                and not self._eof
                and self._error is None
                and not self._stop_event.is_set()
            ):
                remaining = (deadline - time.monotonic()) if deadline is not None else 0.05
                if deadline is not None and remaining <= 0.0:
                    break
                self._cond.wait(timeout=min(0.05, max(0.001, remaining)))

            if self._error is not None:
                raise self._error
            return self._size

    def _pop_tokens_numpy(self, num_tokens: int, *, timeout: float | None = None) -> Any:
        import time

        n_req = int(num_tokens)
        if n_req <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        if n_req > self.capacity_tokens:
            raise ValueError(
                f"Requested num_tokens ({n_req}) exceeds ring capacity ({self.capacity_tokens})."
            )

        deadline = (time.monotonic() + float(timeout)) if timeout is not None else None
        with self._cond:
            while (
                self._size < n_req
                and not self._eof
                and self._error is None
                and not self._stop_event.is_set()
            ):
                remaining = (deadline - time.monotonic()) if deadline is not None else 0.05
                if deadline is not None and remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out waiting for {n_req} tokens in TokenRingBuffer (available={self._size})."
                    )
                self._cond.wait(timeout=min(0.05, max(0.001, remaining)))

            if self._error is not None:
                raise self._error
            if self._size < n_req:
                raise StopIteration

            out = self._np.empty((n_req,), dtype=self._ring_dtype)
            first_part = min(n_req, self.capacity_tokens - self._read_pos)
            out[:first_part] = self._ring[self._read_pos : self._read_pos + first_part]
            second_part = n_req - first_part
            if second_part > 0:
                out[first_part:] = self._ring[:second_part]

            self._read_pos = (self._read_pos + n_req) % self.capacity_tokens
            self._size -= n_req
            self._total_read += n_req
            self._cond.notify_all()
        return out

    def pop_tokens(self, num_tokens: int, *, timeout: float | None = None) -> Any:
        """Pop a contiguous 1D block of `num_tokens` token IDs from the ring buffer."""
        out = self._pop_tokens_numpy(num_tokens, timeout=timeout)
        if self._torch is not None:
            t = self._torch.from_numpy(out)
            if t.dtype != self._torch.int64:
                t = t.to(dtype=self._torch.int64)
            if self.pin_memory and self._cuda_available:
                t = t.pin_memory()
            return t
        return out

    def next_batch(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        *,
        device: Any | None = None,
        non_blocking: bool = True,
    ) -> tuple[Any, Any]:
        """Fetch the next `(x, y)` causal LM batch of shape `(batch_size, seq_len)`."""
        B = int(batch_size) if batch_size is not None else self.batch_size
        T = int(seq_len) if seq_len is not None else self.seq_len
        if B <= 0:
            raise ValueError(f"batch_size must be > 0, got {B}.")
        if T <= 0:
            raise ValueError(f"seq_len must be > 0, got {T}.")
        target_dev = device if device is not None else self.device

        raw_np = self._pop_tokens_numpy(B * (T + 1))
        block_np = raw_np.reshape(B, T + 1)
        if self._torch is not None:
            torch_mod = self._torch
            is_cuda_target = target_dev is not None and (
                (isinstance(target_dev, str) and target_dev.startswith("cuda"))
                or (getattr(target_dev, "type", None) == "cuda")
            )
            if is_cuda_target and self.pin_memory and self._cuda_available:
                with self._dma_lock:
                    slot_idx = self._pinned_slot_idx & 1
                    self._pinned_slot_idx ^= 1
                    pinned_buf, last_ev = self._pinned_slots[slot_idx]
                    if last_ev is not None:
                        last_ev.synchronize()
                    if (
                        pinned_buf is None
                        or tuple(pinned_buf.shape) != (B, T + 1)
                        or pinned_buf.dtype != self._torch_ring_dtype
                    ):
                        pinned_buf = torch_mod.empty(
                            (B, T + 1), dtype=self._torch_ring_dtype, pin_memory=True
                        )
                    pinned_buf.copy_(torch_mod.from_numpy(block_np))
                    with torch_mod.cuda.device(target_dev):
                        dev_block = pinned_buf.to(device=target_dev, non_blocking=bool(non_blocking))
                        ev = torch_mod.cuda.Event()
                        ev.record()
                    self._pinned_slots[slot_idx] = (pinned_buf, ev)
                if dev_block.dtype != torch_mod.int64:
                    dev_block = dev_block.to(dtype=torch_mod.int64)
                return dev_block[:, :-1].contiguous(), dev_block[:, 1:].contiguous()

            block_t = torch_mod.from_numpy(block_np)
            if target_dev is not None:
                dev_block = block_t.to(device=target_dev, non_blocking=bool(non_blocking))
                if dev_block.dtype != torch_mod.int64:
                    dev_block = dev_block.to(dtype=torch_mod.int64)
                return dev_block[:, :-1].contiguous(), dev_block[:, 1:].contiguous()
            if block_t.dtype != torch_mod.int64:
                block_t = block_t.to(dtype=torch_mod.int64)
            x = block_t[:, :-1].contiguous()
            y = block_t[:, 1:].contiguous()
            if self.pin_memory and self._cuda_available:
                x = x.pin_memory()
                y = y.pin_memory()
            return x, y

        return block_np[:, :-1].copy(), block_np[:, 1:].copy()

    pop_batch = next_batch

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        while not self._stop_event.is_set():
            try:
                yield self.next_batch()
            except StopIteration:
                break

    def close(self) -> None:
        """Stop the background producer thread and wake waiting consumers."""
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> TokenRingBuffer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = ["TokenRingBuffer"]
