from __future__ import annotations

from collections.abc import Iterator, Sequence
import struct
from typing import TYPE_CHECKING, Any, overload

from ._ffi import StringTensor as _RawStringTensor

if TYPE_CHECKING:
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import torch


class StringTensor:
    """Zero-overhead columnar string buffer supporting CPU and GPU dual representations.

    1. CPU Layout (Pola 2 - Umbra / DuckDB / Arrow 2.0 StringView):
       - 16-byte compact struct per string:
         - len <= 12: Inlined directly inside the struct (zero heap pointer, zero dereferencing).
         - len > 12: 4-byte prefix + 4-byte offset + 4-byte buffer id.
         - Register-level SIMD equality checks and parallel XXH3-64 hashing without Python GIL.

    2. GPU Layout (Pola 1 - Contiguous Offset + Payload Buffer):
       - Contiguous uint32 offsets buffer [0, offset_1, ..., offset_N] of length N + 1.
       - Contiguous uint8 character payload buffer.
       - Direct zero-copy interop with PyTorch GPU tensors, Vulkan / CUDA compute shaders,
         NumPy arrays, and Apache Arrow string/string_view arrays.
    """

    def __init__(self, items: Sequence[str] | _RawStringTensor | None = None) -> None:
        if isinstance(items, _RawStringTensor):
            self._raw = items
        elif items is None:
            self._raw = _RawStringTensor.from_list([])
        else:
            self._raw = _RawStringTensor.from_list(list(items))

    @classmethod
    def from_list(cls, strings: Sequence[str]) -> StringTensor:
        """Constructs StringTensor from a sequence of Python strings."""
        return cls(_RawStringTensor.from_list(list(strings)))

    @classmethod
    def from_gpu_buffers(
        cls,
        offsets: Sequence[int] | bytes | bytearray | memoryview | Any,
        payload: bytes | bytearray | memoryview | Any,
    ) -> StringTensor:
        """Constructs StringTensor from GPU contiguous buffers (offsets + payload).

        Supports Python lists, raw bytes, NumPy uint32 arrays, and PyTorch int32/uint8 tensors.
        """
        # Convert offsets to list[int]
        offsets_any: Any = offsets
        if hasattr(offsets_any, "cpu"):
            offsets_any = offsets_any.cpu()
        if hasattr(offsets_any, "numpy"):
            offsets_any = offsets_any.numpy()
        if hasattr(offsets_any, "tolist"):
            raw_offsets = [int(x) for x in offsets_any.tolist()]
        elif isinstance(offsets_any, (bytes, bytearray, memoryview)):
            num_offsets = len(offsets_any) // 4
            raw_offsets = list(struct.unpack(f"<{num_offsets}I", bytes(offsets_any)))
        else:
            raw_offsets = [int(x) for x in offsets_any]

        # Convert payload to bytes
        payload_any: Any = payload
        if hasattr(payload_any, "cpu"):
            payload_any = payload_any.cpu()
        if hasattr(payload_any, "numpy"):
            payload_any = payload_any.numpy()
        if hasattr(payload_any, "tobytes"):
            raw_payload = payload_any.tobytes()
        elif not isinstance(payload_any, bytes):
            raw_payload = bytes(payload_any)
        else:
            raw_payload = payload_any

        return cls(_RawStringTensor.from_gpu_buffers(raw_offsets, raw_payload))

    @classmethod
    def from_numpy(cls, array: Any) -> StringTensor:
        """Constructs StringTensor from a NumPy 1D array of strings or objects."""
        return cls.from_list([str(x) for x in array.tolist()])

    @classmethod
    def from_arrow(cls, array: Any) -> StringTensor:
        """Constructs StringTensor from an Apache Arrow string / string_view array."""
        return cls.from_list([str(x) for x in array.to_pylist()])

    def to_gpu_buffers(self) -> tuple[bytes, bytes]:
        """Exports contiguous GPU buffers: (offsets_bytes, payload_bytes).

        offsets_bytes is a little-endian uint32 array of shape (N + 1,).
        payload_bytes is a contiguous UTF-8 byte sequence of total string characters.
        """
        return self._raw.to_gpu_buffers()

    def to_offsets(self) -> list[int]:
        """Returns the list of uint32 offsets of length N + 1."""
        return self._raw.to_offsets()

    def to_payload(self) -> bytes:
        """Returns the raw contiguous UTF-8 payload bytes."""
        return self._raw.to_payload()

    def to_numpy_gpu(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns GPU-ready contiguous buffers as NumPy arrays: (offsets_u32, payload_u8)."""
        import numpy as np

        offset_bytes, payload_bytes = self.to_gpu_buffers()
        offsets = np.frombuffer(offset_bytes, dtype=np.uint32)
        payload = np.frombuffer(payload_bytes, dtype=np.uint8)
        return offsets, payload

    def to_torch_gpu(self, device: str | Any = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Returns GPU-ready contiguous buffers as PyTorch tensors: (offsets_int32, payload_uint8).

        Can be placed directly on any target device (e.g. 'cuda', 'vulkan', 'cpu').
        """
        import torch

        offset_bytes, payload_bytes = self.to_gpu_buffers()
        # Use bytearray to provide a writable buffer and avoid PyTorch non-writable buffer warnings
        offsets_tensor = torch.frombuffer(bytearray(offset_bytes), dtype=torch.int32).to(device)
        payload_tensor = torch.frombuffer(bytearray(payload_bytes), dtype=torch.uint8).to(device)
        return offsets_tensor, payload_tensor

    def to_arrow(self, zero_copy: bool = True) -> pa.Array:
        """Converts to Apache Arrow array directly from contiguous buffers in O(1) zero-copy.

        When zero_copy is True, wraps the contiguous offsets and payload buffers directly via
        pa.StringArray.from_buffers without allocating intermediate Python strings.
        """
        import pyarrow as pa

        if zero_copy:
            off_bytes, pay_bytes = self.to_gpu_buffers()
            return pa.StringArray.from_buffers(
                len(self),
                pa.py_buffer(off_bytes),
                pa.py_buffer(pay_bytes),
            )
        if hasattr(pa, "string_view"):
            return pa.array(self.to_list(), type=pa.string_view())
        return pa.array(self.to_list(), type=pa.string())

    def to_polars(self) -> pl.Series:
        """Converts StringTensor directly to a Polars Series with zero-copy Arrow memory backing."""
        import polars as pl

        res = pl.from_arrow(self.to_arrow(zero_copy=True))
        if isinstance(res, pl.Series):
            return res
        raise TypeError("Expected Polars Series from Arrow array")

    def __len__(self) -> int:
        return len(self._raw)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> StringTensor: ...

    def __getitem__(self, index: int | slice) -> str | StringTensor:
        if isinstance(index, slice):
            return StringTensor(self.to_list()[index])
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("StringTensor index out of range")
        return self._raw[index]

    def get(self, index: int) -> str | None:
        """Returns string at index or None if index out of bounds."""
        if index < 0:
            index += len(self)
        if 0 <= index < len(self):
            return self._raw.get(index)
        return None

    def to_list(self) -> list[str]:
        """Converts StringTensor back to a list of Python strings."""
        return self._raw.to_list()

    def lengths(self) -> list[int]:
        """Returns byte lengths of all strings."""
        return self._raw.lengths()

    def is_inlined_mask(self) -> list[bool]:
        """Returns boolean mask indicating which strings are inlined (<= 12 bytes) on CPU."""
        return self._raw.is_inlined_mask()

    def total_bytes(self) -> int:
        """Returns total character bytes across all strings."""
        return self._raw.total_bytes()

    @property
    def inlined_count(self) -> int:
        """Number of strings inlined directly in the 16-byte CPU StringView struct."""
        return sum(1 for inlined in self._raw.is_inlined_mask() if inlined)

    @property
    def heap_count(self) -> int:
        """Number of strings stored in the referenced memory pool (> 12 bytes)."""
        return len(self._raw) - self.inlined_count

    def filter_by_length(self, min_len: int | None = None, max_len: int | None = None) -> StringTensor:
        """Filters strings by byte length range without Python GIL.

        If min_len or max_len is None, that bound is unconstrained (inactive).
        """
        raw_min = 0 if min_len is None else max(0, min_len)
        raw_max = 0 if max_len is None else max(0, max_len)
        return StringTensor(self._raw.filter_by_length(raw_min, raw_max))

    def filter_by_prefix(self, prefix: str) -> StringTensor:
        """Fast SIMD-accelerated prefix filtering using 4-byte register comparison."""
        return StringTensor(self._raw.filter_by_prefix(prefix))

    def hash_all(self) -> list[int]:
        """Calculates 64-bit XXH3 hashes in parallel across all CPU cores without Python GIL."""
        return self._raw.hash_all()

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self._raw[i]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StringTensor):
            return self.to_list() == other.to_list()
        if isinstance(other, (list, tuple)):
            return self.to_list() == list(other)
        return False

    def __repr__(self) -> str:
        return (
            f"StringTensor(count={len(self)}, inlined={self.inlined_count}, "
            f"heap={self.heap_count}, total_bytes={self.total_bytes()})"
        )


__all__ = ["StringTensor"]
