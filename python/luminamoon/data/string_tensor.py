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
    """Columnar string buffer supporting Arrow Utf8View CPU layout and contiguous GPU buffers.

    1. CPU Layout (Apache Arrow `Utf8View` / `ByteView` binary-compatible 16-byte views):
       - `len <= 12`: `[len: u32][inline: [u8; 12]]` (zero heap pointer, zero dereferencing).
       - `len > 12`:  `[len: u32][prefix: [u8; 4]][buffer_index: u32][offset: u32]`.
       - Implements the Arrow PyCapsule Interface (`__arrow_c_schema__`, `__arrow_c_array__`)
         for zero-copy exchange with PyArrow, Polars, and DuckDB.

    2. GPU Layout (Contiguous Offset + Payload Buffer):
       - Contiguous `uint32` offsets buffer `[0, offset_1, ..., offset_N]` of length `N + 1`.
       - Contiguous `uint8` UTF-8 character payload buffer for CUDA / CubeCL / PyTorch kernels.
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
        """Constructs StringTensor from contiguous buffers `(offsets, payload)`."""
        offsets_any: Any = offsets
        if hasattr(offsets_any, "cpu"):
            offsets_any = offsets_any.cpu()
        if hasattr(offsets_any, "numpy"):
            offsets_any = offsets_any.numpy()
        if hasattr(offsets_any, "tolist"):
            raw_offsets = [int(x) for x in offsets_any.tolist()]
        elif isinstance(offsets_any, (bytes, bytearray, memoryview)):
            if len(offsets_any) % 4 != 0:
                raise ValueError("Corrupted GPU offsets buffer: byte length must be a multiple of 4")
            num_offsets = len(offsets_any) // 4
            raw_offsets = list(struct.unpack(f"<{num_offsets}I", bytes(offsets_any)))
        else:
            raw_offsets = [int(x) for x in offsets_any]

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
        return cls.from_list(["" if x is None else str(x) for x in array.tolist()])

    @classmethod
    def from_arrow(cls, array: Any) -> StringTensor:
        """Constructs StringTensor from any Arrow string / string_view / Polars Series object.

        Uses the Arrow PyCapsule Interface (`__arrow_c_array__`) when supported by the native backend.
        """
        if hasattr(_RawStringTensor, "from_arrow") and hasattr(array, "__arrow_c_array__"):
            return cls(_RawStringTensor.from_arrow(array))
        return cls.from_list(["" if x is None else str(x) for x in array.to_pylist()])

    def __arrow_c_schema__(self) -> Any:
        if hasattr(self._raw, "__arrow_c_schema__"):
            return self._raw.__arrow_c_schema__()
        return self.to_arrow(zero_copy=True).__arrow_c_schema__()

    def __arrow_c_array__(self, requested_schema: Any = None) -> tuple[Any, Any]:
        if hasattr(self._raw, "__arrow_c_array__"):
            return self._raw.__arrow_c_array__(requested_schema)
        return self.to_arrow(zero_copy=True).__arrow_c_array__(requested_schema)

    def to_gpu_buffers(self) -> tuple[bytes, bytes]:
        """Exports contiguous GPU buffers: `(offsets_bytes, payload_bytes)`."""
        return self._raw.to_gpu_buffers()

    def to_offsets(self) -> list[int]:
        """Returns the list of uint32 offsets of length `N + 1`."""
        return self._raw.to_offsets()

    def to_payload(self) -> bytes:
        """Returns the raw contiguous UTF-8 payload bytes."""
        return self._raw.to_payload()

    def to_numpy_gpu(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns GPU-ready contiguous buffers as NumPy arrays: `(offsets_u32, payload_u8)`."""
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                'NumPy is required for to_numpy_gpu(). Install it via `pip install "luminamoon[numpy]"` or `pip install numpy`.'
            ) from exc

        offset_bytes, payload_bytes = self.to_gpu_buffers()
        offsets = np.frombuffer(offset_bytes, dtype=np.uint32)
        payload = np.frombuffer(payload_bytes, dtype=np.uint8)
        return offsets, payload

    def to_torch_gpu(self, device: str | Any = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Returns GPU-ready contiguous buffers as PyTorch tensors: `(offsets_int32, payload_uint8)`."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                'PyTorch is required for to_torch_gpu(). Install it via `pip install "luminamoon[torch]"` or `pip install torch`.'
            ) from exc

        offset_bytes, payload_bytes = self.to_gpu_buffers()
        if len(payload_bytes) > 0x7FFF_FFFF:
            raise ValueError(
                f"StringTensor payload ({len(payload_bytes)} bytes) exceeds signed int32 offset limit for PyTorch tensors."
            )
        if offset_bytes:
            offsets_tensor = torch.frombuffer(bytearray(offset_bytes), dtype=torch.int32).to(device)
        else:
            offsets_tensor = torch.empty((0,), dtype=torch.int32, device=device)
        if payload_bytes:
            payload_tensor = torch.frombuffer(bytearray(payload_bytes), dtype=torch.uint8).to(device)
        else:
            payload_tensor = torch.empty((0,), dtype=torch.uint8, device=device)
        return offsets_tensor, payload_tensor

    def to_arrow(self, zero_copy: bool = True) -> pa.Array:
        """Exports to Apache Arrow array (`string_view` via PyCapsule zero-copy when native)."""
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise ImportError(
                "PyArrow is required for to_arrow(). Install it via `pip install pyarrow`."
            ) from exc

        if zero_copy:
            if hasattr(self._raw, "__arrow_c_array__") and hasattr(pa, "array"):
                try:
                    return pa.array(self)
                except Exception:
                    pass
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
        """Exports StringTensor directly to a Polars Series via Arrow PyCapsule / Arrow memory."""
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Polars is required for to_polars(). Install it via `pip install polars pyarrow`."
            ) from exc

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
            start, stop, step = index.indices(len(self))
            return StringTensor([self._raw[i] for i in range(start, stop, step)])
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
        """Filters strings by byte length range without Python GIL."""
        raw_min = 0 if min_len is None else max(0, int(min_len))
        if max_len is not None and int(max_len) < raw_min:
            return StringTensor([])
        if max_len is not None and int(max_len) == 0:
            return StringTensor([s for s in self.to_list() if len(s) == 0])
        raw_max = None if max_len is None else max(0, int(max_len))
        try:
            return StringTensor(self._raw.filter_by_length(raw_min, raw_max))
        except TypeError:
            return StringTensor(
                self._raw.filter_by_length(raw_min, 0 if raw_max is None else raw_max)
            )

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
        return repr(self._raw)


__all__ = ["StringTensor"]
