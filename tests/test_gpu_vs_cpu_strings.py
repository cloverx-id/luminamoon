from __future__ import annotations

from pathlib import Path
import sys
import time

# Ensure python workspace is on sys.path
ROOT_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon.data import (
    MinHashDeduplicator,
    PIIMasker,
    QualityFilter,
    StreamDeduplicator,
    StringTensor,
    TextCleaner,
    data_cpu_isa,
)


def generate_100_corrupted_strings() -> list[str]:
    """Generates 100 diverse corrupted, noisy, malformed, and adversarial string samples."""
    samples = []

    # 1. Newline & Whitespace corruption (20 samples)
    for i in range(20):
        crlf = "\r\n" * ((i % 5) + 1)
        tabs = "\t" * (i % 3)
        samples.append(f"{tabs}  Dirty text sample #{i} with CRLF corruption{crlf}   trailing spaces   ")

    # 2. Corrupted / Raw JSON payloads (20 samples)
    for i in range(20):
        if i % 2 == 0:
            samples.append(f'{{"payload": {{"content": "Extracted JSON text record #{i}\\r\\nwith inner CRLF"}}}}')
        else:
            samples.append(f'{{"broken_json": "record #{i}", "text": "Some text content without closing brace')

    # 3. Degenerate repetitions & gibberish / low entropy (15 samples)
    for i in range(15):
        char = chr(ord('a') + (i % 10))
        samples.append(char * (50 + i * 10))  # "aaaa...", "bbbb..." (triggers quality repetition filter)

    # 4. Short strings <= 12 bytes: test CPU inlining (Pola 2) vs GPU offsets (15 samples)
    short_items = ["", "ok", "err", "404_not_found", "simd", "gpu", "cpu_core", "a", "12_bytes_ok!", "inlined!", "test", "short_str", "nan", "null", "vulkan"]
    samples.extend(short_items[:15])

    # 5. Strings with PII & Token leaks (10 samples)
    for i in range(10):
        samples.append(f"User #{i} logged in with email test_{i}@malformed.org from IP 192.168.1.{i} and token Bearer sk-abc1234567890def1234567890abcdef")

    # 6. Near-duplicate pairs for MinHash testing (10 samples = 5 pairs)
    for i in range(5):
        base = f"The quick brown fox jumps over the lazy dog in experiment #{i} at midnight."
        noisy = f"The quick brown fox jumps over the lazy dog in experiment #{i} at midnight!"
        samples.extend([base, noisy])

    # 7. Heavy long payloads > 500 bytes (10 samples)
    for i in range(10):
        samples.append(f"LONG_BLOCK_#{i}: " + ("#$@!%^&*()_+~ " * 40) + f" End of block #{i}.")

    assert len(samples) == 100, f"Expected 100 samples, got {len(samples)}"
    return samples


def run_benchmark() -> None:
    print("=" * 70)
    print("BENCHMARK: 100 ADVERSARIAL & CORRUPTED STRINGS (CPU vs GPU LAYOUT)")
    print("=" * 70)

    isa = data_cpu_isa()
    print(f"Detected CPU ISA Hardware: {isa}\n")

    raw_samples = generate_100_corrupted_strings()
    print(f"[Dataset] Generated {len(raw_samples)} test samples.")
    total_raw_bytes = sum(len(s.encode("utf-8")) for s in raw_samples)
    print(f"[Dataset] Total raw text volume: {total_raw_bytes:,} bytes (~{total_raw_bytes / 1024:.2f} KB)\n")

    # -------------------------------------------------------------
    # 1. Direct Evaluation of all 100 Corrupted Strings
    # -------------------------------------------------------------
    print(">>> 1. DIRECT BENCHMARK ON ALL 100 RAW CORRUPTED STRINGS...")
    # CPU StringView (Pola 2)
    t0_st = time.perf_counter_ns()
    st_cpu = StringTensor(raw_samples)
    inlined_mask = st_cpu.is_inlined_mask()
    hashes = st_cpu.hash_all()
    filtered_prefix = st_cpu.filter_by_prefix("User")
    t1_st = time.perf_counter_ns()
    st_duration_us = (t1_st - t0_st) / 1000.0

    print(f"  - StringTensor total strings: {len(st_cpu)}")
    print(f"  - Inlined strings (len <= 12B, ZERO heap pointer): {st_cpu.inlined_count} / {len(st_cpu)} ({st_cpu.inlined_count / len(st_cpu) * 100:.1f}%)")
    print(f"  - Heap pool strings (len > 12B, 4B prefix register): {st_cpu.heap_count} / {len(st_cpu)}")
    print(f"  - Filtered prefix 'User': {len(filtered_prefix)} matches")
    print(f"  - Parallel XXH3-64 hashes computed: {len(hashes)}")
    print(f"  - CPU StringView Processing Time: {st_duration_us:.2f} us ({st_duration_us / 1000.0:.3f} ms)\n")

    # GPU Contiguous Buffer (Pola 1)
    print(">>> 2. GPU MEMORY LAYOUT: Contiguous Offsets + Data Buffer (Pola 1)...")
    t0_gpu = time.perf_counter_ns()
    offset_bytes, payload_bytes = st_cpu.to_gpu_buffers()
    offsets = st_cpu.to_offsets()
    offsets_np, payload_np = st_cpu.to_numpy_gpu()
    offsets_torch, payload_torch = st_cpu.to_torch_gpu()
    t1_gpu = time.perf_counter_ns()
    gpu_duration_us = (t1_gpu - t0_gpu) / 1000.0

    print(f"  - GPU Offsets buffer: {len(offsets)} elements ({len(offset_bytes)} bytes, uint32)")
    print(f"  - GPU Payload buffer: {len(payload_bytes):,} contiguous UTF-8 bytes (uint8)")
    print(f"  - Memory compactness: 0 bytes pointer overhead, 0 heap fragmentation")
    print(f"  - NumPy GPU-ready arrays: offsets shape {offsets_np.shape} ({offsets_np.dtype}), payload shape {payload_np.shape} ({payload_np.dtype})")
    print(f"  - PyTorch GPU-ready tensors: offsets {offsets_torch.shape} ({offsets_torch.dtype}), payload {payload_torch.shape} ({payload_torch.dtype})")
    print(f"  - GPU Packing & Layout Generation Time: {gpu_duration_us:.2f} us ({gpu_duration_us / 1000.0:.3f} ms)\n")

    # -------------------------------------------------------------
    # 3. CPU Cleaning & Quality Pipeline on Corrupted Strings
    # -------------------------------------------------------------
    print(">>> 3. EXECUTING FULL CLEANING PIPELINE ON CORRUPTED STRINGS...")
    cleaner = TextCleaner(trim_whitespace=True, normalize_newlines=True, min_byte_len=1)
    quality = QualityFilter(min_entropy=1.5, max_repetition_ratio=0.4)
    dedup = StreamDeduplicator()
    masker = PIIMasker(mask_emails=True, mask_ips=True, mask_api_keys=True)

    t0_cpu = time.perf_counter_ns()

    # Step A: Clean with SIMD
    cleaned_records = [cleaner.clean_single(s) for s in raw_samples]
    cleaned_records = [s for s in cleaned_records if s is not None]

    # Step B: Mask PII
    masked_records = [masker.mask(s) for s in cleaned_records]

    # Step C: Filter quality (entropy & loop detector)
    quality_records = [s for s in masked_records if quality.is_high_quality(s)]

    # Step D: Exact Deduplication
    unique_records = dedup.filter_batch(quality_records)
    dedup.commit()

    t1_cpu = time.perf_counter_ns()
    cpu_duration_us = (t1_cpu - t0_cpu) / 1000.0

    print(f"  - Input raw samples: {len(raw_samples)}")
    print(f"  - After SIMD cleaning & PII mask: {len(masked_records)}")
    print(f"  - Filtered degenerate loops / gibberish: {len(masked_records) - len(quality_records)} samples rejected")
    print(f"  - Valid high-quality & deduplicated samples: {len(unique_records)}")
    print(f"  - CPU Pipeline Execution Time: {cpu_duration_us:.2f} us ({cpu_duration_us / 1000.0:.3f} ms)")
    print(f"  - CPU Throughput: {len(raw_samples) / (cpu_duration_us / 1_000_000.0):,.0f} records/sec\n")

    # -------------------------------------------------------------
    # 4. Verification & Parity Audit
    # -------------------------------------------------------------
    print(">>> 4. INTEGRITY AUDIT: CPU <-> GPU Roundtrip...")
    # Slicing simulation on GPU buffers
    reconstructed_from_gpu = []
    for i in range(len(st_cpu)):
        start = offsets[i]
        end = offsets[i + 1]
        chunk = payload_bytes[start:end].decode("utf-8")
        reconstructed_from_gpu.append(chunk)

    assert reconstructed_from_gpu == st_cpu.to_list()
    assert len(reconstructed_from_gpu) == len(st_cpu)
    print("  [AUDIT PASS] 100% Bit-exact parity: All 100 corrupted strings sliced from GPU contiguous buffer match CPU StringView identically!")

    # -------------------------------------------------------------
    # 5. Architectural Comparison Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ARCHITECTURAL COMPARISON: CPU (StringView) vs GPU (Offsets + Buffer)")
    print("=" * 70)
    print(f"{'Metrik / Aspek':<30} | {'CPU (StringView Pola 2)':<20} | {'GPU (Contiguous Pola 1)':<20}")
    print("-" * 75)
    print(f"{'Representasi Memori':<30} | {'16B Struct (Inlined/Ref)':<20} | {'Dual Buffer (u32 + u8)':<20}")
    print(f"{'Heap Pointer Dereference':<30} | {f'{st_cpu.inlined_count} item = 0 pointer':<20} | {'0 pointer (flat index)':<20}")
    print(f"{'SIMD / Vector Filter':<30} | {'4B Register Compare':<20} | {'Thread-coalesced access':<20}")
    print(f"{'Kesesuaian Compute':<30} | {'CPU Cache L1/L2 kencang':<20} | {'VRAM GPU / Shader ramah':<20}")
    print(f"{'Total Waktu Pemrosesan':<30} | {f'{st_duration_us:.1f} us':<20} | {f'{gpu_duration_us:.1f} us':<20}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
