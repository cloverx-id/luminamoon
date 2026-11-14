"""LuminaMoon Data Processing Engine Benchmark.

Compares Native Rust SIMD Accelerator (AVX-512 / AVX2 / Rayon) against
Pure Python Fallbacks across micro-benchmarks and end-to-end pipelines.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Ensure local python package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from luminamoon.optimizer import _luminamoon_optimizer_rs as rs
from luminamoon.data import fallbacks as fb


@dataclass
class BenchmarkResult:
    benchmark_name: str
    workload_size: int
    data_bytes: int
    rust_duration_sec: float
    rust_throughput_items_per_sec: float
    rust_throughput_mb_per_sec: float
    fallback_duration_sec: float
    fallback_throughput_items_per_sec: float
    fallback_throughput_mb_per_sec: float
    speedup_ratio: float


def generate_synthetic_dataset(num_records: int = 10000, seed: int = 42) -> Tuple[List[str], int]:
    rng = random.Random(seed)
    templates = [
        "   The quick brown fox jumps over the lazy dog.\r\n   This is line 2 with carriage returns.\r\n",
        "Deep learning optimization requires efficient gradient descent with momentum and adaptive learning rates.\n",
        '{"text": "{\"key\": \"value\", \"nested\": [1, 2, 3]}", "meta": "test record"}\r\n',
        "System security alert: detected unauthorized access attempt from 192.168.1.105 on port 443.\n",
        "Contact user support at alice.walker@enterprise-domain.org or call +1-800-555-0199 for assistance.\r\n",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",  # high repetition
        "Short text.\n",
        "Mathematics: Pade [13/6] approximant ensures subnormal FP16 RNE preservation without overflow.\n",
        "High symbol noise: #$%@!&*()!@#$%^&*()_+{}|:<>?~`!@#$%^&*()_+{}|:<>?~`\n",
        "Large document block: " + ("lorem ipsum dolor sit amet " * 20) + "\r\n",
    ]
    records = []
    total_bytes = 0
    for i in range(num_records):
        base = templates[i % len(templates)]
        noise = f" [doc_id={i}_{rng.randint(100, 999)}]"
        rec = base + noise
        records.append(rec)
        total_bytes += len(rec.encode("utf-8"))
    return records, total_bytes


def run_timed_benchmark(
    name: str,
    rust_fn: Callable[[], Any],
    fallback_fn: Callable[[], Any],
    workload_size: int,
    data_bytes: int,
    warmup_runs: int = 2,
    test_runs: int = 5,
) -> BenchmarkResult:
    # Warmup Rust
    for _ in range(warmup_runs):
        rust_fn()
    # Timed Rust
    rust_times = []
    for _ in range(test_runs):
        t0 = time.perf_counter_ns()
        rust_fn()
        t1 = time.perf_counter_ns()
        rust_times.append((t1 - t0) / 1e9)
    rust_dur = min(rust_times)  # best of N runs

    # Warmup Fallback
    for _ in range(warmup_runs):
        fallback_fn()
    # Timed Fallback
    fallback_times = []
    for _ in range(test_runs):
        t0 = time.perf_counter_ns()
        fallback_fn()
        t1 = time.perf_counter_ns()
        fallback_times.append((t1 - t0) / 1e9)
    fallback_dur = min(fallback_times)

    rust_items_per_sec = workload_size / max(rust_dur, 1e-9)
    rust_mb_per_sec = (data_bytes / (1024 * 1024)) / max(rust_dur, 1e-9)

    fallback_items_per_sec = workload_size / max(fallback_dur, 1e-9)
    fallback_mb_per_sec = (data_bytes / (1024 * 1024)) / max(fallback_dur, 1e-9)

    speedup = fallback_dur / max(rust_dur, 1e-9)

    return BenchmarkResult(
        benchmark_name=name,
        workload_size=workload_size,
        data_bytes=data_bytes,
        rust_duration_sec=rust_dur,
        rust_throughput_items_per_sec=rust_items_per_sec,
        rust_throughput_mb_per_sec=rust_mb_per_sec,
        fallback_duration_sec=fallback_dur,
        fallback_throughput_items_per_sec=fallback_items_per_sec,
        fallback_throughput_mb_per_sec=fallback_mb_per_sec,
        speedup_ratio=speedup,
    )


def benchmark_text_cleaner(records: List[str], total_bytes: int) -> BenchmarkResult:
    def rust_call():
        return rs.clean_batch_records(records, True, True, 10, 10000, False, None)

    def fallback_call():
        return fb.clean_batch_records(records, True, True, 10, 10000, False, None)

    return run_timed_benchmark(
        name="TextCleaner.clean_batch (CRLF + Trimming + Bounds)",
        rust_fn=rust_call,
        fallback_fn=fallback_call,
        workload_size=len(records),
        data_bytes=total_bytes,
    )


def benchmark_quality_filter(records: List[str], total_bytes: int) -> BenchmarkResult:
    rust_filter = rs.QualityFilter(
        min_chars=10,
        max_chars=20000,
        min_words=3,
        min_entropy=2.0,
        max_repetition_ratio=0.3,
        max_symbol_ratio=0.25,
    )
    fallback_filter = fb.QualityFilter(
        min_chars=10,
        max_chars=20000,
        min_words=3,
        min_entropy=2.0,
        max_repetition_ratio=0.3,
        max_symbol_ratio=0.25,
    )

    def rust_call():
        return rust_filter.filter_batch(records)

    def fallback_call():
        return fallback_filter.filter_batch(records)

    return run_timed_benchmark(
        name="QualityFilter.filter_batch (Entropy + Repetition + Symbols)",
        rust_fn=rust_call,
        fallback_fn=fallback_call,
        workload_size=len(records),
        data_bytes=total_bytes,
    )


def benchmark_stream_dedup(records: List[str], total_bytes: int) -> BenchmarkResult:
    # Inject 25% duplicate records
    dup_records = list(records)
    for i in range(len(dup_records) // 4):
        dup_records[i * 4] = dup_records[i * 4 + 1]

    def rust_call():
        dedup = rs.StreamDeduplicator()
        passed = dedup.filter_batch(dup_records)
        dedup.commit()
        return passed

    def fallback_call():
        dedup = fb.StreamDeduplicator()
        passed = dedup.filter_batch(dup_records)
        dedup.commit()
        return passed

    return run_timed_benchmark(
        name="StreamDeduplicator.filter_batch (Exact XXH3-64 Staging & Commit)",
        rust_fn=rust_call,
        fallback_fn=fallback_call,
        workload_size=len(dup_records),
        data_bytes=total_bytes,
    )


def benchmark_minhash_dedup(records: List[str], total_bytes: int) -> BenchmarkResult:
    # Use 500 documents for MinHash permutation benchmarking
    sample_records = records[:500]
    sample_bytes = sum(len(r.encode("utf-8")) for r in sample_records)

    def rust_call():
        dedup = rs.MinHashDeduplicator(0.8, 5, 64)
        for r in sample_records:
            dedup.is_unique(r)
        return dedup.count()

    def fallback_call():
        dedup = fb.MinHashDeduplicator(0.8, 5, 64)
        for r in sample_records:
            dedup.is_unique(r)
        return dedup.count()

    return run_timed_benchmark(
        name="MinHashDeduplicator.is_unique (64-Permutation MinHash Signatures)",
        rust_fn=rust_call,
        fallback_fn=fallback_call,
        workload_size=len(sample_records),
        data_bytes=sample_bytes,
    )


def benchmark_state_tracker() -> BenchmarkResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        rust_wal = Path(tmpdir) / "rust_tracker.wal"
        fb_wal = Path(tmpdir) / "fb_tracker.wal"

        rust_tracker = rs.StateTracker(str(rust_wal))
        fallback_tracker = fb.StateTracker(str(fb_wal))

        num_commits = 2000

        def rust_call():
            for i in range(num_commits):
                rust_tracker.update(1, 1, i * 32, i * 1024, 0)
            rust_tracker.commit()

        def fallback_call():
            for i in range(num_commits):
                fallback_tracker.update(1, 1, i * 32, i * 1024, 0)
            fallback_tracker.commit()

        return run_timed_benchmark(
            name="StateTracker.update & commit (128-Byte Atomic CRC32 WAL)",
            rust_fn=rust_call,
            fallback_fn=fallback_call,
            workload_size=num_commits,
            data_bytes=num_commits * 128,
        )


def benchmark_string_tensor(records: List[str], total_bytes: int) -> BenchmarkResult:
    def rust_call():
        st = rs.StringTensor.from_list(records)
        hashes = st.hash_all()
        filtered = st.filter_by_length(20, 1000)
        return len(filtered.to_list()), len(hashes)

    def fallback_call():
        st = fb.StringTensor.from_list(records)
        hashes = st.hash_all()
        filtered = st.filter_by_length(20, 1000)
        return len(filtered.to_list()), len(hashes)

    return run_timed_benchmark(
        name="StringTensor (Umbra-style Layout, Hash All & Length Filter)",
        rust_fn=rust_call,
        fallback_fn=fallback_call,
        workload_size=len(records),
        data_bytes=total_bytes,
    )


def benchmark_fused_pipeline(records: List[str], total_bytes: int) -> BenchmarkResult:
    def rust_call():
        dedup = rs.StreamDeduplicator()
        st = rs.fused_pipeline_process_batch(
            records=records,
            trim_whitespace=True,
            normalize_newlines=True,
            min_byte_len=10,
            max_byte_len=10000,
            truncate_over_max=False,
            json_extract_key=None,
            filter_quality=True,
            min_chars=10,
            max_chars=10000,
            min_words=3,
            min_entropy=2.0,
            max_repetition_ratio=0.3,
            max_symbol_ratio=0.25,
            deduplicator=dedup,
            commit_dedup=True,
        )
        return st.total_bytes()

    def fallback_call():
        dedup = fb.StreamDeduplicator()
        st = fb.fused_pipeline_process_batch(
            records=records,
            trim_whitespace=True,
            normalize_newlines=True,
            min_byte_len=10,
            max_byte_len=10000,
            truncate_over_max=False,
            json_extract_key=None,
            filter_quality=True,
            min_chars=10,
            max_chars=10000,
            min_words=3,
            min_entropy=2.0,
            max_repetition_ratio=0.3,
            max_symbol_ratio=0.25,
            deduplicator=dedup,
            commit_dedup=True,
        )
        return st.total_bytes()

    return run_timed_benchmark(
        name="FusedPipeline.fused_process_batch (End-to-End SIMD Single Pass)",
        rust_fn=rust_call,
        fallback_fn=fallback_call,
        workload_size=len(records),
        data_bytes=total_bytes,
    )


def benchmark_serialization(records: List[str]) -> List[BenchmarkResult]:
    cols = {
        "id": [str(i) for i in range(len(records))],
        "text": records,
        "score": [str(round(math.sin(i) * 10, 4)) for i in range(len(records))],
    }
    total_data_bytes = sum(len(c.encode("utf-8")) for col in cols.values() for c in col)

    results = []
    for fmt in ["parquet", "arrow", "jsonl", "csv"]:
        def rust_call(fmt=fmt):
            return rs.serialize_table_to_bytes(cols, fmt, False)

        def fallback_call(fmt=fmt):
            return fb.serialize_table_to_bytes(cols, fmt, False)

        res = run_timed_benchmark(
            name=f"Serialization: {fmt.upper()}",
            rust_fn=rust_call,
            fallback_fn=fallback_call,
            workload_size=len(records),
            data_bytes=total_data_bytes,
        )
        results.append(res)
    return results


def main() -> None:
    print("=================================================================")
    print("  LuminaMoon Data Processing Engine Benchmark Suite")
    print(f"  CPU Architecture / SIMD ISA: {rs.dataloader_cpu_isa()}")
    print("=================================================================\n")

    num_records = 10000
    print(f"Generating synthetic workload ({num_records:,} records)...")
    records, total_bytes = generate_synthetic_dataset(num_records=num_records)
    print(f"Generated {len(records):,} records, {total_bytes / (1024*1024):.2f} MB payload.\n")

    benchmarks = [
        ("Text Cleaner", lambda: benchmark_text_cleaner(records, total_bytes)),
        ("Quality Filter", lambda: benchmark_quality_filter(records, total_bytes)),
        ("Stream Deduplicator", lambda: benchmark_stream_dedup(records, total_bytes)),
        ("MinHash Deduplicator", lambda: benchmark_minhash_dedup(records, total_bytes)),
        ("State Tracker", lambda: benchmark_state_tracker()),
        ("String Tensor", lambda: benchmark_string_tensor(records, total_bytes)),
        ("Fused Pipeline", lambda: benchmark_fused_pipeline(records, total_bytes)),
    ]

    all_results: List[BenchmarkResult] = []

    for label, bench_fn in benchmarks:
        print(f"Running benchmark: {label}...")
        res = bench_fn()
        all_results.append(res)
        print(f"  -> Rust:     {res.rust_throughput_items_per_sec:>10,.0f} items/s ({res.rust_throughput_mb_per_sec:>8.2f} MB/s, {res.rust_duration_sec*1000:>7.2f} ms)")
        print(f"  -> Fallback: {res.fallback_throughput_items_per_sec:>10,.0f} items/s ({res.fallback_throughput_mb_per_sec:>8.2f} MB/s, {res.fallback_duration_sec*1000:>7.2f} ms)")
        print(f"  -> Speedup:  {res.speedup_ratio:>7.2f}x\n")

    print("Running Serialization Benchmarks...")
    ser_results = benchmark_serialization(records[:5000])
    for res in ser_results:
        all_results.append(res)
        print(f"  -> {res.benchmark_name:<28}: Rust {res.rust_throughput_mb_per_sec:>8.2f} MB/s vs Fallback {res.fallback_throughput_mb_per_sec:>8.2f} MB/s (Speedup: {res.speedup_ratio:>6.2f}x)")

    # Save output json
    out_dir = Path(__file__).resolve().parent.parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_data_processing_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cpu_isa": rs.dataloader_cpu_isa(),
                "num_records": num_records,
                "total_mb": total_bytes / (1024 * 1024),
                "benchmarks": [asdict(r) for r in all_results],
            },
            f,
            indent=2,
        )
    print(f"\nSaved raw benchmark metrics to {out_file}")


if __name__ == "__main__":
    main()
