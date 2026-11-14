#!/usr/bin/env python3
"""Benchmark and Comparative Suite for LuminaV across all execution and buffer modes.

Tests training performance on highly complex, pathological synthetic image data
(multi-octave fractal turbulence, non-stationary Moire/Chirp waves, chaotic manifolds)
using a deep convolutional vision architecture (ResNet-style ConvNet with residual blocks).
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from luminamoon import LuminaV


# 1. Pathological / Highly Complex Synthetic Vision Dataset Generation
def generate_hard_synthetic_vision_dataset(
    num_samples: int = 256,
    height: int = 64,
    width: int = 64,
    num_classes: int = 8,
    seed: int = 2026,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates an extremely difficult synthetic image dataset.

    Features:
    - Channel 0: Multi-octave Fractal Brownian Motion (fBm) with non-stationary phase shifts.
    - Channel 1: High-frequency chirp & radial Moire vortex with cross-quadrature modulation.
    - Channel 2: Chaotic double-well reaction-diffusion attractor pattern with high spatial curvature.
    - Target: Non-linear topological invariant coupling cross-channel gradients with label ambiguity.
    """
    rng = np.random.RandomState(seed)

    # Coordinate grids normalized to [-2, 2]
    y_coords, x_coords = np.meshgrid(
        np.linspace(-2.0, 2.0, height, dtype=np.float32),
        np.linspace(-2.0, 2.0, width, dtype=np.float32),
        indexing="ij",
    )
    r_sq = x_coords**2 + y_coords**2
    theta = np.arctan2(y_coords, x_coords)

    images = np.zeros((num_samples, 3, height, width), dtype=np.float32)
    labels = np.zeros((num_samples,), dtype=np.int64)

    for i in range(num_samples):
        # Sample-specific chaotic random parameters
        omega = rng.uniform(8.0, 18.0)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        k_oct = rng.uniform(3.0, 7.0)

        # Ch 0: Fractal multi-octave turbulence
        ch0 = np.zeros((height, width), dtype=np.float32)
        for octave in range(1, 5):
            freq = 2.0**octave
            amp = 1.0 / freq
            angle = phi + octave * 0.785
            proj = x_coords * math.cos(angle) + y_coords * math.sin(angle)
            ch0 += amp * np.sin(freq * k_oct * proj + phi)
        # Add high-frequency micro-checkerboard jitter
        checker = ((np.arange(height)[:, None] + np.arange(width)[None, :]) % 2) * 2.0 - 1.0
        ch0 += 0.25 * checker * np.cos(omega * r_sq)

        # Ch 1: High-frequency Chirp / Moire radial vortex
        ch1 = np.sin(omega * r_sq + 4.0 * theta + phi) * np.cos(k_oct * x_coords * y_coords)
        # Cross-quadrature modulation
        ch1 += 0.35 * np.cos(15.0 * (x_coords + y_coords) + phi)

        # Ch 2: Chaotic double-well / non-linear attractor
        poly = (x_coords**3 - 3.0 * x_coords * (y_coords**2)) * 0.5
        ch2 = np.tanh(poly * 2.0 + np.cos(omega * y_coords))
        ch2 += 0.3 * np.sin(20.0 * x_coords + phi) * np.sin(20.0 * y_coords)

        # Combine channels and normalize
        images[i, 0] = (ch0 - ch0.mean()) / (ch0.std() + 1e-6)
        images[i, 1] = (ch1 - ch1.mean()) / (ch1.std() + 1e-6)
        images[i, 2] = (ch2 - ch2.mean()) / (ch2.std() + 1e-6)

        # Compute complex topological functional for target class:
        # Cross-gradient winding energy + multi-channel nonlinear coupling
        dx_ch1 = np.gradient(ch1, axis=1)
        dy_ch2 = np.gradient(ch2, axis=0)
        curl_cross = np.mean(ch0 * (dx_ch1 * dy_ch2))
        vol = np.mean(ch0 * ch1 * ch2)
        quad = np.mean(np.sin(r_sq * 5.0) * ch0)

        # Map to discrete class [0..num_classes-1] via non-linear hash
        hash_val = math.sin(curl_cross * 25.0 + vol * 15.0 + quad * 10.0 + phi)
        class_idx = int(math.floor(((hash_val + 1.0) / 2.0) * num_classes)) % num_classes

        # Inject 10% label ambiguity / boundary noise to test cautious clamping & regularization
        if rng.rand() < 0.10:
            class_idx = rng.randint(0, num_classes)

        labels[i] = class_idx

    return torch.from_numpy(images), torch.from_numpy(labels)


# 2. Deep Convolutional Vision Architecture (ResNet-Lite)
class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.silu(out + res)


class HardVisionConvNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 8):
        super().__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        # Stage 1 (32 channels)
        self.res1 = ResidualConvBlock(32)
        self.pool1 = nn.MaxPool2d(2)  # 64 -> 32

        # Stage 2 (64 channels)
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.res2 = ResidualConvBlock(64)
        self.pool2 = nn.MaxPool2d(2)  # 32 -> 16

        # Stage 3 (128 channels)
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.res3 = ResidualConvBlock(128)
        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))  # 128 x 2 x 2 = 512

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.trans2(x)
        x = self.res3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# 3. Benchmark Training Routine for Single Configuration
def run_benchmark_trial(
    config_name: str,
    optimizer_fn: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    steps: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HardVisionConvNet(in_channels=3, num_classes=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    num_tensors = sum(1 for p in model.parameters())

    losses: List[float] = []
    step_times: List[float] = []
    accuracies: List[float] = []

    num_samples = images.shape[0]

    # Pre-split into deterministic batches
    batch_indices = [
        slice(
            (s * batch_size) % (num_samples - batch_size),
            ((s * batch_size) % (num_samples - batch_size)) + batch_size,
        )
        for s in range(steps)
    ]

    total_start = time.perf_counter()

    for step_idx in range(steps):
        b_slice = batch_indices[step_idx]
        batch_x = images[b_slice]
        batch_y = labels[b_slice]

        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        t1 = time.perf_counter()

        loss_val = float(loss.item())
        losses.append(loss_val)
        step_times.append((t1 - t0) * 1000.0)  # ms

        preds = torch.argmax(logits, dim=1)
        acc = float((preds == batch_y).float().mean().item()) * 100.0
        accuracies.append(acc)

    total_duration_ms = (time.perf_counter() - total_start) * 1000.0

    # Calculate optimizer state memory footprint
    opt_state_bytes = 0
    if hasattr(optimizer, "state"):
        for p, st in optimizer.state.items():
            if isinstance(st, dict):
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        opt_state_bytes += v.numel() * v.element_size()
                    elif isinstance(v, np.ndarray):
                        opt_state_bytes += v.nbytes

    initial_loss = losses[0]
    final_loss = losses[-1]
    pct_drop = ((initial_loss - final_loss) / initial_loss) * 100.0
    avg_step_ms = float(np.mean(step_times))

    return {
        "name": config_name,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": pct_drop,
        "final_accuracy": accuracies[-1],
        "total_time_ms": total_duration_ms,
        "avg_step_ms": avg_step_ms,
        "state_mem_mb": opt_state_bytes / (1024.0 * 1024.0),
        "total_params": total_params,
        "num_tensors": num_tensors,
        "losses": losses,
    }


# 4. Main Suite Comparing All Modes
def main():
    print("=" * 80)
    print("LuminaV Vision Mode Benchmark & Comparative Training Suite")
    print("=" * 80)

    hw = LuminaV.detect_hardware()
    print("[Hardware Environment]")
    print(f"  CPU SIMD Vectorization: {hw.get('simd_backend')}")
    print(f"  Vulkan GPU Acceleration: {hw.get('vulkan_available')} ({hw.get('gpu_device_name')})")
    print(f"  Triton AOT Available:    {hw.get('triton_aot_available')} (arch: {hw.get('triton_arch')})")
    print(f"  Wave32 Hardware Support: {hw.get('wave32_supported')}")
    try:
        import triton
        print(f"  Triton Windows Package:  Version {triton.__version__} ({triton.__file__})")
    except Exception as e:
        print(f"  Triton Windows Package:  Failed ({e})")
    print("-" * 80)

    print("[Dataset Generation] Synthesizing pathological hard vision images (64x64, 3ch)...")
    images, labels = generate_hard_synthetic_vision_dataset(
        num_samples=256, height=64, width=64, num_classes=8, seed=2026
    )
    print(f"  Generated {images.shape[0]} images | Shape: {images.shape} | Labels: {labels.shape}")
    print(f"  Label distribution: {torch.bincount(labels).tolist()}")
    print("-" * 80)

    # Define test configurations across all LuminaV modes and architectures
    configurations = [
        # --- Triton Windows Backends ---
        (
            "LuminaV (triton, Dual-Buffer)",
            lambda p: LuminaV(p, lr=1e-3, buffer=2, execution="triton", bound=True, cautious=True),
        ),
        (
            "LuminaV (triton, Single-Buffer TD-CLM)",
            lambda p: LuminaV(p, lr=1e-3, buffer=1, execution="triton", bound=True, cautious=True),
        ),
        # --- Hardware Backends (Vulkan GPU, SIMD AVX-512, Scalar Core) ---
        (
            "LuminaV (vulkan, Dual-Buffer)",
            lambda p: LuminaV(p, lr=1e-3, buffer=2, execution="vulkan", bound=True, cautious=True),
        ),
        (
            "LuminaV (simd AVX-512, Dual-Buffer)",
            lambda p: LuminaV(p, lr=1e-3, buffer=2, execution="simd", bound=True, cautious=True),
        ),
        (
            "LuminaV (scalar, Dual-Buffer)",
            lambda p: LuminaV(p, lr=1e-3, buffer=2, execution="scalar", bound=True, cautious=True),
        ),
        # --- Buffer Architectures ---
        (
            "LuminaV (auto, Single-Buffer TD-CLM)",
            lambda p: LuminaV(p, lr=1e-3, buffer=1, execution="auto", bound=True, cautious=True),
        ),
        (
            "LuminaV (auto, Dual-Buffer CLM)",
            lambda p: LuminaV(p, lr=1e-3, buffer=2, execution="auto", bound=True, cautious=True),
        ),
        (
            "LuminaV (auto, T1 Single-Buffer Triangle)",
            lambda p: LuminaV(p, lr=1e-3, buffer="T1", execution="auto", bound=True, cautious=True),
        ),
        (
            "LuminaV (auto, T2 Dual-Buffer Triangle)",
            lambda p: LuminaV(p, lr=1e-3, buffer="T2", execution="auto", bound=True, cautious=True),
        ),
        # --- Advanced Vision Features ---
        (
            "LuminaV (auto, fuse_tensors=True)",
            lambda p: LuminaV(p, lr=1e-3, buffer=2, execution="auto", fuse_tensors=True),
        ),
        (
            "LuminaV (auto, radial bound)",
            lambda p: LuminaV(p, lr=1e-3, buffer=2, execution="auto", bound=True, bound_type="radial"),
        ),
        # --- Baseline Comparison ---
        (
            "Baseline AdamW (PyTorch)",
            lambda p: torch.optim.AdamW(p, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01),
        ),
    ]

    results: List[Dict[str, Any]] = []

    print("[Training Phase] Benchmarking 20 optimization steps per configuration on ResNet-Lite...")
    for idx, (name, opt_fn) in enumerate(configurations, 1):
        print(f"\n[{idx:02d}/{len(configurations):02d}] Running trial: {name} ...", flush=True)
        res = run_benchmark_trial(
            name, opt_fn, images, labels, batch_size=64, steps=20, seed=42
        )
        results.append(res)
        print(
            f"      Loss: {res['initial_loss']:.4f} -> {res['final_loss']:.4f} "
            f"({res['loss_reduction_pct']:+.2f}%) | "
            f"Acc: {res['final_accuracy']:.1f}% | "
            f"Step: {res['avg_step_ms']:.2f} ms | "
            f"State Mem: {res['state_mem_mb']:.2f} MB"
        )

    print("\n" + "=" * 105)
    print("COMPREHENSIVE BENCHMARK COMPARISON TABLE")
    print("=" * 105)
    header = (
        f"{'Mode Configuration':<40} | {'Init Loss':<10} | {'Final Loss':<10} | "
        f"{'Reduction':<10} | {'Accuracy':<9} | {'Avg Step':<11} | {'State RAM':<9}"
    )
    print(header)
    print("-" * 105)

    for r in results:
        row = (
            f"{r['name']:<40} | "
            f"{r['initial_loss']:<10.4f} | "
            f"{r['final_loss']:<10.4f} | "
            f"{r['loss_reduction_pct']:>8.2f}% | "
            f"{r['final_accuracy']:>7.1f}% | "
            f"{r['avg_step_ms']:>8.2f} ms | "
            f"{r['state_mem_mb']:>7.2f} MB"
        )
        print(row)

    print("=" * 105)


if __name__ == "__main__":
    main()
