use std::fs;
use std::path::{Path, PathBuf};

use luminav_core::{
    _prng, round_to_nearest_bf16, round_to_nearest_fp16, stochastic_round_bf16,
    stochastic_round_fp16,
};
use luminav_simd::{SimdBackend, detect_simd_backend};

fn sample_edge_inputs<const N: usize>() -> [f32; N] {
    std::array::from_fn(|i| match i {
        0 => 0.0_f32,
        1 => -0.0_f32,
        2 => f32::from_bits(102 << 23),
        3 => f32::from_bits((105 << 23) | 0x123456),
        4 => -f32::from_bits((110 << 23) | 0x654321),
        5 => f32::NAN,
        6 => f32::INFINITY,
        7 => f32::NEG_INFINITY,
        _ => (i as f32) * 0.123_456_7_f32 - 0.5_f32,
    })
}

fn assert_bits_or_nan(actual: f32, expected: f32, label: &str, idx: usize) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "[{label}] lane {idx} expected NaN, got {actual}"
        );
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "[{label}] lane {idx} bit mismatch: actual={actual} expected={expected}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_bitwise_prng_and_sr_fp16_bf16_avx512_exact() {
    if detect_simd_backend() != SimdBackend::Avx512 {
        return;
    }
    unsafe {
        use std::arch::x86_64::{
            _mm512_loadu_ps, _mm512_set_epi32, _mm512_set1_epi32, _mm512_storeu_ps,
            _mm512_storeu_si512,
        };
        let seed = 1337_i32;
        let v_seed = _mm512_set1_epi32(seed);
        let v_offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        let v_prng = luminav_simd::avx512::prng_avx512(v_seed, v_offsets);
        let mut prng_lanes = [0_i32; 16];
        _mm512_storeu_si512(prng_lanes.as_mut_ptr().cast(), v_prng);
        for (idx, &val) in prng_lanes.iter().enumerate() {
            assert_eq!(val, _prng(seed, idx as i32));
        }

        let inputs = sample_edge_inputs::<16>();
        let v_in = _mm512_loadu_ps(inputs.as_ptr());
        let mut bf16_out = [0.0_f32; 16];
        let mut rne_bf16_out = [0.0_f32; 16];
        let mut fp16_out = [0.0_f32; 16];
        let mut rne_fp16_out = [0.0_f32; 16];

        _mm512_storeu_ps(
            bf16_out.as_mut_ptr(),
            luminav_simd::avx512::stochastic_round_bf16_avx512(v_in, v_seed, v_offsets),
        );
        _mm512_storeu_ps(
            rne_bf16_out.as_mut_ptr(),
            luminav_simd::avx512::round_to_nearest_bf16_avx512(v_in),
        );
        _mm512_storeu_ps(
            fp16_out.as_mut_ptr(),
            luminav_simd::avx512::stochastic_round_fp16_avx512(v_in, v_seed, v_offsets),
        );
        _mm512_storeu_ps(
            rne_fp16_out.as_mut_ptr(),
            luminav_simd::avx512::round_to_nearest_fp16_avx512(v_in),
        );

        for i in 0..16 {
            assert_bits_or_nan(
                bf16_out[i],
                stochastic_round_bf16(inputs[i], seed, i as i32),
                "AVX512 SR BF16",
                i,
            );
            assert_bits_or_nan(
                rne_bf16_out[i],
                round_to_nearest_bf16(inputs[i]),
                "AVX512 RNE BF16",
                i,
            );
            assert_bits_or_nan(
                fp16_out[i],
                stochastic_round_fp16(inputs[i], seed, i as i32),
                "AVX512 SR FP16",
                i,
            );
            assert_bits_or_nan(
                rne_fp16_out[i],
                round_to_nearest_fp16(inputs[i]),
                "AVX512 RNE FP16",
                i,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_bitwise_prng_and_sr_fp16_bf16_avx2_exact() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return;
    }
    unsafe {
        use std::arch::x86_64::{
            _mm256_loadu_ps, _mm256_set_epi32, _mm256_set1_epi32, _mm256_storeu_ps,
            _mm256_storeu_si256,
        };
        let seed = 1337_i32;
        let v_seed = _mm256_set1_epi32(seed);
        let v_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

        let v_prng = luminav_simd::avx2::prng_avx2(v_seed, v_offsets);
        let mut prng_lanes = [0_i32; 8];
        _mm256_storeu_si256(prng_lanes.as_mut_ptr().cast(), v_prng);
        for (idx, &val) in prng_lanes.iter().enumerate() {
            assert_eq!(val, _prng(seed, idx as i32));
        }

        let inputs = sample_edge_inputs::<8>();
        let v_in = _mm256_loadu_ps(inputs.as_ptr());
        let mut bf16_out = [0.0_f32; 8];
        let mut rne_bf16_out = [0.0_f32; 8];
        let mut fp16_out = [0.0_f32; 8];
        let mut rne_fp16_out = [0.0_f32; 8];

        _mm256_storeu_ps(
            bf16_out.as_mut_ptr(),
            luminav_simd::avx2::stochastic_round_bf16_avx2(v_in, v_seed, v_offsets),
        );
        _mm256_storeu_ps(
            rne_bf16_out.as_mut_ptr(),
            luminav_simd::avx2::round_to_nearest_bf16_avx2(v_in),
        );
        _mm256_storeu_ps(
            fp16_out.as_mut_ptr(),
            luminav_simd::avx2::stochastic_round_fp16_avx2(v_in, v_seed, v_offsets),
        );
        _mm256_storeu_ps(
            rne_fp16_out.as_mut_ptr(),
            luminav_simd::avx2::round_to_nearest_fp16_avx2(v_in),
        );

        for i in 0..8 {
            assert_bits_or_nan(
                bf16_out[i],
                stochastic_round_bf16(inputs[i], seed, i as i32),
                "AVX2 SR BF16",
                i,
            );
            assert_bits_or_nan(
                rne_bf16_out[i],
                round_to_nearest_bf16(inputs[i]),
                "AVX2 RNE BF16",
                i,
            );
            assert_bits_or_nan(
                fp16_out[i],
                stochastic_round_fp16(inputs[i], seed, i as i32),
                "AVX2 SR FP16",
                i,
            );
            assert_bits_or_nan(
                rne_fp16_out[i],
                round_to_nearest_fp16(inputs[i]),
                "AVX2 RNE FP16",
                i,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_bitwise_prng_and_sr_fp16_bf16_neon_exact() {
    if detect_simd_backend() != SimdBackend::Neon {
        return;
    }
    unsafe {
        use std::arch::aarch64::{vdupq_n_u32, vld1q_f32, vld1q_u32, vst1q_f32, vst1q_u32};
        let seed = 1337_i32;
        let v_seed = vdupq_n_u32(seed as u32);
        let offs_arr: [u32; 4] = [0, 1, 2, 3];
        let v_offsets = vld1q_u32(offs_arr.as_ptr());

        let v_prng = luminav_simd::neon::prng_neon(v_seed, v_offsets);
        let mut prng_lanes = [0_u32; 4];
        vst1q_u32(prng_lanes.as_mut_ptr(), v_prng);
        for (idx, &val) in prng_lanes.iter().enumerate() {
            assert_eq!(val as i32, _prng(seed, idx as i32));
        }

        let inputs = sample_edge_inputs::<8>();
        for chunk in 0..2 {
            let base = chunk * 4;
            let offs_chunk: [u32; 4] = [
                base as u32,
                (base + 1) as u32,
                (base + 2) as u32,
                (base + 3) as u32,
            ];
            let v_off = vld1q_u32(offs_chunk.as_ptr());
            let v_in = vld1q_f32(inputs[base..base + 4].as_ptr());

            let mut bf16_out = [0.0_f32; 4];
            let mut rne_bf16_out = [0.0_f32; 4];
            let mut fp16_out = [0.0_f32; 4];
            let mut rne_fp16_out = [0.0_f32; 4];

            vst1q_f32(
                bf16_out.as_mut_ptr(),
                luminav_simd::neon::stochastic_round_bf16_neon(v_in, v_seed, v_off),
            );
            vst1q_f32(
                rne_bf16_out.as_mut_ptr(),
                luminav_simd::neon::round_to_nearest_bf16_neon(v_in),
            );
            vst1q_f32(
                fp16_out.as_mut_ptr(),
                luminav_simd::neon::stochastic_round_fp16_neon(v_in, v_seed, v_off),
            );
            vst1q_f32(
                rne_fp16_out.as_mut_ptr(),
                luminav_simd::neon::round_to_nearest_fp16_neon(v_in),
            );

            for lane in 0..4 {
                let idx = base + lane;
                assert_bits_or_nan(
                    bf16_out[lane],
                    stochastic_round_bf16(inputs[idx], seed, idx as i32),
                    "NEON SR BF16",
                    idx,
                );
                assert_bits_or_nan(
                    rne_bf16_out[lane],
                    round_to_nearest_bf16(inputs[idx]),
                    "NEON RNE BF16",
                    idx,
                );
                assert_bits_or_nan(
                    fp16_out[lane],
                    stochastic_round_fp16(inputs[idx], seed, idx as i32),
                    "NEON SR FP16",
                    idx,
                );
                assert_bits_or_nan(
                    rne_fp16_out[lane],
                    round_to_nearest_fp16(inputs[idx]),
                    "NEON RNE FP16",
                    idx,
                );
            }
        }
    }
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read crate dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn test_luminav_simd_files_respect_line_budget() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_rs_files(&crate_root.join("src"), &mut files);
    collect_rs_files(&crate_root.join("tests"), &mut files);

    let mut violations = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(crate_root)
            .expect("relative path")
            .to_string_lossy()
            .replace('\\', "/");
        let lines = fs::read_to_string(&path)
            .expect("read rs file")
            .lines()
            .count();
        let ceiling = if rel.starts_with("tests/") {
            650
        } else if rel.starts_with("src/avx2/")
            || rel.starts_with("src/avx512/")
            || rel.starts_with("src/neon/")
            || rel == "src/ra.rs"
        {
            600
        } else {
            450
        };
        if lines > ceiling {
            violations.push(format!("{rel}: {lines} lines (limit {ceiling})"));
        }
    }

    assert!(
        violations.is_empty(),
        "luminav-simd line-count budget violations:\n{}",
        violations.join("\n")
    );
}
