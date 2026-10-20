use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuIsa {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

impl CpuIsa {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Avx512 => "AVX-512",
            Self::Avx2 => "AVX2",
            Self::Neon => "NEON",
            Self::Scalar => "SCALAR",
        }
    }
}

#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct CacheAlignedBuffer {
    pub data: Vec<u8>,
}

impl CacheAlignedBuffer {
    pub fn with_capacity(cap: usize) -> Self {
        let aligned_cap = (cap + 63) & !63;
        Self {
            data: Vec::with_capacity(aligned_cap),
        }
    }
}

#[inline]
pub fn detect_cpu_isa() -> CpuIsa {
    static CACHED_ISA: OnceLock<CpuIsa> = OnceLock::new();
    *CACHED_ISA.get_or_init(|| {
        std::cfg_select! {
            any(target_arch = "x86", target_arch = "x86_64") => {
                if std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                {
                    CpuIsa::Avx512
                } else if std::arch::is_x86_feature_detected!("avx2") {
                    CpuIsa::Avx2
                } else {
                    CpuIsa::Scalar
                }
            }
            target_arch = "aarch64" => {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    CpuIsa::Neon
                } else {
                    CpuIsa::Scalar
                }
            }
            _ => {
                CpuIsa::Scalar
            }
        }
    })
}

#[inline]
pub fn find_byte_fast(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match detect_cpu_isa() {
            CpuIsa::Avx512 => return unsafe { find_byte_avx512(haystack, needle) },
            CpuIsa::Avx2 => return unsafe { find_byte_avx2(haystack, needle) },
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if detect_cpu_isa() == CpuIsa::Neon {
            return unsafe { find_byte_neon(haystack, needle) };
        }
    }
    haystack.iter().position(|&b| b == needle)
}

#[inline]
pub fn rfind_byte_fast(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match detect_cpu_isa() {
            CpuIsa::Avx512 => return unsafe { rfind_byte_avx512(haystack, needle) },
            CpuIsa::Avx2 => return unsafe { rfind_byte_avx2(haystack, needle) },
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if detect_cpu_isa() == CpuIsa::Neon {
            return unsafe { rfind_byte_neon(haystack, needle) };
        }
    }
    haystack.iter().rposition(|&b| b == needle)
}

#[inline]
pub fn contains_byte_fast(haystack: &[u8], needle: u8) -> bool {
    find_byte_fast(haystack, needle).is_some()
}

#[inline]
pub fn find_newline_fast(haystack: &[u8]) -> Option<usize> {
    find_byte_fast(haystack, b'\n')
}

#[inline]
pub fn rfind_newline_fast(haystack: &[u8]) -> Option<usize> {
    rfind_byte_fast(haystack, b'\n')
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn find_byte_avx512(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let v_needle = _mm512_set1_epi8(needle as i8);
    let mut offset = 0usize;

    while offset + 64 <= len {
        let chunk = unsafe { _mm512_loadu_si512(ptr.add(offset).cast()) };
        let mask = _mm512_cmpeq_epi8_mask(chunk, v_needle);
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 64;
    }

    while offset < len {
        if unsafe { *ptr.add(offset) } == needle {
            return Some(offset);
        }
        offset += 1;
    }
    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn rfind_byte_avx512(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let v_needle = _mm512_set1_epi8(needle as i8);
    let mut end = len;

    while end >= 64 {
        let start = end - 64;
        let chunk = unsafe { _mm512_loadu_si512(ptr.add(start).cast()) };
        let mask = _mm512_cmpeq_epi8_mask(chunk, v_needle);
        if mask != 0 {
            let highest_bit = 63 - mask.leading_zeros() as usize;
            return Some(start + highest_bit);
        }
        end = start;
    }

    while end > 0 {
        end -= 1;
        if unsafe { *ptr.add(end) } == needle {
            return Some(end);
        }
    }
    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn find_byte_avx2(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let v_needle = _mm256_set1_epi8(needle as i8);
    let mut offset = 0usize;

    while offset + 32 <= len {
        let chunk = unsafe { _mm256_loadu_si256(ptr.add(offset).cast()) };
        let cmp = _mm256_cmpeq_epi8(chunk, v_needle);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 32;
    }

    while offset < len {
        if unsafe { *ptr.add(offset) } == needle {
            return Some(offset);
        }
        offset += 1;
    }
    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn rfind_byte_avx2(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let v_needle = _mm256_set1_epi8(needle as i8);
    let mut end = len;

    while end >= 32 {
        let start = end - 32;
        let chunk = unsafe { _mm256_loadu_si256(ptr.add(start).cast()) };
        let cmp = _mm256_cmpeq_epi8(chunk, v_needle);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0 {
            let highest_bit = 31 - mask.leading_zeros() as usize;
            return Some(start + highest_bit);
        }
        end = start;
    }

    while end > 0 {
        end -= 1;
        if unsafe { *ptr.add(end) } == needle {
            return Some(end);
        }
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_byte_neon(haystack: &[u8], needle: u8) -> Option<usize> {
    use std::arch::aarch64::*;

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let v_needle = vdupq_n_u8(needle);
    let mut offset = 0usize;

    while offset + 16 <= len {
        let chunk = unsafe { vld1q_u8(ptr.add(offset)) };
        let cmp = vceqq_u8(chunk, v_needle);
        if vmaxvq_u8(cmp) != 0 {
            for lane in 0..16 {
                if unsafe { *ptr.add(offset + lane) } == needle {
                    return Some(offset + lane);
                }
            }
        }
        offset += 16;
    }

    while offset < len {
        if unsafe { *ptr.add(offset) } == needle {
            return Some(offset);
        }
        offset += 1;
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rfind_byte_neon(haystack: &[u8], needle: u8) -> Option<usize> {
    use std::arch::aarch64::*;

    let len = haystack.len();
    let ptr = haystack.as_ptr();
    let v_needle = vdupq_n_u8(needle);
    let mut end = len;

    while end >= 16 {
        let start = end - 16;
        let chunk = unsafe { vld1q_u8(ptr.add(start)) };
        let cmp = vceqq_u8(chunk, v_needle);
        if vmaxvq_u8(cmp) != 0 {
            for lane in (0..16).rev() {
                if unsafe { *ptr.add(start + lane) } == needle {
                    return Some(start + lane);
                }
            }
        }
        end = start;
    }

    while end > 0 {
        end -= 1;
        if unsafe { *ptr.add(end) } == needle {
            return Some(end);
        }
    }
    None
}
