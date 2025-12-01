//! PCLMUL (Carry-less Multiplication) implementations for x86-64
//!
//! These implementations use the PCLMULQDQ instruction for efficient
//! GF(2^16) multiplication with Barrett reduction.
//!
//! The Barrett reduction uses the following constants for polynomial 0x1100B:
//! - Barrett constant: 0x1111a (derived from floor(x^32 / poly))
//! - Reduction polynomial: 0x100b (low 16 bits of 0x1100B)

use crate::galois::simd::{GaloisSimdStrategy, Priority};
use std::arch::x86_64::*;

/// SSE PCLMUL strategy
pub struct PclmulStrategy;

impl GaloisSimdStrategy for PclmulStrategy {
    fn name(&self) -> &'static str {
        "PCLMUL"
    }

    fn is_available(&self) -> bool {
        is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("ssse3")
    }

    fn priority(&self) -> Priority {
        Priority::Advanced
    }

    unsafe fn mul_slice(&self, scalar: u16, data: &mut [u16]) {
        gf_mul_slice_pclmul(scalar, data)
    }

    unsafe fn muladd(&self, dst: &mut [u16], src: &[u16], scalar: u16) {
        gf_muladd_pclmul(dst, src, scalar)
    }

    unsafe fn muladd_region(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    ) {
        gf_muladd_region_pclmul(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        )
    }
}

/// AVX2 + PCLMUL strategy
pub struct Avx2PclmulStrategy;

impl GaloisSimdStrategy for Avx2PclmulStrategy {
    fn name(&self) -> &'static str {
        "AVX2-PCLMUL"
    }

    fn is_available(&self) -> bool {
        is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("ssse3")
    }

    fn priority(&self) -> Priority {
        Priority::Optimal
    }

    unsafe fn mul_slice(&self, scalar: u16, data: &mut [u16]) {
        gf_mul_slice_avx2_pclmul(scalar, data)
    }

    unsafe fn muladd(&self, dst: &mut [u16], src: &[u16], scalar: u16) {
        gf_muladd_avx2_pclmul(dst, src, scalar)
    }

    unsafe fn muladd_region(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    ) {
        gf_muladd_region_avx2_pclmul(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        )
    }
}

/// Core PCLMUL multiplication for 8 u16 values using Barrett reduction
///
/// Uses PCLMULQDQ to perform carryless multiplication followed by
/// Barrett reduction modulo the GF(2^16) primitive polynomial 0x1100B.
///
/// # Algorithm
/// 1. Split 8 u16 values into even/odd positions (4 values each in a qword)
/// 2. Perform carryless multiplication using PCLMULQDQ
/// 3. Blend results back together
/// 4. Apply Barrett reduction to reduce 32-bit products to 16-bit
#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1,ssse3")]
unsafe fn gf_mul_pclmul_8(data1: __m128i, data2: __m128i) -> __m128i {
    // Split into even/odd u16 values for parallel processing
    // wordMask selects even u16 values (positions 0, 2, 4, 6)
    let word_mask = _mm_set1_epi32(0x0000FFFF_u32 as i32);

    // Even indices: keep in place, mask out odd
    let data1_even = _mm_and_si128(word_mask, data1);
    let data2_even = _mm_and_si128(word_mask, data2);

    // Odd indices: keep in place (andnot masks out even)
    let data1_odd = _mm_andnot_si128(word_mask, data1);
    let data2_odd = _mm_andnot_si128(word_mask, data2);

    // Carryless multiplication - produces 32-bit products
    // 0x00: multiply low qwords (positions 0, 2)
    // 0x11: multiply high qwords (positions 4, 6)
    let prod1_even = _mm_clmulepi64_si128(data1_even, data2_even, 0x00);
    let prod2_even = _mm_clmulepi64_si128(data1_even, data2_even, 0x11);
    let prod1_odd = _mm_clmulepi64_si128(data1_odd, data2_odd, 0x00);
    let prod2_odd = _mm_clmulepi64_si128(data1_odd, data2_odd, 0x11);

    // Interleave even/odd results: 0xCC = 0b11001100 selects odd in positions 2,3,6,7
    let prod1 = _mm_blend_epi16(prod1_even, prod1_odd, 0xCC);
    let prod2 = _mm_blend_epi16(prod2_even, prod2_odd, 0xCC);

    // Barrett reduction: reduce 32-bit products to 16-bit GF(2^16) values
    // Split low/high 16-bit halves using shuffle
    // Shuffle pattern: low 16 bits to positions 0,2,4,6; high 16 bits to positions 1,3,5,7
    let shuf_lo_hi = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
    let tmp1 = _mm_shuffle_epi8(prod1, shuf_lo_hi);
    let tmp2 = _mm_shuffle_epi8(prod2, shuf_lo_hi);

    // rem = low 16 bits of each 32-bit product
    // quot = high 16 bits of each 32-bit product (needs reduction)
    let rem = _mm_unpacklo_epi64(tmp1, tmp2);
    let mut quot = _mm_unpackhi_epi64(tmp1, tmp2);

    // Multiply quot by 0x1111a (Barrett constant) and retain high half
    // This computes floor(quot * 0x1111a / 2^16)
    // Using shift+xor is faster than actual multiplication
    // 0x1111a = 2^16 + 2^12 + 2^8 + 2^4 + 2^3 + 2^1
    // Result = q ^ q>>4 ^ q>>8 ^ q>>12 ^ q>>13 ^ q>>15
    let tmp1 = _mm_xor_si128(quot, _mm_srli_epi16(quot, 4));
    let tmp1 = _mm_xor_si128(tmp1, _mm_srli_epi16(tmp1, 8));
    let tmp1 = _mm_xor_si128(tmp1, _mm_srli_epi16(quot, 13));
    quot = _mm_xor_si128(tmp1, _mm_srli_epi16(quot, 15));

    // Multiply quot by 0x100b (irreducible polynomial without leading 1), retain low half
    // 0x100b = 2^12 + 2^3 + 2^1 + 2^0
    let tmp1 = _mm_xor_si128(quot, _mm_slli_epi16(quot, 3));
    let tmp1 = _mm_xor_si128(tmp1, _mm_add_epi16(quot, quot)); // quot * 2
    quot = _mm_xor_si128(tmp1, _mm_slli_epi16(quot, 12));

    // Final result: XOR remainder with reduced quotient
    _mm_xor_si128(quot, rem)
}

/// PCLMUL implementation of multiply slice (128-bit SSE)
#[target_feature(enable = "pclmulqdq,sse4.1,ssse3")]
unsafe fn gf_mul_slice_pclmul(scalar: u16, data: &mut [u16]) {
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    if scalar == 1 {
        return;
    }

    let scalar_vec = _mm_set1_epi16(scalar as i16);

    // Process 8 u16 values at a time
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 8);
        let data_vec = _mm_loadu_si128(ptr as *const __m128i);
        let result = gf_mul_pclmul_8(data_vec, scalar_vec);
        _mm_storeu_si128(ptr as *mut __m128i, result);
    }

    // Handle remainder with scalar
    let offset = chunks * 8;
    for val in data.iter_mut().skip(offset) {
        *val = crate::galois::core::gf_mul(*val, scalar);
    }
}

/// PCLMUL implementation of multiply-add
#[target_feature(enable = "pclmulqdq,sse4.1,ssse3")]
unsafe fn gf_muladd_pclmul(dst: &mut [u16], src: &[u16], scalar: u16) {
    if scalar == 0 {
        return;
    }

    let scalar_vec = _mm_set1_epi16(scalar as i16);
    let len = dst.len().min(src.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let src_ptr = src.as_ptr().add(offset);
        let dst_ptr = dst.as_mut_ptr().add(offset);

        let src_vec = _mm_loadu_si128(src_ptr as *const __m128i);
        let dst_vec = _mm_loadu_si128(dst_ptr as *const __m128i);

        let prod = gf_mul_pclmul_8(src_vec, scalar_vec);
        let result = _mm_xor_si128(dst_vec, prod);

        _mm_storeu_si128(dst_ptr as *mut __m128i, result);
    }

    // Handle remainder
    let offset = chunks * 8;
    for j in offset..len {
        dst[j] ^= crate::galois::core::gf_mul(src[j], scalar);
    }
}

/// AVX2 PCLMUL implementation of multiply slice (256-bit)
///
/// Processes 16 u16 values at a time using two 128-bit PCLMUL operations.
#[target_feature(enable = "avx2,pclmulqdq,sse4.1,ssse3")]
unsafe fn gf_mul_slice_avx2_pclmul(scalar: u16, data: &mut [u16]) {
    if scalar == 0 {
        let zero = _mm256_setzero_si256();
        let chunks = data.len() / 16;
        for i in 0..chunks {
            let ptr = data.as_mut_ptr().add(i * 16);
            _mm256_storeu_si256(ptr as *mut __m256i, zero);
        }
        // Zero remainder
        let offset = chunks * 16;
        for val in &mut data[offset..] {
            *val = 0;
        }
        return;
    }

    if scalar == 1 {
        return;
    }

    let scalar_vec = _mm_set1_epi16(scalar as i16);

    // Process 16 u16 values at a time using two 128-bit ops
    let chunks = data.len() / 16;
    for i in 0..chunks {
        let offset = i * 16;

        // Process first 8 u16
        let data_vec1 = _mm_loadu_si128(data.as_ptr().add(offset) as *const __m128i);
        let result1 = gf_mul_pclmul_8(data_vec1, scalar_vec);
        _mm_storeu_si128(data.as_mut_ptr().add(offset) as *mut __m128i, result1);

        // Process second 8 u16
        let data_vec2 = _mm_loadu_si128(data.as_ptr().add(offset + 8) as *const __m128i);
        let result2 = gf_mul_pclmul_8(data_vec2, scalar_vec);
        _mm_storeu_si128(data.as_mut_ptr().add(offset + 8) as *mut __m128i, result2);
    }

    // Handle remainder
    let offset = chunks * 16;
    for val in data.iter_mut().skip(offset) {
        *val = crate::galois::core::gf_mul(*val, scalar);
    }
}

/// AVX2 PCLMUL implementation of multiply-add
#[target_feature(enable = "avx2,pclmulqdq,sse4.1,ssse3")]
unsafe fn gf_muladd_avx2_pclmul(dst: &mut [u16], src: &[u16], scalar: u16) {
    if scalar == 0 {
        return;
    }

    let scalar_vec = _mm_set1_epi16(scalar as i16);
    let len = dst.len().min(src.len());
    let chunks = len / 16;

    for i in 0..chunks {
        let offset = i * 16;

        // Process first 8 u16
        let src_vec1 = _mm_loadu_si128(src.as_ptr().add(offset) as *const __m128i);
        let dst_vec1 = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
        let prod1 = gf_mul_pclmul_8(src_vec1, scalar_vec);
        let result1 = _mm_xor_si128(dst_vec1, prod1);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result1);

        // Process second 8 u16
        let src_vec2 = _mm_loadu_si128(src.as_ptr().add(offset + 8) as *const __m128i);
        let dst_vec2 = _mm_loadu_si128(dst.as_ptr().add(offset + 8) as *const __m128i);
        let prod2 = gf_mul_pclmul_8(src_vec2, scalar_vec);
        let result2 = _mm_xor_si128(dst_vec2, prod2);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset + 8) as *mut __m128i, result2);
    }

    // Handle remainder
    let offset = chunks * 16;
    for j in offset..len {
        dst[j] ^= crate::galois::core::gf_mul(src[j], scalar);
    }
}

/// PCLMUL region-based multiply-accumulate
#[target_feature(enable = "pclmulqdq,sse4.1,ssse3")]
unsafe fn gf_muladd_region_pclmul(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    let num_dsts = destinations.len();
    let num_srcs = sources.len();

    // Process each destination using PCLMUL SIMD muladd
    for dst_idx in 0..num_dsts {
        let dst = &mut destinations[dst_idx][region_offset..region_offset + region_size];
        let coeff_row = coefficients[dst_idx];

        // Accumulate contributions from all sources using PCLMUL
        for (src_idx, &coeff) in coeff_row.iter().enumerate().take(num_srcs) {
            if coeff == 0 {
                continue;
            }

            let src = &sources[src_idx][region_offset..region_offset + region_size];
            gf_muladd_pclmul(dst, src, coeff);
        }
    }
}

/// AVX2 PCLMUL region-based multiply-accumulate with batched processing
#[target_feature(enable = "avx2,pclmulqdq,sse4.1,ssse3")]
unsafe fn gf_muladd_region_avx2_pclmul(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    let num_dsts = destinations.len();
    let num_srcs = sources.len();

    // Process each destination using AVX2 PCLMUL SIMD muladd
    for dst_idx in 0..num_dsts {
        let dst = &mut destinations[dst_idx][region_offset..region_offset + region_size];
        let coeff_row = coefficients[dst_idx];

        // Accumulate contributions from all sources
        for (src_idx, &coeff) in coeff_row.iter().enumerate().take(num_srcs) {
            if coeff == 0 {
                continue;
            }

            let src = &sources[src_idx][region_offset..region_offset + region_size];
            gf_muladd_avx2_pclmul(dst, src, coeff);
        }
    }
}
