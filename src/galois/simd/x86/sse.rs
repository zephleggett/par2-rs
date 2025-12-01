//! SSE2 and SSSE3 implementations for x86-64
//!
//! These provide basic SIMD optimizations available on most x86-64 processors.

use crate::galois::core::{gf_mul, init_tables, initialize_mul128_table, LOG_TABLE, MUL128_TABLE};
use crate::galois::simd::{GaloisSimdStrategy, Priority};
use std::arch::x86_64::*;

/// SSE2 strategy (baseline x86-64 SIMD)
pub struct Sse2Strategy;

impl GaloisSimdStrategy for Sse2Strategy {
    fn name(&self) -> &'static str {
        "SSE2"
    }

    fn is_available(&self) -> bool {
        is_x86_feature_detected!("sse2")
    }

    fn priority(&self) -> Priority {
        Priority::Basic
    }

    unsafe fn mul_slice(&self, scalar: u16, data: &mut [u16]) {
        gf_mul_slice_sse2(scalar, data)
    }

    unsafe fn muladd(&self, dst: &mut [u16], src: &[u16], scalar: u16) {
        gf_muladd_sse2(dst, src, scalar)
    }

    unsafe fn muladd_region(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    ) {
        // Use SSE2 SIMD muladd in a loop for region processing
        gf_muladd_region_sse2_loop(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        )
    }
}

/// SSSE3 strategy with shuffle-based multiplication
pub struct Ssse3Strategy;

impl GaloisSimdStrategy for Ssse3Strategy {
    fn name(&self) -> &'static str {
        "SSSE3"
    }

    fn is_available(&self) -> bool {
        is_x86_feature_detected!("ssse3")
    }

    fn priority(&self) -> Priority {
        Priority::Enhanced
    }

    unsafe fn mul_slice(&self, scalar: u16, data: &mut [u16]) {
        gf_mul_slice_ssse3(scalar, data)
    }

    unsafe fn muladd(&self, dst: &mut [u16], src: &[u16], scalar: u16) {
        gf_muladd_ssse3(dst, src, scalar)
    }

    unsafe fn muladd_region(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    ) {
        // SSSE3 doesn't have efficient region operations, use scalar
        crate::galois::scalar::gf_muladd_region_scalar(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        )
    }
}

/// SSE2 implementation of multiply slice (basic XMM register usage)
#[target_feature(enable = "sse2")]
unsafe fn gf_mul_slice_sse2(scalar: u16, data: &mut [u16]) {
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    if scalar == 1 {
        return; // No change needed
    }

    init_tables();

    // Process 8 u16 values at a time using SSE2
    let chunks = data.len() / 8;
    let remainder = data.len() % 8;

    // For SSE2, we don't have efficient shuffle, so we process in smaller batches
    // and use the scalar multiplication for each element
    for i in 0..chunks {
        let offset = i * 8;
        for j in 0..8 {
            data[offset + j] = gf_mul(data[offset + j], scalar);
        }
    }

    // Handle remainder
    let offset = chunks * 8;
    for j in 0..remainder {
        data[offset + j] = gf_mul(data[offset + j], scalar);
    }
}

/// SSE2 region-based multiply-accumulate loop
#[target_feature(enable = "sse2")]
unsafe fn gf_muladd_region_sse2_loop(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    let num_dsts = destinations.len();
    let num_srcs = sources.len();

    // Process each destination
    for dst_idx in 0..num_dsts {
        let dst = &mut destinations[dst_idx][region_offset..region_offset + region_size];
        let coeff_row = coefficients[dst_idx];

        // Accumulate contributions from all sources
        for (src_idx, &coeff) in coeff_row.iter().enumerate().take(num_srcs) {
            if coeff == 0 {
                continue;
            }
            let src = &sources[src_idx][region_offset..region_offset + region_size];
            gf_muladd_sse2(dst, src, coeff);
        }
    }
}

/// SSE2 implementation of multiply-add
#[target_feature(enable = "sse2")]
unsafe fn gf_muladd_sse2(dst: &mut [u16], src: &[u16], scalar: u16) {
    if scalar == 0 {
        return;
    }

    init_tables();

    let len = dst.len().min(src.len());

    // Process aligned portions
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        for j in 0..8 {
            dst[offset + j] ^= gf_mul(src[offset + j], scalar);
        }
    }

    // Handle remainder
    let offset = chunks * 8;
    for j in offset..len {
        dst[j] ^= gf_mul(src[j], scalar);
    }
}

/// SSSE3 implementation using shuffle instructions
///
/// Uses proper byte separation to extract nibbles from u16 values.
#[target_feature(enable = "ssse3")]
unsafe fn gf_mul_slice_ssse3(scalar: u16, data: &mut [u16]) {
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    if scalar == 1 {
        return;
    }

    // Initialize multiplication tables
    initialize_mul128_table();

    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let mul_table = MUL128_TABLE.get().expect("MUL128_TABLE not initialized");

    let log_scalar = log_table[scalar as usize] as usize;
    let lut = &mul_table[log_scalar];

    // Load lookup tables into 128-bit registers
    let t0_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[0]).cast::<__m128i>());
    let t1_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[1]).cast::<__m128i>());
    let t2_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[2]).cast::<__m128i>());
    let t3_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[3]).cast::<__m128i>());

    let t0_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[0]).cast::<__m128i>());
    let t1_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[1]).cast::<__m128i>());
    let t2_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[2]).cast::<__m128i>());
    let t3_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[3]).cast::<__m128i>());

    let clr_mask = _mm_set1_epi8(0x0f);

    // Masks to extract even bytes (low byte of each u16) and odd bytes (high byte of each u16)
    // For 8 u16s: bytes 0,2,4,6,8,10,12,14 are low bytes; 1,3,5,7,9,11,13,15 are high bytes
    let byte_mask_lo = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
    );
    let byte_mask_hi = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, -128, -128, -128, -128, -128, -128, -128, -128,
    );

    // Process 8 u16 values at a time (128 bits)
    let chunks = data.len() / 8;
    let remainder = data.len() % 8;

    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 8);
        let v = _mm_loadu_si128(ptr as *const __m128i);

        // Separate low bytes and high bytes of each u16
        let value_lo = _mm_shuffle_epi8(v, byte_mask_lo);
        let value_hi = _mm_shuffle_epi8(v, byte_mask_hi);

        // Extract 4 nibbles from separated byte streams
        let n0 = _mm_and_si128(value_lo, clr_mask);
        let n1 = _mm_and_si128(_mm_srli_epi64(value_lo, 4), clr_mask);
        let n2 = _mm_and_si128(value_hi, clr_mask);
        let n3 = _mm_and_si128(_mm_srli_epi64(value_hi, 4), clr_mask);

        // Perform 4 table lookups and XOR results
        let mut prod_lo = _mm_shuffle_epi8(t0_lo, n0);
        let mut prod_hi = _mm_shuffle_epi8(t0_hi, n0);

        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t1_lo, n1));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t1_hi, n1));

        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t2_lo, n2));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t2_hi, n2));

        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t3_lo, n3));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t3_hi, n3));

        // Reinterleave low/high product bytes back to u16 format
        let result = _mm_unpacklo_epi8(prod_lo, prod_hi);

        _mm_storeu_si128(ptr as *mut __m128i, result);
    }

    // Handle remainder with scalar operations
    let offset = chunks * 8;
    for j in 0..remainder {
        data[offset + j] = gf_mul(data[offset + j], scalar);
    }
}

/// SSSE3 implementation of multiply-add
///
/// Uses proper byte separation to extract nibbles from u16 values.
#[target_feature(enable = "ssse3")]
unsafe fn gf_muladd_ssse3(dst: &mut [u16], src: &[u16], scalar: u16) {
    if scalar == 0 {
        return;
    }

    // Initialize multiplication tables
    initialize_mul128_table();

    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let mul_table = MUL128_TABLE.get().expect("MUL128_TABLE not initialized");

    let log_scalar = log_table[scalar as usize] as usize;
    let lut = &mul_table[log_scalar];

    // Load lookup tables
    let t0_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[0]).cast::<__m128i>());
    let t1_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[1]).cast::<__m128i>());
    let t2_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[2]).cast::<__m128i>());
    let t3_lo = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[3]).cast::<__m128i>());

    let t0_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[0]).cast::<__m128i>());
    let t1_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[1]).cast::<__m128i>());
    let t2_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[2]).cast::<__m128i>());
    let t3_hi = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[3]).cast::<__m128i>());

    let clr_mask = _mm_set1_epi8(0x0f);

    // Masks to extract even/odd bytes
    let byte_mask_lo = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
    );
    let byte_mask_hi = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, -128, -128, -128, -128, -128, -128, -128, -128,
    );

    let len = dst.len().min(src.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let src_ptr = src.as_ptr().add(offset);
        let dst_ptr = dst.as_mut_ptr().add(offset);

        let v = _mm_loadu_si128(src_ptr as *const __m128i);
        let d = _mm_loadu_si128(dst_ptr as *const __m128i);

        // Separate low bytes and high bytes
        let value_lo = _mm_shuffle_epi8(v, byte_mask_lo);
        let value_hi = _mm_shuffle_epi8(v, byte_mask_hi);

        // Extract 4 nibbles
        let n0 = _mm_and_si128(value_lo, clr_mask);
        let n1 = _mm_and_si128(_mm_srli_epi64(value_lo, 4), clr_mask);
        let n2 = _mm_and_si128(value_hi, clr_mask);
        let n3 = _mm_and_si128(_mm_srli_epi64(value_hi, 4), clr_mask);

        // Perform lookups and XOR
        let mut prod_lo = _mm_shuffle_epi8(t0_lo, n0);
        let mut prod_hi = _mm_shuffle_epi8(t0_hi, n0);

        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t1_lo, n1));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t1_hi, n1));

        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t2_lo, n2));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t2_hi, n2));

        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t3_lo, n3));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t3_hi, n3));

        // Reinterleave and XOR with destination
        let product = _mm_unpacklo_epi8(prod_lo, prod_hi);
        let result = _mm_xor_si128(d, product);

        _mm_storeu_si128(dst_ptr as *mut __m128i, result);
    }

    // Handle remainder
    let offset = chunks * 8;
    for j in offset..len {
        dst[j] ^= gf_mul(src[j], scalar);
    }
}
