//! AVX2 implementations for x86-64
//!
//! These provide 256-bit SIMD optimizations using nibble-based shuffle tables.
//!
//! **Region Processing:** This strategy includes a fully optimized region-based
//! multiply-accumulate implementation (`gf_muladd_region_avx2_shuffle`) which
//! provides significant performance improvements for Reed-Solomon reconstruction.

use crate::galois::core::{initialize_mul128_table, LOG_TABLE, MUL128_TABLE};
use crate::galois::simd::{GaloisSimdStrategy, Priority};
use std::arch::x86_64::*;

/// AVX2 shuffle-based strategy
pub struct Avx2ShuffleStrategy;

impl GaloisSimdStrategy for Avx2ShuffleStrategy {
    fn name(&self) -> &'static str {
        "AVX2-Shuffle"
    }

    fn is_available(&self) -> bool {
        is_x86_feature_detected!("avx2")
    }

    fn priority(&self) -> Priority {
        // AVX2 shuffle with native shuffle2x is fastest on most x86 CPUs
        Priority::Optimal
    }

    unsafe fn mul_slice(&self, scalar: u16, data: &mut [u16]) {
        gf_mul_slice_avx2_shuffle(scalar, data)
    }

    unsafe fn muladd(&self, dst: &mut [u16], src: &[u16], scalar: u16) {
        gf_muladd_avx2_shuffle(dst, src, scalar)
    }

    unsafe fn muladd_region(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    ) {
        gf_muladd_region_avx2_shuffle(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        )
    }

    // === Shuffle2x format support ===

    fn supports_shuffle2x(&self) -> bool {
        true
    }

    unsafe fn prepare_shuffle2x(&self, data: &mut [u16]) {
        gf_prepare_shuffle2x_avx2(data)
    }

    unsafe fn finish_shuffle2x(&self, data: &mut [u16]) {
        gf_finish_shuffle2x_avx2(data)
    }

    unsafe fn muladd_shuffle2x(&self, dst: &mut [u16], src: &[u16], scalar: u16) {
        gf_muladd_native_shuffle2x_avx2(dst, src, scalar)
    }

    unsafe fn muladd_region_shuffle2x(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    ) {
        gf_muladd_region_native_shuffle2x_avx2(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        )
    }
}

/// AVX2 shuffle2x multiplication of slice - high performance implementation
///
/// Uses shuffle2x algorithm for efficient GF(2^16) multiplication.
#[target_feature(enable = "avx2")]
unsafe fn gf_mul_slice_avx2_shuffle(scalar: u16, data: &mut [u16]) {
    if scalar == 0 {
        let zero = _mm256_setzero_si256();
        let chunks = data.len() / 16;
        for i in 0..chunks {
            let ptr = data.as_mut_ptr().add(i * 16);
            _mm256_storeu_si256(ptr as *mut __m256i, zero);
        }
        let offset = chunks * 16;
        for val in &mut data[offset..] {
            *val = 0;
        }
        return;
    }

    if scalar == 1 {
        return;
    }

    // Load shuffle2x tables once
    let tables = gf_load_shuffle2x_tables(scalar);
    let mask = _mm256_set1_epi8(0x0f);

    // Shuffle to separate low/high bytes
    let separate_shuf = _mm256_set_epi32(
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
    );

    // Shuffle to recombine bytes
    let recombine_shuf = _mm256_set_epi32(
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
    );

    // Process 16 u16 values at a time
    let chunks = data.len() / 16;
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 16);
        let v = _mm256_loadu_si256(ptr as *const __m256i);

        // Convert to shuffle2x format
        let sep = _mm256_shuffle_epi8(v, separate_shuf);
        let d = _mm256_permute4x64_epi64(sep, 0b11_01_10_00);

        // Extract nibbles and do lookups
        let n_lo = _mm256_and_si256(d, mask);
        let n_hi = _mm256_and_si256(_mm256_srli_epi16(d, 4), mask);

        let mut result = _mm256_shuffle_epi8(tables.norm_lo, n_lo);
        let mut swap = _mm256_shuffle_epi8(tables.swap_lo, n_lo);
        result = _mm256_xor_si256(result, _mm256_shuffle_epi8(tables.norm_hi, n_hi));
        swap = _mm256_xor_si256(swap, _mm256_shuffle_epi8(tables.swap_hi, n_hi));

        // Swap lanes and combine
        swap = _mm256_permute2x128_si256(swap, swap, 0x01);
        result = _mm256_xor_si256(result, swap);

        // Convert back to interleaved format
        let perm = _mm256_permute4x64_epi64(result, 0b11_01_10_00);
        let out = _mm256_shuffle_epi8(perm, recombine_shuf);

        _mm256_storeu_si256(ptr as *mut __m256i, out);
    }

    // Handle remainder with scalar
    let offset = chunks * 16;
    for val in data.iter_mut().skip(offset) {
        *val = crate::galois::core::gf_mul(*val, scalar);
    }
}

/// Shuffle2x tables for efficient GF(2^16) multiplication
///
/// In shuffle2x format, data is organized with all low bytes in lane 0 and all high bytes
/// in lane 1. The norm tables give results that stay in the same lane, while swap tables
/// give results that cross lanes.
#[derive(Clone, Copy)]
struct GfShuffle2xTables {
    norm_lo: __m256i, // nibble 0 → same lane result
    swap_lo: __m256i, // nibble 0 → cross lane result
    norm_hi: __m256i, // nibble 1 → same lane result
    swap_hi: __m256i, // nibble 1 → cross lane result
}

/// Load shuffle2x tables from MUL128_TABLE
///
/// In shuffle2x format:
/// - Lane 0 contains low bytes, processes nibbles 0,1
/// - Lane 1 contains high bytes, processes nibbles 2,3
///
/// norm_lo = [lut.lo[0] | lut.hi[2]] - result stays in same lane
/// swap_lo = [lut.hi[0] | lut.lo[2]] - result crosses to other lane
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn gf_load_shuffle2x_tables(coeff: u16) -> GfShuffle2xTables {
    initialize_mul128_table();
    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let mul_table = MUL128_TABLE.get().expect("MUL128_TABLE not initialized");

    let log_scalar = log_table[coeff as usize] as usize;
    let lut = &mul_table[log_scalar];

    // Load 128-bit tables
    let lo0 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[0]).cast::<__m128i>());
    let lo1 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[1]).cast::<__m128i>());
    let lo2 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[2]).cast::<__m128i>());
    let lo3 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.lo[3]).cast::<__m128i>());

    let hi0 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[0]).cast::<__m128i>());
    let hi1 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[1]).cast::<__m128i>());
    let hi2 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[2]).cast::<__m128i>());
    let hi3 = _mm_loadu_si128(std::ptr::from_ref::<u128>(&lut.hi[3]).cast::<__m128i>());

    // Combine into shuffle2x format:
    // Lane 0 processes low bytes (nibbles 0,1) → contributes to lo result (stays) and hi result (crosses)
    // Lane 1 processes high bytes (nibbles 2,3) → contributes to hi result (stays) and lo result (crosses)
    GfShuffle2xTables {
        // For nibble 0: lane 0 uses lo[0] (→lo byte), lane 1 uses hi[2] (→hi byte)
        norm_lo: _mm256_permute2x128_si256(
            _mm256_castsi128_si256(lo0),
            _mm256_castsi128_si256(hi2),
            0x20,
        ),
        // For nibble 0: lane 0 uses hi[0] (→hi byte), lane 1 uses lo[2] (→lo byte)
        swap_lo: _mm256_permute2x128_si256(
            _mm256_castsi128_si256(hi0),
            _mm256_castsi128_si256(lo2),
            0x20,
        ),
        // For nibble 1: lane 0 uses lo[1] (→lo byte), lane 1 uses hi[3] (→hi byte)
        norm_hi: _mm256_permute2x128_si256(
            _mm256_castsi128_si256(lo1),
            _mm256_castsi128_si256(hi3),
            0x20,
        ),
        // For nibble 1: lane 0 uses hi[1] (→hi byte), lane 1 uses lo[3] (→lo byte)
        swap_hi: _mm256_permute2x128_si256(
            _mm256_castsi128_si256(hi1),
            _mm256_castsi128_si256(lo3),
            0x20,
        ),
    }
}

/// AVX2 shuffle2x multiply-add - high performance implementation
///
/// Based on par2cmdline-turbo's shuffle2x algorithm:
/// 1. Convert data to shuffle2x format (low bytes in lane 0, high bytes in lane 1)
/// 2. Use 4 table lookups (norm_lo, swap_lo, norm_hi, swap_hi)
/// 3. Permute swap result across lanes and XOR
/// 4. Convert result back to interleaved format
#[target_feature(enable = "avx2")]
unsafe fn gf_muladd_avx2_shuffle(dst: &mut [u16], src: &[u16], scalar: u16) {
    if scalar == 0 {
        return;
    }

    // Load shuffle2x tables once
    let tables = gf_load_shuffle2x_tables(scalar);
    let mask = _mm256_set1_epi8(0x0f);

    // Shuffle to separate low/high bytes: bytes 0,2,4,... go to low half, 1,3,5,... go to high half
    let separate_shuf = _mm256_set_epi32(
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
    );

    // Shuffle to recombine: interleave low and high bytes back to u16 format
    let recombine_shuf = _mm256_set_epi32(
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
    );

    let len = dst.len().min(src.len());

    // Process 32 u16 at a time (2x unrolled)
    let chunks2 = len / 32;
    for i in 0..chunks2 {
        let offset = i * 32;
        let src_ptr = src.as_ptr().add(offset);
        let dst_ptr = dst.as_mut_ptr().add(offset);

        // Load two source vectors and two destination vectors
        let s0 = _mm256_loadu_si256(src_ptr as *const __m256i);
        let s1 = _mm256_loadu_si256(src_ptr.add(16) as *const __m256i);
        let d0 = _mm256_loadu_si256(dst_ptr as *const __m256i);
        let d1 = _mm256_loadu_si256(dst_ptr.add(16) as *const __m256i);

        // Convert to shuffle2x format: separate low/high bytes, then permute lanes
        let sep0 = _mm256_shuffle_epi8(s0, separate_shuf);
        let sep1 = _mm256_shuffle_epi8(s1, separate_shuf);
        let data0 = _mm256_permute4x64_epi64(sep0, 0b11_01_10_00); // _MM_SHUFFLE(3,1,2,0)
        let data1 = _mm256_permute4x64_epi64(sep1, 0b11_01_10_00);

        // Process first vector with shuffle2x algorithm
        let n0_lo = _mm256_and_si256(data0, mask);
        let n0_hi = _mm256_and_si256(_mm256_srli_epi16(data0, 4), mask);

        let mut result0 = _mm256_shuffle_epi8(tables.norm_lo, n0_lo);
        let mut swap0 = _mm256_shuffle_epi8(tables.swap_lo, n0_lo);
        result0 = _mm256_xor_si256(result0, _mm256_shuffle_epi8(tables.norm_hi, n0_hi));
        swap0 = _mm256_xor_si256(swap0, _mm256_shuffle_epi8(tables.swap_hi, n0_hi));

        // Swap 128-bit lanes and XOR
        swap0 = _mm256_permute2x128_si256(swap0, swap0, 0x01);
        result0 = _mm256_xor_si256(result0, swap0);

        // Process second vector
        let n1_lo = _mm256_and_si256(data1, mask);
        let n1_hi = _mm256_and_si256(_mm256_srli_epi16(data1, 4), mask);

        let mut result1 = _mm256_shuffle_epi8(tables.norm_lo, n1_lo);
        let mut swap1 = _mm256_shuffle_epi8(tables.swap_lo, n1_lo);
        result1 = _mm256_xor_si256(result1, _mm256_shuffle_epi8(tables.norm_hi, n1_hi));
        swap1 = _mm256_xor_si256(swap1, _mm256_shuffle_epi8(tables.swap_hi, n1_hi));

        swap1 = _mm256_permute2x128_si256(swap1, swap1, 0x01);
        result1 = _mm256_xor_si256(result1, swap1);

        // Convert back to interleaved format: permute lanes, then recombine bytes
        let perm0 = _mm256_permute4x64_epi64(result0, 0b11_01_10_00);
        let perm1 = _mm256_permute4x64_epi64(result1, 0b11_01_10_00);
        let product0 = _mm256_shuffle_epi8(perm0, recombine_shuf);
        let product1 = _mm256_shuffle_epi8(perm1, recombine_shuf);

        // XOR with destination and store
        let out0 = _mm256_xor_si256(d0, product0);
        let out1 = _mm256_xor_si256(d1, product1);
        _mm256_storeu_si256(dst_ptr as *mut __m256i, out0);
        _mm256_storeu_si256(dst_ptr.add(16) as *mut __m256i, out1);
    }

    // Handle remaining 16-element chunk
    let offset = chunks2 * 32;
    if offset + 16 <= len {
        let src_ptr = src.as_ptr().add(offset);
        let dst_ptr = dst.as_mut_ptr().add(offset);

        let s = _mm256_loadu_si256(src_ptr as *const __m256i);
        let d = _mm256_loadu_si256(dst_ptr as *const __m256i);

        let sep = _mm256_shuffle_epi8(s, separate_shuf);
        let data = _mm256_permute4x64_epi64(sep, 0b11_01_10_00);

        let n_lo = _mm256_and_si256(data, mask);
        let n_hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask);

        let mut result = _mm256_shuffle_epi8(tables.norm_lo, n_lo);
        let mut swap = _mm256_shuffle_epi8(tables.swap_lo, n_lo);
        result = _mm256_xor_si256(result, _mm256_shuffle_epi8(tables.norm_hi, n_hi));
        swap = _mm256_xor_si256(swap, _mm256_shuffle_epi8(tables.swap_hi, n_hi));

        swap = _mm256_permute2x128_si256(swap, swap, 0x01);
        result = _mm256_xor_si256(result, swap);

        let perm = _mm256_permute4x64_epi64(result, 0b11_01_10_00);
        let product = _mm256_shuffle_epi8(perm, recombine_shuf);

        let out = _mm256_xor_si256(d, product);
        _mm256_storeu_si256(dst_ptr as *mut __m256i, out);
    }

    // Handle scalar remainder
    let final_offset = (len / 16) * 16;
    for j in final_offset..len {
        dst[j] ^= crate::galois::core::gf_mul(src[j], scalar);
    }
}

/// AVX2 region-based multiply-accumulate with shuffle2x algorithm
///
/// Converts source data to shuffle2x format once per chunk, then reuses it
/// across all destination-coefficient pairs for better efficiency.
#[target_feature(enable = "avx2")]
unsafe fn gf_muladd_region_avx2_shuffle(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    let num_dsts = destinations.len();
    let num_srcs = sources.len();

    const DEST_BATCH: usize = 4;
    const SRC_BATCH: usize = 4;

    let chunks = region_size / 16;
    let remainder = region_size % 16;

    // Shuffle constants
    let mask = _mm256_set1_epi8(0x0f);
    let separate_shuf = _mm256_set_epi32(
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
    );
    let recombine_shuf = _mm256_set_epi32(
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
    );

    let mut dst_base = 0;
    while dst_base < num_dsts {
        let dst_end = (dst_base + DEST_BATCH).min(num_dsts);
        let dst_count = dst_end - dst_base;

        let mut dst_rows: [*mut u16; DEST_BATCH] = [std::ptr::null_mut(); DEST_BATCH];
        let mut coeff_rows: [&[u16]; DEST_BATCH] = [&[]; DEST_BATCH];

        for r in 0..dst_count {
            let idx = dst_base + r;
            let dst_slice = &mut destinations[idx][region_offset..region_offset + region_size];
            dst_rows[r] = dst_slice.as_mut_ptr();
            coeff_rows[r] = coefficients[idx];
        }

        let mut src_batch_start = 0;
        while src_batch_start < num_srcs {
            let src_batch_end = (src_batch_start + SRC_BATCH).min(num_srcs);
            let batch_len = src_batch_end - src_batch_start;

            // Preload coefficients
            let mut any_nonzero = false;
            let mut coeffs: [[u16; SRC_BATCH]; DEST_BATCH] = [[0; SRC_BATCH]; DEST_BATCH];

            for b in 0..batch_len {
                for r in 0..dst_count {
                    coeffs[r][b] = coeff_rows[r][src_batch_start + b];
                    if coeffs[r][b] != 0 {
                        any_nonzero = true;
                    }
                }
            }

            if !any_nonzero {
                src_batch_start = src_batch_end;
                continue;
            }

            // Preload shuffle2x tables
            let mut tables: [[Option<GfShuffle2xTables>; SRC_BATCH]; DEST_BATCH] =
                [[None, None, None, None]; DEST_BATCH];

            for r in 0..dst_count {
                for b in 0..batch_len {
                    if coeffs[r][b] != 0 {
                        tables[r][b] = Some(gf_load_shuffle2x_tables(coeffs[r][b]));
                    }
                }
            }

            // Process chunks
            for chunk_idx in 0..chunks {
                let offset = chunk_idx * 16;

                // Load and convert source data to shuffle2x format ONCE
                let mut src_s2x: [(__m256i, __m256i); SRC_BATCH] =
                    [(_mm256_setzero_si256(), _mm256_setzero_si256()); SRC_BATCH];

                for b in 0..batch_len {
                    let src_ptr = sources[src_batch_start + b][region_offset + offset..].as_ptr();
                    let raw = _mm256_loadu_si256(src_ptr as *const __m256i);
                    let sep = _mm256_shuffle_epi8(raw, separate_shuf);
                    let data = _mm256_permute4x64_epi64(sep, 0b11_01_10_00);
                    let n_lo = _mm256_and_si256(data, mask);
                    let n_hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask);
                    src_s2x[b] = (n_lo, n_hi);
                }

                // Process each destination
                for r in 0..dst_count {
                    let dst_ptr = dst_rows[r].add(offset);
                    let mut accum = _mm256_loadu_si256(dst_ptr as *const __m256i);

                    // Accumulate contributions from each source
                    for b in 0..batch_len {
                        if let Some(ref tbl) = tables[r][b] {
                            let (n_lo, n_hi) = src_s2x[b];

                            // 4 lookups with shuffle2x tables
                            let mut result = _mm256_shuffle_epi8(tbl.norm_lo, n_lo);
                            let mut swap = _mm256_shuffle_epi8(tbl.swap_lo, n_lo);
                            result =
                                _mm256_xor_si256(result, _mm256_shuffle_epi8(tbl.norm_hi, n_hi));
                            swap = _mm256_xor_si256(swap, _mm256_shuffle_epi8(tbl.swap_hi, n_hi));

                            // Swap lanes and combine
                            swap = _mm256_permute2x128_si256(swap, swap, 0x01);
                            result = _mm256_xor_si256(result, swap);

                            // Convert back to interleaved format
                            let perm = _mm256_permute4x64_epi64(result, 0b11_01_10_00);
                            let product = _mm256_shuffle_epi8(perm, recombine_shuf);

                            accum = _mm256_xor_si256(accum, product);
                        }
                    }

                    _mm256_storeu_si256(dst_ptr as *mut __m256i, accum);
                }
            }

            // Handle remainder with scalar
            if remainder > 0 {
                let offset = chunks * 16;
                for r in 0..dst_count {
                    for b in 0..batch_len {
                        if coeffs[r][b] != 0 {
                            for j in 0..remainder {
                                let src_val =
                                    sources[src_batch_start + b][region_offset + offset + j];
                                let dst_idx = region_offset + offset + j;
                                destinations[dst_base + r][dst_idx] ^=
                                    crate::galois::core::gf_mul(src_val, coeffs[r][b]);
                            }
                        }
                    }
                }
            }

            src_batch_start = src_batch_end;
        }

        dst_base = dst_end;
    }
}

// ============================================================================
// Native Shuffle2x Format Operations
// ============================================================================
//
// These functions work with data already in shuffle2x format, avoiding
// per-operation format conversion overhead.

/// Convert interleaved u16 data to shuffle2x format (in-place)
///
/// Shuffle2x format: within each 256-bit block, low bytes go to lane 0,
/// high bytes go to lane 1.
#[target_feature(enable = "avx2")]
unsafe fn gf_prepare_shuffle2x_avx2(data: &mut [u16]) {
    let separate_shuf = _mm256_set_epi32(
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
        0x0f0d0b09_u32 as i32,
        0x07050301_u32 as i32,
        0x0e0c0a08_u32 as i32,
        0x06040200_u32 as i32,
    );

    let chunks = data.len() / 16; // 16 u16 per 256-bit vector
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 16);
        let v = _mm256_loadu_si256(ptr as *const __m256i);

        // Separate low/high bytes within each lane
        let sep = _mm256_shuffle_epi8(v, separate_shuf);
        // Permute lanes so all low bytes are in lane 0, high bytes in lane 1
        let result = _mm256_permute4x64_epi64(sep, 0b11_01_10_00);

        _mm256_storeu_si256(ptr as *mut __m256i, result);
    }

    // Remainder is left in interleaved format (handled by scalar fallback)
}

/// Convert shuffle2x format back to interleaved u16 data (in-place)
#[target_feature(enable = "avx2")]
unsafe fn gf_finish_shuffle2x_avx2(data: &mut [u16]) {
    let recombine_shuf = _mm256_set_epi32(
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
        0x0f070e06_u32 as i32,
        0x0d050c04_u32 as i32,
        0x0b030a02_u32 as i32,
        0x09010800_u32 as i32,
    );

    let chunks = data.len() / 16;
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 16);
        let v = _mm256_loadu_si256(ptr as *const __m256i);

        // Permute lanes to interleave low/high bytes
        let perm = _mm256_permute4x64_epi64(v, 0b11_01_10_00);
        // Recombine bytes into u16 format
        let result = _mm256_shuffle_epi8(perm, recombine_shuf);

        _mm256_storeu_si256(ptr as *mut __m256i, result);
    }
}

/// Native shuffle2x multiply-add (data already in shuffle2x format)
///
/// This is the high-performance path: 4 table lookups per 256-bit vector
/// with NO format conversion overhead.
#[target_feature(enable = "avx2")]
unsafe fn gf_muladd_native_shuffle2x_avx2(dst: &mut [u16], src: &[u16], scalar: u16) {
    if scalar == 0 {
        return;
    }

    let tables = gf_load_shuffle2x_tables(scalar);
    let mask = _mm256_set1_epi8(0x0f);

    let len = dst.len().min(src.len());
    let chunks = len / 16;

    // Process chunks - data is already in shuffle2x format!
    for i in 0..chunks {
        let offset = i * 16;
        let src_ptr = src.as_ptr().add(offset);
        let dst_ptr = dst.as_mut_ptr().add(offset);

        // Load data (already in shuffle2x format)
        let data = _mm256_loadu_si256(src_ptr as *const __m256i);
        let d = _mm256_loadu_si256(dst_ptr as *const __m256i);

        // Extract nibbles directly (no format conversion needed!)
        let n_lo = _mm256_and_si256(data, mask);
        let n_hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask);

        // 4 table lookups
        let mut result = _mm256_shuffle_epi8(tables.norm_lo, n_lo);
        let mut swap = _mm256_shuffle_epi8(tables.swap_lo, n_lo);
        result = _mm256_xor_si256(result, _mm256_shuffle_epi8(tables.norm_hi, n_hi));
        swap = _mm256_xor_si256(swap, _mm256_shuffle_epi8(tables.swap_hi, n_hi));

        // Swap lanes and combine
        swap = _mm256_permute2x128_si256(swap, swap, 0x01);
        result = _mm256_xor_si256(result, swap);

        // XOR with destination (also in shuffle2x format)
        let out = _mm256_xor_si256(d, result);
        _mm256_storeu_si256(dst_ptr as *mut __m256i, out);
    }

    // Handle remainder with scalar (data may not be in shuffle2x format)
    let offset = chunks * 16;
    for j in offset..len {
        dst[j] ^= crate::galois::core::gf_mul(src[j], scalar);
    }
}

/// Native shuffle2x region multiply-add (all data in shuffle2x format)
///
/// This is the highest-performance path for Reed-Solomon operations:
/// - Sources are loaded once and reused across all destinations
/// - No format conversion in the hot loop
/// - 4 table lookups per source-destination pair
#[target_feature(enable = "avx2")]
unsafe fn gf_muladd_region_native_shuffle2x_avx2(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    let num_dsts = destinations.len();
    let num_srcs = sources.len();

    const DEST_BATCH: usize = 4;
    const SRC_BATCH: usize = 4;

    let chunks = region_size / 16;
    let remainder = region_size % 16;
    let mask = _mm256_set1_epi8(0x0f);

    let mut dst_base = 0;
    while dst_base < num_dsts {
        let dst_end = (dst_base + DEST_BATCH).min(num_dsts);
        let dst_count = dst_end - dst_base;

        let mut dst_rows: [*mut u16; DEST_BATCH] = [std::ptr::null_mut(); DEST_BATCH];
        let mut coeff_rows: [&[u16]; DEST_BATCH] = [&[]; DEST_BATCH];

        for r in 0..dst_count {
            let idx = dst_base + r;
            let dst_slice = &mut destinations[idx][region_offset..region_offset + region_size];
            dst_rows[r] = dst_slice.as_mut_ptr();
            coeff_rows[r] = coefficients[idx];
        }

        let mut src_batch_start = 0;
        while src_batch_start < num_srcs {
            let src_batch_end = (src_batch_start + SRC_BATCH).min(num_srcs);
            let batch_len = src_batch_end - src_batch_start;

            // Preload coefficients
            let mut any_nonzero = false;
            let mut coeffs: [[u16; SRC_BATCH]; DEST_BATCH] = [[0; SRC_BATCH]; DEST_BATCH];

            for b in 0..batch_len {
                for r in 0..dst_count {
                    coeffs[r][b] = coeff_rows[r][src_batch_start + b];
                    if coeffs[r][b] != 0 {
                        any_nonzero = true;
                    }
                }
            }

            if !any_nonzero {
                src_batch_start = src_batch_end;
                continue;
            }

            // Preload shuffle2x tables
            let mut tables: [[Option<GfShuffle2xTables>; SRC_BATCH]; DEST_BATCH] =
                [[None, None, None, None]; DEST_BATCH];

            for r in 0..dst_count {
                for b in 0..batch_len {
                    if coeffs[r][b] != 0 {
                        tables[r][b] = Some(gf_load_shuffle2x_tables(coeffs[r][b]));
                    }
                }
            }

            // Process chunks - NO FORMAT CONVERSION!
            for chunk_idx in 0..chunks {
                let offset = chunk_idx * 16;

                // Load source data (already in shuffle2x format)
                // Extract nibbles once and reuse across all destinations
                let mut src_nibbles: [(__m256i, __m256i); SRC_BATCH] =
                    [(_mm256_setzero_si256(), _mm256_setzero_si256()); SRC_BATCH];

                for b in 0..batch_len {
                    let src_ptr = sources[src_batch_start + b][region_offset + offset..].as_ptr();
                    let data = _mm256_loadu_si256(src_ptr as *const __m256i);
                    let n_lo = _mm256_and_si256(data, mask);
                    let n_hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask);
                    src_nibbles[b] = (n_lo, n_hi);
                }

                // Process each destination
                for r in 0..dst_count {
                    let dst_ptr = dst_rows[r].add(offset);
                    let mut accum = _mm256_loadu_si256(dst_ptr as *const __m256i);

                    // Accumulate contributions from each source
                    for b in 0..batch_len {
                        if let Some(ref tbl) = tables[r][b] {
                            let (n_lo, n_hi) = src_nibbles[b];

                            // 4 lookups - no format conversion!
                            let mut result = _mm256_shuffle_epi8(tbl.norm_lo, n_lo);
                            let mut swap = _mm256_shuffle_epi8(tbl.swap_lo, n_lo);
                            result =
                                _mm256_xor_si256(result, _mm256_shuffle_epi8(tbl.norm_hi, n_hi));
                            swap = _mm256_xor_si256(swap, _mm256_shuffle_epi8(tbl.swap_hi, n_hi));

                            swap = _mm256_permute2x128_si256(swap, swap, 0x01);
                            result = _mm256_xor_si256(result, swap);

                            accum = _mm256_xor_si256(accum, result);
                        }
                    }

                    _mm256_storeu_si256(dst_ptr as *mut __m256i, accum);
                }
            }

            // Handle remainder with scalar
            if remainder > 0 {
                let offset = chunks * 16;
                for r in 0..dst_count {
                    for b in 0..batch_len {
                        if coeffs[r][b] != 0 {
                            for j in 0..remainder {
                                let src_val =
                                    sources[src_batch_start + b][region_offset + offset + j];
                                let dst_idx = region_offset + offset + j;
                                destinations[dst_base + r][dst_idx] ^=
                                    crate::galois::core::gf_mul(src_val, coeffs[r][b]);
                            }
                        }
                    }
                }
            }

            src_batch_start = src_batch_end;
        }

        dst_base = dst_end;
    }
}
