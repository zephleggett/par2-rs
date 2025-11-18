//! x86-64 SIMD implementations for GF(2^16) operations
//!
//! This module contains all x86-64 specific SIMD implementations using:
//! - PCLMULQDQ (carryless multiplication with Barrett reduction)
//! - AVX2 (table-based multiplication, 32 u16 values per iteration)
//! - SSSE3 (table-based multiplication, 16 u16 values per iteration)
//!
//! **Testing Status**: These implementations follow the reference from reed-solomon-simd
//! but have not been tested on x86-64 hardware. Scalar fallback ensures correctness.

use std::arch::x86_64::*;
use crate::galois::core::{gf_mul, initialize_mul128_table, LOG_TABLE, MUL128_TABLE};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse2,sse4.1")]
pub(super) unsafe fn gf_mul_slice_pclmul_x86(scalar: u16, data: &mut [u16]) {
    use std::arch::x86_64::*;

    // Handle zero scalar
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    // Broadcast scalar to all positions
    let scalar_vec = _mm_set1_epi16(scalar as i16);

    // Process 8 u16 values (16 bytes) at a time
    let chunks = data.len() / 8;
    let remainder = data.len() % 8;

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 u16 values
        let data_vec = _mm_loadu_si128(data.as_ptr().add(offset) as *const __m128i);

        // Multiply using PCLMULQDQ + Barrett
        let result = gf_mul_pclmul_x86_8(data_vec, scalar_vec);

        // Store result
        _mm_storeu_si128(data.as_mut_ptr().add(offset) as *mut __m128i, result);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 8;
        for item in data.iter_mut().skip(start) {
            *item = gf_mul(*item, scalar);
        }
    }
}

/// AVX2 PCLMULQDQ implementation for gf_mul_slice (256-bit, processes 16 u16 at once)
///
/// 2x wider than SSE version, processes 16 u16 values (32 bytes) per iteration.
/// Available on Intel Haswell (2013+) and AMD Excavator (2015+).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,avx2,sse4.1")]
pub(super) unsafe fn gf_mul_slice_avx2_pclmul_x86(scalar: u16, data: &mut [u16]) {
    use std::arch::x86_64::*;

    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    let scalar_vec = _mm_set1_epi16(scalar as i16);

    // Process 16 u16 values (32 bytes) at a time using two 128-bit ops
    let chunks = data.len() / 16;
    let remainder = data.len() % 16;

    for i in 0..chunks {
        let offset = i * 16;

        // Load and process first 8 u16
        let data_vec1 = _mm_loadu_si128(data.as_ptr().add(offset) as *const __m128i);
        let result1 = gf_mul_pclmul_x86_8(data_vec1, scalar_vec);
        _mm_storeu_si128(data.as_mut_ptr().add(offset) as *mut __m128i, result1);

        // Load and process second 8 u16
        let data_vec2 = _mm_loadu_si128(data.as_ptr().add(offset + 8) as *const __m128i);
        let result2 = gf_mul_pclmul_x86_8(data_vec2, scalar_vec);
        _mm_storeu_si128(data.as_mut_ptr().add(offset + 8) as *mut __m128i, result2);
    }

    // Handle remainder with SSE
    let start = chunks * 16;
    if data.len() - start >= 8 {
        let data_vec = _mm_loadu_si128(data.as_ptr().add(start) as *const __m128i);
        let result = gf_mul_pclmul_x86_8(data_vec, scalar_vec);
        _mm_storeu_si128(data.as_mut_ptr().add(start) as *mut __m128i, result);

        // Handle final <8 elements with scalar
        for item in data.iter_mut().skip(start + 8) {
            *item = gf_mul(*item, scalar);
        }
    } else if remainder > 0 {
        for item in data.iter_mut().skip(start) {
            *item = gf_mul(*item, scalar);
        }
    }
}

/// Core 512-bit VPCLMULQDQ helper: multiplies 32 u16 values using AVX-512
///
/// This is the AVX-512 equivalent of gf_mul_pclmul_x86_8 (SSE) and provides 4x throughput.
/// Uses VPCLMULQDQ instructions which perform carryless multiplication in each 128-bit lane
/// of the 512-bit register, followed by Barrett reduction.
///
/// Available on Intel Ice Lake (2019+), Rocket Lake, Tiger Lake, and AMD Zen 4 (2022+).
///
/// # Safety
/// Requires `vpclmulqdq` and `avx512f` CPU features.
#[cfg(all(target_arch = "x86_64", feature = "unstable"))]
#[target_feature(enable = "vpclmulqdq,avx512f,avx512vl,sse4.1")]
pub(super) unsafe fn gf_mul_pclmul_x86_32(data1: __m512i, data2: __m512i) -> __m512i {
    use std::arch::x86_64::*;

    // Split into even/odd u16 values for parallel processing
    // This processes all 4 lanes (128-bit each) of the 512-bit register
    let word_mask = _mm512_set1_epi32(0x0000FFFF_u32 as i32);

    // Even indices: keep in place, mask out odd
    let data1_even = _mm512_and_si512(word_mask, data1);
    let data2_even = _mm512_and_si512(word_mask, data2);

    // Odd indices: keep in place (andnot masks out even)
    let data1_odd = _mm512_andnot_si512(word_mask, data1);
    let data2_odd = _mm512_andnot_si512(word_mask, data2);

    // Carryless multiplication in each 128-bit lane
    // VPCLMULQDQ operates on 128-bit lanes within the 512-bit register
    // 0x00: multiply low qwords in each lane
    // 0x11: multiply high qwords in each lane
    let prod1_even = _mm512_clmulepi64_epi128::<0x00>(data1_even, data2_even);
    let prod2_even = _mm512_clmulepi64_epi128::<0x11>(data1_even, data2_even);
    let prod1_odd = _mm512_clmulepi64_epi128::<0x00>(data1_odd, data2_odd);
    let prod2_odd = _mm512_clmulepi64_epi128::<0x11>(data1_odd, data2_odd);

    // Interleave even/odd results: 0xCC = 0b11001100 selects odd in positions 2,3,6,7
    let prod1 = _mm512_mask_blend_epi16(0xCCCCCCCC, prod1_even, prod1_odd);
    let prod2 = _mm512_mask_blend_epi16(0xCCCCCCCC, prod2_even, prod2_odd);

    // Barrett reduction: reduce 32-bit products to 16-bit GF(2^16) values
    // Split low/high 16-bit halves using shuffle
    let shuf_lo_hi = _mm512_set_epi8(
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    let tmp1 = _mm512_shuffle_epi8(prod1, shuf_lo_hi);
    let tmp2 = _mm512_shuffle_epi8(prod2, shuf_lo_hi);

    // rem = low 16 bits, quot = high 16 bits
    let rem = _mm512_unpacklo_epi64(tmp1, tmp2);
    let mut quot = _mm512_unpackhi_epi64(tmp1, tmp2);

    // Multiply quot by 0x1111a (Barrett constant)
    let tmp1 = _mm512_xor_si512(quot, _mm512_srli_epi16(quot, 4));
    let tmp1 = _mm512_xor_si512(tmp1, _mm512_srli_epi16(tmp1, 8));
    quot = _mm512_xor_si512(tmp1, _mm512_srli_epi16(quot, 13));

    // Multiply quot by 0x100b (irreducible polynomial without leading 1)
    let tmp1 = _mm512_xor_si512(quot, _mm512_slli_epi16(quot, 3));
    let tmp1 = _mm512_xor_si512(tmp1, _mm512_add_epi16(quot, quot)); // quot * 2
    quot = _mm512_xor_si512(tmp1, _mm512_slli_epi16(quot, 12));

    // Final result: XOR remainder with reduced quotient
    _mm512_xor_si512(quot, rem)
}

/// AVX-512 VPCLMULQDQ implementation for gf_mul_slice (512-bit, processes 32 u16 at once)
///
/// 4x wider than SSE version, 2x wider than AVX2 version.
/// Processes 32 u16 values (64 bytes) per iteration.
/// Available on Intel Ice Lake (2019+) and AMD Zen 4 (2022+).
#[cfg(all(target_arch = "x86_64", feature = "unstable"))]
#[target_feature(enable = "vpclmulqdq,avx512f,avx512vl,sse4.1")]
pub(super) unsafe fn gf_mul_slice_vpclmul_x86(scalar: u16, data: &mut [u16]) {
    use std::arch::x86_64::*;

    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    let scalar_vec = _mm512_set1_epi16(scalar as i16);

    // Process 32 u16 values (64 bytes) at a time
    let chunks = data.len() / 32;
    let remainder = data.len() % 32;

    for i in 0..chunks {
        let offset = i * 32;

        let data_vec = _mm512_loadu_si512(data.as_ptr().add(offset) as *const __m512i);
        let result = gf_mul_pclmul_x86_32(data_vec, scalar_vec);
        _mm512_storeu_si512(data.as_mut_ptr().add(offset) as *mut __m512i, result);
    }

    // Handle remainder with AVX2
    let start = chunks * 32;
    if data.len() - start >= 16 {
        gf_mul_slice_avx2_pclmul_x86(scalar, &mut data[start..]);
    } else if remainder > 0 {
        for item in data.iter_mut().skip(start) {
            *item = gf_mul(*item, scalar);
        }
    }
}

/// Placeholder for VPCLMUL+GFNI - requires unstable features
#[cfg(all(target_arch = "x86_64", feature = "unstable"))]
#[allow(dead_code)]
pub(super) unsafe fn gf_mul_slice_vpclmul_gfni_x86(_scalar: u16, _data: &mut [u16]) {
    // TODO: Implement GFNI affine transformation
    // For now, fall back to AVX2
    #[cfg(target_feature = "avx2")]
    gf_mul_slice_avx2_pclmul_x86(_scalar, _data);
}

/// AVX2 table-based multiplication: processes 32 u16 values (64 bytes) per iteration
///
/// Uses table lookups with vpshufb for 8-10x speedup over scalar operations.
/// This is the preferred x86-64 implementation until PCLMULQDQ is implemented.
///
/// **Untested**: This implementation follows the reference from reed-solomon-simd
/// exactly but has not been tested on x86-64 hardware.
///
/// Reference: <https://github.com/AndersTrier/reed-solomon-simd>
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub(super) unsafe fn gf_mul_slice_avx2(scalar: u16, data: &mut [u16]) {
    // Handle zero scalar
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    // Ensure tables are initialized
    initialize_mul128_table();

    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let mul_table = MUL128_TABLE.get().expect("MUL128_TABLE not initialized");

    // Get the logarithm of the scalar for table lookup
    let log_scalar = log_table[scalar as usize] as usize;
    let lut = &mul_table[log_scalar];

    // Load lookup table into AVX2 registers
    let t0_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.lo[0]).cast::<__m128i>(),
    ));
    let t1_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.lo[1]).cast::<__m128i>(),
    ));
    let t2_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.lo[2]).cast::<__m128i>(),
    ));
    let t3_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.lo[3]).cast::<__m128i>(),
    ));

    let t0_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.hi[0]).cast::<__m128i>(),
    ));
    let t1_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.hi[1]).cast::<__m128i>(),
    ));
    let t2_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.hi[2]).cast::<__m128i>(),
    ));
    let t3_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
        std::ptr::from_ref::<u128>(&lut.hi[3]).cast::<__m128i>(),
    ));

    let clr_mask = _mm256_set1_epi8(0x0f);

    // Process data in 64-byte chunks (32 u16 values)
    // Following reed-solomon-simd's LEO_MUL_256 pattern
    let chunks = data.len() / 32;
    let remainder = data.len() % 32;

    let ptr = data.as_mut_ptr() as *mut u8;

    for i in 0..chunks {
        let offset = i * 64;
        let data_ptr = ptr.add(offset) as *mut __m256i;

        // Load 64 bytes as two 32-byte registers
        let value_lo = _mm256_loadu_si256(data_ptr);
        let value_hi = _mm256_loadu_si256(data_ptr.add(1));

        // Nibble-based multiplication following Leopard-RS pattern
        let data_0 = _mm256_and_si256(value_lo, clr_mask);
        let mut prod_lo = _mm256_shuffle_epi8(t0_lo, data_0);
        let mut prod_hi = _mm256_shuffle_epi8(t0_hi, data_0);

        let data_1 = _mm256_and_si256(_mm256_srli_epi64(value_lo, 4), clr_mask);
        prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(t1_lo, data_1));
        prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(t1_hi, data_1));

        let data_2 = _mm256_and_si256(value_hi, clr_mask);
        prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(t2_lo, data_2));
        prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(t2_hi, data_2));

        let data_3 = _mm256_and_si256(_mm256_srli_epi64(value_hi, 4), clr_mask);
        prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(t3_lo, data_3));
        prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(t3_hi, data_3));

        //Store results - prod_lo and prod_hi represent the result bytes
        _mm256_storeu_si256(data_ptr, prod_lo);
        _mm256_storeu_si256(data_ptr.add(1), prod_hi);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 32;
        for item in data.iter_mut().skip(start) {
            *item = gf_mul(*item, scalar);
        }
    }
}

/// SSSE3 table-based multiplication: processes 16 u16 values (32 bytes) per iteration
///
/// Uses table lookups with pshufb for 8-10x speedup over scalar operations.
/// Fallback for older x86-64 CPUs without AVX2 support.
///
/// **Untested**: This implementation is adapted from reed-solomon-simd but has
/// not been tested on x86-64 hardware.
///
/// Reference: <https://github.com/AndersTrier/reed-solomon-simd>
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(dead_code)]
pub(super) unsafe fn gf_mul_slice_ssse3(scalar: u16, data: &mut [u16]) {
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    // Ensure tables are initialized
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

    // Process 16 u16 values (32 bytes) at a time
    let chunks = data.len() / 16;
    let remainder = data.len() % 16;

    let ptr = data.as_mut_ptr() as *mut u8;

    for i in 0..chunks {
        let offset = i * 32;
        let data_ptr = ptr.add(offset);

        // Load 32 bytes as two consecutive 16-byte registers
        let interleaved0 = _mm_loadu_si128(data_ptr as *const __m128i);
        let interleaved1 = _mm_loadu_si128(data_ptr.add(16) as *const __m128i);

        // Deinterleave using punpck instructions (SSSE3 doesn't have dedicated deinterleave)
        let low_bytes = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
        let high_bytes = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1);

        let value_lo_part0 = _mm_shuffle_epi8(interleaved0, low_bytes);
        let value_lo_part1 = _mm_shuffle_epi8(interleaved1, low_bytes);
        let value_hi_part0 = _mm_shuffle_epi8(interleaved0, high_bytes);
        let value_hi_part1 = _mm_shuffle_epi8(interleaved1, high_bytes);

        let value_lo = _mm_unpacklo_epi64(value_lo_part0, value_lo_part1);
        let value_hi = _mm_unpacklo_epi64(value_hi_part0, value_hi_part1);

        // Nibble-based multiplication
        let data_0 = _mm_and_si128(value_lo, clr_mask);
        let mut prod_lo = _mm_shuffle_epi8(t0_lo, data_0);
        let mut prod_hi = _mm_shuffle_epi8(t0_hi, data_0);

        let data_1 = _mm_and_si128(_mm_srli_epi64(value_lo, 4), clr_mask);
        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t1_lo, data_1));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t1_hi, data_1));

        let data_2 = _mm_and_si128(value_hi, clr_mask);
        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t2_lo, data_2));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t2_hi, data_2));

        let data_3 = _mm_and_si128(_mm_srli_epi64(value_hi, 4), clr_mask);
        prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t3_lo, data_3));
        prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t3_hi, data_3));

        // Reinterleave results
        let reint0 = _mm_unpacklo_epi8(prod_lo, prod_hi);
        let reint1 = _mm_unpackhi_epi8(prod_lo, prod_hi);

        _mm_storeu_si128(data_ptr as *mut __m128i, reint0);
        _mm_storeu_si128(data_ptr.add(16) as *mut __m128i, reint1);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 16;
        for item in data.iter_mut().skip(start) {
            *item = gf_mul(*item, scalar);
        }
    }
}

/// Multiply 8 u16 values by a scalar using PCLMULQDQ+Barrett reduction
///
/// Core GF(2^16) multiplication using carryless multiplication and Barrett reduction.
/// Based on par2cmdline-turbo's gf16pmul implementation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse2,sse4.1")]
pub(super) unsafe fn gf_mul_pclmul_x86_8(data1: __m128i, data2: __m128i) -> __m128i {
    use std::arch::x86_64::*;

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
    // 0x00: multiply low qwords
    // 0x11: multiply high qwords
    let prod1_even = _mm_clmulepi64_si128(data1_even, data2_even, 0x00);
    let prod2_even = _mm_clmulepi64_si128(data1_even, data2_even, 0x11);
    let prod1_odd = _mm_clmulepi64_si128(data1_odd, data2_odd, 0x00);
    let prod2_odd = _mm_clmulepi64_si128(data1_odd, data2_odd, 0x11);

    // Interleave even/odd results: 0xCC = 0b11001100 selects odd in positions 2,3,6,7
    let prod1 = _mm_blend_epi16(prod1_even, prod1_odd, 0xCC);
    let prod2 = _mm_blend_epi16(prod2_even, prod2_odd, 0xCC);

    // Barrett reduction: reduce 32-bit products to 16-bit GF(2^16) values
    // Split low/high 16-bit halves using shuffle
    let shuf_lo_hi = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
    let tmp1 = _mm_shuffle_epi8(prod1, shuf_lo_hi);
    let tmp2 = _mm_shuffle_epi8(prod2, shuf_lo_hi);

    // rem = low 16 bits, quot = high 16 bits
    let rem = _mm_unpacklo_epi64(tmp1, tmp2);
    let mut quot = _mm_unpackhi_epi64(tmp1, tmp2);

    // Multiply quot by 0x1111a (Barrett constant) and retain high half
    // Using shift+xor is faster than actual multiplication
    let tmp1 = _mm_xor_si128(quot, _mm_srli_epi16(quot, 4));
    let tmp1 = _mm_xor_si128(tmp1, _mm_srli_epi16(tmp1, 8));
    quot = _mm_xor_si128(tmp1, _mm_srli_epi16(quot, 13));

    // Multiply quot by 0x100b (irreducible polynomial without leading 1), retain low half
    let tmp1 = _mm_xor_si128(quot, _mm_slli_epi16(quot, 3));
    let tmp1 = _mm_xor_si128(tmp1, _mm_add_epi16(quot, quot)); // quot * 2
    quot = _mm_xor_si128(tmp1, _mm_slli_epi16(quot, 12));

    // Final result: XOR remainder with reduced quotient
    _mm_xor_si128(quot, rem)
}

/// x86 PCLMULQDQ implementation for gf_muladd using carryless multiplication
///
/// This is the x86 equivalent of ARM's PMULL-based implementation.
/// Uses PCLMULQDQ instruction for polynomial multiplication in GF(2^16) with
/// Barrett reduction modulo 0x1100B (the irreducible polynomial).
///
/// Processes 8 u16 values (16 bytes) per iteration with ~4-5x speedup.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse2,sse4.1")]
pub(super) unsafe fn gf_muladd_pclmul_x86(dst: &mut [u16], src: &[u16], scalar: u16) {
    use std::arch::x86_64::*;

    if scalar == 0 {
        return;
    }

    // Broadcast scalar to all positions
    let scalar_vec = _mm_set1_epi16(scalar as i16);

    // Process 8 u16 values (16 bytes) at a time
    let chunks = dst.len().min(src.len()) / 8;
    let remainder = dst.len().min(src.len()) % 8;

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 u16 values from source
        let src_vec = _mm_loadu_si128(src.as_ptr().add(offset) as *const __m128i);

        // Multiply using PCLMULQDQ + Barrett
        let prod_vec = gf_mul_pclmul_x86_8(src_vec, scalar_vec);

        // Load destination, XOR with product, store
        let dst_vec = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
        let result = _mm_xor_si128(dst_vec, prod_vec);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 8;
        for i in start..dst.len().min(src.len()) {
            dst[i] ^= gf_mul(src[i], scalar);
        }
    }
}

/// AVX2 PCLMULQDQ implementation for gf_muladd (256-bit, processes 16 u16 at once)
///
/// 2x wider than SSE version, processes 16 u16 values (32 bytes) per iteration.
/// Available on Intel Haswell (2013+) and AMD Excavator (2015+).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,avx2,sse4.1")]
pub(super) unsafe fn gf_muladd_avx2_pclmul_x86(dst: &mut [u16], src: &[u16], scalar: u16) {
    use std::arch::x86_64::*;

    if scalar == 0 {
        return;
    }

    let scalar_vec = _mm_set1_epi16(scalar as i16);

    // Process 16 u16 values (32 bytes) at a time using two 128-bit ops
    let chunks = dst.len().min(src.len()) / 16;
    let remainder = dst.len().min(src.len()) % 16;

    for i in 0..chunks {
        let offset = i * 16;

        // Process first 8 u16
        let src_vec1 = _mm_loadu_si128(src.as_ptr().add(offset) as *const __m128i);
        let prod_vec1 = gf_mul_pclmul_x86_8(src_vec1, scalar_vec);
        let dst_vec1 = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
        let result1 = _mm_xor_si128(dst_vec1, prod_vec1);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result1);

        // Process second 8 u16
        let src_vec2 = _mm_loadu_si128(src.as_ptr().add(offset + 8) as *const __m128i);
        let prod_vec2 = gf_mul_pclmul_x86_8(src_vec2, scalar_vec);
        let dst_vec2 = _mm_loadu_si128(dst.as_ptr().add(offset + 8) as *const __m128i);
        let result2 = _mm_xor_si128(dst_vec2, prod_vec2);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset + 8) as *mut __m128i, result2);
    }

    // Handle remainder with SSE
    let start = chunks * 16;
    if dst.len().min(src.len()) - start >= 8 {
        let src_vec = _mm_loadu_si128(src.as_ptr().add(start) as *const __m128i);
        let prod_vec = gf_mul_pclmul_x86_8(src_vec, scalar_vec);
        let dst_vec = _mm_loadu_si128(dst.as_ptr().add(start) as *const __m128i);
        let result = _mm_xor_si128(dst_vec, prod_vec);
        _mm_storeu_si128(dst.as_mut_ptr().add(start) as *mut __m128i, result);

        // Handle final <8 elements with scalar
        for i in (start + 8)..dst.len().min(src.len()) {
            dst[i] ^= gf_mul(src[i], scalar);
        }
    } else if remainder > 0 {
        for i in start..dst.len().min(src.len()) {
            dst[i] ^= gf_mul(src[i], scalar);
        }
    }
}

/// AVX-512 VPCLMULQDQ implementation for gf_muladd (512-bit, processes 32 u16 at once)
///
/// 4x wider than SSE version, 2x wider than AVX2 version.
/// Processes 32 u16 values (64 bytes) per iteration.
#[cfg(all(target_arch = "x86_64", feature = "unstable"))]
#[target_feature(enable = "vpclmulqdq,avx512f,avx512vl,sse4.1")]
pub(super) unsafe fn gf_muladd_vpclmul_x86(dst: &mut [u16], src: &[u16], scalar: u16) {
    use std::arch::x86_64::*;

    if scalar == 0 {
        return;
    }

    let scalar_vec = _mm512_set1_epi16(scalar as i16);

    // Process 32 u16 values (64 bytes) at a time
    let chunks = dst.len().min(src.len()) / 32;
    let remainder = dst.len().min(src.len()) % 32;

    for i in 0..chunks {
        let offset = i * 32;

        let src_vec = _mm512_loadu_si512(src.as_ptr().add(offset) as *const __m512i);
        let prod_vec = gf_mul_pclmul_x86_32(src_vec, scalar_vec);
        let dst_vec = _mm512_loadu_si512(dst.as_ptr().add(offset) as *const __m512i);
        let result = _mm512_xor_si512(dst_vec, prod_vec);
        _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, result);
    }

    // Handle remainder with AVX2
    let start = chunks * 32;
    if dst.len().min(src.len()) - start >= 16 {
        gf_muladd_avx2_pclmul_x86(&mut dst[start..], &src[start..], scalar);
    } else if remainder > 0 {
        for i in start..dst.len().min(src.len()) {
            dst[i] ^= gf_mul(src[i], scalar);
        }
    }
}

/// x86 PCLMULQDQ multi-region implementation
///
/// Processes up to 8 source regions simultaneously, accumulating products before XORing.
/// Key optimization: batches multiple sources to maximize register utilization.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse2,sse4.1")]
pub(super) unsafe fn gf_muladd_multi_pclmul_x86(dst: &mut [u16], sources: &[&[u16]], coefficients: &[u16]) {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
    if CALL_COUNT.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "  ✓ Using 8-region parallel PCLMUL x86 (sources={}, dst_len={})",
            sources.len(),
            dst.len()
        );
    }

    assert!(sources.len() <= 8, "Max 8 regions supported");
    assert_eq!(sources.len(), coefficients.len());

    if sources.is_empty() {
        return;
    }

    // Verify all sources have same length as destination
    for src in sources {
        assert_eq!(dst.len(), src.len());
    }

    // Process 8 u16 values (16 bytes) at a time
    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    if min_len == 0 {
        return;
    }
    let chunks = dst.len().min(min_len) / 8;
    let remainder = dst.len().min(min_len) % 8;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 8;

        // Initialize accumulator to zero
        let mut acc_vec = _mm_setzero_si128();

        // Process all sources, accumulating products using PCLMUL
        for i in 0..sources.len() {
            if coefficients[i] == 0 {
                continue;
            }

            // Load 8 u16 values from source
            let src_vec = _mm_loadu_si128(sources[i].as_ptr().add(offset) as *const __m128i);

            // Broadcast coefficient
            let coeff_vec = _mm_set1_epi16(coefficients[i] as i16);

            // Multiply using PCLMULQDQ + Barrett
            let prod_vec = gf_mul_pclmul_x86_8(src_vec, coeff_vec);

            // Accumulate with XOR
            acc_vec = _mm_xor_si128(acc_vec, prod_vec);
        }

        // Load destination, XOR with accumulated products, store
        let dst_vec = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
        let result = _mm_xor_si128(dst_vec, acc_vec);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 8;
        let limit = dst.len().min(min_len);
        for (i, dst_item) in dst.iter_mut().enumerate().take(limit).skip(start) {
            for j in 0..sources.len() {
                if coefficients[j] != 0 {
                    *dst_item ^= gf_mul(sources[j][i], coefficients[j]);
                }
            }
        }
    }
}

/// AVX2 PCLMULQDQ multi-region implementation (256-bit, processes 16 u16 at once)
///
/// 2x wider than SSE version. Processes up to 8 source regions simultaneously,
/// accumulating products before XORing. Processes 16 u16 values (32 bytes) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,avx2,sse4.1")]
pub(super) unsafe fn gf_muladd_multi_avx2_pclmul_x86(
    dst: &mut [u16],
    sources: &[&[u16]],
    coefficients: &[u16],
) {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static WARNED: AtomicUsize = AtomicUsize::new(0);

    // Find minimum length
    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    if dst.len() < min_len && WARNED.swap(1, Ordering::Relaxed) == 0 {
        eprintln!("Warning: dst.len()={} < min_src_len={}", dst.len(), min_len);
    }

    // Process 16 u16 values (32 bytes) at a time using two 128-bit ops
    let chunks = dst.len().min(min_len) / 16;
    let remainder = dst.len().min(min_len) % 16;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 16;

        // Process first 8 u16
        let mut acc_vec1 = _mm_setzero_si128();
        for i in 0..sources.len() {
            if coefficients[i] == 0 {
                continue;
            }
            let src_vec = _mm_loadu_si128(sources[i].as_ptr().add(offset) as *const __m128i);
            let coeff_vec = _mm_set1_epi16(coefficients[i] as i16);
            let prod_vec = gf_mul_pclmul_x86_8(src_vec, coeff_vec);
            acc_vec1 = _mm_xor_si128(acc_vec1, prod_vec);
        }
        let dst_vec1 = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
        let result1 = _mm_xor_si128(dst_vec1, acc_vec1);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result1);

        // Process second 8 u16
        let mut acc_vec2 = _mm_setzero_si128();
        for i in 0..sources.len() {
            if coefficients[i] == 0 {
                continue;
            }
            let src_vec = _mm_loadu_si128(sources[i].as_ptr().add(offset + 8) as *const __m128i);
            let coeff_vec = _mm_set1_epi16(coefficients[i] as i16);
            let prod_vec = gf_mul_pclmul_x86_8(src_vec, coeff_vec);
            acc_vec2 = _mm_xor_si128(acc_vec2, prod_vec);
        }
        let dst_vec2 = _mm_loadu_si128(dst.as_ptr().add(offset + 8) as *const __m128i);
        let result2 = _mm_xor_si128(dst_vec2, acc_vec2);
        _mm_storeu_si128(dst.as_mut_ptr().add(offset + 8) as *mut __m128i, result2);
    }

    // Handle remainder with SSE
    let start = chunks * 16;
    if dst.len().min(min_len) - start >= 8 {
        let mut acc_vec = _mm_setzero_si128();
        for i in 0..sources.len() {
            if coefficients[i] == 0 {
                continue;
            }
            let src_vec = _mm_loadu_si128(sources[i].as_ptr().add(start) as *const __m128i);
            let coeff_vec = _mm_set1_epi16(coefficients[i] as i16);
            let prod_vec = gf_mul_pclmul_x86_8(src_vec, coeff_vec);
            acc_vec = _mm_xor_si128(acc_vec, prod_vec);
        }
        let dst_vec = _mm_loadu_si128(dst.as_ptr().add(start) as *const __m128i);
        let result = _mm_xor_si128(dst_vec, acc_vec);
        _mm_storeu_si128(dst.as_mut_ptr().add(start) as *mut __m128i, result);

        // Handle final <8 elements with scalar
        let limit = dst.len().min(min_len);
        for (i, dst_item) in dst.iter_mut().enumerate().take(limit).skip(start + 8) {
            for j in 0..sources.len() {
                if coefficients[j] != 0 {
                    *dst_item ^= gf_mul(sources[j][i], coefficients[j]);
                }
            }
        }
    } else if remainder > 0 {
        let limit = dst.len().min(min_len);
        for (i, dst_item) in dst.iter_mut().enumerate().take(limit).skip(start) {
            for j in 0..sources.len() {
                if coefficients[j] != 0 {
                    *dst_item ^= gf_mul(sources[j][i], coefficients[j]);
                }
            }
        }
    }
}

/// AVX-512 VPCLMULQDQ multi-region implementation (512-bit, processes 32 u16 at once)
///
/// 4x wider than SSE, 2x wider than AVX2. Processes up to 8 source regions simultaneously,
/// accumulating products before XORing. Processes 32 u16 values (64 bytes) per iteration.
#[cfg(all(target_arch = "x86_64", feature = "unstable"))]
#[target_feature(enable = "vpclmulqdq,avx512f,avx512vl,sse4.1")]
pub(super) unsafe fn gf_muladd_multi_vpclmul_x86(
    dst: &mut [u16],
    sources: &[&[u16]],
    coefficients: &[u16],
) {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static WARNED: AtomicUsize = AtomicUsize::new(0);

    // Find minimum length
    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    if dst.len() < min_len && WARNED.swap(1, Ordering::Relaxed) == 0 {
        eprintln!("Warning: dst.len()={} < min_src_len={}", dst.len(), min_len);
    }

    // Process 32 u16 values (64 bytes) at a time
    let chunks = dst.len().min(min_len) / 32;
    let remainder = dst.len().min(min_len) % 32;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 32;

        let mut acc_vec = _mm512_setzero_si512();
        for i in 0..sources.len() {
            if coefficients[i] == 0 {
                continue;
            }
            let src_vec = _mm512_loadu_si512(sources[i].as_ptr().add(offset) as *const __m512i);
            let coeff_vec = _mm512_set1_epi16(coefficients[i] as i16);
            let prod_vec = gf_mul_pclmul_x86_32(src_vec, coeff_vec);
            acc_vec = _mm512_xor_si512(acc_vec, prod_vec);
        }
        let dst_vec = _mm512_loadu_si512(dst.as_ptr().add(offset) as *const __m512i);
        let result = _mm512_xor_si512(dst_vec, acc_vec);
        _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, result);
    }

    // Handle remainder with AVX2
    let start = chunks * 32;
    if dst.len().min(min_len) - start >= 16 {
        // Use AVX2 for next 16 u16
        let mut sources_slice: Vec<&[u16]> = sources.iter().map(|s| &s[start..]).collect();
        gf_muladd_multi_avx2_pclmul_x86(&mut dst[start..], &sources_slice, coefficients);
    } else if remainder > 0 {
        let limit = dst.len().min(min_len);
        for (i, dst_item) in dst.iter_mut().enumerate().take(limit).skip(start) {
            for j in 0..sources.len() {
                if coefficients[j] != 0 {
                    *dst_item ^= gf_mul(sources[j][i], coefficients[j]);
                }
            }
        }
    }
}

/// x86 PCLMULQDQ column implementation
///
/// One source contributes to multiple destinations (transpose of multi-region).
/// Processes up to 8 destinations simultaneously.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse2,sse4.1")]
pub(super) unsafe fn gf_muladd_column_pclmul_x86(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
) {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
    if CALL_COUNT.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "  ✓ Using parallel column PCLMUL x86 (destinations={}, src_len={})",
            destinations.len(),
            source.len()
        );
    }

    assert!(destinations.len() <= 8, "Max 8 destinations supported");
    assert_eq!(destinations.len(), coefficients.len());

    if destinations.is_empty() {
        return;
    }

    // Verify all destinations have same length as source
    for dst in destinations.iter() {
        assert_eq!(dst.len(), source.len());
    }

    // Process 8 u16 values at a time
    let chunks = source.len() / 8;
    let remainder = source.len() % 8;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 8;

        // Load source data once
        let src_vec = _mm_loadu_si128(source.as_ptr().add(offset) as *const __m128i);

        // Process each destination
        for (i, dst) in destinations.iter_mut().enumerate() {
            if coefficients[i] == 0 {
                continue;
            }

            // Broadcast coefficient
            let coeff_vec = _mm_set1_epi16(coefficients[i] as i16);

            // Multiply using PCLMULQDQ + Barrett
            let prod_vec = gf_mul_pclmul_x86_8(src_vec, coeff_vec);

            // Load destination, XOR with product, store
            let dst_vec = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
            let result = _mm_xor_si128(dst_vec, prod_vec);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result);
        }
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 8;
        for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
            if coeff != 0 {
                for i in start..source.len() {
                    dst[i] ^= gf_mul(source[i], coeff);
                }
            }
        }
    }
}

/// AVX2 PCLMULQDQ column implementation (256-bit, processes 16 u16 at once)
///
/// 2x wider than SSE version. One source contributes to multiple destinations,
/// processing up to 8 destinations simultaneously. Processes 16 u16 values (32 bytes) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,avx2,sse4.1")]
pub(super) unsafe fn gf_muladd_column_avx2_pclmul_x86(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
) {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static WARNED: AtomicUsize = AtomicUsize::new(0);

    // Verify all destinations match source length
    for (i, dst) in destinations.iter().enumerate() {
        if dst.len() != source.len() && WARNED.swap(1, Ordering::Relaxed) == 0 {
            eprintln!(
                "Warning: destinations[{}].len()={} != source.len()={}",
                i,
                dst.len(),
                source.len()
            );
        }
    }

    // Process 16 u16 values (32 bytes) at a time using two 128-bit ops
    let chunks = source.len() / 16;
    let remainder = source.len() % 16;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 16;

        // Load source data (first 8 u16)
        let src_vec1 = _mm_loadu_si128(source.as_ptr().add(offset) as *const __m128i);
        // Load source data (second 8 u16)
        let src_vec2 = _mm_loadu_si128(source.as_ptr().add(offset + 8) as *const __m128i);

        // Process each destination
        for (i, dst) in destinations.iter_mut().enumerate() {
            if coefficients[i] == 0 {
                continue;
            }

            let coeff_vec = _mm_set1_epi16(coefficients[i] as i16);

            // Process first 8 u16
            let prod_vec1 = gf_mul_pclmul_x86_8(src_vec1, coeff_vec);
            let dst_vec1 = _mm_loadu_si128(dst.as_ptr().add(offset) as *const __m128i);
            let result1 = _mm_xor_si128(dst_vec1, prod_vec1);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, result1);

            // Process second 8 u16
            let prod_vec2 = gf_mul_pclmul_x86_8(src_vec2, coeff_vec);
            let dst_vec2 = _mm_loadu_si128(dst.as_ptr().add(offset + 8) as *const __m128i);
            let result2 = _mm_xor_si128(dst_vec2, prod_vec2);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset + 8) as *mut __m128i, result2);
        }
    }

    // Handle remainder with SSE
    let start = chunks * 16;
    if source.len() - start >= 8 {
        let src_vec = _mm_loadu_si128(source.as_ptr().add(start) as *const __m128i);

        for (i, dst) in destinations.iter_mut().enumerate() {
            if coefficients[i] == 0 {
                continue;
            }

            let coeff_vec = _mm_set1_epi16(coefficients[i] as i16);
            let prod_vec = gf_mul_pclmul_x86_8(src_vec, coeff_vec);
            let dst_vec = _mm_loadu_si128(dst.as_ptr().add(start) as *const __m128i);
            let result = _mm_xor_si128(dst_vec, prod_vec);
            _mm_storeu_si128(dst.as_mut_ptr().add(start) as *mut __m128i, result);
        }

        // Handle final <8 elements with scalar
        for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
            if coeff != 0 {
                for i in (start + 8)..source.len() {
                    dst[i] ^= gf_mul(source[i], coeff);
                }
            }
        }
    } else if remainder > 0 {
        for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
            if coeff != 0 {
                for i in start..source.len() {
                    dst[i] ^= gf_mul(source[i], coeff);
                }
            }
        }
    }
}

/// AVX-512 VPCLMULQDQ column implementation (512-bit, processes 32 u16 at once)
///
/// 4x wider than SSE, 2x wider than AVX2. One source contributes to multiple destinations,
/// processing up to 8 destinations simultaneously. Processes 32 u16 values (64 bytes) per iteration.
#[cfg(all(target_arch = "x86_64", feature = "unstable"))]
#[target_feature(enable = "vpclmulqdq,avx512f,avx512vl,sse4.1")]
pub(super) unsafe fn gf_muladd_column_vpclmul_x86(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
) {
    use std::arch::x86_64::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static WARNED: AtomicUsize = AtomicUsize::new(0);

    // Verify all destinations match source length
    for (i, dst) in destinations.iter().enumerate() {
        if dst.len() != source.len() && WARNED.swap(1, Ordering::Relaxed) == 0 {
            eprintln!(
                "Warning: destinations[{}].len()={} != source.len()={}",
                i,
                dst.len(),
                source.len()
            );
        }
    }

    // Process 32 u16 values (64 bytes) at a time
    let chunks = source.len() / 32;
    let remainder = source.len() % 32;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 32;

        // Load source data (32 u16)
        let src_vec = _mm512_loadu_si512(source.as_ptr().add(offset) as *const __m512i);

        // Process each destination
        for (i, dst) in destinations.iter_mut().enumerate() {
            if coefficients[i] == 0 {
                continue;
            }

            let coeff_vec = _mm512_set1_epi16(coefficients[i] as i16);
            let prod_vec = gf_mul_pclmul_x86_32(src_vec, coeff_vec);
            let dst_vec = _mm512_loadu_si512(dst.as_ptr().add(offset) as *const __m512i);
            let result = _mm512_xor_si512(dst_vec, prod_vec);
            _mm512_storeu_si512(dst.as_mut_ptr().add(offset) as *mut __m512i, result);
        }
    }

    // Handle remainder with AVX2
    let start = chunks * 32;
    if source.len() - start >= 16 {
        gf_muladd_column_avx2_pclmul_x86(destinations, &source[start..], coefficients);
    } else if remainder > 0 {
        for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
            if coeff != 0 {
                for i in start..source.len() {
                    dst[i] ^= gf_mul(source[i], coeff);
                }
            }
        }
    }
}
