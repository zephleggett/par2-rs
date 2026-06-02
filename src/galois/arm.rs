//! ARM NEON implementations for GF(2^16) operations
//!
//! This module contains all ARM AArch64 NEON implementations using:
//! - PMULL (polynomial multiplication with Barrett reduction)
//! - NEON (table-based multiplication)
//!
//! **Testing Status**: Fully tested and verified on Apple Silicon with 8-10x speedup
//!
//! # Safety Invariants
//!
//! All unsafe functions in this module require:
//! 1. CPU support for NEON instructions (guaranteed on AArch64)
//! 2. Lookup tables initialized via `galois::init_tables()` (automatic via public API)
//!
//! The public dispatch functions in `galois/mod.rs` handle initialization automatically,
//! so direct callers need only ensure they're on an AArch64 CPU.

use crate::galois::core::{
    debug_assert_tables_initialized, gf_mul, initialize_mul128_table, LOG_TABLE, MUL128_TABLE,
};
use std::arch::aarch64::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// NEON table-based GF(2^16) multiplication on a slice.
///
/// # Safety
///
/// - Caller must ensure CPU supports NEON instructions (always true on AArch64)
/// - Tables are initialized automatically by `initialize_mul128_table()` call
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gf_mul_slice_neon(scalar: u16, data: &mut [u16]) {
    // Handle zero scalar
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    // Ensure tables are initialized (also calls init_tables internally)
    initialize_mul128_table();
    debug_assert_tables_initialized();

    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let mul_table = MUL128_TABLE.get().expect("MUL128_TABLE not initialized");

    // Get the logarithm of the scalar for table lookup
    let log_scalar = log_table[scalar as usize] as usize;
    let lut = &mul_table[log_scalar];

    // Load lookup tables into NEON registers
    let t0_lo = vld1q_u8(std::ptr::from_ref::<u128>(&lut.lo[0]).cast::<u8>());
    let t1_lo = vld1q_u8(std::ptr::from_ref::<u128>(&lut.lo[1]).cast::<u8>());
    let t2_lo = vld1q_u8(std::ptr::from_ref::<u128>(&lut.lo[2]).cast::<u8>());
    let t3_lo = vld1q_u8(std::ptr::from_ref::<u128>(&lut.lo[3]).cast::<u8>());

    let t0_hi = vld1q_u8(std::ptr::from_ref::<u128>(&lut.hi[0]).cast::<u8>());
    let t1_hi = vld1q_u8(std::ptr::from_ref::<u128>(&lut.hi[1]).cast::<u8>());
    let t2_hi = vld1q_u8(std::ptr::from_ref::<u128>(&lut.hi[2]).cast::<u8>());
    let t3_hi = vld1q_u8(std::ptr::from_ref::<u128>(&lut.hi[3]).cast::<u8>());

    let clr_mask = vdupq_n_u8(0x0f);

    // Process 16 u16 values (32 bytes) at a time
    // Note: Standard u16 arrays have interleaved bytes [low0, high0, low1, high1, ...]
    // but nibble-shuffle multiplication expects deinterleaved format [low0..low15, high0..high15]
    // We need to deinterleave using vuzpq_u8 before processing
    let chunks = data.len() / 16;
    let remainder = data.len() % 16;

    let ptr = data.as_mut_ptr() as *mut u8;

    for i in 0..chunks {
        let offset = i * 32;
        let data_ptr = ptr.add(offset);

        // Load 32 bytes as two consecutive 16-byte registers
        let interleaved0 = vld1q_u8(data_ptr); // bytes [0..15]: [low0, high0, ..., low7, high7]
        let interleaved1 = vld1q_u8(data_ptr.add(16)); // bytes [16..31]: [low8, high8, ..., low15, high15]

        // Deinterleave: separate even bytes (lows) from odd bytes (highs)
        let deint0 = vuzpq_u8(interleaved0, interleaved1);
        let value_lo = deint0.0; // All low bytes: [low0..low15]
        let value_hi = deint0.1; // All high bytes: [high0..high15]

        // Process deinterleaved data
        let (prod_lo, prod_hi) = neon_mul_128(
            value_lo, value_hi, &clr_mask, t0_lo, t1_lo, t2_lo, t3_lo, t0_hi, t1_hi, t2_hi, t3_hi,
        );

        // Reinterleave results before storing
        let reint = vzipq_u8(prod_lo, prod_hi);
        vst1q_u8(data_ptr, reint.0);
        vst1q_u8(data_ptr.add(16), reint.1);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 16;
        for item in data.iter_mut().skip(start) {
            *item = gf_mul(*item, scalar);
        }
    }
}

/// Helper function for NEON table-based multiplication - processes 32 bytes
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn neon_mul_128(
    value_lo: uint8x16_t,
    value_hi: uint8x16_t,
    clr_mask: &uint8x16_t,
    t0_lo: uint8x16_t,
    t1_lo: uint8x16_t,
    t2_lo: uint8x16_t,
    t3_lo: uint8x16_t,
    t0_hi: uint8x16_t,
    t1_hi: uint8x16_t,
    t2_hi: uint8x16_t,
    t3_hi: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    // Nibble-based multiplication following Leopard-RS pattern
    let data_0 = vandq_u8(value_lo, *clr_mask);
    let mut prod_lo = vqtbl1q_u8(t0_lo, data_0);
    let mut prod_hi = vqtbl1q_u8(t0_hi, data_0);

    let data_1 = vshrq_n_u8(value_lo, 4);
    prod_lo = veorq_u8(prod_lo, vqtbl1q_u8(t1_lo, data_1));
    prod_hi = veorq_u8(prod_hi, vqtbl1q_u8(t1_hi, data_1));

    let data_2 = vandq_u8(value_hi, *clr_mask);
    prod_lo = veorq_u8(prod_lo, vqtbl1q_u8(t2_lo, data_2));
    prod_hi = veorq_u8(prod_hi, vqtbl1q_u8(t2_hi, data_2));

    let data_3 = vshrq_n_u8(value_hi, 4);
    prod_lo = veorq_u8(prod_lo, vqtbl1q_u8(t3_lo, data_3));
    prod_hi = veorq_u8(prod_hi, vqtbl1q_u8(t3_hi, data_3));

    (prod_lo, prod_hi)
}

// ======================================================================
// PMULL-based Multiplication (ARM64 Polynomial Multiplication)
// ======================================================================

/// PMULL-based GF(2^16) multiplication using ARM polynomial multiply instructions.
///
/// This implementation uses the ARM64 `pmull` instruction for carryless multiplication
/// followed by Barrett reduction for polynomial 0x1100B.
///
/// Uses ARM64 polynomial multiplication with Barrett reduction for GF(2^16) operations.
///
/// **Performance**: ~2-3x faster than table-based multiplication on ARM64.
///
/// **Requirements**: ARM64 with NEON support (standard on all AArch64 CPUs).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_low_asm(a: poly8x16_t, b: poly8x16_t) -> poly16x8_t {
    // Multiply low 8 bytes: pmull v0.8h, v1.8b, v2.8b
    // This performs 8-bit×8-bit→16-bit polynomial multiplication on the low half
    let result: poly16x8_t;
    std::arch::asm!(
        "pmull {0:v}.8h, {1:v}.8b, {2:v}.8b",
        out(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    result
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull2_high_asm(a: poly8x16_t, b: poly8x16_t) -> poly16x8_t {
    // Multiply high 8 bytes: pmull2 v0.8h, v1.16b, v2.16b
    // This performs 8-bit×8-bit→16-bit polynomial multiplication on the high half
    let result: poly16x8_t;
    std::arch::asm!(
        "pmull2 {0:v}.8h, {1:v}.16b, {2:v}.16b",
        out(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    result
}

/// Karatsuba multiplication for GF(2^16) using 8-bit polynomial multiplication.
///
/// Given two GF(2^16) elements represented as bytes (low, high):
/// - a = a_lo + a_hi * x^8
/// - b = b_lo + b_hi * x^8
///
/// Computes three products:
/// 1. low = a_lo * b_lo
/// 2. high = a_hi * b_hi
/// 3. mid = (a_lo ⊕ a_hi) * (b_lo ⊕ b_hi)
///
/// These are later combined in Barrett reduction to produce the final GF(2^16) result.
///
/// Processes 16 u16 values (32 bytes) per call.
///
/// The coefficient parameters should be:
/// - b_lo: scalar low byte broadcast to all lanes
/// - b_hi: scalar high byte broadcast to all lanes
/// - b_mid: (scalar_lo XOR scalar_hi) broadcast to all lanes
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn karatsuba_mul_p8(
    a_lo: poly8x16_t,
    a_hi: poly8x16_t,
    b_lo: poly8x16_t,
    b_hi: poly8x16_t,
    b_mid: poly8x16_t,
) -> (
    poly16x8_t,
    poly16x8_t,
    poly16x8_t,
    poly16x8_t,
    poly16x8_t,
    poly16x8_t,
) {
    // Compute low product: a_lo * b_lo (both halves of 16 bytes)
    let low1 = pmull_low_asm(a_lo, b_lo); // Low 8 bytes
    let low2 = pmull2_high_asm(a_lo, b_lo); // High 8 bytes

    // Compute high product: a_hi * b_hi (both halves)
    let high1 = pmull_low_asm(a_hi, b_hi);
    let high2 = pmull2_high_asm(a_hi, b_hi);

    // Compute middle product: (a_lo ⊕ a_hi) * b_mid
    // This matches reference: mid = veorq_p8(data.val[0], data.val[1]); pmull(mid, coeff[2])
    let mid_a = vreinterpretq_p8_u8(veorq_u8(
        vreinterpretq_u8_p8(a_lo),
        vreinterpretq_u8_p8(a_hi),
    ));
    let mid1 = pmull_low_asm(mid_a, b_mid);
    let mid2 = pmull2_high_asm(mid_a, b_mid);

    (low1, low2, mid1, mid2, high1, high2)
}

/// Barrett reduction for polynomial 0x1100B (PAR2's GF(2^16) primitive polynomial).
///
/// Takes the results of Karatsuba multiplication (low, mid, high products) and
/// reduces them modulo the primitive polynomial to produce final GF(2^16) results.
///
/// Barrett reduction modulo primitive polynomial 0x1100B for PAR2 GF(2^16) operations.
///
/// The reduction is hardcoded for polynomial 0x1100B = x^16 + x^12 + x^3 + x + 1.
/// Reduction coefficients:
/// - First reduction: 0x11110 (derived from polynomial structure)
/// - Second reduction: 0x1a (continuation of reduction)
/// - Final reduction: 0x100b (lower bits of polynomial)
///
/// Returns (low1, low2, high1, high2) which represent the reduced 16-bit results.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn barrett_reduce_0x1100b(
    low1: poly16x8_t,
    low2: poly16x8_t,
    mid1: poly16x8_t,
    mid2: poly16x8_t,
    high1: poly16x8_t,
    high2: poly16x8_t,
) -> (poly16x8_t, poly16x8_t, poly16x8_t, poly16x8_t) {
    // Step 1: Deinterleave results into byte streams
    // vuzpq_u8 separates even bytes (index 0,2,4...) from odd bytes (1,3,5...)
    let hibytes_raw = vuzpq_u8(vreinterpretq_u8_p16(high1), vreinterpretq_u8_p16(high2));
    let mut hibytes_0 = hibytes_raw.0; // Even bytes from high
    let hibytes_1 = hibytes_raw.1; // Odd bytes from high

    let lobytes_raw = vuzpq_u8(vreinterpretq_u8_p16(low1), vreinterpretq_u8_p16(low2));
    let lobytes_0 = lobytes_raw.0; // Even bytes from low
    let mut lobytes_1 = lobytes_raw.1; // Odd bytes from low

    let midbytes_raw = vuzpq_u8(vreinterpretq_u8_p16(mid1), vreinterpretq_u8_p16(mid2));
    let midbytes_0 = midbytes_raw.0;
    let midbytes_1 = midbytes_raw.1;

    // Step 2: Merge middle terms
    // The Karatsuba middle term needs to be XOR'd with low and high
    let libytes = veorq_u8(hibytes_0, lobytes_1);
    lobytes_1 = veorq_u8(veorq_u8(libytes, lobytes_0), midbytes_0);
    hibytes_0 = veorq_u8(veorq_u8(libytes, hibytes_1), midbytes_1);

    // Step 3: Multiply high bytes by reduction coefficient 0x11110
    // This uses bit shifts and XOR to implement the multiplication

    // Combine high bytes with shifts: (hibytes_1 << 4) | (hibytes_0 >> 4)
    let th0 = vsriq_n_u8::<4>(vshlq_n_u8::<4>(hibytes_1), hibytes_0);

    // th1 = hibytes_1 ⊕ (hibytes_1 >> 4)
    let mut th1 = veorq_u8(hibytes_1, vshrq_n_u8::<4>(hibytes_1));

    // th0 = th0 ⊕ th1 ⊕ hibytes_0
    let mut th0 = veorq_u8(veorq_u8(th0, th1), hibytes_0);

    // Step 4: Multiply by 0x1a (shift right by 5)
    th0 = veorq_u8(th0, vshrq_n_u8::<5>(hibytes_1));

    // Step 5: Extract upper 3 bits of th0 for final reduction
    let th0_hi3 = vshrq_n_u8::<5>(th0);

    // Compute th0_hi1 = th0_hi3 >> 2 (needed for final XOR in high1 output)
    let th0_hi1 = vshrq_n_u8::<2>(th0_hi3);

    // Prepare th1 for polynomial multiplication by 0x100b
    // Note: th1 will be XOR'd with th0_hi1 in the final output assembly

    // Step 6: Final reduction with polynomial 0x100b
    // Use vmulq_p8 for 8×8→8 polynomial multiplication (upper bits are discarded)
    let red_l = vdupq_n_p8(0x0b); // Lower byte of polynomial

    // Compute new hibytes_1 value: vsliq_n_u8 shifts left th0 by 4 and inserts into th0_hi3
    let hibytes_1 = vsliq_n_u8::<4>(th0_hi3, th0);

    // Multiply th1 by 0x0b
    th1 = vreinterpretq_u8_p8(vmulq_p8(vreinterpretq_p8_u8(th1), red_l));

    // Multiply th0 by 0x0b
    hibytes_0 = vreinterpretq_u8_p8(vmulq_p8(vreinterpretq_p8_u8(th0), red_l));

    // Step 7: Assemble final result
    // The four outputs are arranged as: low1, low2, high1, high2
    let out_low1 = vreinterpretq_p16_u8(lobytes_0); // Low bytes from low product
    let out_low2 = vreinterpretq_p16_u8(hibytes_0); // Reduced high bytes (part 1)

    // For high1, XOR three values: hibytes_1 ^ th0_hi1 ^ th1
    let out_high1 = vreinterpretq_p16_u8(veorq_u8(veorq_u8(hibytes_1, th0_hi1), th1));

    let out_high2 = vreinterpretq_p16_u8(lobytes_1); // High bytes from low product

    (out_low1, out_low2, out_high1, out_high2)
}

/// PMULL-based scalar multiplication: data[i] *= scalar for all i
///
/// Uses ARM64 polynomial multiplication instructions (pmull/pmull2) for 4.6-5.8x speedup
/// over scalar operations. Processes 16 u16 values (32 bytes) per iteration.
///
/// **Algorithm**:
/// 1. Deinterleave data into low/high bytes using `vld2q_u8`
/// 2. Split scalar into low/high bytes
/// 3. Karatsuba multiplication (3 polynomial products using pmull)
/// 4. Barrett reduction for polynomial 0x1100B
/// 5. Reinterleave results using `vst2q_u8`
/// # Safety
/// Caller must ensure CPU supports required NEON/PMULL instructions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gf_mul_slice_pmull_neon(scalar: u16, data: &mut [u16]) {
    // Handle zero scalar
    if scalar == 0 {
        for val in data.iter_mut() {
            *val = 0;
        }
        return;
    }

    // Split scalar into low/high bytes for Karatsuba multiplication
    let scalar_lo = (scalar & 0xFF) as u8;
    let scalar_hi = (scalar >> 8) as u8;
    let scalar_mid = scalar_lo ^ scalar_hi;

    // Broadcast scalar bytes to NEON registers (matching reference coeff[0], coeff[1], coeff[2])
    let scalar_lo_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_lo));
    let scalar_hi_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_hi));
    let scalar_mid_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_mid));

    // Process 16 u16 values (32 bytes) at a time
    let chunks = data.len() / 16;
    let remainder = data.len() % 16;

    let ptr = data.as_mut_ptr() as *mut u8;

    for i in 0..chunks {
        let offset = i * 32;
        let data_ptr = ptr.add(offset);

        // Load with automatic deinterleaving (stride-2 access)
        // vld2q_u8 loads bytes at even positions into val.0 and odd positions into val.1
        let data_deint = vld2q_u8(data_ptr);
        let data_lo = vreinterpretq_p8_u8(data_deint.0); // Low bytes of each u16
        let data_hi = vreinterpretq_p8_u8(data_deint.1); // High bytes of each u16

        // Karatsuba multiplication with pre-computed XOR coefficient
        let (low1, low2, mid1, mid2, high1, high2) = karatsuba_mul_p8(
            data_lo,
            data_hi,
            scalar_lo_vec,
            scalar_hi_vec,
            scalar_mid_vec,
        );

        // Barrett reduction
        let (out_low1, out_low2, out_high1, out_high2) =
            barrett_reduce_0x1100b(low1, low2, mid1, mid2, high1, high2);

        // Combine outputs: XOR the pairs and store with interleaving
        // out.val[0] = low1 XOR low2, out.val[1] = high1 XOR high2
        let out_val0 = veorq_u8(
            vreinterpretq_u8_p16(out_low1),
            vreinterpretq_u8_p16(out_low2),
        );
        let out_val1 = veorq_u8(
            vreinterpretq_u8_p16(out_high1),
            vreinterpretq_u8_p16(out_high2),
        );

        // Use vst2q_u8 to store with automatic interleaving
        let out = uint8x16x2_t(out_val0, out_val1);
        vst2q_u8(data_ptr, out);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 16;
        for item in data.iter_mut().skip(start) {
            *item = gf_mul(*item, scalar);
        }
    }
}

/// PMULL-based fused multiply-add: dst[i] ^= scalar * src[i] for all i
///
/// Combines PMULL multiplication with XOR to avoid intermediate allocations.
/// Processes 16 u16 values (32 bytes) per iteration with 4.6-5.8x speedup.
/// # Safety
/// Caller must ensure CPU supports required NEON/PMULL instructions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gf_muladd_pmull_neon(dst: &mut [u16], src: &[u16], scalar: u16) {
    if scalar == 0 {
        return;
    }

    // Split scalar into low/high bytes
    let scalar_lo = (scalar & 0xFF) as u8;
    let scalar_hi = (scalar >> 8) as u8;
    let scalar_mid = scalar_lo ^ scalar_hi;

    let scalar_lo_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_lo));
    let scalar_hi_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_hi));
    let scalar_mid_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_mid));

    // Process 16 u16 values (32 bytes) at a time
    let chunks = dst.len().min(src.len()) / 16;
    let remainder = dst.len().min(src.len()) % 16;

    let src_ptr = src.as_ptr() as *const u8;
    let dst_ptr = dst.as_mut_ptr() as *mut u8;

    for i in 0..chunks {
        let offset = i * 32;

        // Load source data with automatic deinterleaving
        let src_deint = vld2q_u8(src_ptr.add(offset));
        let src_lo = vreinterpretq_p8_u8(src_deint.0);
        let src_hi = vreinterpretq_p8_u8(src_deint.1);

        // Karatsuba multiplication with pre-computed XOR coefficient
        let (low1, low2, mid1, mid2, high1, high2) =
            karatsuba_mul_p8(src_lo, src_hi, scalar_lo_vec, scalar_hi_vec, scalar_mid_vec);

        // Barrett reduction
        let (out_low1, out_low2, out_high1, out_high2) =
            barrett_reduce_0x1100b(low1, low2, mid1, mid2, high1, high2);

        // Combine outputs: XOR the pairs
        let prod_val0 = veorq_u8(
            vreinterpretq_u8_p16(out_low1),
            vreinterpretq_u8_p16(out_low2),
        );
        let prod_val1 = veorq_u8(
            vreinterpretq_u8_p16(out_high1),
            vreinterpretq_u8_p16(out_high2),
        );

        // Load destination data (already interleaved)
        let dst_data = vld2q_u8(dst_ptr.add(offset));

        // XOR product with destination
        let result_val0 = veorq_u8(dst_data.0, prod_val0);
        let result_val1 = veorq_u8(dst_data.1, prod_val1);

        // Store with automatic interleaving
        let result = uint8x16x2_t(result_val0, result_val1);
        vst2q_u8(dst_ptr.add(offset), result);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 16;
        let min_len = dst.len().min(src.len());
        for i in start..min_len {
            dst[i] ^= gf_mul(src[i], scalar);
        }
    }
}

/// PMULL-based 8-region parallel multiply-add: dst[i] ^= sum(sources[j][i] * coefficients[j])
///
/// Processes up to 8 sources simultaneously, accumulating all products before
/// XORing to destination. This maximizes register utilization and reduces memory
/// traffic by loading destination values only once.
///
/// Processes 16 u16 values (32 bytes) per iteration with PMULL for each of 8 regions.
/// # Safety
/// Caller must ensure CPU supports required NEON/PMULL instructions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gf_muladd_multi_pmull_neon(
    dst: &mut [u16],
    sources: &[&[u16]],
    coefficients: &[u16],
) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::debug!(
            sources = sources.len(),
            dst_len = dst.len(),
            "Using multi-source PMULL batching"
        );
    }

    assert_eq!(sources.len(), coefficients.len());

    if sources.is_empty() {
        return;
    }

    // Verify all sources have same length as destination
    for src in sources {
        assert_eq!(dst.len(), src.len());
    }

    // Process 16 u16 values (32 bytes) at a time
    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    if min_len == 0 {
        return; // No data to process
    }
    let chunks = dst.len().min(min_len) / 16;
    let remainder = dst.len().min(min_len) % 16;

    let dst_ptr = dst.as_mut_ptr() as *mut u8;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 32;

        // Initialize accumulators for the two interleaved halves (low bytes, high bytes)
        let mut acc_val0 = vdupq_n_u8(0);
        let mut acc_val1 = vdupq_n_u8(0);

        // Process all sources, accumulating products
        for i in 0..sources.len() {
            if coefficients[i] == 0 {
                continue;
            }

            let src_ptr = sources[i].as_ptr() as *const u8;

            // Load source data with automatic deinterleaving
            let src_deint = vld2q_u8(src_ptr.add(offset));
            let src_lo = vreinterpretq_p8_u8(src_deint.0);
            let src_hi = vreinterpretq_p8_u8(src_deint.1);

            // Prepare coefficient for this source (compute on-the-fly to support >8 sources)
            let scalar = coefficients[i];
            let scalar_lo = (scalar & 0xFF) as u8;
            let scalar_hi = (scalar >> 8) as u8;
            let scalar_mid = scalar_lo ^ scalar_hi;
            let scalar_lo_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_lo));
            let scalar_hi_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_hi));
            let scalar_mid_vec = vreinterpretq_p8_u8(vdupq_n_u8(scalar_mid));

            // Karatsuba multiplication
            let (low1, low2, mid1, mid2, high1, high2) =
                karatsuba_mul_p8(src_lo, src_hi, scalar_lo_vec, scalar_hi_vec, scalar_mid_vec);

            // Barrett reduction
            let (out_low1, out_low2, out_high1, out_high2) =
                barrett_reduce_0x1100b(low1, low2, mid1, mid2, high1, high2);

            // Combine outputs: XOR the pairs to get product
            let prod_val0 = veorq_u8(
                vreinterpretq_u8_p16(out_low1),
                vreinterpretq_u8_p16(out_low2),
            );
            let prod_val1 = veorq_u8(
                vreinterpretq_u8_p16(out_high1),
                vreinterpretq_u8_p16(out_high2),
            );

            // Accumulate product into accumulators
            acc_val0 = veorq_u8(acc_val0, prod_val0);
            acc_val1 = veorq_u8(acc_val1, prod_val1);
        }

        // Load destination data (already interleaved)
        let dst_data = vld2q_u8(dst_ptr.add(offset));

        // XOR accumulated products with destination
        let result_val0 = veorq_u8(dst_data.0, acc_val0);
        let result_val1 = veorq_u8(dst_data.1, acc_val1);

        // Store with automatic interleaving
        let result = uint8x16x2_t(result_val0, result_val1);
        vst2q_u8(dst_ptr.add(offset), result);
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 16;
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

/// PMULL-based column multiply-add: one source contributes to multiple destinations
///
/// This is the transpose operation of `gf_muladd_multi_pmull_neon`:
/// - gf_muladd_multi: dst[i] ^= sum(coeffs[j] * sources[j][i])  // many sources → one dest
/// - gf_muladd_column: dests[j][i] ^= source[i] * coeffs[j]     // one source → many dests
///
/// Processes up to 8 destinations simultaneously using PMULL for optimal register utilization.
/// # Safety
/// Caller must ensure CPU supports required NEON/PMULL instructions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gf_muladd_column_neon(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
) {
    static mut CALL_COUNT: usize = 0;
    CALL_COUNT += 1;
    if CALL_COUNT == 1 {
        tracing::debug!(
            destinations = destinations.len(),
            src_len = source.len(),
            "Using parallel column PMULL"
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

    // Pre-compute scalar vectors for all coefficients
    let mut scalar_lo_vecs: [poly8x16_t; 8] = [vdupq_n_p8(0); 8];
    let mut scalar_hi_vecs: [poly8x16_t; 8] = [vdupq_n_p8(0); 8];
    let mut scalar_mid_vecs: [poly8x16_t; 8] = [vdupq_n_p8(0); 8];

    for i in 0..destinations.len() {
        let scalar = coefficients[i];
        let scalar_lo = (scalar & 0xFF) as u8;
        let scalar_hi = (scalar >> 8) as u8;
        let scalar_mid = scalar_lo ^ scalar_hi;

        scalar_lo_vecs[i] = vreinterpretq_p8_u8(vdupq_n_u8(scalar_lo));
        scalar_hi_vecs[i] = vreinterpretq_p8_u8(vdupq_n_u8(scalar_hi));
        scalar_mid_vecs[i] = vreinterpretq_p8_u8(vdupq_n_u8(scalar_mid));
    }

    // Process 16 u16 values (32 bytes) at a time
    let chunks = source.len() / 16;
    let remainder = source.len() % 16;

    let src_ptr = source.as_ptr() as *const u8;

    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 32;

        // Load source data once with automatic deinterleaving
        let src_deint = vld2q_u8(src_ptr.add(offset));
        let src_lo = vreinterpretq_p8_u8(src_deint.0);
        let src_hi = vreinterpretq_p8_u8(src_deint.1);

        // For each destination, compute source[i] * coeff[j] and accumulate
        for j in 0..destinations.len() {
            if coefficients[j] == 0 {
                continue;
            }

            let dst_ptr = destinations[j].as_mut_ptr() as *mut u8;

            // Karatsuba multiplication: source * coefficient[j]
            let (low1, low2, mid1, mid2, high1, high2) = karatsuba_mul_p8(
                src_lo,
                src_hi,
                scalar_lo_vecs[j],
                scalar_hi_vecs[j],
                scalar_mid_vecs[j],
            );

            // Barrett reduction
            let (out_low1, out_low2, out_high1, out_high2) =
                barrett_reduce_0x1100b(low1, low2, mid1, mid2, high1, high2);

            // Combine outputs: XOR the pairs to get product
            let prod_val0 = veorq_u8(
                vreinterpretq_u8_p16(out_low1),
                vreinterpretq_u8_p16(out_low2),
            );
            let prod_val1 = veorq_u8(
                vreinterpretq_u8_p16(out_high1),
                vreinterpretq_u8_p16(out_high2),
            );

            // Load destination data (already interleaved)
            let dst_data = vld2q_u8(dst_ptr.add(offset));

            // XOR product with destination
            let result_val0 = veorq_u8(dst_data.0, prod_val0);
            let result_val1 = veorq_u8(dst_data.1, prod_val1);

            // Store with automatic interleaving
            let result = uint8x16x2_t(result_val0, result_val1);
            vst2q_u8(dst_ptr.add(offset), result);
        }
    }

    // Handle remainder with scalar code
    if remainder > 0 {
        let start = chunks * 16;
        #[allow(clippy::needless_range_loop)]
        for i in start..source.len() {
            let src_val = source[i];
            for j in 0..destinations.len() {
                if coefficients[j] != 0 {
                    destinations[j][i] ^= gf_mul(src_val, coefficients[j]);
                }
            }
        }
    }
}

/// Convert bytes to u16 array with SIMD optimization
///
/// Converts packed byte array to u16 array, interpreting each pair of bytes as
/// little-endian u16. This is a hot path in PAR2 reconstruction as every block
/// chunk must be converted from bytes to GF(2^16) elements.
///
/// # Performance
///
/// - NEON (AArch64): 8 u16 values per iteration (~4-8x faster than scalar)
/// - Scalar fallback: Safe on all platforms
///
/// # Arguments
///
/// * `bytes` - Input byte array (must have even length)
/// * `output` - Output u16 array (length must be bytes.len() / 2)
#[inline]
/// # Safety
/// Caller must ensure CPU supports required NEON/PMULL instructions.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bytes_to_u16_neon(bytes: &[u8], output: &mut [u16]) {
    let len = output.len();
    let chunks = len / 8; // Process 8 u16 at a time (16 bytes)

    let src_ptr = bytes.as_ptr();
    let dst_ptr = output.as_mut_ptr();

    // Process 8 u16 values (16 bytes) per iteration
    for i in 0..chunks {
        let offset = i * 8;
        let byte_offset = i * 16;

        // Load 16 bytes
        let bytes_vec = vld1q_u8(src_ptr.add(byte_offset));

        // Reinterpret as u16 (works on little-endian)
        let u16_vec = vreinterpretq_u16_u8(bytes_vec);

        // Store 8 u16 values
        vst1q_u16(dst_ptr.add(offset), u16_vec);
    }

    // Handle remainder with scalar code
    let remainder_start = chunks * 8;
    if remainder_start < len {
        bytes_to_u16_scalar(
            &bytes[remainder_start * 2..],
            &mut output[remainder_start..],
        );
    }
}

/// Scalar fallback for byte to u16 conversion
#[inline]
fn bytes_to_u16_scalar(bytes: &[u8], output: &mut [u16]) {
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        output[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
    }
}
