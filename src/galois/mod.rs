//! GF(2^16) Galois Field arithmetic for PAR2
//!
//! This module implements finite field arithmetic over GF(2^16) using the primitive
//! polynomial 0x1100B (x^16 + x^12 + x^3 + x + 1) as specified in the PAR2 standard.
//!
//! The implementation uses precomputed logarithm and exponential tables for efficient
//! multiplication and division operations. Tables are initialized once on first use
//! and are thread-safe.
//!
//! # SIMD Optimization
//!
//! SIMD-optimized batch operations are available on x86-64 (AVX2, SSSE3, PCLMUL)
//! and AArch64 (NEON, PMULL) with runtime feature detection and automatic fallback to scalar code.
//!
//! ## Platform Support
//!
//! - **ARM (AArch64)**: NEON, PMULL with crypto extensions
//! - **x86-64**: SSE2, SSSE3, AVX2, PCLMUL
//! - **Scalar**: Portable fallback for all architectures
//!
//! All implementations use runtime feature detection with automatic fallback.
//!
//! # Module Organization
//!
//! - `core`: Core GF(2^16) operations, tables, and constants
//! - `scalar`: Scalar (non-SIMD) implementations
//! - `simd`: Platform-specific SIMD implementations with unified dispatch
//! - `region`: Region-based processing for cache-efficient Reed-Solomon
//! - `tests`: Test suite

// Submodules
pub(crate) mod core;
pub(crate) mod scalar;
pub mod simd;

// ARM SIMD implementations (restored)
#[cfg(target_arch = "aarch64")]
pub mod arm;

// Region-based processing
pub mod region;

// Streaming reconstruction with cache optimization
pub mod streaming;

#[cfg(test)]
mod tests;

// Re-export public API from core
pub use core::{debug_assert_tables_initialized, gf_div, gf_mul, gf_pow, init_tables};

use simd::get_registry;

// ======================================================================
// SIMD Batch Operations - Public Dispatch Functions
// ======================================================================

/// Multiply a slice of u16 values by a scalar in GF(2^16) using SIMD when available
///
/// This function automatically selects the best implementation based on runtime CPU feature detection.
/// The SIMD registry automatically selects the optimal strategy based on CPU features.
#[inline]
pub fn gf_mul_slice(scalar: u16, data: &mut [u16]) {
    init_tables(); // Ensure tables are initialized

    // Use platform-specific optimized implementations
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // Use PMULL-based implementation on ARM (fastest)
        arm::gf_mul_slice_pmull_neon(scalar, data);
    }

    // Use SIMD registry for other platforms
    #[cfg(not(target_arch = "aarch64"))]
    {
        let registry = get_registry();
        if let Some(strategy) = registry.get_selected() {
            unsafe {
                strategy.mul_slice(scalar, data);
            }
        } else {
            scalar::gf_mul_slice_scalar(scalar, data);
        }
    }
}

/// Multiply-accumulate in GF(2^16) using SIMD when available
///
/// Performs: dst[i] ^= src[i] * scalar for all i
///
/// Automatically selects the best implementation based on runtime CPU features.
#[inline]
pub fn gf_muladd(dst: &mut [u16], src: &[u16], scalar: u16) {
    init_tables(); // Ensure tables are initialized

    #[cfg(target_arch = "aarch64")]
    unsafe {
        arm::gf_muladd_pmull_neon(dst, src, scalar);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let registry = get_registry();
        if let Some(strategy) = registry.get_selected() {
            unsafe {
                strategy.muladd(dst, src, scalar);
            }
        } else {
            scalar::gf_muladd_scalar(dst, src, scalar);
        }
    }
}

/// Column-wise multiply-accumulate
///
/// For each destination row i: dst[i] ^= source * coefficients[i]
#[inline]
pub fn gf_muladd_column(destinations: &mut [&mut [u16]], source: &[u16], coefficients: &[u16]) {
    init_tables();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        arm::gf_muladd_column_neon(destinations, source, coefficients);
    }

    // On x86, use SIMD muladd for each destination row
    #[cfg(not(target_arch = "aarch64"))]
    {
        let registry = get_registry();
        if let Some(strategy) = registry.get_selected() {
            for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
                if coeff != 0 {
                    unsafe {
                        strategy.muladd(dst, source, coeff);
                    }
                }
            }
        } else {
            scalar::gf_muladd_column_scalar(destinations, source, coefficients);
        }
    }
}

/// Multi-source multiply-accumulate
///
/// Accumulates contributions from multiple sources: dst ^= Σ(src[i] * coeff[i])
#[inline]
pub fn gf_muladd_multi(dst: &mut [u16], sources: &[&[u16]], coefficients: &[u16]) {
    init_tables();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        arm::gf_muladd_multi_pmull_neon(dst, sources, coefficients);
    }

    // On x86, use SIMD muladd for each source
    #[cfg(not(target_arch = "aarch64"))]
    {
        let registry = get_registry();
        if let Some(strategy) = registry.get_selected() {
            for (src, &coeff) in sources.iter().zip(coefficients.iter()) {
                if coeff != 0 {
                    unsafe {
                        strategy.muladd(dst, src, coeff);
                    }
                }
            }
        } else {
            scalar::gf_muladd_multi_scalar(dst, sources, coefficients);
        }
    }
}

/// Convert bytes to u16 values using SIMD when available
///
/// Converts little-endian byte pairs to u16 values.
#[inline]
pub fn bytes_to_u16_simd(bytes: &[u8], output: &mut [u16]) {
    debug_assert_eq!(
        bytes.len(),
        output.len() * 2,
        "bytes.len() must be 2x output.len()"
    );

    // Scalar implementation is sufficient - byte conversion is not a bottleneck
    // compared to GF multiplication operations
    scalar::bytes_to_u16_scalar(bytes, output);
}

/// Convert bytes to u16 while checking if all zeros
///
/// Returns true if every input byte was zero.
pub fn bytes_to_u16_simd_with_zero_flag(bytes: &[u8], output: &mut [u16]) -> bool {
    debug_assert_eq!(
        bytes.len(),
        output.len() * 2,
        "bytes.len() must be 2x output.len()"
    );

    // Scalar implementation with zero check - adequate performance for this operation
    let mut all_zero = true;
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        let val = u16::from_le_bytes([chunk[0], chunk[1]]);
        output[i] = val;
        if val != 0 {
            all_zero = false;
        }
    }
    all_zero
}

/// Convert u16 values to bytes using SIMD when available
#[inline]
pub fn u16_to_bytes_simd(input: &[u16], bytes: &mut [u8]) {
    debug_assert_eq!(
        bytes.len(),
        input.len() * 2,
        "bytes.len() must be 2x input.len()"
    );

    // For now, use scalar implementation
    scalar::u16_to_bytes_scalar(input, bytes);
}

// ======================================================================
// Configuration and Feature Detection
// ======================================================================

/// Initialize SIMD registry with environment configuration
pub fn configure_simd() {
    // Check for forced strategy from environment
    if let Ok(strategy) = std::env::var("PAR2_SIMD_STRATEGY") {
        simd::init_registry_with_config(Some(&strategy));
    }
}

/// Get list of available SIMD strategies
pub fn list_simd_strategies() -> Vec<(&'static str, simd::Priority)> {
    get_registry().list_available()
}

/// Get currently selected SIMD strategy name
pub fn get_selected_simd_strategy() -> Option<&'static str> {
    get_registry().get_selected().map(|s| s.name())
}

// ======================================================================
// Shuffle2x Format Support
// ======================================================================

/// Check if the current SIMD strategy supports native shuffle2x format
///
/// Shuffle2x format stores data with all low bytes in one half and all high
/// bytes in the other half. This allows more efficient table lookups on x86
/// AVX2, nearly doubling throughput compared to interleaved format.
pub fn supports_shuffle2x() -> bool {
    get_registry()
        .get_selected()
        .map(|s| s.supports_shuffle2x())
        .unwrap_or(false)
}

/// Convert interleaved u16 data to shuffle2x format in place
///
/// Shuffle2x format: within each 256-bit block, all low bytes are in the
/// first 128 bits and all high bytes are in the second 128 bits.
///
/// This should be called once before a sequence of muladd operations.
///
/// # Safety
///
/// This is safe to call on any slice, but the data will only be meaningful
/// if later processed with shuffle2x operations and then converted back.
#[inline]
pub fn prepare_shuffle2x(data: &mut [u16]) {
    if let Some(strategy) = get_registry().get_selected() {
        if strategy.supports_shuffle2x() {
            unsafe {
                strategy.prepare_shuffle2x(data);
            }
        }
    }
}

/// Convert shuffle2x format back to interleaved u16 data in place
///
/// This must be called after shuffle2x operations to restore data to
/// normal interleaved format for reading/writing.
#[inline]
pub fn finish_shuffle2x(data: &mut [u16]) {
    if let Some(strategy) = get_registry().get_selected() {
        if strategy.supports_shuffle2x() {
            unsafe {
                strategy.finish_shuffle2x(data);
            }
        }
    }
}

/// Multiply-accumulate on shuffle2x format data
///
/// Both dst and src must be in shuffle2x format. This is the high-performance
/// path that avoids per-operation format conversion overhead.
#[inline]
pub fn gf_muladd_shuffle2x(dst: &mut [u16], src: &[u16], scalar: u16) {
    init_tables();

    if let Some(strategy) = get_registry().get_selected() {
        if strategy.supports_shuffle2x() {
            unsafe {
                strategy.muladd_shuffle2x(dst, src, scalar);
            }
            return;
        }
    }
    // Fallback to regular muladd if shuffle2x not supported
    gf_muladd(dst, src, scalar);
}

/// Column-wise multiply-accumulate on shuffle2x format data
///
/// For each destination row i: dst[i] ^= source * coefficients[i]
/// Both destinations and source must be in shuffle2x format.
///
/// This is the high-performance path for creation encoding on x86 AVX2,
/// providing ~55% better throughput than the interleaved format.
#[inline]
pub fn gf_muladd_column_shuffle2x(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
) {
    init_tables();

    if let Some(strategy) = get_registry().get_selected() {
        if strategy.supports_shuffle2x() {
            for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
                if coeff != 0 {
                    unsafe {
                        strategy.muladd_shuffle2x(dst, source, coeff);
                    }
                }
            }
            return;
        }
    }
    // Fallback to regular column muladd if shuffle2x not supported
    gf_muladd_column(destinations, source, coefficients);
}
