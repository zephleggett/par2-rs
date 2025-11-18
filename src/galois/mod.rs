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
//! SIMD-optimized batch operations are available on x86-64 (AVX2, SSSE3) and AArch64 (NEON)
//! with runtime feature detection and automatic fallback to scalar code.
//!
//! The SIMD multiplication algorithm is adapted from the reed-solomon-simd crate
//! by Anders Trier Olesen (<https://github.com/AndersTrier/reed-solomon-simd>),
//! which implements the Leopard-RS algorithm. We've adapted it for PAR2's polynomial 0x1100B.
//!
//! ## Testing Status
//!
//! - **NEON (AArch64)**: Fully tested and verified on Apple Silicon (8-10x speedup)
//! - **AVX2 (x86-64)**: Implemented following reference algorithm but untested
//! - **SSSE3 (x86-64)**: Implemented following reference algorithm but untested
//!
//! **Note**: x86-64 SIMD implementations use table-based multiplication adapted from
//! reed-solomon-simd. They follow the reference implementation exactly but have not
//! been validated on actual x86-64 hardware. Scalar fallback ensures correctness on
//! all platforms. If you encounter issues on x86-64, please report them.
//!
//! # Safety
//!
//! This module uses `unsafe` code for SIMD operations. All unsafe code is protected by:
//!
//! 1. **Runtime CPU feature detection**: Using `is_x86_feature_detected!()` and
//!    `is_aarch64_feature_detected!()` macros to verify CPU support before calling
//!    SIMD functions.
//! 2. **Target feature attributes**: All SIMD functions are marked with `#[target_feature]`
//!    to ensure they're only compiled when the feature is available.
//! 3. **Memory safety**: All pointer operations respect slice bounds and alignment
//!    requirements. SIMD loads/stores use unaligned operations (`loadu`/`storeu`)
//!    to handle arbitrary data alignment.
//! 4. **Automatic fallback**: If SIMD features are not available, the code automatically
//!    falls back to safe scalar implementations.
//!
//! # Module Organization
//!
//! - `core`: Core GF(2^16) operations, tables, and scalar fallbacks
//! - `x86`: x86-64 SIMD implementations (PCLMULQDQ, AVX2, SSSE3)
//! - `arm`: ARM NEON implementations (PMULL, table-based)
//! - `tests`: Test suite

// Submodules
pub(crate) mod core;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod arm;

#[cfg(test)]
mod tests;

// Re-export public API from core
pub use core::{gf_div, gf_mul, gf_pow, init_tables};

// Import private items needed for dispatch
use core::{
    bytes_to_u16_scalar, gf_mul_slice_scalar, gf_muladd_column_scalar, gf_muladd_multi_scalar,
    gf_muladd_scalar,
};

#[cfg(target_arch = "x86_64")]
use x86::{
    gf_mul_slice_avx2_pclmul_x86, gf_mul_slice_pclmul_x86, gf_muladd_avx2_pclmul_x86,
    gf_muladd_column_avx2_pclmul_x86, gf_muladd_column_pclmul_x86, gf_muladd_multi_avx2_pclmul_x86,
    gf_muladd_multi_pclmul_x86, gf_muladd_pclmul_x86,
};

#[cfg(all(target_arch = "x86_64", feature = "unstable"))]
use x86::{
    gf_mul_slice_vpclmul_gfni_x86, gf_mul_slice_vpclmul_x86, gf_muladd_column_vpclmul_x86,
    gf_muladd_multi_vpclmul_x86, gf_muladd_vpclmul_x86,
};

#[cfg(target_arch = "aarch64")]
use arm::{
    bytes_to_u16_neon, gf_mul_slice_neon, gf_mul_slice_pmull_neon, gf_muladd_column_neon,
    gf_muladd_multi_pmull_neon, gf_muladd_pmull_neon,
};

// ======================================================================
// SIMD Batch Operations - Public Dispatch Functions
// ======================================================================

/// Multiply a slice of u16 values by a scalar in GF(2^16) using SIMD when available
///
/// This function automatically selects the best implementation based on runtime CPU feature detection:
///
/// ## ARM64 (aarch64)
/// - **PMULL (preferred)**: Native polynomial multiplication with Barrett reduction
///   - Uses `pmull`/`pmull2` instructions with Karatsuba algorithm
///   - Processes 16 u16 values (32 bytes) per iteration
///   - Performance: 4.6-5.8x speedup over scalar
///
/// ## x86-64
/// - **AVX2** (if available): Table-based multiplication, 32 u16 values per iteration
/// - **SSSE3** (fallback): Table-based multiplication, 16 u16 values per iteration
/// - Performance: 8-10x speedup over scalar
///
/// ## Scalar fallback
/// - Used on unsupported architectures or when SIMD is unavailable
/// - Simple loop-based multiplication using lookup tables
#[inline]
pub fn gf_mul_slice(scalar: u16, data: &mut [u16]) {
    init_tables(); // Ensure tables are initialized

    #[cfg(target_arch = "x86_64")]
    {
        // Prefer widest/fastest PCLMUL variant available
        // Priority: VPCLMUL+GFNI > VPCLMUL > AVX2 > SSE
        #[cfg(feature = "unstable")]
        {
            if is_x86_feature_detected!("vpclmulqdq")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("gfni")
            {
                unsafe { gf_mul_slice_vpclmul_gfni_x86(scalar, data) };
                return;
            }
            if is_x86_feature_detected!("vpclmulqdq") && is_x86_feature_detected!("avx512f") {
                unsafe { gf_mul_slice_vpclmul_x86(scalar, data) };
                return;
            }
        }
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("sse4.1")
        {
            unsafe { gf_mul_slice_avx2_pclmul_x86(scalar, data) };
            return;
        }
        if is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1") {
            unsafe { gf_mul_slice_pclmul_x86(scalar, data) };
            return;
        }
        // TODO: SSSE3 table-based fallback for 2006-2010 CPUs without PCLMUL
        // if is_x86_feature_detected!("ssse3") {
        //     unsafe { gf_mul_slice_ssse3(scalar, data) };
        //     return;
        // }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Prefer PMULL (requires crypto extensions) over table lookup for better performance
        if std::arch::is_aarch64_feature_detected!("aes") {
            unsafe { gf_mul_slice_pmull_neon(scalar, data) };
            return;
        }
        // Fallback to table-lookup NEON for processors without crypto extensions
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { gf_mul_slice_neon(scalar, data) };
            return;
        }
    }

    // Scalar fallback
    gf_mul_slice_scalar(scalar, data);
}

// ======================================================================
// Multiply-Add Operations (Fused)
// ======================================================================

/// Fused multiply-add: `dst[i] ^= scalar * src[i]` for all i
///
/// This is the core operation for Reed-Solomon encoding/decoding.
/// Instead of separate mul+add passes, this performs both in a single
/// pass over the data, significantly reducing memory traffic.
///
/// Equivalent to:
/// ```ignore
/// for i in 0..dst.len() {
///     dst[i] ^= gf_mul(src[i], scalar);
/// }
/// ```
///
/// ## Current Implementations
/// - **ARM64**: PMULL-based (4.6-5.8x speedup)
/// - **x86-64**: Scalar fallback (TODO: add AVX2/SSSE3)
#[inline]
pub fn gf_muladd(dst: &mut [u16], src: &[u16], scalar: u16) {
    init_tables(); // Ensure tables are initialized

    assert_eq!(dst.len(), src.len(), "dst and src must have same length");

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { gf_muladd_pmull_neon(dst, src, scalar) };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "unstable")]
        {
            if is_x86_feature_detected!("vpclmulqdq")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
            {
                unsafe { gf_muladd_vpclmul_x86(dst, src, scalar) };
                return;
            }
        }
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("sse4.1")
        {
            unsafe { gf_muladd_avx2_pclmul_x86(dst, src, scalar) };
            return;
        }
        if is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1") {
            unsafe { gf_muladd_pclmul_x86(dst, src, scalar) };
            return;
        }
    }

    // Scalar fallback
    gf_muladd_scalar(dst, src, scalar);
}

/// Multi-region multiply-add: `dst[i] ^= sum(coefficients[j] * sources[j][i])`
///
/// Key optimization: instead of processing each source separately, we process
/// multiple sources together to maximize register usage and reduce memory traffic.
///
/// ## Current Implementations
/// - **ARM64**: PMULL-based, processes up to 8 regions simultaneously
/// - **x86-64**: PCLMULQDQ-based, processes up to 8 regions simultaneously
#[inline]
pub fn gf_muladd_multi(dst: &mut [u16], sources: &[&[u16]], coefficients: &[u16]) {
    init_tables(); // Ensure tables are initialized

    assert_eq!(
        sources.len(),
        coefficients.len(),
        "sources and coefficients must match"
    );
    if sources.is_empty() {
        return;
    }

    // Verify all sources have same length as destination
    for src in sources {
        assert_eq!(
            dst.len(),
            src.len(),
            "all sources must have same length as dst"
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && sources.len() <= 8 {
            unsafe { gf_muladd_multi_pmull_neon(dst, sources, coefficients) };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "unstable")]
        {
            if is_x86_feature_detected!("vpclmulqdq")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && sources.len() <= 8
            {
                unsafe { gf_muladd_multi_vpclmul_x86(dst, sources, coefficients) };
                return;
            }
        }
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("sse4.1")
            && sources.len() <= 8
        {
            unsafe { gf_muladd_multi_avx2_pclmul_x86(dst, sources, coefficients) };
            return;
        }
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("sse4.1")
            && sources.len() <= 8
        {
            unsafe { gf_muladd_multi_pclmul_x86(dst, sources, coefficients) };
            return;
        }
    }

    // Scalar fallback
    gf_muladd_multi_scalar(dst, sources, coefficients);
}

/// Column-wise multiply-add: `destinations[j][i] ^= source[i] * coefficients[j]` for all j
///
/// This is the inverse of gf_muladd_multi - one source contributes to multiple destinations.
/// Optimized for column-wise matrix operations in PAR2 reconstruction.
///
/// ## Current Implementations
/// - **ARM64**: PMULL-based parallel processing (up to 8 destinations)
/// - **x86-64**: PCLMULQDQ-based parallel processing (up to 8 destinations)
#[inline]
pub fn gf_muladd_column(destinations: &mut [&mut [u16]], source: &[u16], coefficients: &[u16]) {
    init_tables(); // Ensure tables are initialized

    assert_eq!(
        destinations.len(),
        coefficients.len(),
        "destinations and coefficients must match"
    );
    if destinations.is_empty() {
        return;
    }

    // Verify all destinations have same length as source
    for dst in destinations.iter() {
        assert_eq!(
            dst.len(),
            source.len(),
            "all destinations must have same length as source"
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && destinations.len() <= 8 {
            unsafe { gf_muladd_column_neon(destinations, source, coefficients) };
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "unstable")]
        {
            if is_x86_feature_detected!("vpclmulqdq")
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512vl")
                && destinations.len() <= 8
            {
                unsafe { gf_muladd_column_vpclmul_x86(destinations, source, coefficients) };
                return;
            }
        }
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("sse4.1")
            && destinations.len() <= 8
        {
            unsafe { gf_muladd_column_avx2_pclmul_x86(destinations, source, coefficients) };
            return;
        }
        if is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("sse4.1")
            && destinations.len() <= 8
        {
            unsafe { gf_muladd_column_pclmul_x86(destinations, source, coefficients) };
            return;
        }
    }

    // Scalar fallback for unsupported platforms or feature sets
    gf_muladd_column_scalar(destinations, source, coefficients);
}

/// Convert byte slice to u16 slice using SIMD when available
///
/// Efficiently converts little-endian byte pairs to u16 values.
/// Uses NEON on ARM64 for ~2-3x speedup over scalar conversion.
pub fn bytes_to_u16_simd(bytes: &[u8], output: &mut [u16]) {
    debug_assert_eq!(
        bytes.len(),
        output.len() * 2,
        "bytes.len() must be 2x output.len()"
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                bytes_to_u16_neon(bytes, output);
            }
            return;
        }
    }

    // Scalar fallback for all platforms
    bytes_to_u16_scalar(bytes, output);
}
