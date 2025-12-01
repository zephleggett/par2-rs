//! Region-based Galois Field operations for Reed-Solomon reconstruction
//!
//! This module implements high-performance region-based processing that:
//! 1. Batches multiple source blocks together (region size)
//! 2. Processes them with interleaved memory layout for cache efficiency
//! 3. Uses multi-source operations to maximize throughput
//!
//! # Key Concept: Region-Based Processing
//!
//! Instead of processing one source at a time across all outputs:
//! ```text
//! for each source:
//!     for each output:
//!         output += source * coefficient
//! ```
//!
//! We process multiple sources together in "regions" that fit in cache:
//! ```text
//! for each region:
//!     for each output:
//!         accumulate all (source * coefficient) in region
//! ```
//!
//! This provides:
//! - **Better cache utilization**: Working set fits in L2 cache
//! - **Higher ILP**: CPU can parallelize operations better
//! - **Reduced memory traffic**: Fewer cache misses and memory accesses
//!
//! # Configuration
//!
//! Region size can be configured via the `PAR2_REGION_SIZE` environment variable.
//! Default is 128KB which works well for typical L2 cache sizes (256KB-1MB).

use super::{gf_mul, simd};
use std::sync::OnceLock;

/// Default region size in bytes (128KB)
/// This should fit comfortably in L2 cache (typically 256KB-1MB)
/// For x86 with 512KB L2, use 128KB to leave room for output accumulators
pub const REGION_SIZE_BYTES: usize = 128 * 1024;

/// Maximum number of sources to batch together
/// Limited by register pressure and cache capacity
pub const MAX_SOURCES_PER_BATCH: usize = 16;

/// Get the configured region size in bytes
///
/// Can be overridden via `PAR2_REGION_SIZE` environment variable.
/// Must be at least 4KB and even-sized for alignment.
pub fn region_size_bytes() -> usize {
    static REGION_SIZE: OnceLock<usize> = OnceLock::new();
    *REGION_SIZE.get_or_init(|| {
        match std::env::var("PAR2_REGION_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            Some(v) if v >= 4096 && v % 2 == 0 => v,
            _ => REGION_SIZE_BYTES,
        }
    })
}

/// Region-based multiply-accumulate for Reed-Solomon reconstruction
///
/// Processes multiple source blocks simultaneously for better cache performance.
///
/// # Algorithm
///
/// For each region of data:
/// 1. Load region from all source blocks
/// 2. For each destination:
///    - Accumulate: dest += src[0] * coeff[0] + src[1] * coeff[1] + ...
/// 3. Write accumulated results back
///
/// # Arguments
///
/// * `destinations` - Output accumulators (one per missing block to reconstruct)
/// * `sources` - Available source blocks (recovery + intact data blocks)
/// * `coefficients` - Coefficient matrix for Reed-Solomon decoding
///   - `coefficients[dst_idx][src_idx]` is the coefficient for source `src_idx`
///     when computing destination `dst_idx`
/// * `region_offset` - Starting offset in the block (in u16 elements)
/// * `region_size` - Number of u16 elements to process in this region
///
/// # Performance Notes
///
/// This function automatically uses the best available SIMD implementation
/// via the SIMD strategy registry. Strategies like AVX2 shuffle provide
/// significant performance improvements over scalar code.
pub fn gf_muladd_region(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    // Validate inputs
    debug_assert_eq!(
        destinations.len(),
        coefficients.len(),
        "destinations and coefficients must have same length"
    );
    debug_assert!(
        !destinations.is_empty(),
        "must have at least one destination"
    );
    debug_assert!(!sources.is_empty(), "must have at least one source");

    // Use SIMD strategy if available
    let registry = simd::get_registry();
    if let Some(strategy) = registry.get_selected() {
        unsafe {
            strategy.muladd_region(
                destinations,
                sources,
                coefficients,
                region_offset,
                region_size,
            );
        }
    } else {
        // Fallback to scalar if no SIMD strategy available
        gf_muladd_region_scalar(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        );
    }
}

/// Scalar implementation of region-based multiply-accumulate
///
/// This is the fallback implementation that works on all platforms.
/// While not SIMD-optimized, it still benefits from cache locality
/// due to the region-based processing pattern.
fn gf_muladd_region_scalar(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    let num_dsts = destinations.len();
    let num_srcs = sources.len();

    // Process each destination output
    for dst_idx in 0..num_dsts {
        let dst = &mut destinations[dst_idx][region_offset..region_offset + region_size];
        let coeff_row = coefficients[dst_idx];

        // Accumulate contributions from all sources
        for (src_idx, &coeff) in coeff_row.iter().enumerate().take(num_srcs) {
            if coeff == 0 {
                continue; // Skip zero coefficients
            }

            let src = &sources[src_idx][region_offset..region_offset + region_size];

            // Multiply-accumulate: dst[i] ^= src[i] * coeff
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                if s != 0 {
                    *d ^= gf_mul(s, coeff);
                }
            }
        }
    }
}

/// Helper to process an entire block using region-based approach
///
/// Automatically divides the block into regions and processes them sequentially.
///
/// # Arguments
///
/// * `destinations` - Output blocks to reconstruct
/// * `sources` - Available source blocks
/// * `coefficients` - Reed-Solomon coefficient matrix
/// * `block_size` - Total size of each block (in u16 elements)
///
/// # Example
///
/// ```ignore
/// // Reconstruct 2 missing blocks from 5 available blocks
/// let mut dest1 = vec![0u16; 1024];
/// let mut dest2 = vec![0u16; 1024];
/// let mut destinations = vec![dest1.as_mut_slice(), dest2.as_mut_slice()];
///
/// let sources: Vec<&[u16]> = /* ... 5 source blocks ... */;
/// let coefficients: Vec<&[u16]> = /* ... 2x5 coefficient matrix ... */;
///
/// gf_muladd_block_regions(&mut destinations, &sources, &coefficients, 1024);
/// ```
pub fn gf_muladd_block_regions(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    block_size: usize,
) {
    let region_bytes = region_size_bytes();
    let region_u16s = region_bytes / 2; // Convert bytes to u16 count

    let mut offset = 0;
    while offset < block_size {
        let region_size = (block_size - offset).min(region_u16s);

        gf_muladd_region(destinations, sources, coefficients, offset, region_size);

        offset += region_size;
    }
}

/// Helper to process an entire block using region-based approach with shuffle2x format
///
/// All destinations and sources must be in shuffle2x format.
/// This is the high-performance path (~55% faster) for platforms that support shuffle2x.
///
/// # Arguments
///
/// * `destinations` - Output blocks to reconstruct (in shuffle2x format)
/// * `sources` - Available source blocks (in shuffle2x format)
/// * `coefficients` - Reed-Solomon coefficient matrix
/// * `block_size` - Total size of each block (in u16 elements)
pub fn gf_muladd_block_regions_shuffle2x(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    block_size: usize,
) {
    let region_bytes = region_size_bytes();
    let region_u16s = region_bytes / 2;

    let mut offset = 0;
    while offset < block_size {
        let region_size = (block_size - offset).min(region_u16s);

        gf_muladd_region_shuffle2x(destinations, sources, coefficients, offset, region_size);

        offset += region_size;
    }
}

/// Region-based multiply-accumulate for shuffle2x format data
///
/// All destinations and sources must be in shuffle2x format.
/// This is the high-performance path for platforms that support shuffle2x.
pub fn gf_muladd_region_shuffle2x(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    debug_assert_eq!(
        destinations.len(),
        coefficients.len(),
        "destinations and coefficients must have same length"
    );
    debug_assert!(
        !destinations.is_empty(),
        "must have at least one destination"
    );
    debug_assert!(!sources.is_empty(), "must have at least one source");

    let registry = simd::get_registry();
    if let Some(strategy) = registry.get_selected() {
        if strategy.supports_shuffle2x() {
            unsafe {
                strategy.muladd_region_shuffle2x(
                    destinations,
                    sources,
                    coefficients,
                    region_offset,
                    region_size,
                );
            }
            return;
        }
    }

    // Fallback to regular region processing if shuffle2x not supported
    gf_muladd_region(
        destinations,
        sources,
        coefficients,
        region_offset,
        region_size,
    );
}

/// Column-wise region processing
///
/// Alternative to region-based processing for when you want to process
/// one source at a time but still benefit from region-based cache locality.
///
/// # Arguments
///
/// * `destinations` - Output blocks (multiple)
/// * `source` - Single source block to process
/// * `coefficients` - Coefficients for each destination (one per destination)
/// * `region_offset` - Starting offset in u16 elements
/// * `region_size` - Number of u16 elements to process
pub fn gf_muladd_column_region(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
    region_offset: usize,
    region_size: usize,
) {
    debug_assert_eq!(
        destinations.len(),
        coefficients.len(),
        "destinations and coefficients must have same length"
    );

    let source_region = &source[region_offset..region_offset + region_size];

    // Process each destination
    for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
        if coeff == 0 {
            continue;
        }

        let dst_region = &mut dst[region_offset..region_offset + region_size];

        // Multiply-accumulate for this column
        for (d, &s) in dst_region.iter_mut().zip(source_region.iter()) {
            if s != 0 {
                *d ^= gf_mul(s, coeff);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::galois::init_tables;

    #[test]
    fn test_region_basic() {
        init_tables();

        let src1 = vec![1u16, 2, 3, 4];
        let src2 = vec![5u16, 6, 7, 8];
        let sources = vec![src1.as_slice(), src2.as_slice()];

        let mut dst = vec![0u16; 4];
        let mut destinations = vec![dst.as_mut_slice()];

        // Coefficients: dst = src1 * 2 + src2 * 3
        let coeff = vec![2u16, 3u16];
        let coefficients = vec![coeff.as_slice()];

        gf_muladd_region(&mut destinations, &sources, &coefficients, 0, 4);

        // Verify results
        for i in 0..4 {
            let expected = gf_mul(src1[i], 2) ^ gf_mul(src2[i], 3);
            assert_eq!(dst[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_region_multiple_destinations() {
        init_tables();

        let src1 = vec![1u16, 2, 3, 4];
        let src2 = vec![5u16, 6, 7, 8];
        let sources = vec![src1.as_slice(), src2.as_slice()];

        let mut dst1 = vec![0u16; 4];
        let mut dst2 = vec![0u16; 4];
        let mut destinations = vec![dst1.as_mut_slice(), dst2.as_mut_slice()];

        // dst1 = src1 * 2 + src2 * 3
        // dst2 = src1 * 5 + src2 * 7
        let coeff1 = vec![2u16, 3u16];
        let coeff2 = vec![5u16, 7u16];
        let coefficients = vec![coeff1.as_slice(), coeff2.as_slice()];

        gf_muladd_region(&mut destinations, &sources, &coefficients, 0, 4);

        // Verify dst1
        for i in 0..4 {
            let expected = gf_mul(src1[i], 2) ^ gf_mul(src2[i], 3);
            assert_eq!(dst1[i], expected, "dst1 mismatch at index {}", i);
        }

        // Verify dst2
        for i in 0..4 {
            let expected = gf_mul(src1[i], 5) ^ gf_mul(src2[i], 7);
            assert_eq!(dst2[i], expected, "dst2 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_region_with_offset() {
        init_tables();

        let src = vec![1u16, 2, 3, 4, 5, 6, 7, 8];
        let sources = vec![src.as_slice()];

        let mut dst = vec![0u16; 8];
        let mut destinations = vec![dst.as_mut_slice()];

        let coeff = vec![3u16];
        let coefficients = vec![coeff.as_slice()];

        // Process only middle 4 elements (indices 2-5)
        gf_muladd_region(&mut destinations, &sources, &coefficients, 2, 4);

        // First 2 elements should be zero
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 0);

        // Middle 4 should be processed
        for i in 2..6 {
            let expected = gf_mul(src[i], 3);
            assert_eq!(dst[i], expected, "Mismatch at index {}", i);
        }

        // Last 2 elements should be zero
        assert_eq!(dst[6], 0);
        assert_eq!(dst[7], 0);
    }

    #[test]
    fn test_block_regions() {
        init_tables();

        // Create a larger block to test region division
        let block_size = 256;
        let src = (0..block_size)
            .map(|i| (i % 256) as u16)
            .collect::<Vec<_>>();
        let sources = vec![src.as_slice()];

        let mut dst = vec![0u16; block_size];
        let mut destinations = vec![dst.as_mut_slice()];

        let coeff = vec![5u16];
        let coefficients = vec![coeff.as_slice()];

        gf_muladd_block_regions(&mut destinations, &sources, &coefficients, block_size);

        // Verify all elements were processed
        for i in 0..block_size {
            let expected = gf_mul(src[i], 5);
            assert_eq!(dst[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_column_region() {
        init_tables();

        let src = vec![1u16, 2, 3, 4];

        let mut dst1 = vec![0u16; 4];
        let mut dst2 = vec![0u16; 4];
        let mut destinations = vec![dst1.as_mut_slice(), dst2.as_mut_slice()];

        let coefficients = vec![2u16, 3u16];

        gf_muladd_column_region(&mut destinations, &src, &coefficients, 0, 4);

        // Verify dst1 = src * 2
        for i in 0..4 {
            assert_eq!(dst1[i], gf_mul(src[i], 2));
        }

        // Verify dst2 = src * 3
        for i in 0..4 {
            assert_eq!(dst2[i], gf_mul(src[i], 3));
        }
    }
}
