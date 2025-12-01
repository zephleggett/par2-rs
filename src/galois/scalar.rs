//! Scalar (non-SIMD) implementations of Galois field operations
//!
//! These implementations serve as both fallback for platforms without
//! SIMD support and as reference implementations for correctness testing.

use crate::galois::core::{gf_mul, init_tables};

/// Scalar multiplication - multiply slice by scalar using lookup tables
#[allow(dead_code)] // Used by x86 fallback paths
#[inline]
pub fn gf_mul_slice_scalar(scalar: u16, data: &mut [u16]) {
    init_tables(); // Ensure tables are initialized

    for val in data.iter_mut() {
        *val = gf_mul(*val, scalar);
    }
}

/// Scalar multiply-add: dst[i] ^= src[i] * scalar
#[allow(dead_code)] // Used by x86 fallback paths
#[inline]
pub fn gf_muladd_scalar(dst: &mut [u16], src: &[u16], scalar: u16) {
    init_tables(); // Ensure tables are initialized

    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= gf_mul(s, scalar);
    }
}

/// Scalar column multiply-add
///
/// For each destination row i: dst[i] ^= source * coefficients[i]
#[allow(dead_code)] // Used by non-ARM platforms
#[inline]
pub fn gf_muladd_column_scalar(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
) {
    init_tables(); // Ensure tables are initialized

    for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
        if coeff != 0 {
            for (d, &s) in dst.iter_mut().zip(source.iter()) {
                *d ^= gf_mul(s, coeff);
            }
        }
    }
}

/// Scalar multi-source multiply-add
///
/// Optimized to load destination once, accumulate contributions from all sources,
/// then store once per element to minimize memory traffic.
#[allow(dead_code)] // Used by non-ARM platforms
#[inline]
pub fn gf_muladd_multi_scalar(dst: &mut [u16], sources: &[&[u16]], coefficients: &[u16]) {
    init_tables(); // Ensure tables are initialized

    // Process element-by-element to enable better optimization
    for i in 0..dst.len() {
        let mut accum = dst[i];

        // Accumulate contributions from all sources
        for (src, &coeff) in sources.iter().zip(coefficients.iter()) {
            if coeff != 0 {
                let src_val = src[i];
                if src_val != 0 {
                    accum ^= gf_mul(src_val, coeff);
                }
            }
        }

        dst[i] = accum;
    }
}

/// Scalar region-based multiply-accumulate
///
/// Process multiple destinations and sources in a region for better cache locality
#[allow(dead_code)] // Used by SIMD fallback paths
pub fn gf_muladd_region_scalar(
    destinations: &mut [&mut [u16]],
    sources: &[&[u16]],
    coefficients: &[&[u16]],
    region_offset: usize,
    region_size: usize,
) {
    init_tables(); // Ensure tables are initialized

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

            // Multiply-accumulate
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d ^= gf_mul(s, coeff);
            }
        }
    }
}

/// Scalar byte-to-u16 conversion
#[inline]
pub fn bytes_to_u16_scalar(bytes: &[u8], output: &mut [u16]) {
    debug_assert_eq!(
        bytes.len(),
        output.len() * 2,
        "bytes.len() must be 2x output.len()"
    );

    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        output[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
    }
}

/// Scalar u16-to-bytes conversion
#[inline]
pub fn u16_to_bytes_scalar(input: &[u16], bytes: &mut [u8]) {
    debug_assert_eq!(
        bytes.len(),
        input.len() * 2,
        "bytes.len() must be 2x input.len()"
    );

    for (i, &val) in input.iter().enumerate() {
        let offset = i * 2;
        bytes[offset] = val as u8;
        bytes[offset + 1] = (val >> 8) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_mul_slice() {
        let mut data = vec![1, 2, 3, 4, 5];
        let scalar = 7;
        gf_mul_slice_scalar(scalar, &mut data);

        // Verify each element was multiplied
        assert_eq!(data[0], gf_mul(1, 7));
        assert_eq!(data[1], gf_mul(2, 7));
        assert_eq!(data[2], gf_mul(3, 7));
    }

    #[test]
    fn test_scalar_muladd() {
        let mut dst = vec![10, 20, 30];
        let src = vec![1, 2, 3];
        let scalar = 5;

        gf_muladd_scalar(&mut dst, &src, scalar);

        // Verify multiply-accumulate
        assert_eq!(dst[0], 10 ^ gf_mul(1, 5));
        assert_eq!(dst[1], 20 ^ gf_mul(2, 5));
        assert_eq!(dst[2], 30 ^ gf_mul(3, 5));
    }

    #[test]
    fn test_bytes_conversion() {
        let bytes = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut u16_data = vec![0u16; 3];

        bytes_to_u16_scalar(&bytes, &mut u16_data);

        assert_eq!(u16_data[0], 0x0201); // Little-endian
        assert_eq!(u16_data[1], 0x0403);
        assert_eq!(u16_data[2], 0x0605);

        let mut bytes_out = vec![0u8; 6];
        u16_to_bytes_scalar(&u16_data, &mut bytes_out);

        assert_eq!(bytes, bytes_out);
    }
}
