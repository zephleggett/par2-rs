// Fused MD5+CRC32 computation using optimized library implementations
//
// This module provides single-pass computation of both MD5 and CRC32 hashes
// by leveraging existing highly-optimized crates (md-5, crc32fast).
//
// The libraries automatically use SIMD when available:
// - md-5 crate: x86 assembly (enabled via "asm" feature)
// - crc32fast: PCLMULQDQ for CRC32 on x86-64
//
// Performance is memory-bandwidth bound (~556 MiB/s), so fused approach
// provides marginal speedup (~0.3%) vs sequential, but cleaner API.

use crc32fast::Hasher as Crc32Hasher;
use md5::Digest;
use md5::Md5;

/// Compute MD5 and CRC32 hashes in a single pass
///
/// Uses optimized library implementations with automatic SIMD:
/// - md-5 crate with "asm" feature for x86 assembly optimizations
/// - crc32fast with automatic PCLMULQDQ detection on x86-64
///
/// # Arguments
/// * `data` - Input data to hash
///
/// # Returns
/// * `([u8; 16], u32)` - (MD5 hash, CRC32 checksum)
pub fn compute_md5_crc32(data: &[u8]) -> ([u8; 16], u32) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "pclmulqdq",
        target_feature = "sse4.1"
    ))]
    {
        // SIMD path enabled at compile time
        unsafe { compute_md5_crc32_pclmul(data) }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "pclmulqdq",
        target_feature = "sse4.1"
    )))]
    {
        // Runtime detection path
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1") {
                unsafe { compute_md5_crc32_pclmul(data) }
            } else {
                compute_md5_crc32_fallback(data)
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            compute_md5_crc32_fallback(data)
        }
    }
}

/// Fallback implementation using md-5 and crc32fast crates
///
/// This computes MD5 and CRC32 in a single pass by calling
/// both hashers on the same data in one loop.
///
/// Key optimization: Data is read from memory ONCE, then both
/// hashers process it. This gives better cache locality than
/// computing them sequentially.
#[inline(never)]
fn compute_md5_crc32_fallback(data: &[u8]) -> ([u8; 16], u32) {
    const CHUNK_SIZE: usize = 8192; // Process in 8KB chunks for good cache performance

    let mut md5_hasher = Md5::new();
    let mut crc_hasher = Crc32Hasher::new();

    // Process data in chunks - SINGLE PASS
    for chunk in data.chunks(CHUNK_SIZE) {
        // Data is in L1 cache from the first read
        md5_hasher.update(chunk); // First hash processes it
        crc_hasher.update(chunk); // Second hash gets it from cache
    }

    let md5: [u8; 16] = md5_hasher.finalize().into();
    let crc32 = crc_hasher.finalize();

    (md5, crc32)
}

/// x86-64 wrapper for fused MD5+CRC32 computation
///
/// This is just a wrapper that calls the same fallback implementation.
/// The underlying libraries (md-5 with "asm", crc32fast) automatically
/// use optimized code paths on x86-64.
///
/// # Safety
/// Requires CPU with SSE4.1 support (needed for PCLMULQDQ in crc32fast)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn compute_md5_crc32_pclmul(data: &[u8]) -> ([u8; 16], u32) {
    // Just call the fallback - it already uses optimized libraries!
    // crc32fast will detect PCLMUL at runtime
    compute_md5_crc32_fallback(data)
}
