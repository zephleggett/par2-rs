//! Fused hash functions for PAR2 verification
//!
//! This module provides fused MD5+CRC32 computation by leveraging
//! existing highly-optimized crates:
//! - md-5 crate: x86 assembly optimizations (via "asm" feature)
//! - crc32fast: PCLMULQDQ for CRC32 on x86-64, NEON on ARM64
//!
//! The fused single-pass implementation provides:
//! 1. Single memory read for both hashes (instead of two sequential reads)
//! 2. Better cache locality (both hashers process same chunks)
//! 3. Cleaner API and more maintainable code
//! 4. Automatic SIMD optimizations from underlying libraries
//!
//! Benchmarks show marginal speedup (~0.3%) vs sequential MD5 then CRC32
//! because both approaches are memory-bandwidth bound on modern CPUs (~556 MiB/s)

mod md5_crc32_fused;

pub use md5_crc32_fused::compute_md5_crc32;

/// Compute MD5 hash only (for when CRC is already verified)
pub fn compute_md5(data: &[u8]) -> [u8; 16] {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(data);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        let data = b"";
        let (md5, crc32) = compute_md5_crc32(data);

        // MD5 of empty string
        assert_eq!(
            md5,
            [
                0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8,
                0x42, 0x7e
            ]
        );

        // CRC32 of empty string
        assert_eq!(crc32, 0);
    }

    #[test]
    fn test_abc() {
        let data = b"abc";
        let (md5, crc32) = compute_md5_crc32(data);

        // MD5("abc") from RFC 1321
        assert_eq!(
            md5,
            [
                0x90, 0x01, 0x50, 0x98, 0x3c, 0xd2, 0x4f, 0xb0, 0xd6, 0x96, 0x3f, 0x7d, 0x28, 0xe1,
                0x7f, 0x72
            ]
        );

        // CRC32("abc")
        assert_eq!(crc32, 0x352441c2);
    }

    #[test]
    fn test_long_message() {
        // MD5 test vector from RFC 1321: "message digest"
        let data = b"message digest";
        let (md5, crc32) = compute_md5_crc32(data);

        assert_eq!(
            md5,
            [
                0xf9, 0x6b, 0x69, 0x7d, 0x7c, 0xb7, 0x93, 0x8d, 0x52, 0x5a, 0x2f, 0x31, 0xaa, 0xf1,
                0x61, 0xd0
            ]
        );

        // Verify CRC32 is computed
        assert_ne!(crc32, 0);
    }

    #[test]
    fn test_compare_with_reference() {
        // Compare our implementation against md-5 and crc32fast crates
        use crc32fast::Hasher as Crc32Hasher;
        use md5::{Digest, Md5};

        let test_data = b"The quick brown fox jumps over the lazy dog";

        // Reference computation
        let mut md5_hasher = Md5::new();
        md5_hasher.update(test_data);
        let expected_md5: [u8; 16] = md5_hasher.finalize().into();

        let mut crc_hasher = Crc32Hasher::new();
        crc_hasher.update(test_data);
        let expected_crc = crc_hasher.finalize();

        // Our computation
        let (md5, crc32) = compute_md5_crc32(test_data);

        assert_eq!(md5, expected_md5);
        assert_eq!(crc32, expected_crc);
    }

    #[test]
    fn test_block_sized_data() {
        // Test with data that's exactly 64 bytes (one MD5 block)
        let data = vec![0x42u8; 64];

        use crc32fast::Hasher as Crc32Hasher;
        use md5::{Digest, Md5};

        let mut md5_hasher = Md5::new();
        md5_hasher.update(&data);
        let expected_md5: [u8; 16] = md5_hasher.finalize().into();

        let mut crc_hasher = Crc32Hasher::new();
        crc_hasher.update(&data);
        let expected_crc = crc_hasher.finalize();

        let (md5, crc32) = compute_md5_crc32(&data);

        assert_eq!(md5, expected_md5);
        assert_eq!(crc32, expected_crc);
    }

    #[test]
    fn test_large_data() {
        // Test with data larger than one MD5 block
        let data = vec![0x5au8; 512 * 1024]; // 512KB

        use crc32fast::Hasher as Crc32Hasher;
        use md5::{Digest, Md5};

        let mut md5_hasher = Md5::new();
        md5_hasher.update(&data);
        let expected_md5: [u8; 16] = md5_hasher.finalize().into();

        let mut crc_hasher = Crc32Hasher::new();
        crc_hasher.update(&data);
        let expected_crc = crc_hasher.finalize();

        let (md5, crc32) = compute_md5_crc32(&data);

        assert_eq!(md5, expected_md5);
        assert_eq!(crc32, expected_crc);
    }
}
