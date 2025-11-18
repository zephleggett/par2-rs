//! Tests for GF(2^16) Galois Field operations

use super::*;
use crate::galois::core::GF_SIZE;

#[test]
fn test_gf_initialization() {
    // Can be called multiple times safely
    init_tables();
    init_tables();
}

#[test]
fn test_gf_multiplication() {
    init_tables();

    // Identity
    assert_eq!(gf_mul(1, 5), 5);
    assert_eq!(gf_mul(5, 1), 5);

    // Zero
    assert_eq!(gf_mul(0, 5), 0);
    assert_eq!(gf_mul(5, 0), 0);

    // Commutative
    assert_eq!(gf_mul(3, 7), gf_mul(7, 3));

    // Generator
    assert_eq!(gf_mul(2, 2), gf_pow(2, 2));
}

#[test]
fn test_gf_division() {
    init_tables();

    // a / a = 1
    assert_eq!(gf_div(5, 5), 1);

    // a / 1 = a
    assert_eq!(gf_div(5, 1), 5);

    // 0 / a = 0
    assert_eq!(gf_div(0, 5), 0);

    // Division is inverse of multiplication
    let a = 123u16;
    let b = 456u16;
    assert_eq!(gf_div(gf_mul(a, b), b), a);
}

#[test]
#[should_panic(expected = "Division by zero")]
fn test_gf_division_by_zero() {
    init_tables();
    gf_div(5, 0);
}

#[test]
fn test_gf_power() {
    init_tables();

    // a^0 = 1
    assert_eq!(gf_pow(5, 0), 1);
    assert_eq!(gf_pow(0, 0), 1);

    // a^1 = a
    assert_eq!(gf_pow(5, 1), 5);

    // 0^n = 0 for n > 0
    assert_eq!(gf_pow(0, 5), 0);

    // a^2 = a * a
    let a = 123u16;
    assert_eq!(gf_pow(a, 2), gf_mul(a, a));

    // a^3 = a * a * a
    assert_eq!(gf_pow(a, 3), gf_mul(gf_mul(a, a), a));
}

#[test]
fn test_field_properties() {
    init_tables();

    // Test that multiplication by inverse gives 1
    let a = 42u16;
    let a_inv = gf_pow(a, GF_SIZE - 2); // a^(p-1) = 1, so a^(p-2) = a^(-1)
    assert_eq!(gf_mul(a, a_inv), 1);
}

#[test]
fn test_simd_mul_slice() {
    init_tables();

    let scalar = 123u16;
    let mut data = vec![
        1u16, 2, 3, 4, 5, 100, 200, 300, 1000, 2000, 5000, 10000, 30000, 60000,
    ];
    let mut expected = data.clone();

    // Compute expected results using scalar multiplication
    for val in expected.iter_mut() {
        *val = gf_mul(*val, scalar);
}

    // Use SIMD multiplication
    gf_mul_slice(scalar, &mut data);

    // Results should match
    assert_eq!(data, expected, "SIMD multiplication should match scalar");
}

#[test]
fn test_bytes_to_u16_simd() {
    // Test with various sizes to ensure both SIMD and scalar paths work
    let test_cases = vec![
        vec![0x12, 0x34, 0x56, 0x78, 0xAB, 0xCD, 0xEF, 0x01], // 4 u16 values
        vec![0x00, 0x01, 0xFF, 0xFE, 0x12, 0x34],             // 3 u16 values
        vec![0xAA; 32], // 16 u16 values (SIMD path on NEON)
        vec![
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10, 0x11, 0x12,
        ], // 9 u16 values (mixed SIMD + scalar)
    ];

    for bytes in test_cases {
        let mut output = vec![0u16; bytes.len() / 2];
        let mut expected = vec![0u16; bytes.len() / 2];

        // Compute expected using scalar method
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            expected[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }

        // Use SIMD method
        bytes_to_u16_simd(&bytes, &mut output);

        assert_eq!(
            output, expected,
            "SIMD byte conversion should match scalar for input {:?}",
            bytes
        );
}
}

#[test]
fn test_simd_mul_slice_32() {
    init_tables();

    let scalar = 123u16;
    let mut data: Vec<u16> = (0..32).collect(); // Exactly one SIMD chunk
    let mut expected = data.clone();

    for val in expected.iter_mut() {
        *val = gf_mul(*val, scalar);
}

    gf_mul_slice(scalar, &mut data);

    assert_eq!(
        data, expected,
        "SIMD on 32-element array should match scalar"
    );
}

#[test]
fn test_simd_mul_slice_large() {
    init_tables();

    let scalar = 999u16;
    let mut data: Vec<u16> = (0..1024).map(|i| (i * 37) as u16).collect();
    let mut expected = data.clone();

    for val in expected.iter_mut() {
        *val = gf_mul(*val, scalar);
}

    gf_mul_slice(scalar, &mut data);

    assert_eq!(data, expected, "SIMD on large array should match scalar");
}

#[test]
fn test_simd_feature_detection() {
    // Print which SIMD features are available for debugging/verification
    println!("\n=== SIMD Feature Detection ===");

    #[cfg(target_arch = "x86_64")]
    {
        println!("Platform: x86-64");

        let has_vpclmulqdq = std::is_x86_feature_detected!("vpclmulqdq");
        let has_avx512f = std::is_x86_feature_detected!("avx512f");
        let has_avx512vl = std::is_x86_feature_detected!("avx512vl");
        let has_gfni = std::is_x86_feature_detected!("gfni");
        let has_avx2 = std::is_x86_feature_detected!("avx2");
        let has_pclmulqdq = std::is_x86_feature_detected!("pclmulqdq");
        let has_sse41 = std::is_x86_feature_detected!("sse4.1");
        let has_ssse3 = std::is_x86_feature_detected!("ssse3");

        println!("  VPCLMULQDQ: {}", has_vpclmulqdq);
        println!("  AVX-512F:   {}", has_avx512f);
        println!("  AVX-512VL:  {}", has_avx512vl);
        println!("  GFNI:       {}", has_gfni);
        println!("  AVX2:       {}", has_avx2);
        println!("  PCLMULQDQ:  {}", has_pclmulqdq);
        println!("  SSE4.1:     {}", has_sse41);
        println!("  SSSE3:      {}", has_ssse3);

        // Determine which implementation will be used
        #[cfg(feature = "unstable")]
        {
            if has_vpclmulqdq && has_avx512f && has_avx512vl && has_gfni {
                println!("  → Using: VPCLMUL + GFNI (AVX-512) [32 u16/iter]");
            } else if has_vpclmulqdq && has_avx512f && has_avx512vl {
                println!("  → Using: VPCLMUL (AVX-512) [32 u16/iter]");
            } else if has_pclmulqdq && has_avx2 && has_sse41 {
                println!("  → Using: AVX2 PCLMUL [16 u16/iter]");
            } else if has_pclmulqdq && has_sse41 {
                println!("  → Using: SSE PCLMUL [8 u16/iter]");
            } else {
                println!("  → Using: Scalar fallback [1 u16/iter]");
            }
        }
        #[cfg(not(feature = "unstable"))]
        {
            if has_pclmulqdq && has_avx2 && has_sse41 {
                println!("  → Using: AVX2 PCLMUL [16 u16/iter]");
            } else if has_pclmulqdq && has_sse41 {
                println!("  → Using: SSE PCLMUL [8 u16/iter]");
            } else {
                println!("  → Using: Scalar fallback [1 u16/iter]");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("Platform: ARM64");

        let has_neon = std::arch::is_aarch64_feature_detected!("neon");
        let has_pmull = std::arch::is_aarch64_feature_detected!("neon");

        println!("  NEON:  {}", has_neon);
        println!("  PMULL: {} (assumed with NEON crypto)", has_pmull);

        if has_neon {
            println!("  → Using: PMULL (NEON) [16 u16/iter]");
        } else {
            println!("  → Using: Scalar fallback [1 u16/iter]");
        }
    }

    println!("==============================\n");
}
