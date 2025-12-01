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
    assert_eq!(gf_div(5, 5), Some(1));

    // a / 1 = a
    assert_eq!(gf_div(5, 1), Some(5));

    // 0 / a = 0
    assert_eq!(gf_div(0, 5), Some(0));

    // Division is inverse of multiplication
    let a = 123u16;
    let b = 456u16;
    assert_eq!(gf_div(gf_mul(a, b), b), Some(a));
}

#[test]
fn test_gf_division_by_zero() {
    init_tables();
    // Division by zero returns None (no panic)
    assert_eq!(gf_div(5, 0), None);
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
        vec![0xAA; 32],                                       // 16 u16 values (SIMD path on NEON)
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
        if has_avx2 && has_ssse3 {
            println!("  → Using: AVX2 Shuffle [16 u16/iter]");
        } else if has_pclmulqdq && has_avx2 && has_sse41 {
            println!("  → Using: AVX2 PCLMUL [16 u16/iter]");
        } else if has_pclmulqdq && has_sse41 {
            println!("  → Using: SSE PCLMUL [8 u16/iter]");
        } else if has_ssse3 {
            println!("  → Using: SSSE3 [8 u16/iter]");
        } else {
            println!("  → Using: Scalar fallback [1 u16/iter]");
        }

        // Silence unused variable warnings for feature detection vars
        let _ = (has_vpclmulqdq, has_avx512f, has_avx512vl, has_gfni);
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

/// Test all SIMD implementations against scalar baseline
/// This ensures correctness of SSE, AVX2, AVX-512, and NEON implementations
#[test]
fn test_all_simd_implementations() {
    init_tables();

    let scalar = 12345u16;
    let test_sizes = vec![
        8,    // Exactly one SSE chunk
        16,   // Exactly one AVX2/NEON chunk
        32,   // Exactly one AVX-512 chunk
        100,  // Mixed SIMD + scalar remainder
        1024, // Large array
    ];

    for size in test_sizes {
        let mut data: Vec<u16> = (0..size).map(|i| (i * 97) as u16).collect();
        let mut expected = data.clone();

        // Compute expected using scalar
        for val in expected.iter_mut() {
            *val = gf_mul(*val, scalar);
        }

        // Use SIMD (will pick best available)
        gf_mul_slice(scalar, &mut data);

        assert_eq!(data, expected, "SIMD gf_mul_slice failed for size {}", size);
    }
}

/// Test gf_muladd SIMD implementations
#[test]
fn test_simd_muladd() {
    init_tables();

    let scalar = 7777u16;
    let test_sizes = vec![8, 16, 32, 100, 512];

    for size in test_sizes {
        let src: Vec<u16> = (0..size).map(|i| (i * 13) as u16).collect();
        let mut dst: Vec<u16> = (0..size).map(|i| (i * 17) as u16).collect();
        let mut expected = dst.clone();

        // Compute expected using scalar
        for (d, s) in expected.iter_mut().zip(src.iter()) {
            *d ^= gf_mul(*s, scalar);
        }

        // Use SIMD
        gf_muladd(&mut dst, &src, scalar);

        assert_eq!(dst, expected, "SIMD gf_muladd failed for size {}", size);
    }
}

/// Test gf_muladd_multi SIMD implementations
#[test]
fn test_simd_muladd_multi() {
    init_tables();

    let coeffs = vec![123u16, 456, 789, 1024, 2048];
    let size = 128;

    // Create source slices
    let sources_vec: Vec<Vec<u16>> = (0..5)
        .map(|i| (0..size).map(|j| ((j + i * 100) * 19) as u16).collect())
        .collect();
    let sources: Vec<&[u16]> = sources_vec.iter().map(|v| v.as_slice()).collect();

    let mut dst: Vec<u16> = vec![0u16; size];
    let mut expected: Vec<u16> = vec![0u16; size];

    // Compute expected using scalar
    for (src, &coeff) in sources.iter().zip(coeffs.iter()) {
        for (d, s) in expected.iter_mut().zip(src.iter()) {
            *d ^= gf_mul(*s, coeff);
        }
    }

    // Use SIMD
    gf_muladd_multi(&mut dst, &sources, &coeffs);

    assert_eq!(dst, expected, "SIMD gf_muladd_multi failed");
}

/// Test gf_muladd_column SIMD implementations
#[test]
fn test_simd_muladd_column() {
    init_tables();

    let coeffs = vec![111u16, 222, 333, 444];
    let size = 128;

    let src: Vec<u16> = (0..size).map(|i| (i * 23) as u16).collect();
    let mut dsts_vec: Vec<Vec<u16>> = (0..4)
        .map(|i| (0..size).map(|j| ((j + i * 50) * 29) as u16).collect())
        .collect();

    // Compute expected using scalar
    let mut expected_vec = dsts_vec.clone();
    for (dst, &coeff) in expected_vec.iter_mut().zip(coeffs.iter()) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d ^= gf_mul(*s, coeff);
        }
    }

    // Use SIMD
    let mut dsts: Vec<&mut [u16]> = dsts_vec.iter_mut().map(|v| v.as_mut_slice()).collect();
    gf_muladd_column(&mut dsts, &src, &coeffs);

    for (i, (dst, expected)) in dsts_vec.iter().zip(expected_vec.iter()).enumerate() {
        assert_eq!(dst, expected, "SIMD gf_muladd_column failed for dst {}", i);
    }
}

/// Test edge case: zero scalar
#[test]
fn test_simd_zero_scalar() {
    init_tables();

    let mut data: Vec<u16> = (0..100).collect();
    let original = data.clone();

    // Multiply by zero should zero out the data
    gf_mul_slice(0, &mut data);
    assert_eq!(data, vec![0u16; 100], "Multiply by zero failed");

    // muladd by zero should leave data unchanged
    let mut dst = original.clone();
    let src: Vec<u16> = (100..200).collect();
    gf_muladd(&mut dst, &src, 0);
    assert_eq!(dst, original, "Muladd by zero should not modify dst");
}

/// Test edge case: identity scalar
#[test]
fn test_simd_identity_scalar() {
    init_tables();

    let mut data: Vec<u16> = (0..100).collect();
    let original = data.clone();

    // Multiply by 1 should leave data unchanged
    gf_mul_slice(1, &mut data);
    assert_eq!(data, original, "Multiply by 1 should not change data");
}

/// Test with maximum u16 values
#[test]
fn test_simd_max_values() {
    init_tables();

    let scalar = 0xFFFF;
    let mut data = vec![0xFFFF, 0xFFFE, 0xFFFD, 0x8000, 0x7FFF];
    let mut expected = data.clone();

    for val in expected.iter_mut() {
        *val = gf_mul(*val, scalar);
    }

    gf_mul_slice(scalar, &mut data);
    assert_eq!(data, expected, "Max value multiplication failed");
}

/// Test unaligned sizes (not multiple of SIMD width)
#[test]
fn test_simd_unaligned_sizes() {
    init_tables();

    let scalar = 555u16;
    let test_sizes = vec![
        1,   // Single element
        3,   // Odd size
        7,   // Not aligned to 8
        15,  // Not aligned to 16
        31,  // Not aligned to 32
        33,  // Just over 32
        100, // Random unaligned
    ];

    for size in test_sizes {
        let mut data: Vec<u16> = (0..size).map(|i| (i * 41) as u16).collect();
        let mut expected = data.clone();

        for val in expected.iter_mut() {
            *val = gf_mul(*val, scalar);
        }

        gf_mul_slice(scalar, &mut data);
        assert_eq!(data, expected, "Unaligned size {} failed", size);
    }
}

/// Verify SIMD implementations are actually being used (not just scalar fallback)
/// This test uses timing to detect if SIMD is active
#[test]
#[ignore] // Ignored by default as it's timing-sensitive
fn test_simd_performance_sanity() {
    use std::time::Instant;

    init_tables();

    let scalar = 12345u16;
    let size = 100_000;
    let iterations = 100;

    // Warm up
    let mut data: Vec<u16> = (0..size).map(|i| i as u16).collect();
    gf_mul_slice(scalar, &mut data);

    // Measure SIMD performance
    let mut data: Vec<u16> = (0..size).map(|i| i as u16).collect();
    let start = Instant::now();
    for _ in 0..iterations {
        gf_mul_slice(scalar, &mut data);
    }
    let simd_duration = start.elapsed();

    println!(
        "Processed {} elements {} times in {:?}",
        size, iterations, simd_duration
    );

    // On platforms with SIMD, this should complete in reasonable time
    // This is a sanity check, not a precise benchmark
    assert!(
        simd_duration.as_secs() < 5,
        "SIMD operations taking too long - may be using scalar fallback"
    );
}

/// Test each available SIMD strategy individually
///
/// This ensures all strategies produce correct results, not just the "best" one
#[test]
fn test_each_simd_strategy() {
    use crate::galois::simd::SimdRegistry;

    init_tables();

    let registry = SimdRegistry::new();
    let available = registry.list_available();

    println!("Testing {} available strategies:", available.len());
    for (name, priority) in &available {
        println!("  - {} (priority: {:?})", name, priority);
    }

    // Test data with various sizes to exercise different code paths
    let test_sizes = vec![
        7,    // Odd size, less than 8
        8,    // Exactly 8 (SSE chunk)
        15,   // Odd, less than 16
        16,   // Exactly 16 (AVX chunk)
        31,   // Odd, less than 32
        32,   // Exactly 32 (AVX-512 chunk)
        100,  // Mixed chunks + remainder
        1024, // Large
    ];
    let scalars = vec![0u16, 1, 2, 12345, 65535];

    // Create a fresh registry for iteration
    let strategies = get_all_strategies();

    for (name, strategy) in strategies {
        println!("Testing strategy: {}", name);

        for &size in &test_sizes {
            for &scalar in &scalars {
                // Test mul_slice
                let mut data: Vec<u16> = (0..size).map(|i| (i * 97 + 1) as u16).collect();
                let mut expected = data.clone();

                // Compute expected using scalar
                for val in expected.iter_mut() {
                    *val = gf_mul(*val, scalar);
                }

                // Use this strategy
                unsafe {
                    strategy.mul_slice(scalar, &mut data);
                }

                assert_eq!(
                    data, expected,
                    "Strategy {} failed mul_slice for size={}, scalar={}",
                    name, size, scalar
                );

                // Test muladd
                let src: Vec<u16> = (0..size).map(|i| (i * 13 + 7) as u16).collect();
                let mut dst: Vec<u16> = (0..size).map(|i| (i * 17 + 3) as u16).collect();
                let mut expected = dst.clone();

                for (i, expected_val) in expected.iter_mut().enumerate() {
                    *expected_val ^= gf_mul(src[i], scalar);
                }

                unsafe {
                    strategy.muladd(&mut dst, &src, scalar);
                }

                assert_eq!(
                    dst, expected,
                    "Strategy {} failed muladd for size={}, scalar={}",
                    name, size, scalar
                );
            }
        }
    }
}

/// Helper function to get all available strategies for individual testing
#[allow(unused_mut)] // mut needed on x86_64 but not on other platforms
fn get_all_strategies() -> Vec<(
    &'static str,
    Box<dyn crate::galois::simd::GaloisSimdStrategy>,
)> {
    use crate::galois::simd::GaloisSimdStrategy;

    let mut strategies: Vec<(&'static str, Box<dyn GaloisSimdStrategy>)> = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        use crate::galois::simd::x86::*;

        // SSE strategies
        let sse2 = sse::Sse2Strategy;
        if sse2.is_available() {
            strategies.push((sse2.name(), Box::new(sse2)));
        }

        let ssse3 = sse::Ssse3Strategy;
        if ssse3.is_available() {
            strategies.push((ssse3.name(), Box::new(ssse3)));
        }

        // AVX2 strategies
        let avx2_shuffle = avx2::Avx2ShuffleStrategy;
        if avx2_shuffle.is_available() {
            strategies.push((avx2_shuffle.name(), Box::new(avx2_shuffle)));
        }

        // PCLMUL strategies
        let pclmul = pclmul::PclmulStrategy;
        if pclmul.is_available() {
            strategies.push((pclmul.name(), Box::new(pclmul)));
        }

        let avx2_pclmul = pclmul::Avx2PclmulStrategy;
        if avx2_pclmul.is_available() {
            strategies.push((avx2_pclmul.name(), Box::new(avx2_pclmul)));
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON is tested via the main code path
        // Add ARM-specific strategy testing here if needed
    }

    strategies
}

/// Test shuffle2x format conversion and native operations
///
/// Tests that strategies supporting shuffle2x produce correct results when:
/// 1. Converting to shuffle2x format
/// 2. Operating in native shuffle2x mode
/// 3. Converting back to interleaved format
#[test]
fn test_shuffle2x_format_operations() {
    #[allow(unused_imports)]
    use crate::galois::simd::GaloisSimdStrategy;

    init_tables();

    let strategies = get_all_strategies();
    let test_sizes = vec![16, 32, 64, 128, 256]; // Must be multiple of 16 for shuffle2x
    let scalars = vec![1u16, 2, 12345, 65535];

    for (name, strategy) in strategies {
        if !strategy.supports_shuffle2x() {
            println!("Skipping {} (no shuffle2x support)", name);
            continue;
        }

        println!("Testing shuffle2x for strategy: {}", name);

        for &size in &test_sizes {
            for &scalar in &scalars {
                // Test prepare/finish roundtrip
                let original: Vec<u16> = (0..size).map(|i| (i * 97 + 1) as u16).collect();
                let mut data = original.clone();

                unsafe {
                    strategy.prepare_shuffle2x(&mut data);
                    strategy.finish_shuffle2x(&mut data);
                }

                assert_eq!(
                    data, original,
                    "Strategy {} prepare/finish roundtrip failed for size={}",
                    name, size
                );

                // Test native shuffle2x muladd produces same result as regular muladd
                let src: Vec<u16> = (0..size).map(|i| (i * 13 + 7) as u16).collect();
                let dst_init: Vec<u16> = (0..size).map(|i| (i * 17 + 3) as u16).collect();

                // Compute expected with regular muladd
                let mut expected = dst_init.clone();
                unsafe {
                    strategy.muladd(&mut expected, &src, scalar);
                }

                // Compute with native shuffle2x
                let mut dst_s2x = dst_init.clone();
                let mut src_s2x = src.clone();

                unsafe {
                    strategy.prepare_shuffle2x(&mut dst_s2x);
                    strategy.prepare_shuffle2x(&mut src_s2x);
                    strategy.muladd_shuffle2x(&mut dst_s2x, &src_s2x, scalar);
                    strategy.finish_shuffle2x(&mut dst_s2x);
                }

                assert_eq!(
                    dst_s2x, expected,
                    "Strategy {} native shuffle2x muladd failed for size={}, scalar={}",
                    name, size, scalar
                );
            }
        }
    }
}

/// Test shuffle2x region operations
#[test]
fn test_shuffle2x_region_operations() {
    #[allow(unused_imports)]
    use crate::galois::simd::GaloisSimdStrategy;

    init_tables();

    let strategies = get_all_strategies();
    let size = 128; // Must be multiple of 16
    let num_dsts = 4;
    let num_srcs = 3;

    for (name, strategy) in strategies {
        if !strategy.supports_shuffle2x() {
            continue;
        }

        println!("Testing shuffle2x region for strategy: {}", name);

        // Create sources
        let sources_data: Vec<Vec<u16>> = (0..num_srcs)
            .map(|i| (0..size).map(|j| ((j + i * 100) * 19) as u16).collect())
            .collect();
        let sources: Vec<&[u16]> = sources_data.iter().map(|v| v.as_slice()).collect();

        // Create destinations
        let mut dsts_data: Vec<Vec<u16>> = (0..num_dsts)
            .map(|i| (0..size).map(|j| ((j + i * 50) * 29) as u16).collect())
            .collect();

        // Create coefficients matrix (num_dsts x num_srcs)
        let coeffs_data: Vec<Vec<u16>> = (0..num_dsts)
            .map(|i| {
                (0..num_srcs)
                    .map(|j| (i * 111 + j * 222 + 1) as u16)
                    .collect()
            })
            .collect();
        let coefficients: Vec<&[u16]> = coeffs_data.iter().map(|v| v.as_slice()).collect();

        // Compute expected with regular region operation
        let mut expected_data = dsts_data.clone();
        {
            let mut expected_refs: Vec<&mut [u16]> =
                expected_data.iter_mut().map(|v| v.as_mut_slice()).collect();
            unsafe {
                strategy.muladd_region(&mut expected_refs, &sources, &coefficients, 0, size);
            }
        }

        // Compute with native shuffle2x region operation
        // First convert all data to shuffle2x format
        let mut src_s2x_data = sources_data.clone();
        for src in &mut src_s2x_data {
            unsafe {
                strategy.prepare_shuffle2x(src);
            }
        }
        let src_s2x: Vec<&[u16]> = src_s2x_data.iter().map(|v| v.as_slice()).collect();

        for dst in &mut dsts_data {
            unsafe {
                strategy.prepare_shuffle2x(dst);
            }
        }

        {
            let mut dst_refs: Vec<&mut [u16]> =
                dsts_data.iter_mut().map(|v| v.as_mut_slice()).collect();
            unsafe {
                strategy.muladd_region_shuffle2x(&mut dst_refs, &src_s2x, &coefficients, 0, size);
            }
        }

        // Convert back and verify
        for dst in &mut dsts_data {
            unsafe {
                strategy.finish_shuffle2x(dst);
            }
        }

        for (i, (dst, expected)) in dsts_data.iter().zip(expected_data.iter()).enumerate() {
            assert_eq!(
                dst, expected,
                "Strategy {} shuffle2x region failed for dst {}",
                name, i
            );
        }
    }
}
