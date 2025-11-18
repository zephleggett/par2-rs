//! Core GF(2^16) Galois Field operations
//!
//! This module contains the fundamental Galois Field arithmetic operations including:
//! - Table initialization
//! - Scalar multiplication, division, and exponentiation
//! - Constants and lookup tables

use std::sync::OnceLock;

/// PAR2 uses primitive polynomial 0x1100B for GF(2^16)
/// This is x^16 + x^12 + x^3 + x + 1
pub(crate) const PRIMITIVE_POLY: u32 = 0x1100B;

/// GF(2^16) field size
pub(crate) const GF_SIZE: usize = 65536;

/// Precomputed logarithm table for GF(2^16)
/// log_table[i] = log_α(i) where α is the generator (2)
pub(crate) static LOG_TABLE: OnceLock<Box<[u16; GF_SIZE]>> = OnceLock::new();

/// Precomputed exponential table for GF(2^16)
/// exp_table[i] = α^i where α is the generator (2)
/// Doubled in size for wrap-around to simplify modulo operations
pub(crate) static EXP_TABLE: OnceLock<Box<[u16; GF_SIZE * 2]>> = OnceLock::new();

/// SIMD multiplication lookup table for AVX2/SSSE3/NEON engines
/// Contains precomputed products for nibble-based SIMD multiplication
///
/// Algorithm adapted from reed-solomon-simd by Anders Trier Olesen
/// <https://github.com/AndersTrier/reed-solomon-simd>
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct Multiply128Lut {
    /// Lower byte of products
    pub lo: [u128; 4],
    /// Upper byte of products
    pub hi: [u128; 4],
}

/// SIMD lookup table: one entry per possible logarithm value
/// Lazily initialized only on platforms that need it (x86_64 primarily)
#[allow(dead_code)]
pub(crate) static MUL128_TABLE: OnceLock<Box<[Multiply128Lut; GF_SIZE]>> = OnceLock::new();

/// Initialize lookup tables for GF(2^16) arithmetic.
///
/// This function can be called multiple times safely; initialization
/// only happens once. Thread-safe using `OnceLock`.
#[inline]
pub fn init_tables() {
    // Initialize LOG_TABLE
    LOG_TABLE.get_or_init(|| {
        let mut log_table = Box::new([0u16; GF_SIZE]);
        let mut exp_table_temp = vec![0u16; GF_SIZE - 1];

        let mut b: u32 = 1;
        for (log, exp_entry) in exp_table_temp.iter_mut().enumerate().take(GF_SIZE - 1) {
            *exp_entry = b as u16;
            log_table[b as usize] = log as u16;

            // Multiply by α (generator=2) in GF(2^16)
            b <<= 1;
            if (b & 0x10000) != 0 {
                b ^= PRIMITIVE_POLY;
            }
        }

        // log(0) is undefined, but set to 0 for convenience
        log_table[0] = 0;

        log_table
    });

    // Initialize EXP_TABLE
    EXP_TABLE.get_or_init(|| {
        let mut exp_table = Box::new([0u16; GF_SIZE * 2]);
        let mut b: u32 = 1;

        for log in 0..GF_SIZE - 1 {
            exp_table[log] = b as u16;
            exp_table[log + GF_SIZE - 1] = b as u16; // Wrap-around for modulo

            // Multiply by α (generator=2) in GF(2^16)
            b <<= 1;
            if (b & 0x10000) != 0 {
                b ^= PRIMITIVE_POLY;
            }
        }

        exp_table
    });
}

/// Initialize the SIMD multiplication lookup table
///
/// Algorithm adapted from reed-solomon-simd by Anders Trier Olesen,
/// modified to use PAR2's primitive polynomial 0x1100B instead of Leopard-RS's 0x1002D.
///
/// Reference: <https://github.com/AndersTrier/reed-solomon-simd>
#[allow(dead_code)]
pub(crate) fn initialize_mul128_table() {
    MUL128_TABLE.get_or_init(|| {
        // Ensure LOG/EXP tables are initialized first
        init_tables();

        let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
        let exp_table = EXP_TABLE.get().expect("EXP_TABLE not initialized");

        let mut mul128 = Vec::with_capacity(GF_SIZE);

        for log_m in 0..GF_SIZE {
            let mut lut = Multiply128Lut {
                lo: [0; 4],
                hi: [0; 4],
            };

            // For each of 4 nibble positions (0, 4, 8, 12 bits)
            for i in 0..4 {
                let mut prod_lo = [0u8; 16];
                let mut prod_hi = [0u8; 16];

                // For each possible nibble value (0-15)
                for x in 0..16 {
                    let val = (x << (i * 4)) as u16;

                    // Multiply using logarithms: log(val * m) = log(val) + log_m
                    let prod = if val == 0 {
                        0
                    } else {
                        let log_val = log_table[val as usize] as usize;
                        let log_result = (log_val + log_m) % (GF_SIZE - 1);
                        exp_table[log_result]
                    };

                    prod_lo[x] = prod as u8;
                    prod_hi[x] = (prod >> 8) as u8;
                }

                lut.lo[i] = u128::from_le_bytes(prod_lo);
                lut.hi[i] = u128::from_le_bytes(prod_hi);
            }

            mul128.push(lut);
        }

        mul128.into_boxed_slice().try_into().unwrap()
    });
}

/// Multiply two elements in GF(2^16).
///
/// Uses logarithm tables: log(a*b) = log(a) + log(b)
///
/// # Examples
/// ```ignore
/// assert_eq!(gf_mul(1, 5), 5);
/// assert_eq!(gf_mul(0, 5), 0);
/// ```
#[inline]
pub fn gf_mul(a: u16, b: u16) -> u16 {
    init_tables(); // Ensure tables are initialized

    if a == 0 || b == 0 {
        return 0;
    }

    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let exp_table = EXP_TABLE.get().expect("EXP_TABLE not initialized");

    let log_a = log_table[a as usize] as usize;
    let log_b = log_table[b as usize] as usize;
    let log_result = log_a + log_b;
    exp_table[log_result]
}

/// Divide two elements in GF(2^16).
///
/// Uses logarithm tables: log(a/b) = log(a) - log(b)
///
/// # Panics
/// Panics if `b == 0` (division by zero)
///
/// # Examples
/// ```ignore
/// assert_eq!(gf_div(10, 2), gf_mul(10, gf_pow(2, GF_SIZE - 2)));
/// ```
#[inline]
pub fn gf_div(a: u16, b: u16) -> u16 {
    init_tables(); // Ensure tables are initialized

    if a == 0 {
        return 0;
    }
    if b == 0 {
        panic!("Division by zero in GF(2^16)");
    }

    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let exp_table = EXP_TABLE.get().expect("EXP_TABLE not initialized");

    let log_a = log_table[a as usize] as usize;
    let log_b = log_table[b as usize] as usize;

    // Compute log(a/b) = log(a) - log(b) mod (GF_SIZE - 1)
    let log_result = if log_a >= log_b {
        log_a - log_b
    } else {
        log_a + (GF_SIZE - 1) - log_b
    };

    exp_table[log_result]
}

/// Raise element to a power in GF(2^16).
///
/// Uses logarithm tables: log(a^n) = n * log(a)
///
/// # Examples
/// ```ignore
/// assert_eq!(gf_pow(2, 0), 1);
/// assert_eq!(gf_pow(2, 1), 2);
/// ```
#[inline]
pub fn gf_pow(a: u16, n: usize) -> u16 {
    init_tables(); // Ensure tables are initialized

    if a == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    if n == 0 {
        return 1;
    }

    let log_table = LOG_TABLE.get().expect("LOG_TABLE not initialized");
    let exp_table = EXP_TABLE.get().expect("EXP_TABLE not initialized");

    let log_a = log_table[a as usize] as usize;
    let log_result = (log_a * n) % (GF_SIZE - 1);
    exp_table[log_result]
}

/// Scalar multiplication fallback - multiply slice by scalar using lookup tables
#[inline]
pub(crate) fn gf_mul_slice_scalar(scalar: u16, data: &mut [u16]) {
    for val in data.iter_mut() {
        *val = gf_mul(*val, scalar);
    }
}

/// Scalar fallback for multiply-add
#[inline]
pub(crate) fn gf_muladd_scalar(dst: &mut [u16], src: &[u16], scalar: u16) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= gf_mul(s, scalar);
    }
}

/// Scalar fallback for column multiply-add
#[inline]
pub(crate) fn gf_muladd_column_scalar(
    destinations: &mut [&mut [u16]],
    source: &[u16],
    coefficients: &[u16],
) {
    for (dst, &coeff) in destinations.iter_mut().zip(coefficients.iter()) {
        if coeff != 0 {
            for (d, &s) in dst.iter_mut().zip(source.iter()) {
                *d ^= gf_mul(s, coeff);
            }
        }
    }
}

/// Scalar fallback for multi-region multiply-add
#[inline]
pub(crate) fn gf_muladd_multi_scalar(dst: &mut [u16], sources: &[&[u16]], coefficients: &[u16]) {
    for (src, &coeff) in sources.iter().zip(coefficients.iter()) {
        if coeff != 0 {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d ^= gf_mul(s, coeff);
            }
        }
    }
}

/// Scalar implementation for byte-to-u16 conversion
#[inline]
pub(crate) fn bytes_to_u16_scalar(bytes: &[u8], output: &mut [u16]) {
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        output[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
    }
}
