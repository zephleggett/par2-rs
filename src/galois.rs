// GF(2^16) Galois Field arithmetic for PAR2
// Uses primitive polynomial 0x1100B as specified in PAR2 specification

/// PAR2 uses primitive polynomial 0x1100B for GF(2^16)
/// This is x^16 + x^12 + x^3 + x + 1
const PRIMITIVE_POLY: u32 = 0x1100B;

/// GF(2^16) field size
const GF_SIZE: usize = 65536;

/// Precomputed logarithm table for GF(2^16)
/// log_table[i] = log_α(i) where α is the generator
static mut LOG_TABLE: [u16; GF_SIZE] = [0; GF_SIZE];

/// Precomputed exponential table for GF(2^16)
/// exp_table[i] = α^i where α is the generator
static mut EXP_TABLE: [u16; GF_SIZE * 2] = [0; GF_SIZE * 2];

/// Flag to track if tables have been initialized
static mut TABLES_INITIALIZED: bool = false;

/// Initialize lookup tables for GF(2^16) arithmetic
pub fn init_tables() {
    unsafe {
        if TABLES_INITIALIZED {
            return;
        }

        let mut b: u32 = 1;
        for log in 0..GF_SIZE - 1 {
            EXP_TABLE[log] = b as u16;
            EXP_TABLE[log + GF_SIZE - 1] = b as u16; // Wrap-around for easier computation
            LOG_TABLE[b as usize] = log as u16;

            // Multiply by α (generator=2) in GF(2^16) with primitive poly 0x1100B
            b <<= 1;
            if (b & 0x10000) != 0 {
                b ^= PRIMITIVE_POLY;
            }
        }

        LOG_TABLE[0] = 0; // log(0) is undefined, but we set it to 0 for convenience

        TABLES_INITIALIZED = true;
    }
}

/// Multiply two elements in GF(2^16)
#[inline]
pub fn gf_mul(a: u16, b: u16) -> u16 {
    if a == 0 || b == 0 {
        return 0;
    }

    unsafe {
        let log_a = LOG_TABLE[a as usize] as usize;
        let log_b = LOG_TABLE[b as usize] as usize;
        let log_result = log_a + log_b;
        EXP_TABLE[log_result]
    }
}

/// Divide two elements in GF(2^16)
#[inline]
pub fn gf_div(a: u16, b: u16) -> u16 {
    if a == 0 {
        return 0;
    }
    if b == 0 {
        panic!("Division by zero in GF(2^16)");
    }

    unsafe {
        let log_a = LOG_TABLE[a as usize] as usize;
        let log_b = LOG_TABLE[b as usize] as usize;
        let log_result = if log_a >= log_b {
            log_a - log_b
        } else {
            log_a + (GF_SIZE - 1) - log_b
        };
        EXP_TABLE[log_result]
    }
}

/// Raise element to a power in GF(2^16)
#[inline]
pub fn gf_pow(a: u16, n: usize) -> u16 {
    if a == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    if n == 0 {
        return 1;
    }

    unsafe {
        let log_a = LOG_TABLE[a as usize] as usize;
        let log_result = (log_a * n) % (GF_SIZE - 1);
        EXP_TABLE[log_result]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf_basics() {
        init_tables();

        // Test multiplication
        assert_eq!(gf_mul(1, 5), 5);
        assert_eq!(gf_mul(0, 5), 0);
        assert_eq!(gf_mul(5, 0), 0);

        // Test division
        assert_eq!(gf_div(10, 2), gf_mul(10, gf_pow(2, GF_SIZE - 2))); // a/b = a * b^-1

        // Test power
        assert_eq!(gf_pow(2, 0), 1);
        assert_eq!(gf_pow(2, 1), 2);
    }
}
