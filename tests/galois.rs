use par2_rs::galois::{gf_div, gf_mul, gf_pow, init_tables};

#[test]
fn test_galois_field_multiplication() {
    init_tables();

    // Identity
    assert_eq!(gf_mul(1, 5), 5);
    assert_eq!(gf_mul(5, 1), 5);

    // Zero
    assert_eq!(gf_mul(0, 5), 0);
    assert_eq!(gf_mul(5, 0), 0);

    // Commutative
    assert_eq!(gf_mul(3, 7), gf_mul(7, 3));
    assert_eq!(gf_mul(123, 456), gf_mul(456, 123));

    // Associative
    let a = 123u16;
    let b = 456u16;
    let c = 789u16;
    assert_eq!(gf_mul(gf_mul(a, b), c), gf_mul(a, gf_mul(b, c)));
}

#[test]
fn test_galois_field_division() {
    init_tables();

    // a / a = 1
    assert_eq!(gf_div(5, 5), 1);
    assert_eq!(gf_div(1234, 1234), 1);

    // a / 1 = a
    assert_eq!(gf_div(5, 1), 5);
    assert_eq!(gf_div(1234, 1), 1234);

    // 0 / a = 0
    assert_eq!(gf_div(0, 5), 0);
    assert_eq!(gf_div(0, 1234), 0);

    // Division is inverse of multiplication
    let a = 123u16;
    let b = 456u16;
    assert_eq!(gf_div(gf_mul(a, b), b), a);
    assert_eq!(gf_div(gf_mul(a, b), a), b);
}

#[test]
#[should_panic(expected = "Division by zero")]
fn test_galois_field_division_by_zero() {
    init_tables();
    gf_div(5, 0);
}

#[test]
fn test_galois_field_power() {
    init_tables();

    // a^0 = 1
    assert_eq!(gf_pow(5, 0), 1);
    assert_eq!(gf_pow(1234, 0), 1);

    // 0^0 = 1 (by convention)
    assert_eq!(gf_pow(0, 0), 1);

    // a^1 = a
    assert_eq!(gf_pow(5, 1), 5);
    assert_eq!(gf_pow(1234, 1), 1234);

    // 0^n = 0 for n > 0
    assert_eq!(gf_pow(0, 5), 0);
    assert_eq!(gf_pow(0, 100), 0);

    // a^2 = a * a
    let a = 123u16;
    assert_eq!(gf_pow(a, 2), gf_mul(a, a));

    // a^3 = a * a * a
    assert_eq!(gf_pow(a, 3), gf_mul(gf_mul(a, a), a));

    // Check distributive property: a^(m+n) = a^m * a^n
    assert_eq!(gf_pow(a, 7), gf_mul(gf_pow(a, 3), gf_pow(a, 4)));
}

#[test]
fn test_galois_field_inverse() {
    init_tables();

    // Test that multiplication by inverse gives 1
    // In GF(2^16), multiplicative inverse of a is a^(2^16 - 2) = a^65534
    let a = 42u16;
    let a_inv = gf_pow(a, 65534); // a^(p-2) where p = 2^16
    assert_eq!(gf_mul(a, a_inv), 1);

    // Another value
    let b = 1234u16;
    let b_inv = gf_pow(b, 65534);
    assert_eq!(gf_mul(b, b_inv), 1);
}

#[test]
fn test_galois_generator() {
    init_tables();

    // Generator is 2
    // Test that powers of 2 cycle through field elements
    assert_eq!(gf_pow(2, 0), 1);
    assert_eq!(gf_pow(2, 1), 2);
    assert_eq!(gf_pow(2, 2), 4);
    assert_eq!(gf_pow(2, 3), 8);
    assert_eq!(gf_pow(2, 4), 16);
}

#[test]
fn test_par2_constants_generation() {
    init_tables();

    // PAR2 constants are powers of 2 where exponent satisfies:
    // n % 3 != 0 && n % 5 != 0 && n % 17 != 0 && n % 257 != 0
    // First valid exponents: 1, 2, 4, 7, 8, 11, 13, 14, 16, 19, ...

    // Verify first constant is 2^1 = 2
    assert_eq!(gf_pow(2, 1), 2);

    // Verify second constant is 2^2 = 4
    assert_eq!(gf_pow(2, 2), 4);

    // Verify third constant is 2^4 = 16
    assert_eq!(gf_pow(2, 4), 16);
}
