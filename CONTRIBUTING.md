# Contributing to par2-rs

Thank you for your interest in contributing to par2-rs! This document provides guidelines
for contributing to the project.

## Code of Conduct

Be respectful and constructive in all interactions with the project.

## Getting Started

### Prerequisites

- Rust 1.76.0 or later (as specified in `Cargo.toml`)
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/zephleggett/par2-rs.git
cd par2-rs

# Build the project
cargo build

# Run tests
cargo test

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt

# Check documentation
cargo doc --no-deps --open
```

## Development Guidelines

### Code Quality

1. **Follow Rust conventions**: Use `cargo fmt` to format code and `cargo clippy` to
   catch common mistakes.

2. **Write tests**: All new features should include tests. Aim for high code coverage.

3. **Document public APIs**: All public functions, structs, and modules should have
   documentation comments.

4. **Handle errors properly**: Use `Result` types and avoid `unwrap()` in library code.
   In tests and examples, `unwrap()` is acceptable.

5. **Minimize unsafe code**: Only use `unsafe` when necessary for performance-critical
   SIMD operations. Always document safety invariants.

### Testing

```bash
# Run all tests
cargo test

# Run tests with logging enabled
RUST_LOG=debug cargo test

# Run tests for a specific module
cargo test --test repair

# Run with all features
cargo test --all-features
```

### Benchmarking

```bash
# Run benchmarks
cargo bench

# Profile with perf (Linux)
cargo build --release
perf record --call-graph=dwarf ./target/release/repair test.par2
perf report
```

### Code Review Checklist

Before submitting a pull request, ensure:

- [ ] Code compiles without warnings (`cargo build`)
- [ ] All tests pass (`cargo test`)
- [ ] Clippy passes with no warnings (`cargo clippy --all-targets --all-features -- -D warnings`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] Documentation builds without warnings (`cargo doc --no-deps`)
- [ ] New features include tests
- [ ] Public APIs are documented
- [ ] Security implications are considered (especially for unsafe code)

## Pull Request Process

1. **Fork the repository** and create a new branch for your changes
2. **Make your changes** following the guidelines above
3. **Write or update tests** to cover your changes
4. **Update documentation** if you're changing public APIs
5. **Run the full test suite** to ensure nothing is broken
6. **Submit a pull request** with a clear description of the changes

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for Unicode filenames in PAR2 files

- Implement Unicode filename packet parsing
- Add tests for international character sets
- Update documentation

Fixes #123
```

## Areas for Contribution

We welcome contributions in these areas:

### High Priority

- **x86-64 Performance Testing**: Validate and benchmark PCLMUL implementations on
  Intel/AMD hardware
- **SIMD Bug Fixes**: Fix the SSSE3 table-based implementation (currently disabled)
- **Unicode Support**: Implement Unicode filename packet support

### Medium Priority

- **Command-line Interface**: Improve CLI tools with better progress reporting and
  error messages
- **Additional Tests**: Add more edge case tests and fuzzing
- **Documentation**: Improve examples and API documentation

### Low Priority

- **Creator Packet Support**: Implement creator packet parsing (informational only)
- **Comment Packet Support**: Implement comment packet parsing (informational only)
- **Performance Optimizations**: Further optimize hot paths

## Unsafe Code Guidelines

When contributing unsafe code:

1. **Justify the use**: Explain why unsafe is necessary (usually for SIMD performance)
2. **Document safety invariants**: Use `// SAFETY:` comments to explain why the unsafe
   code is safe
3. **Use runtime feature detection**: Always check CPU features before calling SIMD
   functions
4. **Provide safe fallbacks**: Ensure scalar alternatives exist
5. **Test thoroughly**: SIMD code requires extensive testing on target platforms

Example:

```rust
#[target_feature(enable = "avx2")]
unsafe fn simd_operation(data: &mut [u16]) {
    // SAFETY: This function is only called after runtime CPU feature detection
    // verifies AVX2 support. Pointer operations respect slice bounds.
    // Unaligned loads are used to handle arbitrary alignment.
    ...
}

pub fn public_api(data: &mut [u16]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { simd_operation(data) };
    } else {
        scalar_fallback(data);
    }
}
```

## Release Process

(For maintainers)

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Run full test suite including `cargo test --all-features`
4. Create a git tag: `git tag -a v0.x.y -m "Release v0.x.y"`
5. Push tag: `git push origin v0.x.y`
6. Publish to crates.io: `cargo publish`

## Questions?

If you have questions about contributing, feel free to:

- Open an issue for discussion
- Check existing issues for similar questions
- Reach out to the maintainers

Thank you for contributing to par2-rs!
