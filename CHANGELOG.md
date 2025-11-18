# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive Rust best practices configuration
  - `clippy.toml` for linting configuration
  - `rustfmt.toml` for code formatting standards
  - `deny.toml` for dependency security and license checking
- Security documentation (`SECURITY.md`)
- Contributing guidelines (`CONTRIBUTING.md`)
- Enhanced CI/CD workflows
  - MSRV (Minimum Supported Rust Version) testing
  - Security audit with cargo-audit
  - Documentation validation
  - Proper GitHub Actions token permissions

### Changed
- Updated `.cargo/config.toml` with portability warnings
- Enhanced `Cargo.toml` metadata for better crates.io discoverability
- Improved documentation for unsafe SIMD code with comprehensive safety notes

### Fixed
- Fixed 6 rustdoc broken intra-doc link warnings in `src/galois/mod.rs`
- Fixed GitHub Actions security by adding minimal GITHUB_TOKEN permissions

## [0.1.0] - Initial Release

### Added
- Pure Rust PAR2 file verification and repair
- Multi-volume PAR2 support
- Parallel processing using Rayon
- SIMD-optimized Galois field arithmetic
  - ARM64 NEON (PMULL) support
  - x86-64 PCLMULQDQ support (SSE, AVX2, AVX-512)
- Content-based file matching
- Memory-efficient streaming for large files
- Command-line tools for repair and creation
- Comprehensive test suite (104 tests, 76% coverage)

### Performance
- 8.1× faster than par2cmdline (standard)
- 44% faster than par2cmdline-turbo on ARM64
- Adaptive parallelism based on CPU core count

[Unreleased]: https://github.com/zephleggett/par2-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zephleggett/par2-rs/releases/tag/v0.1.0
