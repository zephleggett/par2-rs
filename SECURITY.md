# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in par2-rs, please report it by:

1. **DO NOT** open a public issue
2. Email the maintainer (see AUTHORS or Cargo.toml for contact information)
3. Provide a detailed description of the vulnerability and steps to reproduce

We will respond to security reports within 48 hours and work with you to address the issue.

## Security Considerations

### Unsafe Code

This library uses `unsafe` code for SIMD optimizations. All unsafe code follows these principles:

1. **Runtime CPU feature detection**: All SIMD operations are protected by runtime checks
   using `is_x86_feature_detected!()` and `is_aarch64_feature_detected!()`.

2. **Target feature attributes**: SIMD functions are marked with `#[target_feature]` to
   ensure proper compilation and feature verification.

3. **Memory safety**: All pointer operations respect slice bounds and use unaligned
   loads/stores to handle arbitrary data alignment.

4. **Automatic fallback**: When SIMD features are unavailable, code automatically falls
   back to safe scalar implementations.

### Dependencies

We monitor dependencies for known security vulnerabilities. To check for vulnerabilities:

```bash
# Install cargo-audit
cargo install cargo-audit

# Run security audit
cargo audit
```

We also provide a `deny.toml` configuration for `cargo-deny` to automatically check for:
- Known security vulnerabilities
- Unmaintained dependencies
- License compatibility issues

```bash
# Install cargo-deny
cargo install cargo-deny

# Run comprehensive checks
cargo deny check
```

### Input Validation

- PAR2 files are parsed with bounds checking and validation
- File sizes are validated before allocation
- MD5 checksums are verified on all packets
- Block counts and sizes are validated to prevent overflow

### Memory Safety

- No use of `unwrap()` or `expect()` in critical paths (checked during code review)
- All file I/O uses standard library functions with proper error handling
- Memory allocations are bounded by PAR2 file metadata

## Supported Versions

We currently support the latest version of par2-rs. Security updates will be applied
to the main branch and released as patch versions.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Best Practices for Users

1. **Verify file integrity**: Always verify PAR2 file checksums before trusting repairs
2. **Keep backups**: Never rely solely on PAR2 for data protection
3. **Update regularly**: Keep par2-rs updated to receive security fixes
4. **Isolate untrusted input**: When processing PAR2 files from untrusted sources,
   consider running in a sandboxed environment
