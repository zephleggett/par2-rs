# par2-rs

[![CI](https://github.com/zephleggett/par2-rs/workflows/CI/badge.svg)](https://github.com/zephleggett/par2-rs/actions)
[![codecov](https://codecov.io/gh/zephleggett/par2-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/zephleggett/par2-rs)

> WARNING: This library is currently in alpha status. Testing has been performed on macOS ARM64 systems. Platform compatibility and behavior with diverse file types has not been extensively verified. No warranties are provided. Maintain backups of critical data.

This is a native Rust implementation of the PAR2 (Parchive 2.0) specification for file verification and repair using Reed-Solomon error correction codes over GF(2^16). The implementation provides repair performance approximately 44% faster than par2cmdline-turbo (a C++ implementation with hand-optimized ARM NEON assembly) on ARM64 hardware while maintaining comparable memory overhead.

PAR2 archives enable verification of data integrity and reconstruction of corrupted or missing file segments through forward error correction. Applications include protection against bit rot, recovery from storage media degradation, and validation of network file transfers.

## Installation

```toml
[dependencies]
par2-rs = "0.1"
```

## Quick Start

```rust
use par2_rs::{Par2Repairer, Result};
use std::path::Path;

fn main() -> Result<()> {
    let repairer = Par2Repairer::new(Path::new("myfiles.par2"))?;

    // Verify only
    repairer.repair(false)?;

    // Verify and repair
    repairer.repair(true)?;

    Ok(())
}
```

### With Progress Tracking

```rust
use par2_rs::{Par2Repairer, Par2Operation};
use std::sync::Arc;

let progress = Arc::new(|operation: Par2Operation, current: u64, total: u64| {
    match operation {
        Par2Operation::Loading => println!("Loading: {}%", current * 100 / total),
        Par2Operation::Verifying => println!("Verifying: {}%", current * 100 / total),
        Par2Operation::Repairing => println!("Repairing: {}%", current * 100 / total),
        _ => {}
    }
});

repairer.repair_with_progress(true, false, Some(progress))?;
```

### Command Line Tools

**Repair:**
```bash
# Verify and repair (default)
cargo run --release --bin repair myfiles.par2

# Verify only
cargo run --release --bin repair myfiles.par2 verify

# Enable detailed logging (shows adaptive parallelism settings, timing)
RUST_LOG=info cargo run --release --bin repair myfiles.par2

# Adjust parallelism for different systems (adaptive by default)
PAR2_PARALLELISM=1 cargo run --release --bin repair myfiles.par2  # Minimal parallelism
PAR2_PARALLELISM=10 cargo run --release --bin repair myfiles.par2  # Maximum speed
```

**Create:**
```bash
# Create PAR2 files with 10% redundancy
cargo run --release --bin create --redundancy 10 file1.bin file2.bin

# Create with 5% redundancy in a single volume
cargo run --release --bin create --redundancy 5 --scheme single archive.zip

# Create with exponential volume distribution (like par2cmdline)
cargo run --release --bin create --redundancy 20 --scheme exponential data/
```

## Performance

Benchmark configuration: M1 Pro (10-core) system, repairing 95MB of corrupted data (35 damaged blocks) from a 2.8GB archive.

| Implementation | Repair Time | Data Repaired | Peak Memory | vs Standard |
|---------------|-------------|---------------|-------------|-------------|
| par2cmdline (standard) | 33.4s | 95MB | 146MB | baseline |
| par2cmdline-turbo (C++) | 7.4s | 95MB | 154MB | 4.5× faster |
| par2-rs (Rust) | 4.1s | 95MB | 170MB | 8.1× faster, 44% faster than turbo |

Implementation Strategy:

The performance characteristics result from several architectural decisions. First, the repair algorithm uses a streaming architecture that processes file blocks in 100KB chunks (worked best for me, let me know). Gaussian elimination is computed once and the resulting transformation matrix is reused across all chunks.

Second, verification employs an IFSC-first strategy. When block-level checksums (IFSC packets) verify successfully, the expensive full-file MD5 computation is skipped. This reduces verification time by in the common case where files are intact or have isolated damage.

Third, parallelism settings adapt automatically based on detected CPU core count. Systems with eight or more cores use a 10× multiplier with aggressive concurrency. Mid-range systems (4-7 cores) apply a 6× multiplier for balanced performance. Low-end systems (fewer than 4 cores) use a conservative 3× multiplier to avoid thread overhead. _**this could be improved upon*_

Fourth, Galois field arithmetic on ARM64 platforms uses NEON PMULL instructions for polynomial multiplication in GF(2^16). The SIMD implementation includes vectorized byte-to-u16 conversion and provides 4.6-5.8× speedup compared to scalar operations.

Fifth, the reconstruction matrix is processed column-wise rather than row-wise. This approach, inspired by ParPar, improves cache locality and reduces function call overhead during the critical repair path.

The combination of these techniques produces repair performance 44% faster than par2cmdline-turbo while maintaining memory overhead within 10% (170MB vs 154MB).

### Parallelism Configuration

Parallelism parameters adapt automatically based on detected CPU core count using the multipliers described above. Manual override is available through the PAR2_PARALLELISM environment variable.

| Setting | Use Case | Memory | Speed |
|---------|----------|--------|-------|
| `PAR2_PARALLELISM=1` | Minimal parallelism | ~100MB | ~10-12s |
| `PAR2_PARALLELISM=6` (adaptive default for 10-core) | Balanced performance | ~170MB | ~4.1s |
| `PAR2_PARALLELISM=10` | Maximum throughput | ~300MB | ~4.0s |

The streaming architecture constrains memory growth even under high parallelism, permitting aggressive concurrency settings on systems with adequate RAM.

## Platform Support

ARM64 (Apple Silicon, AWS Graviton): The ARM64 codepath is fully optimized and tested. Galois field arithmetic uses PMULL instructions for polynomial multiplication in GF(2^16). The implementation employs Karatsuba algorithm for 16-bit multiplication and Barrett reduction for modular arithmetic. Measured performance shows 4.6-5.8× speedup compared to scalar operations.

x86-64: AVX2 and SSSE3 implementations are provided for table-based Galois field multiplication. Expected performance is 8-10× speedup over scalar operations. These codepaths have not been tested on physical hardware. Users requiring x86-64 support should open an issue for testing coordination.

Other architectures: A portable scalar implementation is available and functions on all platforms.

## Features

The library implements parallel file verification and repair using the Rayon work-stealing scheduler. PAR2 file creation supports configurable redundancy levels and both single-volume and multi-volume output schemes. Multi-volume archives are loaded automatically by recovery set ID (matching .vol*.par2 files). File matching uses content-based hashing rather than filenames, providing resilience against renamed or obfuscated files. Packet integrity is verified using MD5 checksums. Large files are processed using memory-efficient streaming to bound working set size. Core operations comply with the PAR2 2.0 specification.

## Specification Compliance

This implementation conforms to the PAR2 2.0 specification (https://parchive.sourceforge.net/docs/specifications/parity-volume-spec/article-spec.html) with the following coverage:

Reading and Verification: The parser handles Main packets (recovery set metadata), File Description packets, Input File Slice Checksum (IFSC) packets, and Recovery Slice packets. Multi-volume archives are loaded automatically by recovery set ID. MD5 packet integrity verification is performed on all packets. File verification uses MD5 hashing with both full-file and 16KB quick-check modes. Reed-Solomon reconstruction operates over GF(2^16) using the standard polynomial 0x1100B.

Creation: PAR2 file generation supports configurable redundancy percentages. Output schemes include single-volume and multi-volume modes with exponential volume size distribution matching par2cmdline behavior. Block size is calculated automatically based on file size. Block-level checksums are written as IFSC packets for fast verification.

Omitted Features: Creator packets (informational metadata only), Unicode filename packets (ASCII filenames are sufficient for most use cases), and Comment packets (informational metadata only) are not implemented. These packets are optional extensions and their absence does not affect interoperability with standard PAR2 tools.

## Testing

The test suite comprises 104 unit and integration tests covering core functionality with 76% code coverage. Run the complete suite with `cargo test`. Code coverage analysis is available through `./scripts/check_coverage.sh` (terminal output) or `./scripts/check_coverage.sh --html` (HTML report). Detailed test documentation is provided in tests/README.md.

## Contributing

Contributions are accepted. The following areas would benefit from additional development:

- x86-64 Testing: The AVX2 and SSSE3 implementations require validation on Intel and AMD hardware. Access to physical x86-64 systems would enable performance verification and correctness testing.

- Command-Line Interface: The repair and create tools provide basic functionality but could be enhanced with improved progress reporting, error diagnostics, and command-line ergonomics.

- Benchmark Coverage: Current benchmarks focus on a single file size and damage pattern. Additional test cases covering diverse file sizes, block counts, and corruption scenarios would provide better performance characterization.

- Unicode Filenames: The current implementation supports ASCII filenames only. Adding Unicode filename packet support would improve compatibility with international character sets.

## Acknowledgments

SIMD Galois field arithmetic is adapted from reed-solomon-simd (https://github.com/AndersTrier/reed-solomon-simd) by Anders Trier Olesen, which implements the Leopard-RS algorithm. The code has been modified to use PAR2's primitive polynomial (0x1100B) and integrate with the PAR2 packet format.

Architectural optimizations draw inspiration from ParPar (https://github.com/animetosho/ParPar) by Anime Tosho, particularly the column-wise matrix processing approach, fused multiply-add operations, and multi-region processing patterns.

## License

Dual-licensed under MIT or Apache 2.0, at your option.
