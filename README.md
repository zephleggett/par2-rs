# par2-rs

[![CI](https://github.com/zephleggett/par2-rs/workflows/CI/badge.svg)](https://github.com/zephleggett/par2-rs/actions)
[![codecov](https://codecov.io/gh/zephleggett/par2-rs/branch/main/graph/badge.svg)](https://codecov.io/gh/zephleggett/par2-rs)

A Rust implementation of PAR2 (Parchive 2.0) for file verification and repair using Reed-Solomon error correction over GF(2^16).

> **Alpha software.** Tested on Apple Silicon and x86 GitHub CI. Platform compatibility with diverse file types has not been extensively verified. Keep backups of important data.

## Building

```bash
git clone https://github.com/zephleggett/par2-rs
cd par2-rs
cargo build --release
```

Requires Rust 1.80.0 or later.

## Usage

### Command Line

**Verify and repair:**

```bash
cargo run --release --bin repair myfiles.par2
```

**Verify only:**

```bash
cargo run --release --bin repair myfiles.par2 verify
```

**Create PAR2 files:**

```bash
# 10% redundancy (default is 5%)
cargo run --release --bin create -r 10 file1.bin file2.bin

# Single volume instead of exponential distribution
cargo run --release --bin create -r 10 -s file1.bin file2.bin
```

**Enable logging:**

```bash
RUST_LOG=info cargo run --release --bin repair myfiles.par2
```

### Library

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

**With progress callback:**

```rust
use par2_rs::{Par2Repairer, Par2Operation};
use std::sync::Arc;

let progress = Arc::new(|op: Par2Operation, current: u64, total: u64| {
    let pct = current * 100 / total.max(1);
    match op {
        Par2Operation::Loading => println!("Loading: {}%", pct),
        Par2Operation::Verifying => println!("Verifying: {}%", pct),
        Par2Operation::Repairing => println!("Repairing: {}%", pct),
        _ => {}
    }
});

repairer.repair_with_progress(true, false, Some(progress))?;
```

## Performance

On ARM64 (Apple Silicon), repair performance is comparable to or faster than par2cmdline-turbo. x86 SIMD implementations (PCLMUL, AVX2, SSSE3) pass correctness tests but have not been benchmarked on physical hardware.

## Platform Support

| Platform | Status |
|----------|--------|
| ARM64 (Apple Silicon, Graviton) | NEON SIMD, tested |
| x86-64 (Intel, AMD) | Multiple SIMD paths, CI tested |
| Other | Scalar fallback |

## SIMD Strategies

The library automatically selects the best available SIMD implementation at runtime. Strategies are ranked by priority (highest wins).

### ARM64

| Strategy | Description |
|----------|-------------|
| NEON + PMULL | Polynomial multiplication with Karatsuba algorithm and Barrett reduction. Primary path on AArch64. |
| NEON table | Nibble-based shuffle table lookup. Fallback if PMULL unavailable. |

### x86-64

| Strategy | Priority | Requirements |
|----------|----------|--------------|
| AVX2-Shuffle | Optimal | AVX2. 256-bit nibble shuffle tables with shuffle2x format. |
| AVX2-PCLMUL | Optimal | AVX2 + PCLMUL. 256-bit carry-less multiplication. |
| PCLMUL | Advanced | PCLMUL + SSE4.1. 128-bit carry-less multiplication with Barrett reduction. |
| SSSE3 | Enhanced | SSSE3. Shuffle-based table lookup. |
| SSE2 | Basic | SSE2. Baseline x86-64 SIMD. |

When multiple strategies share the same priority, the first registered one wins. On most modern x86 CPUs, AVX2-Shuffle is selected.

## Features

- Parallel file processing via Rayon
- Multi-volume PAR2 support (auto-loads `.vol*.par2` files)
- Content-based file matching (works with renamed files)
- Memory-efficient streaming for large files
- PAR2 creation with configurable redundancy

## Specification Compliance

Implements the [PAR2 2.0 specification](https://parchive.sourceforge.net/docs/specifications/parity-volume-spec/article-spec.html):

**Supported:** Main packets, File Description packets, IFSC packets, Recovery Slice packets, multi-volume archives, MD5 verification, Reed-Solomon GF(2^16) with polynomial 0x1100B.

**Not implemented:** Creator packets, Unicode filename packets, Comment packets. These are optional metadata and don't affect interoperability.

## Testing

```bash
cargo test
```

See [tests/README.md](tests/README.md) for details.

## Contributing

Contributions welcome. Areas that could use work:

- **x86 benchmarking** - SIMD paths need performance testing on real Intel/AMD hardware
- **CLI improvements** - Better progress reporting and error messages
- **Unicode filenames** - Currently ASCII only

## Acknowledgments

SIMD Galois field arithmetic adapted from [reed-solomon-simd](https://github.com/AndersTrier/reed-solomon-simd) by Anders Trier Olesen, modified for PAR2's primitive polynomial.

Architectural patterns inspired by [ParPar](https://github.com/animetosho/ParPar) by Anime Tosho.

## License

MIT OR Apache-2.0
