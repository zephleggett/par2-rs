# par2-rs

Pure Rust PAR2 file verification and repair library with SIMD optimizations.

## What is this?

PAR2 files let you verify and repair damaged downloads. This library implements the PAR2 format in pure Rust with no C++ dependencies, using runtime SIMD for performance (AVX2, NEON depending on your CPU).

## Installation

```toml
[dependencies]
par2-rs = "0.1"
```

## Quick Start

```rust
use par2_rs::Par2Repairer;

// Verify files
let repairer = Par2Repairer::new("file.par2")?;
repairer.repair(false)?; // just verify

// Or verify and repair if damaged
repairer.repair(true)?;
```

## Features

- Fast Reed-Solomon repair using O(n log n) algorithm
- Matches files by content hash (works with obfuscated Usenet filenames)
- Multi-volume PAR2 support (`.vol*.par2` files)
- Progress callbacks
- Cross-platform (Linux, macOS, Windows, ARM64)

## What's Implemented

Based on the [PAR 2.0 spec](https://parchive.sourceforge.net/docs/specifications/parity-volume-spec/article-spec.html):

- Main, File Description, and Recovery Slice packets
- MD5 verification (full file + 16KB quick-check)
- Reed-Solomon GF(2^16) error correction
- Multi-volume loading by recovery set ID

## What's Missing

- Input File Slice Checksum (IFSC) packets - we reconstruct directly from file blocks
- Creator, Unicode Filename, Comment packets - not critical for repair
- Packet checksum validation - should add this

## Performance

On Apple Silicon M1: ~500 MB/s verification, ~1.3 GB/s repair
On Intel/AMD AVX2: ~600 MB/s verification, ~1.5 GB/s repair

## License

MIT or Apache-2.0, your choice.

## Contributing

PRs welcome. Priority items:
- IFSC packet support (CRC32 slice verification)
- Packet MD5 checksum validation
- More tests with real PAR2 files
