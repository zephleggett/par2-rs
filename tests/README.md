# Testing

## Structure

The test suite is organized into focused modules:

- **galois.rs** - Galois field arithmetic (multiplication, division, powers)
- **repair.rs** - File verification and repair workflows
- **create.rs** - PAR2 file creation with different configurations
- **edge_cases.rs** - Error handling and boundary cases
- **end_to_end.rs** - Complete create/corrupt/repair scenarios
- **progress.rs** - Progress callback functionality
- **error_paths.rs** - Corrupted files and error conditions

Common utilities live in `common/mod.rs` for creating test files, corrupting data, and computing hashes.

## Running Tests

```bash
# Run everything (104 tests)
cargo test

# Specific test file
cargo test --test repair

# Watch mode for development
cargo watch -x test
```

## Coverage

Current coverage: 76% of core library. Use the coverage script to check:

```bash
./scripts/check_coverage.sh        # Terminal output
./scripts/check_coverage.sh --html # Open HTML report
```

Critical modules (repair, verify, creator, parser) have good coverage. Most uncovered code is platform-specific SIMD paths and rare error conditions.

## Writing Tests

Keep tests fast by using small files (5-10KB). Use `TempDir` for isolation. The common utilities make this easy:

```rust
use tempfile::TempDir;
use common::create_pattern_file;

#[test]
fn test_something() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("test.bin");
    create_pattern_file(&file, b"DATA", 10_000).unwrap();

    // Test your feature

    // TempDir cleans up automatically
}
```

Always verify actual data content (MD5 hashes) rather than just checking that functions don't error.
