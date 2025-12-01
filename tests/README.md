# Testing

## Structure

| Module | Description |
|--------|-------------|
| galois.rs | GF(2^16) arithmetic |
| verify.rs | Block-level verification |
| repair.rs | File repair workflows |
| create.rs | PAR2 creation |
| end_to_end.rs | Full create/corrupt/repair |
| edge_cases.rs | Boundary conditions |
| error_paths.rs | Error handling |
| progress.rs | Progress callbacks |

Utilities in `common/mod.rs`.

## Commands

```bash
cargo test                    # All tests
cargo test --test repair      # Single module
./scripts/check_coverage.sh   # Coverage report
```

## Guidelines

- Small files (5-10KB) with `TempDir`
- Verify MD5 hashes, not just success
