// Benchmark comparing single-pass fused MD5+CRC32 vs sequential two-pass approach

use crc32fast::Hasher as Crc32Hasher;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use md5::{Digest, Md5};

/// Sequential two-pass approach: compute MD5, then compute CRC32
fn compute_sequential(data: &[u8]) -> ([u8; 16], u32) {
    // First pass: compute MD5
    let mut md5_hasher = Md5::new();
    md5_hasher.update(data);
    let md5: [u8; 16] = md5_hasher.finalize().into();

    // Second pass: compute CRC32
    let mut crc_hasher = Crc32Hasher::new();
    crc_hasher.update(data);
    let crc32 = crc_hasher.finalize();

    (md5, crc32)
}

/// Optimized: Single-pass fused MD5+CRC32
/// Data is read once, both hashers process it from cache
fn compute_fused(data: &[u8]) -> ([u8; 16], u32) {
    const CHUNK_SIZE: usize = 8192;

    let mut md5_hasher = Md5::new();
    let mut crc_hasher = Crc32Hasher::new();

    // Single pass - data read once
    for chunk in data.chunks(CHUNK_SIZE) {
        md5_hasher.update(chunk); // Data in L1 cache
        crc_hasher.update(chunk); // Gets from cache
    }

    let md5: [u8; 16] = md5_hasher.finalize().into();
    let crc32 = crc_hasher.finalize();

    (md5, crc32)
}

fn benchmark_hash_approaches(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_comparison");

    // Test with different data sizes typical for PAR2 blocks
    let sizes = vec![
        512 * 1024,      // 512KB (typical PAR2 block size)
        1024 * 1024,     // 1MB
        4 * 1024 * 1024, // 4MB
    ];

    for size in sizes {
        let data = vec![0x42u8; size];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("sequential", size), &data, |b, data| {
            b.iter(|| black_box(compute_sequential(black_box(data))));
        });

        group.bench_with_input(BenchmarkId::new("fused", size), &data, |b, data| {
            b.iter(|| black_box(compute_fused(black_box(data))));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_hash_approaches);
criterion_main!(benches);
