// Benchmark parallel hashing throughput (simulating actual verification workload)

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rayon::prelude::*;

/// Simulate real PAR2 verification: hash many blocks in parallel
fn parallel_hash_blocks(block_size: usize, num_blocks: usize) -> Vec<([u8; 16], u32)> {
    // Create blocks of data
    let data: Vec<Vec<u8>> = (0..num_blocks)
        .map(|i| vec![(i % 256) as u8; block_size])
        .collect();

    // Hash all blocks in parallel (like real verification does)
    data.par_iter()
        .map(|block| par2_rs::hash::compute_md5_crc32(block))
        .collect()
}

fn benchmark_parallel_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_hash");

    // Typical PAR2 scenario: 2000 blocks of 512KB each = 1GB total
    let block_size = 512 * 1024;
    let num_blocks = 2000;
    let total_bytes = block_size * num_blocks;

    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.sample_size(10); // Fewer samples since this is slow

    group.bench_function("parallel_2000_blocks_512kb", |b| {
        b.iter(|| {
            black_box(parallel_hash_blocks(
                black_box(block_size),
                black_box(num_blocks),
            ))
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_parallel_throughput);
criterion_main!(benches);
