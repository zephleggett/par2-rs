// Benchmark MD5 alone to find the bottleneck

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use md5::{Digest, Md5};

fn benchmark_md5_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("md5_standalone");

    let sizes = vec![
        512 * 1024,      // 512KB
        1024 * 1024,     // 1MB
        4 * 1024 * 1024, // 4MB
    ];

    for size in sizes {
        let data = vec![0x42u8; size];

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("md5", size), &data, |b, data| {
            b.iter(|| {
                let mut hasher = Md5::new();
                hasher.update(black_box(data));
                black_box(hasher.finalize())
            });
        });
    }

    group.finish();
}

use criterion::BenchmarkId;
criterion_group!(benches, benchmark_md5_only);
criterion_main!(benches);
