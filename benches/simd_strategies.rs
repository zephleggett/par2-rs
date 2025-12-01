use criterion::BatchSize;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use par2_rs::galois;
use par2_rs::galois::simd::GaloisSimdStrategy;

fn bench_gf_muladd_column(c: &mut Criterion) {
    galois::init_tables();

    let sizes = vec![16384, 65536, 262144]; // 32KB, 128KB, 512KB in u16
    let dest_counts = vec![1, 4, 8];

    let mut group = c.benchmark_group("gf_muladd_column");

    for size in &sizes {
        for &num_dests in &dest_counts {
            let source = vec![0x1234u16; *size];
            let mut destinations: Vec<Vec<u16>> =
                (0..num_dests).map(|_| vec![0u16; *size]).collect();
            let coefficients: Vec<u16> = (0..num_dests).map(|i| (i as u16 + 1) * 0x1111).collect();

            // Rough throughput: num_dests * size symbols (2 bytes each)
            let bytes = (*size as u64) * 2 * (num_dests as u64);
            group.throughput(Throughput::Bytes(bytes));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}u16_{}dests", size, num_dests)),
                &(size, num_dests),
                |b, _| {
                    b.iter(|| {
                        let mut dest_refs: Vec<&mut [u16]> =
                            destinations.iter_mut().map(|d| d.as_mut_slice()).collect();
                        galois::gf_muladd_column(
                            black_box(&mut dest_refs),
                            black_box(&source),
                            black_box(&coefficients),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_gf_mul_slice(c: &mut Criterion) {
    galois::init_tables();

    let sizes = vec![16384, 65536, 262144]; // 32KB, 128KB, 512KB in u16

    let mut group = c.benchmark_group("gf_mul_slice");

    for size in &sizes {
        let mut data = vec![0x1234u16; *size];
        let scalar = 0x9876u16;

        group.throughput(Throughput::Bytes((*size as u64) * 2));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}u16", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    galois::gf_mul_slice(black_box(scalar), black_box(&mut data));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark each available SIMD strategy for mul_slice
fn bench_strategies_mul_slice(c: &mut Criterion) {
    galois::init_tables();

    let size = 65536usize; // 128KB
    let scalar = 0x9876u16;

    let mut group = c.benchmark_group("strategy_mul_slice");
    group.throughput(Throughput::Bytes((size as u64) * 2));

    // Collect strategies
    let strategies = get_all_strategies();

    for (name, strategy) in strategies {
        group.bench_with_input(BenchmarkId::from_parameter(name), &size, |b, _| {
            b.iter_batched(
                || vec![0x1234u16; size],
                |mut data| unsafe {
                    strategy.mul_slice(black_box(scalar), black_box(&mut data));
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Benchmark each available SIMD strategy for muladd
fn bench_strategies_muladd(c: &mut Criterion) {
    galois::init_tables();

    let size = 65536usize; // 128KB
    let scalar = 0x9876u16;

    let mut group = c.benchmark_group("strategy_muladd");
    group.throughput(Throughput::Bytes((size as u64) * 2));

    // Collect strategies
    let strategies = get_all_strategies();

    for (name, strategy) in strategies {
        let src: Vec<u16> = (0..size).map(|i| (i * 13 + 7) as u16).collect();

        group.bench_with_input(BenchmarkId::from_parameter(name), &size, |b, _| {
            b.iter_batched(
                || vec![0x1234u16; size],
                |mut dst| unsafe {
                    strategy.muladd(black_box(&mut dst), black_box(&src), black_box(scalar));
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Helper function to get all available strategies for benchmarking
#[allow(unused_mut)] // mut needed on x86_64 but not on other platforms
fn get_all_strategies() -> Vec<(&'static str, Box<dyn GaloisSimdStrategy>)> {
    let mut strategies: Vec<(&'static str, Box<dyn GaloisSimdStrategy>)> = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        use par2_rs::galois::simd::x86::*;

        // SSE strategies
        let sse2 = sse::Sse2Strategy;
        if sse2.is_available() {
            strategies.push((sse2.name(), Box::new(sse2)));
        }

        let ssse3 = sse::Ssse3Strategy;
        if ssse3.is_available() {
            strategies.push((ssse3.name(), Box::new(ssse3)));
        }

        // AVX2 strategies
        let avx2_shuffle = avx2::Avx2ShuffleStrategy;
        if avx2_shuffle.is_available() {
            strategies.push((avx2_shuffle.name(), Box::new(avx2_shuffle)));
        }

        // PCLMUL strategies
        let pclmul = pclmul::PclmulStrategy;
        if pclmul.is_available() {
            strategies.push((pclmul.name(), Box::new(pclmul)));
        }

        let avx2_pclmul = pclmul::Avx2PclmulStrategy;
        if avx2_pclmul.is_available() {
            strategies.push((avx2_pclmul.name(), Box::new(avx2_pclmul)));
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON uses direct dispatch, not registry
        // Add ARM-specific strategy if needed
    }

    strategies
}

/// Benchmark native shuffle2x muladd vs regular muladd
///
/// This compares the performance when data is pre-converted to shuffle2x format
/// versus on-the-fly format conversion.
fn bench_shuffle2x_muladd(c: &mut Criterion) {
    galois::init_tables();

    let size = 65536usize; // 128KB
    let scalar = 0x9876u16;

    let mut group = c.benchmark_group("shuffle2x_muladd");
    group.throughput(Throughput::Bytes((size as u64) * 2));

    // Get strategies that support shuffle2x
    let strategies = get_all_strategies();

    for (name, strategy) in strategies {
        if !strategy.supports_shuffle2x() {
            continue;
        }

        // Prepare source data in shuffle2x format once
        let mut src_s2x: Vec<u16> = (0..size).map(|i| (i * 13 + 7) as u16).collect();
        unsafe {
            strategy.prepare_shuffle2x(&mut src_s2x);
        }

        // Benchmark with pre-converted data (native shuffle2x)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_native_s2x", name)),
            &size,
            |b, _| {
                b.iter_batched(
                    || {
                        // Prepare destination in shuffle2x format
                        let mut dst = vec![0x1234u16; size];
                        unsafe {
                            strategy.prepare_shuffle2x(&mut dst);
                        }
                        dst
                    },
                    |mut dst| unsafe {
                        strategy.muladd_shuffle2x(
                            black_box(&mut dst),
                            black_box(&src_s2x),
                            black_box(scalar),
                        );
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Benchmark regular muladd (with on-the-fly conversion)
        let src_regular: Vec<u16> = (0..size).map(|i| (i * 13 + 7) as u16).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_regular", name)),
            &size,
            |b, _| {
                b.iter_batched(
                    || vec![0x1234u16; size],
                    |mut dst| unsafe {
                        strategy.muladd(
                            black_box(&mut dst),
                            black_box(&src_regular),
                            black_box(scalar),
                        );
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gf_muladd_column,
    bench_gf_mul_slice,
    bench_strategies_mul_slice,
    bench_strategies_muladd,
    bench_shuffle2x_muladd
);
criterion_main!(benches);
