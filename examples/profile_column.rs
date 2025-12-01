use par2_rs::galois;
use std::time::Instant;

fn main() {
    galois::init_tables();

    // Simulate a chunk from 1GB repair: 256KB u16 = 512KB bytes
    let chunk_size = 262144; // 256K u16 symbols
    let num_destinations = 8; // Typical batch size
    let num_iterations = 100; // Run multiple times for profiling

    println!("Initializing data...");
    let source = vec![0x1234u16; chunk_size];
    let mut destinations: Vec<Vec<u16>> = (0..num_destinations)
        .map(|_| vec![0u16; chunk_size])
        .collect();
    let coefficients: Vec<u16> = (0..num_destinations)
        .map(|i| (i as u16 + 1) * 0x1111)
        .collect();

    println!(
        "Running {} iterations of gf_muladd_column...",
        num_iterations
    );
    println!(
        "Chunk size: {} u16 ({} KB)",
        chunk_size,
        chunk_size * 2 / 1024
    );
    println!("Destinations: {}", num_destinations);

    let start = Instant::now();
    for _ in 0..num_iterations {
        let mut dest_refs: Vec<&mut [u16]> =
            destinations.iter_mut().map(|d| d.as_mut_slice()).collect();
        galois::gf_muladd_column(&mut dest_refs, &source, &coefficients);
    }
    let elapsed = start.elapsed();

    let total_bytes = chunk_size * 2 * num_destinations * num_iterations;
    let throughput = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000_000.0;

    println!("Time: {:?}", elapsed);
    println!("Throughput: {:.2} GB/s", throughput);
    println!(
        "Per iteration: {:.2} ms",
        elapsed.as_secs_f64() * 1000.0 / num_iterations as f64
    );

    // Prevent optimization of results
    let sum: u64 = destinations[0].iter().map(|&x| x as u64).sum();
    println!("Checksum: {}", sum);
}
