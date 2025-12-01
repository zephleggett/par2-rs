//! PAR2-specific Reed-Solomon implementation
//!
//! Uses GF(2^16) with primitive polynomial 0x1100B (x^16 + x^12 + x^3 + x + 1)
//! as specified in the PAR2 standard. Implements Vandermonde matrix construction
//! for encoding and Gaussian elimination for decoding/repair.
//!
//! # Key Types
//!
//! - [`Par2ReedSolomon`] - Main encoder/decoder with matrix operations

use crate::galois::region::{gf_muladd_block_regions, gf_muladd_block_regions_shuffle2x};
use crate::galois::{
    self, gf_mul, gf_muladd, gf_muladd_column, gf_muladd_column_shuffle2x, gf_pow,
};
use rayon::prelude::*;

/// SAFETY: Caller must ensure the returned slice does not alias with any
/// other mutable reference and that `len` does not exceed the capacity of `vec`.
#[inline]
unsafe fn as_mut_slice_unchecked<T>(vec: &mut Vec<T>, len: usize) -> &mut [T] {
    std::slice::from_raw_parts_mut(vec.as_mut_ptr(), len)
}

/// Generate PAR2 constants - powers of 2 with special exponents
/// PAR2 spec: "the first constant is the first power of two that has order 65535"
/// Valid exponents n satisfy: n%3 != 0 && n%5 != 0 && n%17 != 0 && n%257 != 0
fn generate_par2_constants(count: usize) -> Vec<u16> {
    // Per spec: use powers of 2 whose exponent has order 65535
    // i.e. n % 3 != 0 && n % 5 != 0 && n % 17 != 0 && n % 257 != 0
    let mut constants = Vec::with_capacity(count);
    let mut n: u32 = 1; // Start from 1, not 0 (0^n would be invalid)
    while constants.len() < count {
        if n % 3 != 0 && n % 5 != 0 && n % 17 != 0 && n % 257 != 0 {
            constants.push(gf_pow(2, n as usize));
        }
        n += 1;
    }
    constants
}

/// Precomputed Gaussian elimination transformation for reconstruction
/// This is computed once and reused for all chunks to avoid redundant work
pub struct ReconstructionTransform {
    pub row_order: Vec<usize>,
    pub scale_factors: Vec<u16>,
    pub elimination_factors: Vec<Vec<u16>>,
    pub vandermonde_coeffs: Vec<Vec<u16>>,
}

/// Calculate optimal data batch size based on system characteristics
fn optimal_data_batch_size(recovery_blocks: usize, _block_size: usize) -> usize {
    // Use available_parallelism() for accurate container/cgroup detection
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Target enough work per thread to amortize synchronization overhead
    // More recovery blocks = more work per data block = can use larger batches
    let base_batch = if recovery_blocks >= 100 {
        // Many recovery blocks: larger batches for better cache locality
        num_cpus * 8
    } else if recovery_blocks >= 50 {
        num_cpus * 4
    } else {
        // Few recovery blocks: smaller batches to maximize parallelism
        num_cpus * 2
    };

    // Clamp to reasonable range
    base_batch.clamp(16, 128)
}

/// Calculate optimal recovery batch size for parallel processing
fn optimal_recovery_batch_size(recovery_blocks: usize) -> usize {
    // Use available_parallelism() for accurate container/cgroup detection
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Want at least num_cpus parallel work units
    // recovery_blocks / batch_size >= num_cpus
    // batch_size <= recovery_blocks / num_cpus
    let max_batch = (recovery_blocks / num_cpus).max(1);

    // But also want batches big enough to amortize overhead
    // Minimum 4 recovery blocks per batch for efficiency
    // Maximum 8 due to SIMD register constraints in gf_muladd_column
    max_batch.clamp(4, 8)
}

/// Streaming encoder for PAR2 creation
/// Processes data in chunks to minimize memory usage while maintaining performance
pub struct StreamingEncoder {
    /// Number of data blocks to encode
    data_blocks: usize,
    /// Size of each block in bytes
    block_size: usize,
    /// Size of each block in u16 symbols
    symbols_per_block: usize,
    /// Precomputed coefficient matrix [data_idx][recovery_idx]
    coeff_matrix: Vec<Vec<u16>>,
    /// Accumulated recovery block data (recovery_blocks × block_size)
    recovery_data: Vec<Vec<u16>>,
    /// Whether recovery blocks are stored in shuffle2x format
    use_shuffle2x: bool,
    /// Pre-allocated buffers for batch processing (reused to avoid allocation churn)
    /// Each buffer holds one data block in shuffle2x format
    batch_buffers: Vec<Vec<u16>>,
    /// Indices of data blocks currently in batch_buffers
    batch_indices: Vec<usize>,
    /// Dynamic batch size for data blocks
    data_batch_size: usize,
    /// Dynamic batch size for recovery blocks
    recovery_batch_size: usize,
}

/// PAR2 Reed-Solomon decoder
pub struct Par2ReedSolomon {
    data_blocks: usize,
    constants: Vec<u16>,
}

impl Par2ReedSolomon {
    /// Create a new PAR2 Reed-Solomon codec
    pub fn new(data_blocks: usize, _recovery_blocks: usize) -> Self {
        galois::init_tables();
        let constants = generate_par2_constants(data_blocks);

        // Debug: Log first few constants to verify they match the spec
        if data_blocks > 0 {
            tracing::debug!(
                constants = ?&constants[0..constants.len().min(10)],
                "Generated PAR2 constants"
            );
        }

        Self {
            data_blocks,
            constants,
        }
    }

    /// Generate recovery blocks from input data blocks
    ///
    /// This is the encoding operation: B = A*X where:
    /// - A: Matrix of PAR2 constants raised to recovery exponents
    /// - X: Input data blocks
    /// - B: Recovery blocks (output)
    ///
    /// Much simpler than reconstruction since we're just doing matrix-vector multiplication,
    /// not solving a linear system.
    ///
    /// # Arguments
    /// * `data_blocks` - Input data blocks (all must be present)
    /// * `num_recovery` - Number of recovery blocks to generate
    /// * `block_size` - Size of each block in bytes (must be multiple of 2)
    ///
    /// # Returns
    /// Vector of recovery blocks with exponents 0, 1, 2, ... num_recovery-1
    #[allow(dead_code)]
    pub fn encode(
        &self,
        data_blocks: &[Vec<u8>],
        num_recovery: usize,
        block_size: usize,
    ) -> Result<Vec<(u32, Vec<u8>)>, String> {
        use std::time::Instant;
        let start = Instant::now();

        if data_blocks.len() != self.data_blocks {
            return Err(format!(
                "Expected {} data blocks, got {}",
                self.data_blocks,
                data_blocks.len()
            ));
        }

        if block_size % 2 != 0 {
            return Err("Block size must be multiple of 2".to_string());
        }

        let symbols = block_size / 2;
        tracing::info!(
            data_blocks = self.data_blocks,
            recovery_blocks = num_recovery,
            block_size,
            symbols_per_block = symbols,
            "Encoding PAR2 recovery blocks"
        );

        // OPTIMIZATION 1: Pre-convert ALL data blocks to u16 ONCE
        let convert_start = Instant::now();
        let mut data_blocks_u16: Vec<Vec<u16>> = Vec::with_capacity(data_blocks.len());
        for data_block in data_blocks {
            if data_block.len() != block_size {
                return Err(format!(
                    "Data block size mismatch: expected {}, got {}",
                    block_size,
                    data_block.len()
                ));
            }
            let u16_data: Vec<u16> = data_block
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            data_blocks_u16.push(u16_data);
        }
        tracing::info!(
            duration_secs = convert_start.elapsed().as_secs_f64(),
            blocks = data_blocks_u16.len(),
            "Converted data blocks to u16"
        );

        // OPTIMIZATION 2 & 3: Parallel column-wise processing with precomputed coefficients
        // Precompute all coefficients to avoid repeated gf_pow calls
        let encode_start = Instant::now();

        // Precompute coefficient matrix: coeff[data_idx][recovery_idx] = base[data_idx]^recovery_idx
        // Parallelize this since we have 2000 data blocks
        let coeff_matrix: Vec<Vec<u16>> = self
            .constants
            .par_iter()
            .map(|&base| {
                let mut coeffs = Vec::with_capacity(num_recovery);
                let mut coeff = 1u16; // base^0
                for _ in 0..num_recovery {
                    coeffs.push(coeff);
                    coeff = gf_mul(coeff, base); // Fast: base^(i+1) = base^i * base
                }
                coeffs
            })
            .collect();

        // Process recovery blocks in parallel
        let recovery_blocks: Vec<Vec<u16>> = (0..num_recovery)
            .into_par_iter()
            .map(|recovery_idx| {
                let mut recovery_block = vec![0u16; symbols];

                // Process each data block's contribution to this recovery block
                for (data_idx, data_u16) in data_blocks_u16.iter().enumerate() {
                    let coeff = coeff_matrix[data_idx][recovery_idx];

                    // Use SIMD-accelerated multiply-add: recovery[i] ^= data[i] * coeff
                    gf_muladd(&mut recovery_block, data_u16, coeff);
                }

                recovery_block
            })
            .collect();

        tracing::info!(
            duration_secs = encode_start.elapsed().as_secs_f64(),
            data_blocks = data_blocks.len(),
            recovery_blocks = num_recovery,
            total_ops = data_blocks.len() * num_recovery,
            "Encoding complete"
        );

        // Convert recovery blocks back to bytes
        let mut result: Vec<(u32, Vec<u8>)> = Vec::with_capacity(num_recovery);
        for (exponent, recovery_u16) in recovery_blocks.into_iter().enumerate() {
            let mut recovery_bytes = Vec::with_capacity(block_size);
            for &val in &recovery_u16 {
                recovery_bytes.extend_from_slice(&val.to_le_bytes());
            }
            result.push((exponent as u32, recovery_bytes));
        }

        tracing::info!(
            duration_secs = start.elapsed().as_secs_f64(),
            "Total encoding time"
        );

        Ok(result)
    }

    /// Compute Gaussian elimination transformation once for all chunks
    ///
    /// This function performs the expensive O(m³) Gaussian elimination setup
    /// that is independent of chunk data. The result can be reused for all chunks.
    pub fn compute_reconstruction_transform(
        &self,
        missing_indices: &[usize],
        present_data_indices: &[usize],
        present_recovery_indices: &[usize],
        recovery_exponents: &[u32],
    ) -> Result<ReconstructionTransform, String> {
        let m = missing_indices.len();
        if m == 0 {
            return Err("No missing blocks to reconstruct".into());
        }

        if present_recovery_indices.len() < m {
            return Err(format!(
                "Need {} recovery blocks, have {}",
                m,
                present_recovery_indices.len()
            ));
        }

        tracing::debug!(
            missing = m,
            present_data = present_data_indices.len(),
            "Computing Gaussian elimination transformation (ONCE for all chunks)"
        );

        // Build Vandermonde matrix for missing blocks
        #[allow(non_snake_case)]
        let mut A: Vec<Vec<u16>> = vec![vec![0u16; m]; m];
        for (row, &rec_idx) in present_recovery_indices.iter().take(m).enumerate() {
            let exponent = recovery_exponents[rec_idx - self.data_blocks] as usize;
            for (col, &miss_idx) in missing_indices.iter().enumerate() {
                A[row][col] = gf_pow(self.constants[miss_idx], exponent);
            }
        }

        // Track transformation operations for later application
        let mut row_order: Vec<usize> = (0..m).collect();
        let mut scale_factors: Vec<u16> = vec![1; m];
        let mut elimination_factors: Vec<Vec<u16>> = vec![vec![0; m]; m];

        // Perform Gaussian elimination to compute transformation
        for k in 0..m {
            if A[k][k] == 0 {
                if let Some(r) = ((k + 1)..m).find(|&r| A[r][k] != 0) {
                    A.swap(k, r);
                    row_order.swap(k, r);
                    scale_factors.swap(k, r);
                    elimination_factors.swap(k, r);
                } else {
                    return Err("Singular matrix".into());
                }
            }

            let pivot = A[k][k];
            scale_factors[k] = pivot;
            if pivot != 1 {
                #[allow(clippy::needless_range_loop)]
                for j in k..m {
                    if A[k][j] != 0 {
                        // Safe: pivot is non-zero (we checked pivot != 1, and 0 != 1)
                        A[k][j] = galois::gf_div(A[k][j], pivot).unwrap();
                    }
                }
            }

            for r in 0..m {
                if r == k {
                    continue;
                }
                let factor = A[r][k];
                elimination_factors[r][k] = factor;
                if factor == 0 {
                    continue;
                }
                #[allow(clippy::needless_range_loop)]
                for j in k..m {
                    if A[k][j] != 0 {
                        A[r][j] ^= galois::gf_mul(A[k][j], factor);
                    }
                }
            }
        }

        // Pre-compute vandermonde coefficients for present data blocks
        let mut vandermonde_coeffs: Vec<Vec<u16>> = Vec::with_capacity(m);
        for &rec_idx in present_recovery_indices.iter().take(m) {
            let exponent = recovery_exponents[rec_idx - self.data_blocks] as usize;
            let mut row_coeffs = Vec::with_capacity(present_data_indices.len());
            for &data_idx in present_data_indices {
                row_coeffs.push(gf_pow(self.constants[data_idx], exponent));
            }
            vandermonde_coeffs.push(row_coeffs);
        }

        tracing::debug!("Gaussian elimination transformation computed");

        Ok(ReconstructionTransform {
            row_order,
            scale_factors,
            elimination_factors,
            vandermonde_coeffs,
        })
    }

    /// Reconstruct missing blocks using streaming architecture
    ///
    /// This processes ONE BLOCK AT A TIME, never loading all blocks into memory.
    /// Memory usage: (m + overhead) × chunk_size instead of n × chunk_size
    ///
    /// # Arguments
    /// * `missing_indices` - Indices of missing data blocks
    /// * `present_data_indices` - Indices of present data blocks
    /// * `present_recovery_indices` - Indices of present recovery blocks (must have at least m)
    /// * `transform` - Precomputed Gaussian elimination transformation
    /// * `chunk_offset` - Byte offset within block
    /// * `chunk_size` - Size of chunk to process
    /// * `read_block` - Callback to read a chunk from a block: (block_idx, offset, size) -> data
    /// * `write_result` - Callback to write reconstructed chunk: (missing_block_idx, data) -> ()
    ///
    /// # Returns
    /// Ok(()) if reconstruction successful
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct_streaming_chunk<R, W>(
        &self,
        missing_indices: &[usize],
        present_data_indices: &[usize],
        present_recovery_indices: &[usize],
        transform: &ReconstructionTransform,
        chunk_offset: usize,
        chunk_size: usize,
        mut read_block: R,
        mut write_result: W,
    ) -> Result<(), String>
    where
        R: FnMut(usize, usize, usize) -> Result<Vec<u8>, String>,
        W: FnMut(usize, Vec<u8>) -> Result<(), String>,
    {
        let m = missing_indices.len();
        if m == 0 {
            return Ok(());
        }

        if present_recovery_indices.len() < m {
            return Err(format!(
                "Need {} recovery blocks, have {}",
                m,
                present_recovery_indices.len()
            ));
        }

        let chunk_symbols = chunk_size / 2;

        tracing::debug!(
            missing = m,
            present_data = present_data_indices.len(),
            chunk_offset = chunk_offset,
            chunk_size = chunk_size,
            "Starting streaming reconstruction for chunk"
        );

        // Use precomputed transformation matrices
        let row_order = &transform.row_order;
        let scale_factors = &transform.scale_factors;
        let elimination_factors = &transform.elimination_factors;
        let vandermonde_coeffs = &transform.vandermonde_coeffs;

        let mut accumulators: Vec<Vec<u16>> = vec![vec![0u16; chunk_symbols]; m];

        for (original_row, &rec_idx) in present_recovery_indices.iter().take(m).enumerate() {
            let rec_chunk = read_block(rec_idx, chunk_offset, chunk_size)?;

            let swapped_row = row_order
                .iter()
                .position(|&r| r == original_row)
                .ok_or_else(|| format!("Row {} not found in row_order", original_row))?;

            // Convert bytes to u16 using SIMD
            galois::bytes_to_u16_simd(&rec_chunk, &mut accumulators[swapped_row]);
        }

        // Check if shuffle2x format is supported for faster GF operations (~55% speedup)
        let use_shuffle2x = galois::supports_shuffle2x();

        // Convert accumulators to shuffle2x format if supported
        if use_shuffle2x {
            for acc in accumulators.iter_mut() {
                galois::prepare_shuffle2x(acc);
            }
        }

        // PHASE 2: Stream through present data blocks using region-batched SIMD operations
        // Pre-filter present data columns to only those that are used by any missing row.
        // This can skip work if the elimination produced zeros for some columns.
        // Store (data_idx, col_pos) tuples to avoid HashMap lookup later
        let mut used_data_cols: Vec<(usize, usize)> =
            Vec::with_capacity(present_data_indices.len());
        for (data_col, &data_idx) in present_data_indices.iter().enumerate() {
            let any_nz = vandermonde_coeffs
                .iter()
                .take(m)
                .any(|row_coeffs| row_coeffs[data_col] != 0);
            if any_nz {
                used_data_cols.push((data_idx, data_col));
            }
        }

        // Process sources in batches to maximize cache/SIMD reuse
        const SRC_BATCH: usize = 8; // tuned for x86 register pressure
        let mut data_chunks_u16: Vec<Vec<u16>> = Vec::with_capacity(SRC_BATCH);

        let mut batch_start_col = 0usize;
        while batch_start_col < used_data_cols.len() {
            let batch_end_col = (batch_start_col + SRC_BATCH).min(used_data_cols.len());
            let batch_len = batch_end_col - batch_start_col;

            // Read batch of sources; do not perform zero-detection (include all)
            data_chunks_u16.clear();
            // Store col_pos for each item in this batch (avoids HashMap lookup)
            let mut batch_col_positions: Vec<usize> = Vec::with_capacity(batch_len);
            for &(data_idx, col_pos) in &used_data_cols[batch_start_col..batch_end_col] {
                let data_chunk_bytes = read_block(data_idx, chunk_offset, chunk_size)?;

                // Convert to u16 using SIMD and keep
                let mut buf = vec![0u16; chunk_symbols];
                galois::bytes_to_u16_simd(&data_chunk_bytes, &mut buf);

                // Convert to shuffle2x format if supported
                if use_shuffle2x {
                    galois::prepare_shuffle2x(&mut buf);
                }

                data_chunks_u16.push(buf);
                batch_col_positions.push(col_pos);
            }

            // Build region inputs
            let mut sources: Vec<&[u16]> = Vec::with_capacity(data_chunks_u16.len());
            for chunk in &data_chunks_u16 {
                sources.push(&chunk[..]);
            }

            // Coefficients matrix per destination row for this batch
            let mut coeff_rows: Vec<Vec<u16>> = Vec::with_capacity(m);
            for vand_row in vandermonde_coeffs.iter().take(m) {
                let mut row_coeffs = Vec::with_capacity(batch_col_positions.len());
                for &col_pos in &batch_col_positions {
                    row_coeffs.push(vand_row[col_pos]);
                }
                coeff_rows.push(row_coeffs);
            }
            // coeff_rows are consumed below via slices per row-chunk; no separate refs needed here.

            // Process all destination rows sequentially - outer parallelism handles chunk-level
            // parallelism, so nested par_chunks_mut here causes thread contention overhead

            // Build destination refs for all rows
            let mut dst_refs: Vec<&mut [u16]> = accumulators
                .iter_mut()
                .map(|row| row.as_mut_slice())
                .collect();

            // Build coefficient refs for all rows
            let coeff_refs: Vec<&[u16]> = coeff_rows.iter().map(|r| r.as_slice()).collect();

            // Perform cache-optimized region processing
            // Automatically subdivides into 128KB regions for L2 cache efficiency
            if !sources.is_empty() {
                if use_shuffle2x {
                    // Use faster shuffle2x path (~55% faster on x86 AVX2)
                    gf_muladd_block_regions_shuffle2x(
                        &mut dst_refs,
                        &sources,
                        &coeff_refs,
                        chunk_symbols,
                    );
                } else {
                    gf_muladd_block_regions(&mut dst_refs, &sources, &coeff_refs, chunk_symbols);
                }
            }

            batch_start_col = batch_end_col;
        }

        // Convert accumulators back from shuffle2x to interleaved format before Gaussian elimination
        if use_shuffle2x {
            for acc in accumulators.iter_mut() {
                galois::finish_shuffle2x(acc);
            }
        }

        // PHASE 4: Apply pre-computed Gaussian elimination to accumulators
        for k in 0..m {
            let pivot = scale_factors[k];
            if pivot != 1 {
                #[allow(clippy::needless_range_loop)]
                for s in 0..chunk_symbols {
                    if accumulators[k][s] != 0 {
                        // Safe: pivot is non-zero (we checked pivot != 1, and 0 != 1)
                        accumulators[k][s] = galois::gf_div(accumulators[k][s], pivot).unwrap();
                    }
                }
            }

            for r in 0..k {
                let factor = elimination_factors[r][k];
                if factor != 0 {
                    let (first, second) = accumulators.split_at_mut(k);
                    galois::gf_muladd(
                        &mut first[r][..chunk_symbols],
                        &second[0][..chunk_symbols],
                        factor,
                    );
                }
            }

            #[allow(clippy::needless_range_loop)]
            // Need r index for both elimination_factors[r] and split_at_mut(r)
            for r in (k + 1)..m {
                let factor = elimination_factors[r][k];
                if factor != 0 {
                    let (first, second) = accumulators.split_at_mut(r);
                    galois::gf_muladd(
                        &mut second[0][..chunk_symbols],
                        &first[k][..chunk_symbols],
                        factor,
                    );
                }
            }
        }

        // PHASE 5: Write results via callback
        for (row, &miss_idx) in missing_indices.iter().enumerate() {
            let mut output = Vec::with_capacity(chunk_size);
            for &val in &accumulators[row] {
                output.extend_from_slice(&val.to_le_bytes());
            }
            write_result(miss_idx, output)?;
        }

        Ok(())
    }

    /// Create a streaming encoder for memory-efficient PAR2 creation
    ///
    /// This allows processing large files in small chunks without loading everything into memory.
    /// The encoder accumulates recovery blocks progressively as chunks are processed.
    ///
    /// # Arguments
    /// * `data_blocks` - Total number of data blocks in the file set
    /// * `recovery_blocks` - Number of recovery blocks to generate
    /// * `block_size` - Size of each block in bytes (must be multiple of 2)
    /// * `chunk_size` - Size of each chunk for streaming (must be multiple of 2, typically 100KB)
    pub fn create_streaming_encoder(
        &self,
        recovery_blocks: usize,
        block_size: usize,
        chunk_size: usize,
    ) -> Result<StreamingEncoder, String> {
        if block_size % 2 != 0 {
            return Err("Block size must be multiple of 2".to_string());
        }
        if chunk_size % 2 != 0 {
            return Err("Chunk size must be multiple of 2".to_string());
        }
        if chunk_size > block_size {
            return Err("Chunk size cannot exceed block size".to_string());
        }

        let num_chunks = block_size.div_ceil(chunk_size);
        let symbols_per_block = block_size / 2;

        // Precompute coefficient matrix in parallel (same as encoding)
        let coeff_matrix: Vec<Vec<u16>> = self
            .constants
            .par_iter()
            .map(|&base| {
                let mut coeffs = Vec::with_capacity(recovery_blocks);
                let mut coeff = 1u16;
                for _ in 0..recovery_blocks {
                    coeffs.push(coeff);
                    coeff = gf_mul(coeff, base);
                }
                coeffs
            })
            .collect();

        // Initialize recovery block accumulators (all zeros)
        let mut recovery_data = vec![vec![0u16; symbols_per_block]; recovery_blocks];

        // Check if shuffle2x format is supported for faster encoding
        let use_shuffle2x = galois::supports_shuffle2x();

        // Convert recovery blocks to shuffle2x format upfront if supported
        // This avoids repeated conversions during processing
        if use_shuffle2x {
            for recovery_block in &mut recovery_data {
                galois::prepare_shuffle2x(recovery_block);
            }
        }

        // Calculate optimal batch sizes for this system
        let data_batch_size = optimal_data_batch_size(recovery_blocks, block_size);
        let recovery_batch_size = optimal_recovery_batch_size(recovery_blocks);

        // Pre-allocate buffers for batch processing (reused to avoid allocation churn)
        let batch_buffers: Vec<Vec<u16>> = (0..data_batch_size)
            .map(|_| vec![0u16; symbols_per_block])
            .collect();
        let batch_indices = Vec::with_capacity(data_batch_size);

        tracing::info!(
            data_blocks = self.data_blocks,
            recovery_blocks,
            block_size,
            chunk_size,
            num_chunks,
            use_shuffle2x,
            data_batch_size,
            recovery_batch_size,
            num_cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            "Created streaming encoder with adaptive batch sizes"
        );

        Ok(StreamingEncoder {
            data_blocks: self.data_blocks,
            block_size,
            symbols_per_block,
            coeff_matrix,
            recovery_data,
            use_shuffle2x,
            batch_buffers,
            batch_indices,
            data_batch_size,
            recovery_batch_size,
        })
    }
}

impl StreamingEncoder {
    /// Process a data block using batched processing for better cache efficiency
    ///
    /// Data blocks are buffered and processed together in batches. This improves
    /// cache efficiency by keeping recovery blocks in cache while processing
    /// multiple data blocks, instead of cycling through all recovery blocks
    /// for each data block individually.
    ///
    /// If shuffle2x is supported, the source data is converted once and
    /// the high-performance shuffle2x path is used (~55% faster on x86 AVX2).
    pub fn process_block(&mut self, data_block_idx: usize, block_data_u16: &[u16]) {
        if data_block_idx >= self.data_blocks {
            return;
        }

        let batch_slot = self.batch_indices.len();

        // Copy data to pre-allocated buffer and convert to shuffle2x if supported
        let buffer = &mut self.batch_buffers[batch_slot];
        buffer[..block_data_u16.len()].copy_from_slice(block_data_u16);
        if self.use_shuffle2x {
            galois::prepare_shuffle2x(&mut buffer[..block_data_u16.len()]);
        }

        // Track which data block is in this slot
        self.batch_indices.push(data_block_idx);

        // Flush when batch is full
        if self.batch_indices.len() >= self.data_batch_size {
            self.flush_batch();
        }
    }

    /// Flush the buffered data blocks, processing them together for cache efficiency
    ///
    /// This implements the cache-optimized pattern:
    /// ```text
    /// for each recovery_block_batch:      // Stays in cache
    ///     for each data_block in batch:   // Process all together
    ///         recovery ^= data * coeff
    /// ```
    fn flush_batch(&mut self) {
        if self.batch_indices.is_empty() {
            return;
        }

        let use_shuffle2x = self.use_shuffle2x;
        let symbols_per_block = self.symbols_per_block;
        let recovery_batch_size = self.recovery_batch_size;

        // Get references for use in parallel closure
        let batch_indices = &self.batch_indices;
        let batch_buffers = &self.batch_buffers;
        let coeff_matrix = &self.coeff_matrix;

        // Process recovery blocks in parallel batches
        // Each thread processes a batch of recovery blocks against ALL data blocks in our batch
        self.recovery_data
            .par_chunks_mut(recovery_batch_size)
            .enumerate()
            .for_each(|(batch_idx, recovery_batch)| {
                let batch_start = batch_idx * recovery_batch_size;

                // Process each data block in our batch against this recovery batch
                for (slot, &data_idx) in batch_indices.iter().enumerate() {
                    let data_block = &batch_buffers[slot][..symbols_per_block];
                    let coeffs = &coeff_matrix[data_idx];

                    // Build batch of destination references and coefficients
                    let mut destinations: Vec<&mut [u16]> =
                        Vec::with_capacity(recovery_batch.len());
                    let mut batch_coeffs: Vec<u16> = Vec::with_capacity(recovery_batch.len());

                    for (offset, recovery_block) in recovery_batch.iter_mut().enumerate() {
                        let idx = batch_start + offset;
                        batch_coeffs.push(coeffs[idx]);

                        // SAFETY: Each recovery block is distinct in the batch
                        unsafe {
                            destinations
                                .push(as_mut_slice_unchecked(recovery_block, data_block.len()));
                        }
                    }

                    // Use shuffle2x-aware column operation for better throughput
                    if use_shuffle2x {
                        gf_muladd_column_shuffle2x(&mut destinations, data_block, &batch_coeffs);
                    } else {
                        gf_muladd_column(&mut destinations, data_block, &batch_coeffs);
                    }
                }
            });

        // Clear the batch indices for next batch (buffers are reused)
        self.batch_indices.clear();
    }

    /// Finalize and return the recovery blocks
    ///
    /// Flushes any remaining buffered data blocks, then converts the accumulated
    /// u16 recovery data back to bytes. If shuffle2x was used, converts back to
    /// interleaved format first.
    /// This consumes the encoder.
    ///
    /// Returns: Vec<(exponent, data)> where exponent is the recovery block index
    pub fn finalize(mut self) -> Vec<(u32, Vec<u8>)> {
        // Flush any remaining buffered data blocks
        self.flush_batch();

        // Convert recovery blocks back from shuffle2x to interleaved format if needed
        if self.use_shuffle2x {
            for recovery_block in &mut self.recovery_data {
                galois::finish_shuffle2x(recovery_block);
            }
        }

        self.recovery_data
            .into_iter()
            .enumerate()
            .map(|(exponent, recovery_u16)| {
                let mut recovery_bytes = Vec::with_capacity(self.block_size);
                for &val in &recovery_u16 {
                    recovery_bytes.extend_from_slice(&val.to_le_bytes());
                }
                (exponent as u32, recovery_bytes)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_basic() {
        // Create codec for 3 data blocks, generate 2 recovery blocks
        let rs = Par2ReedSolomon::new(3, 2);

        // Create test data blocks (8 bytes each = 4 u16 symbols)
        let data_blocks = vec![
            vec![1, 0, 2, 0, 3, 0, 4, 0],    // [1, 2, 3, 4] in u16
            vec![5, 0, 6, 0, 7, 0, 8, 0],    // [5, 6, 7, 8] in u16
            vec![9, 0, 10, 0, 11, 0, 12, 0], // [9, 10, 11, 12] in u16
        ];

        let result = rs.encode(&data_blocks, 2, 8).unwrap();

        assert_eq!(result.len(), 2, "Should generate 2 recovery blocks");
        assert_eq!(
            result[0].0, 0,
            "First recovery block should have exponent 0"
        );
        assert_eq!(
            result[1].0, 1,
            "Second recovery block should have exponent 1"
        );
        assert_eq!(result[0].1.len(), 8, "Recovery blocks should be 8 bytes");
        assert_eq!(result[1].1.len(), 8, "Recovery blocks should be 8 bytes");
    }

    #[test]
    fn test_encode_wrong_block_count() {
        let rs = Par2ReedSolomon::new(3, 2);
        let data_blocks = vec![
            vec![1, 0, 2, 0, 3, 0, 4, 0],
            vec![5, 0, 6, 0, 7, 0, 8, 0],
            // Missing third block
        ];

        let result = rs.encode(&data_blocks, 2, 8);
        assert!(result.is_err(), "Should fail with wrong block count");
    }

    #[test]
    fn test_encode_wrong_block_size() {
        let rs = Par2ReedSolomon::new(2, 1);
        let data_blocks = vec![
            vec![1, 0, 2, 0, 3, 0, 4, 0], // 8 bytes
            vec![5, 0, 6, 0],             // Only 4 bytes - mismatch!
        ];

        let result = rs.encode(&data_blocks, 1, 8);
        assert!(result.is_err(), "Should fail with mismatched block sizes");
    }

    #[test]
    fn test_encode_odd_block_size() {
        let rs = Par2ReedSolomon::new(2, 1);
        let data_blocks = vec![
            vec![1, 2, 3], // 3 bytes - odd number
            vec![4, 5, 6],
        ];

        let result = rs.encode(&data_blocks, 1, 3);
        assert!(result.is_err(), "Should fail with odd block size");
    }
}
