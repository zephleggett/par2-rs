// PAR2-specific Reed-Solomon implementation using GF(2^16) with polynomial 0x1100B
// Based on Vandermonde matrix construction as specified in PAR2 spec

use crate::galois::{self, gf_mul, gf_muladd_column, gf_pow};

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

        // Initialize recovery blocks (all zeros initially)
        let mut recovery_blocks: Vec<Vec<u16>> = vec![vec![0u16; symbols]; num_recovery];

        // OPTIMIZATION 2 & 3: Column-wise processing
        // For each data block (column), contribute to ALL recovery blocks
        let encode_start = Instant::now();

        for (data_idx, data_u16) in data_blocks_u16.iter().enumerate() {
            // Prepare coefficients for all recovery blocks using repeated multiply
            let mut coeffs: Vec<u16> = Vec::with_capacity(num_recovery);
            let base = self.constants[data_idx];
            let mut coeff = 1u16; // base^0
            for _ in 0..num_recovery {
                coeffs.push(coeff);
                coeff = gf_mul(coeff, base);
            }

            // Get mutable references to all recovery blocks
            let mut recovery_refs: Vec<&mut [u16]> = Vec::with_capacity(num_recovery);
            for recovery_block in recovery_blocks.iter_mut() {
                // SAFETY: Each recovery_block is a distinct Vec<u16> allocation.
                // Converting to raw pointers allows collecting multiple mutable references.
                // This is safe because:
                // 1. Each pointer targets non-overlapping memory (different Vec allocations)
                // 2. Pointers are only used immediately in gf_muladd_column below
                // 3. No other code accesses recovery_blocks during this operation
                unsafe {
                    let ptr = recovery_block.as_mut_ptr();
                    let len = recovery_block.len();
                    recovery_refs.push(std::slice::from_raw_parts_mut(ptr, len));
                }
            }

            // Apply column-wise multiply-add: recovery[j][i] += data[i] * coeffs[j]
            gf_muladd_column(&mut recovery_refs, data_u16, &coeffs);
        }

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
                for j in k..m {
                    if A[k][j] != 0 {
                        A[k][j] = galois::gf_div(A[k][j], pivot);
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
        use std::time::Instant;
        let start = Instant::now();

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

        let init_start = Instant::now();

        let mut accumulators: Vec<Vec<u16>> = vec![vec![0u16; chunk_symbols]; m];

        for (original_row, &rec_idx) in present_recovery_indices.iter().take(m).enumerate() {
            let rec_chunk = read_block(rec_idx, chunk_offset, chunk_size)?;

            let swapped_row = row_order.iter().position(|&r| r == original_row)
                .ok_or_else(|| format!("Row {} not found in row_order", original_row))?;

            // Convert bytes to u16 using SIMD
            galois::bytes_to_u16_simd(&rec_chunk, &mut accumulators[swapped_row]);
        }

        tracing::debug!(
            init_ms = init_start.elapsed().as_millis(),
            "Accumulators initialized"
        );

        // PHASE 2: Stream through present data blocks ONE AT A TIME
        let stream_start = Instant::now();

        // Pre-allocate buffer for data chunk conversion (reused for each block)
        let mut data_chunk_buf: Vec<u16> = vec![0u16; chunk_symbols];

        for (data_col, &data_idx) in present_data_indices.iter().enumerate() {
            // Read ONE block from disk via callback
            let data_chunk_bytes = read_block(data_idx, chunk_offset, chunk_size)?;

            // Convert to u16 in-place using SIMD (HOT PATH - called 1000s of times)
            galois::bytes_to_u16_simd(&data_chunk_bytes, &mut data_chunk_buf[..chunk_symbols]);

            // Process in row batches for SIMD
            const ROW_BATCH_SIZE: usize = 8;
            let mut row_batch_start = 0;
            while row_batch_start < m {
                let row_batch_end = (row_batch_start + ROW_BATCH_SIZE).min(m);
                let batch_size = row_batch_end - row_batch_start;

                let mut coeffs: Vec<u16> = Vec::with_capacity(batch_size);
                #[allow(clippy::needless_range_loop)] // Need row index for 2D array access
                for row in row_batch_start..row_batch_end {
                    coeffs.push(vandermonde_coeffs[row][data_col]);
                }

                let mut acc_refs: Vec<&mut [u16]> = Vec::with_capacity(batch_size);
                #[allow(clippy::needless_range_loop)]
                // Need index for unsafe raw pointer manipulation
                for row in row_batch_start..row_batch_end {
                    // SAFETY: Each accumulators[row] is a distinct Vec<u16> allocation.
                    // Converting to raw pointers allows collecting multiple mutable slices.
                    // Safe because each row targets non-overlapping memory (different Vec allocations).
                    unsafe {
                        let ptr = accumulators[row].as_mut_ptr();
                        acc_refs.push(std::slice::from_raw_parts_mut(ptr, chunk_symbols));
                    }
                }

                gf_muladd_column(&mut acc_refs, &data_chunk_buf[..chunk_symbols], &coeffs);
                row_batch_start = row_batch_end;
            }
        }

        tracing::debug!(
            stream_ms = stream_start.elapsed().as_millis(),
            blocks_processed = present_data_indices.len(),
            "Streaming subtraction complete"
        );

        // PHASE 4: Apply pre-computed Gaussian elimination to accumulators
        let ge_start = Instant::now();

        for k in 0..m {
            let pivot = scale_factors[k];
            if pivot != 1 {
                for s in 0..chunk_symbols {
                    if accumulators[k][s] != 0 {
                        accumulators[k][s] = galois::gf_div(accumulators[k][s], pivot);
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

        tracing::debug!(
            ge_ms = ge_start.elapsed().as_millis(),
            "Gaussian elimination applied"
        );

        // PHASE 5: Write results via callback
        let write_start = Instant::now();

        for (row, &miss_idx) in missing_indices.iter().enumerate() {
            let mut output = Vec::with_capacity(chunk_size);
            for &val in &accumulators[row] {
                output.extend_from_slice(&val.to_le_bytes());
            }
            write_result(miss_idx, output)?;
        }

        tracing::debug!(
            write_ms = write_start.elapsed().as_millis(),
            total_ms = start.elapsed().as_millis(),
            "Streaming reconstruction complete"
        );

        Ok(())
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
