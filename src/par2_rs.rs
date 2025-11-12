// PAR2-specific Reed-Solomon implementation using GF(2^16) with polynomial 0x1100B
// Based on Vandermonde matrix construction as specified in PAR2 spec

use crate::galois::{self, gf_div, gf_mul, gf_pow};

/// PAR2 Reed-Solomon decoder
pub struct Par2ReedSolomon {
    data_blocks: usize,
    recovery_blocks: usize,
    constants: Vec<u16>, // PAR2-specific constants for Vandermonde matrix
}

/// Generate PAR2 constants: powers of 2 where exponent has order 65535
/// Exponent e has order 65535 if: e%3 != 0 && e%5 != 0 && e%17 != 0 && e%257 != 0
fn generate_par2_constants(count: usize) -> Vec<u16> {
    let mut constants = Vec::with_capacity(count);
    let mut exponent = 0u32;

    while constants.len() < count && exponent < 65536 {
        // Check if this exponent has order 65535
        if exponent % 3 != 0 && exponent % 5 != 0 && exponent % 17 != 0 && exponent % 257 != 0 {
            // constant = 2^exponent in GF(2^16)
            let constant = gf_pow(2, exponent as usize);
            constants.push(constant);
        }
        exponent += 1;
    }

    constants
}

impl Par2ReedSolomon {
    /// Create a new PAR2 Reed-Solomon codec
    pub fn new(data_blocks: usize, recovery_blocks: usize) -> Self {
        galois::init_tables();
        let constants = generate_par2_constants(data_blocks);
        Self {
            data_blocks,
            recovery_blocks,
            constants,
        }
    }

    /// Reconstruct missing data blocks from available data and recovery blocks
    ///
    /// # Arguments
    /// * `blocks` - All blocks (data + recovery), where None indicates missing blocks
    /// * `recovery_exponents` - Exponents for each recovery block (sorted same as recovery blocks)
    /// * `block_size` - Size of each block in bytes
    ///
    /// # Returns
    /// Ok(()) if reconstruction successful, Err otherwise
    pub fn reconstruct(&self, blocks: &mut [Option<Vec<u8>>], recovery_exponents: &[u32], block_size: usize) -> Result<(), String> {
        let total_blocks = self.data_blocks + self.recovery_blocks;

        if blocks.len() != total_blocks {
            return Err(format!("Expected {} blocks, got {}", total_blocks, blocks.len()));
        }

        // Count available and missing blocks
        let mut available_indices = Vec::new();
        let mut missing_indices = Vec::new();

        for i in 0..self.data_blocks {
            if blocks[i].is_some() {
                available_indices.push(i);
            } else {
                missing_indices.push(i);
            }
        }

        // Add recovery block indices to available
        for i in self.data_blocks..total_blocks {
            if blocks[i].is_some() {
                available_indices.push(i);
            }
        }

        if missing_indices.is_empty() {
            return Ok(()); // Nothing to reconstruct
        }

        if available_indices.len() < self.data_blocks {
            return Err(format!(
                "Insufficient blocks: need {}, have {}",
                self.data_blocks,
                available_indices.len()
            ));
        }

        // Build matrices following par2cmdline's layout:
        // We need datamissing rows from PRESENT recovery blocks

        let datamissing = missing_indices.len();
        let datapresent = available_indices.iter().filter(|&&idx| idx < self.data_blocks).count();

        // Separate available blocks into present data and present recovery
        let present_data_indices: Vec<usize> = available_indices.iter()
            .filter(|&&idx| idx < self.data_blocks)
            .copied()
            .collect();
        let present_recovery_indices: Vec<usize> = available_indices.iter()
            .filter(|&&idx| idx >= self.data_blocks)
            .copied()
            .collect();

        if present_recovery_indices.len() < datamissing {
            return Err(format!("Need {} present recovery blocks, have {}",
                datamissing, present_recovery_indices.len()));
        }

        let incount = datapresent + datamissing;
        let mut left_matrix = vec![vec![0u16; incount]; datamissing];
        let mut right_matrix = vec![vec![0u16; datamissing]; datamissing];
        let mut result_data: Vec<Vec<u16>> = Vec::new();

        // Build first datamissing rows using PRESENT recovery blocks
        for row in 0..datamissing {
            let recovery_block_idx = present_recovery_indices[row];
            let recovery_idx = recovery_block_idx - self.data_blocks;
            let exponent = recovery_exponents[recovery_idx] as usize;

            // Left matrix columns: present data blocks (Vandermonde) + identity for present recovery
            // Columns for present data blocks
            for col in 0..datapresent {
                let data_idx = present_data_indices[col];
                left_matrix[row][col] = gf_pow(self.constants[data_idx], exponent);
            }
            // Identity columns for present recovery blocks being used
            for col in 0..datamissing {
                left_matrix[row][datapresent + col] = if row == col { 1 } else { 0 };
            }

            // Right matrix columns: missing data blocks (Vandermonde)
            for col in 0..datamissing {
                let missing_data_idx = missing_indices[col];
                right_matrix[row][col] = gf_pow(self.constants[missing_data_idx], exponent);
            }

            // Get the actual block data and convert to u16
            if let Some(ref block_data) = blocks[recovery_block_idx] {
                let mut row_data = Vec::with_capacity(block_size / 2);
                for chunk in block_data.chunks_exact(2) {
                    row_data.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                result_data.push(row_data);
            } else {
                return Err("Missing recovery block data".to_string());
            }
        }

        let mut result = result_data;

        // Solve using augmented Gaussian elimination
        gauss_elim_augmented(&mut left_matrix, &mut right_matrix, &mut result)?;

        // Debug: Check if right_matrix is identity after solving
        tracing::info!("After Gaussian elimination:");
        for row in 0..datamissing.min(3) {
            let mut right_row = String::new();
            for col in 0..datamissing.min(3) {
                right_row.push_str(&format!("{:04x} ", right_matrix[row][col]));
            }
            tracing::info!("  right_matrix[{}]: {}", row, right_row);
        }

        // Debug: Check first few values of left_matrix
        for row in 0..datamissing.min(3) {
            let mut left_row = String::new();
            for col in 0..(datapresent + datamissing).min(5) {
                left_row.push_str(&format!("{:04x} ", left_matrix[row][col]));
            }
            tracing::info!("  left_matrix[{}]: {}", row, left_row);
        }

        // After Gaussian elimination, result now contains data that when combined with
        // the leftmatrix coefficients gives us the missing blocks
        // We need to compute: missing[i] = sum(leftmatrix[i][j] * present_data[j]) + result[i]

        // Actually, after reviewing par2cmdline more carefully, the result vector
        // after Gaussian elimination should directly contain contributions from the
        // present recovery blocks. We need to add contributions from present data blocks.

        for (row, &missing_idx) in missing_indices.iter().enumerate() {
            let mut reconstructed = vec![0u16; block_size / 2];

            // Add contributions from present data blocks
            for (col, &data_idx) in present_data_indices.iter().enumerate() {
                let coeff = left_matrix[row][col];
                if coeff != 0 {
                    if let Some(ref block_data) = blocks[data_idx] {
                        for (i, chunk) in block_data.chunks_exact(2).enumerate() {
                            let data_val = u16::from_le_bytes([chunk[0], chunk[1]]);
                            let contribution = gf_mul(coeff, data_val);
                            reconstructed[i] ^= contribution; // XOR is addition in GF(2^n)
                        }
                    }
                }
            }

            // Add the result from Gaussian elimination (contributions from recovery blocks)
            for (i, &val) in result[row].iter().enumerate() {
                reconstructed[i] ^= val;
            }

            // Convert back to bytes
            let mut block_bytes = Vec::with_capacity(block_size);
            for &val in &reconstructed {
                block_bytes.extend_from_slice(&val.to_le_bytes());
            }
            blocks[missing_idx] = Some(block_bytes);
        }

        Ok(())
    }
}

/// Augmented Gaussian elimination matching par2cmdline's algorithm
/// Solves the system using two matrices: left (coefficients) and right (identity -> solution)
fn gauss_elim_augmented(
    left_matrix: &mut [Vec<u16>],
    right_matrix: &mut [Vec<u16>],
    result: &mut [Vec<u16>],
) -> Result<Vec<Vec<u16>>, String> {
    let rows = right_matrix.len();
    let leftcols = left_matrix[0].len();

    // Solve one row at a time (only the first 'rows' rows, which correspond to missing data)
    for row in 0..rows {
        // Get the pivot value from the RIGHT matrix diagonal
        let pivotvalue = right_matrix[row][row];

        if pivotvalue == 0 {
            return Err("RS computation error: pivot is zero".to_string());
        }

        // If the pivot value is not 1, scale the entire row
        if pivotvalue != 1 {
            // Scale left matrix
            for col in 0..leftcols {
                if left_matrix[row][col] != 0 {
                    left_matrix[row][col] = gf_div(left_matrix[row][col], pivotvalue);
                }
            }

            // Scale right matrix
            right_matrix[row][row] = 1;
            for col in (row + 1)..rows {
                if right_matrix[row][col] != 0 {
                    right_matrix[row][col] = gf_div(right_matrix[row][col], pivotvalue);
                }
            }

            // Scale result
            for i in 0..result[row].len() {
                if result[row][i] != 0 {
                    result[row][i] = gf_div(result[row][i], pivotvalue);
                }
            }
        }

        // For every OTHER row in the matrix
        for row2 in 0..rows {
            if row != row2 {
                // Get the scaling factor from the right matrix
                let scalevalue = right_matrix[row2][row];

                if scalevalue == 1 {
                    // If scaling factor is 1, just subtract rows (XOR in GF(2^n))
                    for col in 0..leftcols {
                        if left_matrix[row][col] != 0 {
                            left_matrix[row2][col] ^= left_matrix[row][col];
                        }
                    }

                    for col in row..rows {
                        if right_matrix[row][col] != 0 {
                            right_matrix[row2][col] ^= right_matrix[row][col];
                        }
                    }

                    for i in 0..result[row].len() {
                        if result[row][i] != 0 {
                            result[row2][i] ^= result[row][i];
                        }
                    }
                } else if scalevalue != 0 {
                    // If scaling factor is not 0, multiply and subtract
                    for col in 0..leftcols {
                        if left_matrix[row][col] != 0 {
                            let product = gf_mul(left_matrix[row][col], scalevalue);
                            left_matrix[row2][col] ^= product;
                        }
                    }

                    for col in row..rows {
                        if right_matrix[row][col] != 0 {
                            let product = gf_mul(right_matrix[row][col], scalevalue);
                            right_matrix[row2][col] ^= product;
                        }
                    }

                    for i in 0..result[row].len() {
                        if result[row][i] != 0 {
                            let product = gf_mul(result[row][i], scalevalue);
                            result[row2][i] ^= product;
                        }
                    }
                }
            }
        }
    }

    Ok(result.to_vec())
}
