// PAR2 file repair using Reed-Solomon error correction
// Uses custom PAR2-compatible Reed-Solomon with GF(2^16) polynomial 0x1100B

use super::parser::{Par2File, FileHash};
use super::verify::VerificationResult;
use super::{Par2Operation, ProgressCallback};
use crate::error::{Par2Error, Result};
use crate::par2_rs::Par2ReedSolomon;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;



/// Repair damaged or missing files using PAR2 recovery blocks
pub fn repair_files(
    par2_data: &Par2File,
    verification_result: &VerificationResult,
    base_path: &Path,
    progress_callback: Option<ProgressCallback>,
) -> Result<()> {
    let block_size = par2_data.block_size as usize;

    // Build a map of file_id -> file info for easier lookup
    let file_map = &par2_data.files;

    // Calculate the total number of data blocks across all files
    // Each file is divided into blocks of block_size
    // IMPORTANT: Files must be processed in a consistent order (sorted by file_id)
    // to ensure block indices match between PAR2 creation and repair
    let mut total_blocks = 0usize;
    let mut file_block_map: HashMap<FileHash, (usize, usize)> = HashMap::new(); // file_id -> (start_block, num_blocks)

    // Use the file order from the PAR2 file itself (NOT sorted by name or file_id)
    // This is critical for Reed-Solomon to work correctly!
    for file_info in &par2_data.files_in_order {
        let num_blocks = ((file_info.length + par2_data.block_size - 1) / par2_data.block_size) as usize;
        file_block_map.insert(file_info.file_id, (total_blocks, num_blocks));
        total_blocks += num_blocks;
    }

    // Calculate how many blocks need to be repaired (not just file count!)
    let mut total_blocks_needed = 0usize;
    for file_id in verification_result.damaged_files.iter().chain(verification_result.missing_files.iter()) {
        if let Some(&(_start_block, num_blocks)) = file_block_map.get(file_id) {
            total_blocks_needed += num_blocks;
        }
    }

    if total_blocks_needed == 0 {
        // Nothing to repair
        return Ok(());
    }

    tracing::info!("Need to repair {} blocks across {} damaged/missing files",
        total_blocks_needed,
        verification_result.damaged_files.len() + verification_result.missing_files.len());
    tracing::info!("Have {} recovery blocks available", par2_data.recovery_blocks.len());

    for file_id in &verification_result.damaged_files {
        if let Some(file_info) = file_map.get(file_id) {
            tracing::info!("Damaged file: {}", file_info.name);
        }
    }
    for file_id in &verification_result.missing_files {
        if let Some(file_info) = file_map.get(file_id) {
            tracing::info!("Missing file: {}", file_info.name);
        }
    }

    if par2_data.recovery_blocks.len() < total_blocks_needed {
        return Err(Par2Error::RepairFailed(format!(
            "Insufficient recovery blocks: need {} blocks, have {} recovery blocks",
            total_blocks_needed,
            par2_data.recovery_blocks.len()
        ))
        );
    }

    if let Some(ref cb) = progress_callback {
        cb(Par2Operation::Repairing, 0, total_blocks as u64);
    }

    // Create data shards: one shard per data block
    // We'll process this block-by-block across all files
    let mut data_shards: Vec<Option<Vec<u8>>> = vec![None; total_blocks];
    let mut damaged_block_indices: Vec<usize> = Vec::new();

    // Read all verified file blocks into data shards
    for (file_id, file_path) in &verification_result.verified_files {
        if let Some(&(start_block, num_blocks)) = file_block_map.get(file_id) {
            let mut file = File::open(file_path)?;

            for block_idx in 0..num_blocks {
                let mut block = vec![0u8; block_size];
                let bytes_read = file.read(&mut block)?;

                // Pad incomplete blocks with zeros (important for last block of file)
                if bytes_read < block_size {
                    block[bytes_read..].fill(0);
                }

                data_shards[start_block + block_idx] = Some(block);
            }
        }
    }

    // Mark damaged/missing blocks as None and track their indices
    for file_id in verification_result.damaged_files.iter().chain(verification_result.missing_files.iter()) {
        if let Some(&(start_block, num_blocks)) = file_block_map.get(file_id) {
            for block_idx in 0..num_blocks {
                let global_idx = start_block + block_idx;
                data_shards[global_idx] = None;
                damaged_block_indices.push(global_idx);
            }
        }
    }

    // Prepare recovery count - we need enough recovery blocks to restore damaged blocks
    let recovery_count = par2_data.recovery_blocks.len().min(damaged_block_indices.len());

    if recovery_count < damaged_block_indices.len() {
        return Err(Par2Error::RepairFailed(format!(
            "Insufficient recovery blocks for repair: need {}, have {}",
            damaged_block_indices.len(),
            recovery_count
        )));
    }

    // Create PAR2-specific Reed-Solomon codec
    // In PAR2, we have N original data blocks and M recovery blocks
    // We can recover up to M missing data blocks
    let rs = Par2ReedSolomon::new(total_blocks, par2_data.recovery_blocks.len());

    // Build blocks vector: first N are data blocks, next M are recovery blocks
    // Use Option<Vec<u8>> where None = missing/damaged
    let mut blocks: Vec<Option<Vec<u8>>> = Vec::with_capacity(total_blocks + par2_data.recovery_blocks.len());

    // Add data blocks (0..total_blocks)
    for data_block in data_shards.iter() {
        blocks.push(data_block.clone());
    }

    // Sort recovery blocks by exponent to get sequential ordering
    let mut sorted_recovery_blocks = par2_data.recovery_blocks.clone();
    sorted_recovery_blocks.sort_by_key(|rb| rb.exponent);

    // Extract exponents and add recovery blocks
    let recovery_exponents: Vec<u32> = sorted_recovery_blocks.iter().map(|rb| rb.exponent).collect();

    for recovery_block in sorted_recovery_blocks.iter() {
        blocks.push(Some(recovery_block.data.clone()));
    }

    tracing::info!("Reconstructing {} damaged blocks using {} recovery blocks",
        damaged_block_indices.len(), sorted_recovery_blocks.len());

    // Perform reconstruction with recovery exponents
    rs.reconstruct(&mut blocks, &recovery_exponents, block_size)
        .map_err(|e| Par2Error::RepairFailed(format!("Reed-Solomon reconstruction failed: {}", e)))?;

    tracing::info!("Reed-Solomon reconstruction complete");

    // Extract repaired blocks and write them back to files
    let mut repaired_count = 0usize;

    for &damaged_idx in &damaged_block_indices {
        // Get the repaired block from blocks vector (first total_blocks elements are data)
        if let Some(Some(repaired_block)) = blocks.get(damaged_idx) {

            // Find which file this block belongs to
            for (file_id, &(start_block, num_blocks)) in &file_block_map {
                if damaged_idx >= start_block && damaged_idx < start_block + num_blocks {
                    // This block belongs to this file
                    if let Some(file_info) = file_map.get(file_id) {
                        let file_path = base_path.join(&file_info.name);
                        let block_offset_in_file = damaged_idx - start_block;
                        let byte_offset = block_offset_in_file * block_size;

                        // Create or open the file for writing
                        let mut file = OpenOptions::new()
                            .read(true)
                            .write(true)
                            .create(true)
                            .open(&file_path)?;

                        file.seek(SeekFrom::Start(byte_offset as u64))?;

                        // Write only the actual file data (not padding for last block)
                        let bytes_to_write = if byte_offset + block_size > file_info.length as usize {
                            file_info.length as usize - byte_offset
                        } else {
                            block_size
                        };

                        file.write_all(&repaired_block[..bytes_to_write])?;
                        file.flush()?;

                        repaired_count += 1;

                        if let Some(ref cb) = progress_callback {
                            cb(Par2Operation::Repairing, repaired_count as u64, damaged_block_indices.len() as u64);
                        }
                    }
                    break;
                }
            }
        }
    }


    if repaired_count < damaged_block_indices.len() {
        return Err(Par2Error::RepairFailed(format!(
            "Repair incomplete: repaired {}/{} blocks",
            repaired_count,
            damaged_block_indices.len()
        ))
        );
    }

    Ok(())
}
