// Parallel PAR2 file repair implementation
// Processes chunks concurrently across all CPU cores

use super::parser::{FileHash, Par2File};
use super::verify::VerificationResult;
use super::{Par2Operation, ProgressCallback};
use crate::error::{Par2Error, Result};
use crate::par2_rs::Par2ReedSolomon;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Parallel repair with optimal CPU utilization
pub fn repair_files_parallel(
    par2_data: &Par2File,
    verification_result: &VerificationResult,
    base_path: &Path,
    progress_callback: Option<ProgressCallback>,
) -> Result<()> {
    let block_size = par2_data.block_size as usize;
    let file_map = &par2_data.files;

    // Build file_id -> (start_block, num_blocks) map
    let mut total_blocks = 0usize;
    let mut file_block_map: HashMap<FileHash, (usize, usize)> = HashMap::new();

    for file_info in &par2_data.files_in_order {
        let num_blocks =
            file_info.length.div_ceil(par2_data.block_size) as usize;
        file_block_map.insert(file_info.file_id, (total_blocks, num_blocks));
        total_blocks += num_blocks;
    }

    // Identify damaged blocks
    let mut damaged_block_indices: Vec<usize> = Vec::new();

    for file_id in &verification_result.missing_files {
        if let Some(&(start_block, num_blocks)) = file_block_map.get(file_id) {
            for block_idx in 0..num_blocks {
                damaged_block_indices.push(start_block + block_idx);
            }
        }
    }

    for file_id in &verification_result.damaged_files {
        if let Some(block_damage) = verification_result.block_damages.get(file_id) {
            if let Some(&(start_block, _num_blocks)) = file_block_map.get(file_id) {
                for &block_idx in &block_damage.damaged_block_indices {
                    damaged_block_indices.push(start_block + block_idx);
                }
            }
        } else if let Some(&(start_block, num_blocks)) = file_block_map.get(file_id) {
            for block_idx in 0..num_blocks {
                damaged_block_indices.push(start_block + block_idx);
            }
        }
    }

    if damaged_block_indices.is_empty() {
        return Ok(());
    }

    tracing::info!(
        damaged_blocks = damaged_block_indices.len(),
        total_blocks,
        "Starting parallel repair"
    );

    if let Some(ref cb) = progress_callback {
        cb(Par2Operation::Repairing, 0, total_blocks as u64);
    }

    // Sort recovery blocks by exponent
    let mut recovery_indices: Vec<usize> = (0..par2_data.recovery_blocks.len()).collect();
    recovery_indices.sort_by_key(|&i| par2_data.recovery_blocks[i].exponent);

    // Only use as many recovery blocks as we have damaged blocks
    let num_recovery_needed = damaged_block_indices.len().min(recovery_indices.len());
    let recovery_exponents: Vec<u32> = recovery_indices
        .iter()
        .take(num_recovery_needed)
        .map(|&i| par2_data.recovery_blocks[i].exponent)
        .collect();

    // Store metadata for recovery blocks we need (no data pre-loading!)
    // We'll read the data on-demand during reconstruction
    let recovery_blocks: Arc<Vec<crate::parser::RecoveryBlock>> = Arc::new(
        recovery_indices
            .iter()
            .take(num_recovery_needed)
            .map(|&idx| par2_data.recovery_blocks[idx].clone())
            .collect(),
    );

    // Create Reed-Solomon codec with full recovery count for proper matrix dimensions
    let total_recovery_count = par2_data.recovery_blocks.len();
    let rs = Arc::new(Par2ReedSolomon::new(total_blocks, total_recovery_count));

    // Chunk size for optimal balance between syscalls and parallelism
    let num_cpus = num_cpus::get();
    const TARGET_CHUNK_SIZE: usize = 100_000; // 100KB chunks
    let chunk_size = (TARGET_CHUNK_SIZE.min(block_size) / 2) * 2;
    let num_chunks = block_size.div_ceil(chunk_size);

    let num_damaged = damaged_block_indices.len();
    tracing::info!(
        chunk_size,
        chunk_size_kb = chunk_size / 1000,
        num_chunks,
        num_cpus,
        num_damaged,
        "Optimized chunk size for performance"
    );

    // Identify ALL good block indices
    // Reed-Solomon reconstruction needs ALL present blocks for correct math
    let damaged_set: std::collections::HashSet<usize> =
        damaged_block_indices.iter().copied().collect();
    let good_indices: Vec<usize> = (0..total_blocks)
        .filter(|idx| !damaged_set.contains(idx))
        .collect();

    tracing::debug!(
        good_blocks = good_indices.len(),
        damaged_blocks = damaged_block_indices.len(),
        recovery_blocks = num_recovery_needed,
        "Block allocation"
    );

    // Build reverse map: block_idx -> (file_id, block_in_file)
    let mut block_to_file: HashMap<usize, (FileHash, usize)> = HashMap::new();
    for (file_id, &(start_block, num_blocks)) in &file_block_map {
        for local_idx in 0..num_blocks {
            block_to_file.insert(start_block + local_idx, (*file_id, local_idx));
        }
    }

    // Build file paths map
    // Include both verified files AND damaged files (we need to read good blocks from damaged files!)
    let mut file_paths: HashMap<FileHash, PathBuf> = verification_result
        .verified_files
        .iter()
        .map(|(id, path)| (*id, path.clone()))
        .collect();

    // Add damaged files (they're on disk, just have some bad blocks)
    for file_id in &verification_result.damaged_files {
        if let Some(file_info) = file_map.get(file_id) {
            let file_path = base_path.join(&file_info.name);
            file_paths.insert(*file_id, file_path);
        }
    }

    // Open file handles once for writing (reduce file open/close overhead)
    let mut output_files: HashMap<FileHash, File> = HashMap::new();

    // Missing files: create new, truncate is OK since file doesn't exist
    for file_id in &verification_result.missing_files {
        if let Some(file_info) = file_map.get(file_id) {
            let file_path = base_path.join(&file_info.name);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&file_path)?;

            // Pre-allocate file size to ensure proper file growth
            // This prevents issues with sparse files and ensures the file has the correct size
            file.set_len(file_info.length)?;

            output_files.insert(*file_id, file);
        }
    }

    // Damaged files: DON'T truncate! We need to read good blocks first
    for file_id in &verification_result.damaged_files {
        if let Some(file_info) = file_map.get(file_id) {
            let file_path = base_path.join(&file_info.name);
            let file = OpenOptions::new().read(true).write(true).open(&file_path)?;
            output_files.insert(*file_id, file);
        }
    }

    // Adaptive parallelism based on data density and memory constraints
    // Calculate estimated memory per parallel chunk
    let blocks_loaded_per_chunk =
        good_indices.len() + damaged_block_indices.len().min(recovery_blocks.len());
    let est_mb_per_chunk = (blocks_loaded_per_chunk * chunk_size) / (1024 * 1024);

    // Adaptive parallelism based on system resources
    // Maximum performance settings - memory usage is acceptable for modern systems
    let default_multiplier = if good_indices.len() < 100 {
        // Sparse case: can use more parallelism
        (num_cpus * 2).clamp(8, 20) // Scale with cores, cap at reasonable max
    } else {
        // Dense case: very aggressive for maximum speed
        if num_cpus >= 8 {
            10 // High-core systems: very aggressive (4-5s repair time)
        } else if num_cpus >= 4 {
            6 // Mid-range: aggressive
        } else {
            3 // Low-core systems: moderate
        }
    };

    // Memory-aware maximum parallel chunks - adaptive to system size
    // Very generous budgets for maximum performance (~200-300MB actual usage)
    let memory_budget_mb = if num_cpus >= 8 {
        6000 // High-end systems: 6GB budget (actual usage ~200-300MB, estimates are 3× high)
    } else if num_cpus >= 4 {
        3000 // Mid-range: 3GB budget
    } else {
        1500 // Low-end: 1.5GB budget
    };

    let memory_limit_chunks = if est_mb_per_chunk > 10 {
        // Allow maximum parallelism - estimates are conservative (3× actual usage)
        (memory_budget_mb / est_mb_per_chunk).max(num_cpus)
    } else {
        // Sparse case: scale with cores very aggressively
        num_cpus * (if num_cpus >= 8 { 8 } else { 6 })
    };

    let parallelism_multiplier = std::env::var("PAR2_PARALLELISM")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_multiplier);

    // Allow memory limit override: PAR2_MAX_PARALLEL_CHUNKS
    let max_parallel = std::env::var("PAR2_MAX_PARALLEL_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(memory_limit_chunks);

    let batch_size = (num_cpus * parallelism_multiplier)
        .max(4)
        .min(max_parallel)
        .min(num_chunks);

    tracing::info!(
        num_cpus,
        parallelism_multiplier,
        memory_budget_mb,
        est_mb_per_chunk,
        memory_limit_chunks,
        batch_size,
        blocks_per_chunk = blocks_loaded_per_chunk,
        "Adaptive parallelism configured for system"
    );

    // Prepare indices for streaming API
    let present_recovery_indices: Vec<usize> = (0..recovery_blocks.len())
        .map(|i| total_blocks + i)
        .collect();

    // CRITICAL: Compute Gaussian elimination transformation ONCE for all chunks
    // This is the key optimization - avoid O(m³) work per chunk!
    let transform = rs
        .compute_reconstruction_transform(
            &damaged_block_indices,
            &good_indices,
            &present_recovery_indices,
            &recovery_exponents,
        )
        .map_err(|e| Par2Error::RepairFailed(format!("Failed to compute transformation: {}", e)))?;

    tracing::info!(
        "Gaussian elimination computed once (will be reused for all {} chunks)",
        num_chunks
    );

    let mut total_writes = 0;
    for batch_start in (0..num_chunks).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_chunks);

        tracing::debug!(batch_start, batch_end, num_chunks, "Processing chunk batch");

        // Process this batch of chunks in parallel using STREAMING API
        let batch_writes: Result<Vec<_>> = (batch_start..batch_end)
            .into_par_iter()
            .map(|chunk_idx| -> Result<Vec<(usize, u64, Vec<u8>)>> {
        let chunk_offset = chunk_idx * chunk_size;
        let this_chunk_size = (block_size - chunk_offset).min(chunk_size);

        // Storage for reconstructed chunks
        let mut reconstructed_chunks: HashMap<usize, Vec<u8>> = HashMap::new();

        // Callback to read a block chunk on-demand
        // NOTE: Opens file, reads, and closes immediately to avoid file descriptor exhaustion
        // when running many parallel tasks
        let mut read_block = |block_idx: usize, offset: usize, size: usize| -> std::result::Result<Vec<u8>, String> {
            // Check if it's a data block or recovery block
            if block_idx < total_blocks {
                // Data block - read from file
                if let Some(&(file_id, block_in_file)) = block_to_file.get(&block_idx) {
                    // Open file fresh for each read to avoid FD exhaustion across parallel tasks
                    let file_path = file_paths.get(&file_id)
                        .ok_or_else(|| "File ID not found in file_paths map".to_string())?;
                    let mut file = File::open(file_path)
                        .map_err(|err| format!("File open failed: {}", err))?;

                    let block_byte_offset = (block_in_file * block_size) as u64;
                    let chunk_byte_offset = block_byte_offset + offset as u64;
                    file.seek(SeekFrom::Start(chunk_byte_offset))
                        .map_err(|err| format!("Seek failed: {}", err))?;

                    let mut buffer = vec![0u8; size];
                    let bytes_read = file.read(&mut buffer)
                        .map_err(|err| format!("Read failed: {}", err))?;
                    if bytes_read < size {
                        buffer[bytes_read..].fill(0);
                    }
                    // File is automatically closed when `file` goes out of scope here
                    Ok(buffer)
                } else {
                    Err(format!("Block {} not found in block_to_file map", block_idx))
                }
            } else {
                // Recovery block
                let rec_idx = block_idx - total_blocks;
                if rec_idx < recovery_blocks.len() {
                    let mut chunk = recovery_blocks[rec_idx].read_chunk(offset, size)
                        .map_err(|err| format!("Recovery block read failed: {}", err))?;
                    chunk.resize(size, 0);
                    Ok(chunk)
                } else {
                    Err(format!("Recovery block {} out of range", rec_idx))
                }
            }
        };

        // Callback to write reconstructed chunk
        let mut write_result = |block_idx: usize, data: Vec<u8>| -> std::result::Result<(), String> {
            reconstructed_chunks.insert(block_idx, data);
            Ok(())
        };

        // Call streaming reconstruction with precomputed transformation
        rs.reconstruct_streaming_chunk(
            &damaged_block_indices,
            &good_indices,
            &present_recovery_indices,
            &transform,
            chunk_offset,
            this_chunk_size,
            &mut read_block,
            &mut write_result,
        )
        .map_err(|e| Par2Error::RepairFailed(format!("Streaming RS reconstruction failed: {}", e)))?;

        // Collect reconstructed chunks to return
        let mut writes = Vec::new();
        for &damaged_idx in &damaged_block_indices {
            if let Some(reconstructed_chunk) = reconstructed_chunks.get(&damaged_idx) {
                writes.push((damaged_idx, chunk_offset as u64, reconstructed_chunk.clone()));
            } else {
                return Err(Par2Error::RepairFailed(format!(
                    "BUG: Block {} was not reconstructed by RS but no error was returned (chunk_offset={}, chunk_idx={})",
                    damaged_idx, chunk_offset, chunk_idx
                )));
            }
        }

        Ok(writes)
            })
            .collect();

        // Write all chunks from this batch
        for chunk_writes in batch_writes? {
            for (block_idx, chunk_offset, data) in chunk_writes {
                if let Some(&(file_id, block_in_file)) = block_to_file.get(&block_idx) {
                    if let Some(file) = output_files.get_mut(&file_id) {
                        if let Some(file_info) = file_map.get(&file_id) {
                            let block_byte_offset = (block_in_file * block_size) as u64;
                            let write_offset = block_byte_offset + chunk_offset;

                            file.seek(SeekFrom::Start(write_offset))?;

                            let file_remaining = file_info.length.saturating_sub(write_offset);
                            let bytes_to_write = (data.len() as u64).min(file_remaining) as usize;

                            if bytes_to_write > 0 {
                                file.write_all(&data[..bytes_to_write])?;
                                total_writes += 1;
                            } else {
                                tracing::warn!(
                                    block_idx,
                                    chunk_offset,
                                    data_len = data.len(),
                                    file_remaining,
                                    "Skipped write: bytes_to_write = 0"
                                );
                            }
                        } else {
                            tracing::error!(
                                block_idx,
                                "Skipped write: file_info not found in file_map"
                            );
                        }
                    } else {
                        tracing::error!(block_idx, "Skipped write: file not found in output_files");
                    }
                } else {
                    tracing::error!(
                        block_idx,
                        "Skipped write: block_idx not found in block_to_file"
                    );
                }
            }
        }

        // Progress tracking
        if batch_end % 10 == 0 || batch_end == num_chunks {
            tracing::debug!(
                completed = batch_end,
                total = num_chunks,
                progress_pct = (batch_end * 100) / num_chunks,
                total_writes,
                "Processing chunks"
            );
        }
    }

    // Verify chunks were written (note: total_writes may be less than theoretical max due to partial last blocks)
    let theoretical_max_writes = num_chunks * damaged_block_indices.len();
    tracing::info!(
        total_writes,
        theoretical_max = theoretical_max_writes,
        num_chunks,
        damaged_blocks = damaged_block_indices.len(),
        "Chunk writes completed"
    );

    if total_writes < theoretical_max_writes {
        tracing::debug!(
            diff = theoretical_max_writes - total_writes,
            "Some chunks were skipped (likely due to partial last blocks)"
        );
    }

    // Flush and sync all output files to disk
    for (_file_id, file) in output_files.iter_mut() {
        file.flush()?;
        file.sync_all()?;
    }

    tracing::info!(
        repaired_blocks = damaged_block_indices.len(),
        "Parallel repair complete"
    );

    if let Some(ref cb) = progress_callback {
        cb(
            Par2Operation::Repairing,
            damaged_block_indices.len() as u64,
            damaged_block_indices.len() as u64,
        );
    }

    Ok(())
}
