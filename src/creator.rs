//! PAR2 file creation module
//!
//! Creates PAR2 protection files for a set of input files using Reed-Solomon
//! error correction over GF(2^16). Supports configurable redundancy levels
//! and block sizes with automatic optimization for typical use cases.
//!
//! # Main Entry Point
//!
//! - [`Par2Creator`] - Create PAR2 files from a set of input files

use crate::error::{Par2Error, Result};
use crate::par2_rs::Par2ReedSolomon;
use crate::parser::{FileHash, FileInfo, SliceChecksum};
use crate::volumes::{split_into_volumes, VolumeInfo, VolumeScheme};
use crate::writer::{self, compute_file_id, compute_recovery_set_id};
use crc32fast::Hasher as Crc32Hasher;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

/// Calculate optimal block size based on total file size
/// Optimal block size calculation: target ~2000 blocks with 4MB default cap,
/// but allow larger blocks if needed to stay within GF(2^16) limits.
///
/// This provides good balance between:
/// - Recovery granularity (smaller blocks = finer repair control)
/// - Encoding performance (with SIMD parallelization, more blocks is acceptable)
/// - Memory usage (4MB default cap, but may be exceeded for huge sets)
fn calculate_block_size(total_size: u64) -> u64 {
    const MIN_BLOCK_SIZE: u64 = 2048; // 2KB
    const MAX_BLOCK_SIZE: u64 = 4 * 1024 * 1024; // 4MB
    const TARGET_BLOCKS: u64 = 2000; // Optimal for performance and memory usage
    const MAX_BLOCKS: u64 = 65_535; // GF(2^16) limit

    if total_size == 0 {
        return MIN_BLOCK_SIZE;
    }

    let mut block_size = total_size / TARGET_BLOCKS;
    let min_for_field = total_size.div_ceil(MAX_BLOCKS);

    // Round up to multiple of 4
    if block_size % 4 != 0 {
        block_size = ((block_size / 4) + 1) * 4;
    }

    // Ensure we don't exceed the GF(2^16) block count limit
    let mut min_block = min_for_field.max(MIN_BLOCK_SIZE);
    if min_block % 4 != 0 {
        min_block = ((min_block / 4) + 1) * 4;
    }

    if min_block > MAX_BLOCK_SIZE {
        // For very large data sets, allow block sizes larger than the usual cap
        block_size = min_block;
    } else {
        // Clamp to default range
        block_size = block_size.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE);
        block_size = block_size.max(min_block);
    }

    // Ensure it's a multiple of 4
    assert_eq!(block_size % 4, 0);

    block_size
}

/// Pre-computed file metadata from parallel hashing pass
struct FileMetadata {
    path: PathBuf,
    full_hash: [u8; 16],
    hash_16k: [u8; 16],
    length: u64,
    name: String,
}

/// Compute file hashes in parallel for all input files
/// This is Phase 1 of creation - metadata collection is parallelized
fn compute_file_hashes_parallel(input_files: &[PathBuf]) -> Result<Vec<FileMetadata>> {
    use md5::{Digest, Md5};

    tracing::info!(
        num_files = input_files.len(),
        "Computing file hashes in parallel"
    );

    input_files
        .par_iter()
        .map(|path| {
            let mut file = File::open(path).map_err(|e| {
                Par2Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to open {}: {}", path.display(), e),
                ))
            })?;

            let file_size = file.metadata()?.len();
            let mut full_hash_ctx = Md5::new();
            let mut hash_16k_ctx = Md5::new();
            let mut bytes_hashed_16k = 0u64;

            // Read file in chunks to compute hashes
            let mut buffer = vec![0u8; 64 * 1024]; // 64KB read buffer
            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }

                full_hash_ctx.update(&buffer[..bytes_read]);

                if bytes_hashed_16k < 16384 {
                    let to_hash = bytes_read.min((16384 - bytes_hashed_16k) as usize);
                    hash_16k_ctx.update(&buffer[..to_hash]);
                    bytes_hashed_16k += to_hash as u64;
                }
            }

            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| Par2Error::InvalidFormat("Invalid filename".to_string()))?
                .to_string();

            Ok(FileMetadata {
                path: path.clone(),
                full_hash: full_hash_ctx.finalize().into(),
                hash_16k: hash_16k_ctx.finalize().into(),
                length: file_size,
                name,
            })
        })
        .collect()
}

/// PAR2 file creator
pub struct Par2Creator {
    input_files: Vec<PathBuf>,
    output_path: PathBuf,
    block_size: Option<u64>,
    redundancy_percent: f32,
    volume_scheme: VolumeScheme,
}

impl Par2Creator {
    /// Create a new PAR2 creator for the given input files
    ///
    /// Defaults:
    /// - 5% redundancy (standard default)
    /// - Automatic block size (2000-4000 blocks target)
    /// - Exponential volume scheme
    /// - Output path derived from first input file
    pub fn new(input_files: Vec<PathBuf>) -> Result<Self> {
        if input_files.is_empty() {
            return Err(Par2Error::InvalidFormat("No input files".to_string()));
        }

        // Validate that all input files exist
        for file in &input_files {
            if !file.exists() {
                return Err(Par2Error::InvalidFormat(format!(
                    "Input file does not exist: {}",
                    file.display()
                )));
            }
        }

        let output_path = input_files[0].with_extension("").with_extension("par2");

        Ok(Self {
            input_files,
            output_path,
            block_size: None,
            redundancy_percent: 5.0,
            volume_scheme: VolumeScheme::Exponential,
        })
    }

    /// Set explicit block size (must be multiple of 4 and at least MIN_BLOCK_SIZE)
    pub fn with_block_size(mut self, size: u64) -> Result<Self> {
        const MIN_BLOCK_SIZE: u64 = 2048; // 2KB minimum

        if size == 0 {
            return Err(Par2Error::InvalidFormat(
                "Block size cannot be zero".to_string(),
            ));
        }
        if size > (usize::MAX as u64) {
            return Err(Par2Error::InvalidFormat(
                "Block size exceeds platform limits".to_string(),
            ));
        }
        if size < MIN_BLOCK_SIZE {
            return Err(Par2Error::InvalidFormat(format!(
                "Block size must be at least {} bytes (got {})",
                MIN_BLOCK_SIZE, size
            )));
        }
        if size % 4 != 0 {
            return Err(Par2Error::InvalidFormat(
                "Block size must be multiple of 4".to_string(),
            ));
        }
        self.block_size = Some(size);
        Ok(self)
    }

    /// Set redundancy percentage (default: 5.0)
    pub fn with_redundancy(mut self, percent: f32) -> Self {
        self.redundancy_percent = percent;
        self
    }

    /// Set output path (default: first_file.par2)
    pub fn with_output_path(mut self, path: PathBuf) -> Self {
        self.output_path = path;
        self
    }

    /// Set volume splitting scheme
    pub fn with_volume_scheme(mut self, scheme: VolumeScheme) -> Self {
        self.volume_scheme = scheme;
        self
    }

    /// Create PAR2 files using streaming (memory-efficient)
    pub fn create(&self) -> Result<Vec<PathBuf>> {
        use md5::{Digest, Md5};
        const MAX_BLOCKS: usize = 65_535;

        tracing::info!(
            input_files = self.input_files.len(),
            redundancy = self.redundancy_percent,
            "Creating PAR2 files (streaming mode)"
        );

        // Step 1: Compute file hashes in parallel (Phase 1 optimization)
        let file_metadata = compute_file_hashes_parallel(&self.input_files)?;
        let total_size: u64 = file_metadata.iter().map(|m| m.length).sum();

        let block_size = self
            .block_size
            .unwrap_or_else(|| calculate_block_size(total_size));
        if block_size == 0 {
            return Err(Par2Error::InvalidFormat(
                "Block size cannot be zero".to_string(),
            ));
        }
        if block_size > (usize::MAX as u64) {
            return Err(Par2Error::InvalidFormat(
                "Block size exceeds platform limits".to_string(),
            ));
        }
        tracing::info!(block_size, total_size, "Calculated block size");

        // Count total blocks across all files
        let mut num_data_blocks = 0usize;
        let mut file_block_counts = Vec::new();
        for meta in &file_metadata {
            let blocks_u64 = meta.length.div_ceil(block_size);
            let blocks = usize::try_from(blocks_u64).map_err(|_| {
                Par2Error::InvalidFormat(format!("File requires too many blocks: {}", meta.name))
            })?;
            file_block_counts.push(blocks);
            num_data_blocks = num_data_blocks
                .checked_add(blocks)
                .ok_or_else(|| Par2Error::InvalidFormat("Too many total blocks".to_string()))?;
        }
        if num_data_blocks > MAX_BLOCKS {
            return Err(Par2Error::InvalidFormat(format!(
                "Total data blocks ({}) exceed GF(2^16) limit ({})",
                num_data_blocks, MAX_BLOCKS
            )));
        }

        // Step 2: Calculate number of recovery blocks
        let data_bytes = (num_data_blocks as u64)
            .checked_mul(block_size)
            .ok_or_else(|| Par2Error::InvalidFormat("Data size overflow".to_string()))?;
        let recovery_bytes =
            ((data_bytes as f64) * (self.redundancy_percent as f64 / 100.0)) as u64;
        let num_recovery_blocks_u64 = recovery_bytes.div_ceil(block_size);
        let num_recovery_blocks = usize::try_from(num_recovery_blocks_u64)
            .map_err(|_| Par2Error::InvalidFormat("Too many recovery blocks".to_string()))?;
        if num_recovery_blocks > MAX_BLOCKS {
            return Err(Par2Error::InvalidFormat(format!(
                "Recovery blocks ({}) exceed GF(2^16) limit ({})",
                num_recovery_blocks, MAX_BLOCKS
            )));
        }
        if num_data_blocks
            .checked_add(num_recovery_blocks)
            .ok_or_else(|| Par2Error::InvalidFormat("Block count overflow".to_string()))?
            > MAX_BLOCKS
        {
            return Err(Par2Error::InvalidFormat(format!(
                "Total blocks (data + recovery) exceed GF(2^16) limit ({})",
                MAX_BLOCKS
            )));
        }

        tracing::info!(
            data_blocks = num_data_blocks,
            recovery_blocks = num_recovery_blocks,
            recovery_bytes,
            "Calculated block counts"
        );

        // Step 3: Create streaming encoder
        // Use chunk size that's at most 100KB or the block size, whichever is smaller
        let chunk_size = 100_000.min(block_size as usize);
        let rs = Par2ReedSolomon::new(num_data_blocks, num_recovery_blocks);
        let mut encoder = rs
            .create_streaming_encoder(num_recovery_blocks, block_size as usize, chunk_size)
            .map_err(|e| Par2Error::RepairFailed(format!("Failed to create encoder: {}", e)))?;

        // Step 4: Stream through files, compute block checksums and encode
        // File-level hashes already computed in parallel, just need IFSC checksums
        tracing::info!("Processing blocks for IFSC and encoding");
        let mut file_infos = Vec::new();
        let mut slice_checksums_map = std::collections::HashMap::new();
        let mut global_block_idx = 0usize;

        for (file_idx, meta) in file_metadata.iter().enumerate() {
            let mut file = File::open(&meta.path)?;
            let file_size = meta.length;
            let num_blocks = file_block_counts[file_idx];

            let mut block_checksums = Vec::with_capacity(num_blocks);

            // Process each block in this file sequentially (no seeks needed!)
            for block_in_file in 0..num_blocks {
                let block_offset = (block_in_file as u64) * block_size;
                let block_actual_size = ((file_size - block_offset).min(block_size)) as usize;

                // Read entire block sequentially (no seek - file cursor is already positioned)
                let mut block_buffer = vec![0u8; block_actual_size];
                file.read_exact(&mut block_buffer)?;

                // Pad block to full size if needed
                if block_buffer.len() < block_size as usize {
                    block_buffer.resize(block_size as usize, 0);
                }

                // Compute IFSC checksum for this block
                // Note: Hash the FULL padded block (PAR2 spec requires zero-padding)
                let mut block_hash_ctx = Md5::new();
                let mut block_crc = Crc32Hasher::new();
                block_hash_ctx.update(&block_buffer);
                block_crc.update(&block_buffer);

                let block_md5: [u8; 16] = block_hash_ctx.finalize().into();
                let block_crc32 = block_crc.finalize();
                block_checksums.push(SliceChecksum {
                    md5: block_md5,
                    crc32: block_crc32,
                });

                // Convert entire block to u16 for encoding
                let block_u16: Vec<u16> = block_buffer
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();

                // Process entire block (internally splits into chunks and parallelizes)
                encoder.process_block(global_block_idx, &block_u16);

                global_block_idx += 1;
            }

            // Use pre-computed file hashes from parallel pass
            let file_id = compute_file_id(&meta.hash_16k, file_size, &meta.name);

            file_infos.push(FileInfo {
                file_id,
                hash: meta.full_hash,
                hash_16k: meta.hash_16k,
                length: file_size,
                name: meta.name.clone(),
            });

            slice_checksums_map.insert(file_id, block_checksums);
        }

        // Step 5: Finalize encoding and get recovery blocks
        tracing::info!("Finalizing recovery blocks");
        let recovery_result = encoder.finalize();

        // Step 6: Build main packet body and compute recovery set ID
        let file_ids: Vec<FileHash> = file_infos.iter().map(|f| f.file_id).collect();
        let mut main_body = Vec::new();
        main_body.extend_from_slice(&block_size.to_le_bytes());
        main_body.extend_from_slice(&(file_infos.len() as u32).to_le_bytes());

        let mut sorted_ids = file_ids.clone();
        sorted_ids.sort_unstable();
        for id in &sorted_ids {
            main_body.extend_from_slice(id);
        }

        let recovery_set_id = compute_recovery_set_id(&main_body);

        // Step 7: Split recovery blocks into volumes
        let mut volumes =
            split_into_volumes(recovery_result, self.volume_scheme, &self.output_path)?;

        // For Single scheme, put all recovery blocks in the main file
        // For other schemes, create a separate main .par2 index file
        if matches!(self.volume_scheme, VolumeScheme::Single) {
            // Single scheme: put everything in the main file
            volumes[0].path = self.output_path.clone();
        } else {
            // Multiple volumes: create a main .par2 index file (with just metadata, no recovery data)
            let main_volume = VolumeInfo {
                path: self.output_path.clone(),
                exponent_start: 0,
                exponent_end: 0,
                recovery_blocks: vec![],
            };
            volumes.insert(0, main_volume);
        }

        tracing::info!(volumes = volumes.len(), "Split into volumes");

        // Step 8: Write volumes
        let mut created_files = Vec::new();

        for volume in volumes {
            tracing::info!(
                path = ?volume.path,
                blocks = volume.recovery_blocks.len(),
                "Writing volume"
            );

            let mut file = File::create(&volume.path)?;

            // Write creator packet
            writer::write_creator_packet(
                &mut file,
                &recovery_set_id,
                "par2-rs v0.1.0 https://github.com/zephleggett/par2-rs",
            )?;

            // Write main packet
            writer::write_main_packet(&mut file, &recovery_set_id, block_size, &file_ids)?;

            // Write file description packets
            for file_info in &file_infos {
                writer::write_file_desc_packet(&mut file, &recovery_set_id, file_info)?;
            }

            // Write IFSC packets (Input File Slice Checksum)
            for file_info in &file_infos {
                if let Some(checksums) = slice_checksums_map.get(&file_info.file_id) {
                    writer::write_ifsc_packet(
                        &mut file,
                        &recovery_set_id,
                        &file_info.file_id,
                        checksums,
                    )?;
                }
            }

            // Write recovery slice packets
            for recovery_block in &volume.recovery_blocks {
                writer::write_recovery_slice_packet(&mut file, &recovery_set_id, recovery_block)?;
            }

            file.flush()?;
            created_files.push(volume.path);
        }

        tracing::info!(files = created_files.len(), "PAR2 creation complete");

        Ok(created_files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_calculate_block_size() {
        assert_eq!(calculate_block_size(0), 2048);
        assert_eq!(calculate_block_size(1_000_000), 2048); // 1MB -> 2KB min
        assert_eq!(calculate_block_size(10_000_000), 5000); // 10MB -> 5KB
        assert_eq!(calculate_block_size(1_000_000_000), 500_000); // 1GB -> 500KB

        // Verify all results are multiples of 4
        for size in [0, 1000, 1_000_000, 100_000_000, 10_000_000_000] {
            let block_size = calculate_block_size(size);
            assert_eq!(block_size % 4, 0, "Block size must be multiple of 4");
            assert!(block_size >= 2048, "Block size must be at least 2KB");
            assert!(
                block_size <= 4 * 1024 * 1024,
                "Block size must be at most 4MB"
            );
        }
    }

    #[test]
    fn test_creator_new_empty_files() {
        let result = Par2Creator::new(vec![]);
        assert!(result.is_err(), "Should fail with empty file list");
    }

    #[test]
    fn test_creator_invalid_block_size() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.bin");
        File::create(&file_path).unwrap();

        let creator = Par2Creator::new(vec![file_path]).unwrap();
        let result = creator.with_block_size(2047); // Not multiple of 4

        assert!(result.is_err(), "Should fail with invalid block size");
    }
}
