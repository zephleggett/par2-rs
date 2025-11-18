// PAR2 file creation module
// Creates PAR2 protection files for a set of input files

use crate::error::{Par2Error, Result};
use crate::par2_rs::Par2ReedSolomon;
use crate::parser::{FileHash, FileInfo, SliceChecksum};
use crate::volumes::{split_into_volumes, VolumeInfo, VolumeScheme};
use crate::writer::{self, compute_file_id, compute_recovery_set_id};
use crc32fast::Hasher as Crc32Hasher;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Calculate optimal block size based on total file size
/// Aims for 2000-4000 blocks total, with block size as multiple of 4
fn calculate_block_size(total_size: u64) -> u64 {
    const TARGET_BLOCKS: u64 = 2000;
    const MIN_BLOCK_SIZE: u64 = 2048; // 2KB
    const MAX_BLOCK_SIZE: u64 = 4 * 1024 * 1024; // 4MB

    if total_size == 0 {
        return MIN_BLOCK_SIZE;
    }

    let mut block_size = total_size / TARGET_BLOCKS;

    // Round up to multiple of 4
    if block_size % 4 != 0 {
        block_size = ((block_size / 4) + 1) * 4;
    }

    // Clamp to range
    block_size = block_size.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE);

    // Ensure it's a multiple of 4
    assert_eq!(block_size % 4, 0);

    block_size
}

/// Compute MD5 hash of first 16KB of a file
fn hash_file_16k(path: &Path) -> Result<FileHash> {
    use md5::{Digest, Md5};
    let mut file = File::open(path)?;
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 16384];
    let bytes_read = file.read(&mut buffer)?;
    hasher.update(&buffer[..bytes_read]);
    Ok(hasher.finalize().into())
}

/// Compute full MD5 hash of a file
fn hash_file(path: &Path) -> Result<FileHash> {
    use md5::{Digest, Md5};
    let mut file = File::open(path)?;
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 1048576]; // 1MB buffer for better performance

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hasher.finalize().into())
}

/// Read file data in blocks
fn read_file_blocks(path: &Path, block_size: u64) -> Result<Vec<Vec<u8>>> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let num_blocks = file_size.div_ceil(block_size) as usize;

    let mut blocks = Vec::with_capacity(num_blocks);
    let mut buffer = vec![0u8; block_size as usize];

    for _ in 0..num_blocks {
        let bytes_read = file.read(&mut buffer)?;

        // Pad incomplete blocks with zeros
        if bytes_read < block_size as usize {
            buffer[bytes_read..].fill(0);
        }

        blocks.push(buffer[..block_size as usize].to_vec());
    }

    Ok(blocks)
}

/// Compute IFSC (Input File Slice Checksum) for file blocks
/// Returns MD5 and CRC32 checksums for each block
fn compute_slice_checksums(blocks: &[Vec<u8>]) -> Vec<SliceChecksum> {
    blocks
        .iter()
        .map(|block| {
            // Compute MD5 of block
            use md5::{Digest, Md5};
            let mut hasher = Md5::new();
            hasher.update(block);
            let md5: [u8; 16] = hasher.finalize().into();

            // Compute CRC32 of block
            let mut hasher = Crc32Hasher::new();
            hasher.update(block);
            let crc32 = hasher.finalize();

            SliceChecksum { md5, crc32 }
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
    /// - 5% redundancy (par2cmdline standard)
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

    /// Create PAR2 files
    pub fn create(&self) -> Result<Vec<PathBuf>> {
        tracing::info!(
            input_files = self.input_files.len(),
            redundancy = self.redundancy_percent,
            "Creating PAR2 files"
        );

        // Step 1: Compute file hashes and metadata
        tracing::info!("Computing file hashes");
        let mut file_infos = Vec::new();
        let mut total_size = 0u64;

        for path in &self.input_files {
            let full_hash = hash_file(path)?;
            let hash_16k = hash_file_16k(path)?;
            let length = path.metadata()?.len();
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| Par2Error::InvalidFormat("Invalid filename".to_string()))?
                .to_string();

            let file_id = compute_file_id(&hash_16k, length, &name);

            file_infos.push(FileInfo {
                file_id,
                hash: full_hash,
                hash_16k,
                length,
                name,
            });

            total_size += length;
        }

        // Step 2: Calculate block size
        let block_size = self
            .block_size
            .unwrap_or_else(|| calculate_block_size(total_size));
        tracing::info!(block_size, total_size, "Calculated block size");

        // Step 3: Read all file data into blocks and compute IFSC checksums
        tracing::info!("Reading input files into blocks");
        let mut all_data_blocks = Vec::new();
        let mut slice_checksums_map = std::collections::HashMap::new();

        for (i, path) in self.input_files.iter().enumerate() {
            let blocks = read_file_blocks(path, block_size)?;

            // Compute IFSC checksums for this file's blocks
            let checksums = compute_slice_checksums(&blocks);
            slice_checksums_map.insert(file_infos[i].file_id, checksums);

            all_data_blocks.extend(blocks);
        }

        let num_data_blocks = all_data_blocks.len();
        tracing::info!(data_blocks = num_data_blocks, "Read all input data");

        // Step 4: Calculate number of recovery blocks
        let data_bytes = (num_data_blocks as u64) * block_size;
        let recovery_bytes =
            ((data_bytes as f64) * (self.redundancy_percent as f64 / 100.0)) as u64;
        let num_recovery_blocks = recovery_bytes.div_ceil(block_size) as usize;

        tracing::info!(
            recovery_blocks = num_recovery_blocks,
            recovery_bytes,
            "Calculated recovery block count"
        );

        // Step 5: Generate recovery blocks using Reed-Solomon encoding
        tracing::info!("Generating recovery blocks");
        let rs = Par2ReedSolomon::new(num_data_blocks, num_recovery_blocks);
        let recovery_result = rs
            .encode(&all_data_blocks, num_recovery_blocks, block_size as usize)
            .map_err(|e| Par2Error::RepairFailed(format!("Encoding failed: {}", e)))?;

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

        // Special case: if no recovery blocks (0% redundancy), create a main .par2 file with just metadata
        if volumes.is_empty() {
            volumes.push(VolumeInfo {
                path: self.output_path.clone(),
                exponent_start: 0,
                exponent_end: 0,
                recovery_blocks: vec![],
            });
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
    use std::io::Write;
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
    fn test_hash_file_16k() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.bin");

        // Create a test file with more than 16KB
        let mut file = File::create(&file_path).unwrap();
        let data = vec![0x42u8; 20000]; // 20KB
        file.write_all(&data).unwrap();
        drop(file);

        let hash = hash_file_16k(&file_path).unwrap();
        // Hash should only cover first 16KB
        assert_ne!(hash, [0u8; 16]);

        // Verify it's deterministic
        let hash2 = hash_file_16k(&file_path).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_hash_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.bin");

        let mut file = File::create(&file_path).unwrap();
        let data = b"Hello, PAR2!";
        file.write_all(data).unwrap();
        drop(file);

        let hash = hash_file(&file_path).unwrap();
        assert_ne!(hash, [0u8; 16]);

        // Verify it's deterministic
        let hash2 = hash_file(&file_path).unwrap();
        assert_eq!(hash, hash2);
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
