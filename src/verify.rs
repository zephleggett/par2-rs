//! PAR2 file verification with IFSC-first strategy
//!
//! This module verifies protected files against PAR2 checksums using an optimized
//! two-phase approach:
//!
//! 1. **IFSC-first**: Check block-level CRC32/MD5 checksums when available
//! 2. **Full MD5 fallback**: Only compute full-file MD5 when IFSC data is missing
//!
//! This strategy is significantly faster for intact files since IFSC verification
//! can skip expensive full-file MD5 computation.
//!
//! # Block-Level Verification
//!
//! For damaged files, the module identifies exactly which blocks are corrupted,
//! enabling minimal repair (only damaged blocks are reconstructed).
//!
//! # Obfuscated Filename Support
//!
//! Files can be matched by hash even when filenames don't match (common on Usenet),
//! and are automatically renamed to their correct names.
//!
//! # Related Modules
//!
//! - [`crate::repair`]: Uses verification results to identify blocks needing repair
//! - [`crate::parser`]: Provides IFSC checksum data

use super::parser::{FileHash, FileInfo, Par2File};
use super::{MessageCallback, MessageLevel, Par2Operation, ProgressCallback};
use crate::error::{Par2Error, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

/// Block-level damage information for a file
#[derive(Debug, Clone)]
pub struct BlockDamage {
    pub file_id: FileHash,
    pub damaged_block_indices: Vec<usize>,
}

/// Result of file verification
#[derive(Debug)]
pub struct VerificationResult {
    pub all_verified: bool,
    pub verified_files: HashMap<FileHash, PathBuf>,
    pub missing_files: Vec<FileHash>,
    pub damaged_files: Vec<FileHash>,
    pub block_damages: HashMap<FileHash, BlockDamage>, // Block-level damage details
}

/// Verify files against PAR2 data
#[allow(dead_code)]
pub fn verify_files(
    par2_data: &Par2File,
    extra_files: &[PathBuf],
    base_path: &Path,
    progress_callback: Option<ProgressCallback>,
) -> Result<VerificationResult> {
    verify_files_with_messages(par2_data, extra_files, base_path, progress_callback, None)
}

/// Verify files against PAR2 data with message callback
pub fn verify_files_with_messages(
    par2_data: &Par2File,
    extra_files: &[PathBuf],
    base_path: &Path,
    progress_callback: Option<ProgressCallback>,
    message_callback: Option<MessageCallback>,
) -> Result<VerificationResult> {
    // Calculate total bytes for byte-level progress
    let total_bytes: u64 = par2_data.files.values().map(|f| f.length).sum();
    let bytes_verified = Arc::new(AtomicU64::new(0));

    tracing::info!(
        files = par2_data.files.len(),
        total_bytes = total_bytes,
        block_size = par2_data.block_size,
        "Starting verification"
    );

    // Use Arc<Mutex> for thread-safe collections
    let verified_files = Arc::new(Mutex::new(HashMap::new()));
    let damaged_files = Arc::new(Mutex::new(Vec::new()));

    let mut renamed_files = Vec::new();

    // First pass: Try to match files by name (parallel)
    // OPTIMIZATION: Use IFSC block-level checksums when available to skip expensive full-file MD5
    let files_vec: Vec<_> = par2_data.files.iter().collect();

    files_vec.par_iter().for_each(|(file_id, file_info)| {
        let expected_path = base_path.join(&file_info.name);

        if expected_path.exists() {
            // Try IFSC-first verification if available
            let slice_checksums = par2_data
                .slice_checksums
                .get(*file_id)
                .map(|v| v.as_slice());

            // Create intra-file progress callback that updates as bytes are read
            let bytes_verified_ref = &bytes_verified;
            let progress_callback_ref = &progress_callback;
            let intra_progress = |bytes: u64| {
                let done = bytes_verified_ref.fetch_add(bytes, Ordering::Relaxed) + bytes;
                if let Some(ref cb) = progress_callback_ref {
                    cb(Par2Operation::Verifying, done, total_bytes);
                }
            };

            match verify_file_smart(
                &expected_path,
                file_info,
                slice_checksums,
                par2_data.block_size,
                Some(&intra_progress),
            ) {
                Ok(true) => {
                    tracing::debug!(
                        path = %expected_path.display(),
                        length = file_info.length,
                        used_ifsc = slice_checksums.is_some(),
                        "File OK by name"
                    );
                    if let Ok(mut files) = verified_files.lock() {
                        files.insert(**file_id, expected_path);
                    }
                }
                Ok(false) => {
                    tracing::warn!(path = %expected_path.display(), "File damaged");
                    if let Some(ref msg_cb) = message_callback {
                        msg_cb(
                            MessageLevel::Warning,
                            &format!("File damaged: {}", expected_path.display()),
                        );
                    }
                    // Store path so repair can find it
                    if let Ok(mut files) = verified_files.lock() {
                        files.insert(**file_id, expected_path);
                    }
                    if let Ok(mut files) = damaged_files.lock() {
                        files.push(**file_id);
                    }
                }
                Err(_) => {
                    tracing::debug!(
                        path = %expected_path.display(),
                        "Error reading file by name"
                    );
                }
            }
        }
    });

    // Convert Arc<Mutex<T>> back to owned values
    let mut verified_files = Arc::try_unwrap(verified_files)
        .map_err(|_| Par2Error::RepairFailed("Failed to unwrap verified_files Arc".to_string()))?
        .into_inner()
        .map_err(|_| Par2Error::RepairFailed("Mutex poisoned: verified_files".to_string()))?;
    let damaged_files = Arc::try_unwrap(damaged_files)
        .map_err(|_| Par2Error::RepairFailed("Failed to unwrap damaged_files Arc".to_string()))?
        .into_inner()
        .map_err(|_| Par2Error::RepairFailed("Mutex poisoned: damaged_files".to_string()))?;
    let mut missing_files = Vec::new();

    // Second pass: Try to match remaining files by hash (for obfuscated names)
    // OPTIMIZATION: Build a size-based index for fast filtering (most files have unique sizes)
    let unmatched_file_ids: Vec<_> = par2_data
        .files
        .keys()
        .filter(|&&id| !verified_files.contains_key(&id) && !damaged_files.iter().any(|d| d == &id))
        .copied()
        .collect();

    if !unmatched_file_ids.is_empty() {
        // Build size -> file_ids map for O(1) size filtering
        let mut size_to_file_ids: HashMap<u64, Vec<FileHash>> = HashMap::new();
        for file_id in &unmatched_file_ids {
            if let Some(file_info) = par2_data.files.get(file_id) {
                size_to_file_ids
                    .entry(file_info.length)
                    .or_default()
                    .push(*file_id);
            }
        }

        // Filter extra_files to only those not already matched and get their sizes
        let extra_files_filtered: Vec<_> = extra_files
            .iter()
            .filter(|p| !verified_files.values().any(|v| v == *p))
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| (p.clone(), m.len())))
            .collect();

        // Process extra files in parallel for faster obfuscated file detection
        // Note: We pass None for progress here since we're just matching, not doing final verification
        // Progress will be updated after a match is found
        let matches: Vec<_> = extra_files_filtered
            .par_iter()
            .filter_map(|(extra_path, extra_size)| {
                // Quick filter: only check file_ids with matching size
                let candidate_ids = size_to_file_ids.get(extra_size)?;

                // Create intra-file progress callback
                let bytes_verified_ref = &bytes_verified;
                let progress_callback_ref = &progress_callback;
                let intra_progress = |bytes: u64| {
                    let done = bytes_verified_ref.fetch_add(bytes, Ordering::Relaxed) + bytes;
                    if let Some(ref cb) = progress_callback_ref {
                        cb(Par2Operation::Verifying, done, total_bytes);
                    }
                };

                for file_id in candidate_ids {
                    if let Some(file_info) = par2_data.files.get(file_id) {
                        let slice_checksums =
                            par2_data.slice_checksums.get(file_id).map(|v| v.as_slice());
                        if let Ok(true) = verify_file_smart(
                            extra_path,
                            file_info,
                            slice_checksums,
                            par2_data.block_size,
                            Some(&intra_progress),
                        ) {
                            return Some((*file_id, extra_path.clone(), file_info.name.clone()));
                        }
                    }
                }
                None
            })
            .collect();

        // Apply matches (need to dedupe in case multiple files matched same file_id)
        let mut matched_file_ids: std::collections::HashSet<FileHash> =
            std::collections::HashSet::new();
        for (file_id, extra_path, correct_name) in matches {
            if matched_file_ids.insert(file_id) {
                verified_files.insert(file_id, extra_path.clone());
                renamed_files.push((extra_path.clone(), correct_name.clone()));
                tracing::info!(
                    current = %extra_path.display(),
                    correct = %correct_name,
                    "Found obfuscated file"
                );
                if let Some(ref msg_cb) = message_callback {
                    msg_cb(
                        MessageLevel::Info,
                        &format!(
                            "Found obfuscated file: {} -> {}",
                            extra_path.file_name().unwrap_or_default().to_string_lossy(),
                            correct_name
                        ),
                    );
                }
            }
        }
    }

    // Rename obfuscated files to their correct names
    for (current_path, correct_name) in &renamed_files {
        let target_path = base_path.join(correct_name);

        // Only rename if target doesn't already exist
        if !target_path.exists() {
            if let Err(e) = std::fs::rename(current_path, &target_path) {
                tracing::warn!(
                    "Failed to rename {} to {}: {}",
                    current_path.display(),
                    correct_name,
                    e
                );
                if let Some(ref msg_cb) = message_callback {
                    msg_cb(
                        MessageLevel::Warning,
                        &format!(
                            "Failed to rename {} to {}: {}",
                            current_path.display(),
                            correct_name,
                            e
                        ),
                    );
                }
            } else {
                tracing::debug!("Renamed {} to {}", current_path.display(), correct_name);
            }
        }
    }

    // Third pass: Identify truly missing files
    for file_id in par2_data.files.keys() {
        if !verified_files.contains_key(file_id) && !damaged_files.iter().any(|id| id == file_id) {
            missing_files.push(*file_id);
        }
    }

    // Fourth pass: For damaged files, perform block-level verification if IFSC data available
    let mut block_damages = HashMap::new();

    for file_id in &damaged_files {
        // Check if IFSC data exists for this file
        if let Some(slice_checksums) = par2_data.slice_checksums.get(file_id) {
            if let Some(file_info) = par2_data.files.get(file_id) {
                // Find the file path (check both verified_files and base_path)
                let file_path = verified_files.get(file_id).cloned().or_else(|| {
                    let path = base_path.join(&file_info.name);
                    if path.exists() {
                        Some(path)
                    } else {
                        None
                    }
                });

                if let Some(path) = file_path {
                    // Perform block-level verification (no progress callback needed here)
                    match verify_file_blocks(
                        &path,
                        file_info,
                        slice_checksums,
                        par2_data.block_size,
                        None,
                    ) {
                        Ok(damaged_block_indices) => {
                            let total_blocks = slice_checksums.len();

                            if !damaged_block_indices.is_empty() {
                                tracing::info!(
                                    file = %file_info.name,
                                    damaged = damaged_block_indices.len(),
                                    total = total_blocks,
                                    "Block-level damage"
                                );
                                if let Some(ref msg_cb) = message_callback {
                                    msg_cb(
                                        MessageLevel::Info,
                                        &format!(
                                            "{}: {}/{} blocks damaged",
                                            file_info.name,
                                            damaged_block_indices.len(),
                                            total_blocks
                                        ),
                                    );
                                }

                                let block_damage = BlockDamage {
                                    file_id: *file_id,
                                    damaged_block_indices,
                                };
                                // Validate consistency: file_id should match the map key
                                debug_assert_eq!(
                                    block_damage.file_id, *file_id,
                                    "BlockDamage file_id must match HashMap key"
                                );
                                block_damages.insert(*file_id, block_damage);

                                // Store the path in verified_files so repair can read good blocks
                                verified_files.insert(*file_id, path);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(file = %file_info.name, error = %e, "Block verification failed");
                            if let Some(ref msg_cb) = message_callback {
                                msg_cb(
                                    MessageLevel::Warning,
                                    &format!(
                                        "Block verification failed for {}: {}",
                                        file_info.name, e
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    let all_verified = damaged_files.is_empty() && missing_files.is_empty();

    // Report missing files via message callback
    if !missing_files.is_empty() {
        if let Some(ref msg_cb) = message_callback {
            for file_id in &missing_files {
                if let Some(file_info) = par2_data.files.get(file_id) {
                    msg_cb(MessageLevel::Error, &format!("Missing: {}", file_info.name));
                }
            }
        }
    }

    tracing::info!(
        verified = verified_files.len(),
        damaged = damaged_files.len(),
        missing = missing_files.len(),
        block_level_damages = block_damages.len(),
        "Verification summary"
    );

    if let Some(ref cb) = progress_callback {
        cb(Par2Operation::Verifying, total_bytes, total_bytes);
    }

    Ok(VerificationResult {
        all_verified,
        verified_files,
        missing_files,
        damaged_files,
        block_damages,
    })
}

/// Progress callback for intra-file progress reporting
type ByteProgressCallback<'a> = Option<&'a dyn Fn(u64)>;

/// Verify individual blocks of a file using IFSC checksums
/// Returns: Vec of damaged block indices (empty = all blocks are good)
///
/// Uses chunked reading to handle large files efficiently without loading entire file into memory
fn verify_file_blocks(
    path: &Path,
    file_info: &FileInfo,
    slice_checksums: &[crate::parser::SliceChecksum],
    block_size: u64,
    byte_progress: ByteProgressCallback<'_>,
) -> Result<Vec<usize>> {
    use rayon::prelude::*;
    use std::io::{BufReader, Read, Seek};

    // Process blocks in batches for memory efficiency
    // Cap batch memory at 64MB regardless of block size
    const MAX_BATCH_MEMORY: usize = 64 * 1024 * 1024; // 64MB hard cap
    let block_size_usize = block_size as usize;
    let blocks_per_batch = (MAX_BATCH_MEMORY / block_size_usize).clamp(1, 64);
    let num_blocks = slice_checksums.len();

    let mut file = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut all_damaged = Vec::new();

    // Process in batches
    for batch_start in (0..num_blocks).step_by(blocks_per_batch) {
        let batch_end = (batch_start + blocks_per_batch).min(num_blocks);
        let batch_size = batch_end - batch_start;

        // Read batch of blocks sequentially
        file.seek(SeekFrom::Start(batch_start as u64 * block_size))?;
        let bytes_to_read = batch_size * block_size_usize;
        let mut batch_data = vec![0u8; bytes_to_read];
        let bytes_read = file.read(&mut batch_data)?;

        // Pad last block if needed
        if bytes_read < bytes_to_read {
            batch_data[bytes_read..].fill(0);
        }

        // Report progress after reading this batch
        if let Some(cb) = byte_progress {
            cb(bytes_read as u64);
        }

        // Verify blocks in this batch in parallel
        let batch_damaged: Vec<usize> = (0..batch_size)
            .into_par_iter()
            .filter_map(|local_idx| {
                let block_idx = batch_start + local_idx;
                let block_offset = local_idx * block_size_usize;
                let block_end = block_offset + block_size_usize;

                let block_data = &batch_data[block_offset..block_end];
                let expected = &slice_checksums[block_idx];

                // Fast path: Check CRC32 first (significantly cheaper than MD5)
                let block_crc32 = crc32fast::hash(block_data);

                // Early rejection if CRC doesn't match - skip expensive MD5
                if block_crc32 != expected.crc32 {
                    tracing::debug!(
                        file = %file_info.name,
                        block = block_idx,
                        "Block damaged (CRC mismatch)"
                    );
                    return Some(block_idx);
                }

                // CRC matches - verify with MD5 to confirm
                let block_md5 = crate::hash::compute_md5(block_data);

                // Check if damaged
                if block_md5 != expected.md5 {
                    tracing::debug!(
                        file = %file_info.name,
                        block = block_idx,
                        "Block damaged"
                    );
                    Some(block_idx)
                } else {
                    None
                }
            })
            .collect();

        all_damaged.extend(batch_damaged);
    }

    Ok(all_damaged)
}

/// Smart file verification: Use IFSC block-level checksums when available to avoid expensive full-file MD5
///
/// Returns:
/// - Ok(true) if file is valid
/// - Ok(false) if file exists but is damaged
/// - Err if file size doesn't match or can't be read
fn verify_file_smart(
    path: &Path,
    file_info: &FileInfo,
    slice_checksums: Option<&[crate::parser::SliceChecksum]>,
    block_size: u64,
    byte_progress: ByteProgressCallback<'_>,
) -> Result<bool> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;

    // Check file size first (quick check)
    if metadata.len() != file_info.length {
        return Err(Par2Error::RepairFailed(format!(
            "File size mismatch: {} (expected {}, got {})",
            path.display(),
            file_info.length,
            metadata.len()
        )));
    }

    // OPTIMIZATION: If we have IFSC block-level checksums, use them instead of full-file MD5!
    // This is much faster: single pass verifying blocks vs two passes (16K + full MD5)
    if let Some(checksums) = slice_checksums {
        tracing::trace!(
            file = %path.display(),
            blocks = checksums.len(),
            "Using IFSC block-level verification (skipping full MD5)"
        );

        let damaged_blocks =
            verify_file_blocks(path, file_info, checksums, block_size, byte_progress)?;

        // If all blocks verified successfully, file is good!
        return Ok(damaged_blocks.is_empty());
    }

    // No IFSC data available: fall back to traditional full-file MD5 verification
    tracing::trace!(
        file = %path.display(),
        "No IFSC data, using traditional MD5 verification"
    );

    // For small files (< 16KB), just do full hash
    if file_info.length < 16384 {
        let result = verify_full_hash(&mut file, &file_info.hash);
        if let Some(cb) = byte_progress {
            cb(file_info.length);
        }
        return result;
    }

    // For larger files, first check 16K hash (quick check)
    let hash_16k_matches = verify_16k_hash(&mut file, &file_info.hash_16k)?;
    if let Some(cb) = byte_progress {
        cb(16384); // Report 16K progress
    }

    if !hash_16k_matches {
        // Still report remaining bytes for progress
        if let Some(cb) = byte_progress {
            cb(file_info.length - 16384);
        }
        return Ok(false); // File is damaged
    }

    // 16K hash matches, now verify full file hash
    file.seek(SeekFrom::Start(0))?;

    verify_full_hash_with_progress(&mut file, &file_info.hash, byte_progress)
}

/// Verify first 16KB of file
fn verify_16k_hash(file: &mut File, expected_hash: &FileHash) -> Result<bool> {
    use md5::{Digest, Md5};

    file.seek(SeekFrom::Start(0))?;

    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 16384];
    let bytes_read = file.read(&mut buffer)?;

    hasher.update(&buffer[..bytes_read]);
    let hash: [u8; 16] = hasher.finalize().into();

    Ok(hash == *expected_hash)
}

/// Verify full file hash
fn verify_full_hash(file: &mut File, expected_hash: &FileHash) -> Result<bool> {
    verify_full_hash_with_progress(file, expected_hash, None)
}

/// Verify full file hash with progress reporting
fn verify_full_hash_with_progress(
    file: &mut File,
    expected_hash: &FileHash,
    byte_progress: ByteProgressCallback<'_>,
) -> Result<bool> {
    use md5::{Digest, Md5};

    file.seek(SeekFrom::Start(0))?;

    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 65536]; // 64KB buffer
    let mut bytes_since_report = 0u64;
    const REPORT_INTERVAL: u64 = 1024 * 1024; // Report every 1MB

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);

        // Report progress periodically
        bytes_since_report += bytes_read as u64;
        if bytes_since_report >= REPORT_INTERVAL {
            if let Some(cb) = byte_progress {
                cb(bytes_since_report);
            }
            bytes_since_report = 0;
        }
    }

    // Report any remaining bytes
    if bytes_since_report > 0 {
        if let Some(cb) = byte_progress {
            cb(bytes_since_report);
        }
    }

    let hash: [u8; 16] = hasher.finalize().into();
    Ok(hash == *expected_hash)
}
