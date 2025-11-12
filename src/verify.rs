// PAR2 file verification with hash-based matching for obfuscated filenames

use super::parser::{FileHash, FileInfo, Par2File};
use super::{Par2Operation, ProgressCallback};
use crate::error::{Par2Error, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};



/// Result of file verification
#[derive(Debug)]
pub struct VerificationResult {
    pub all_verified: bool,
    pub verified_files: HashMap<FileHash, PathBuf>,
    pub missing_files: Vec<FileHash>,
    pub damaged_files: Vec<FileHash>,
    pub renamed_files: Vec<(PathBuf, String)>, // (current_path, correct_name)
}

/// Verify files against PAR2 data
pub fn verify_files(
    par2_data: &Par2File,
    extra_files: &[PathBuf],
    base_path: &Path,
    progress_callback: Option<ProgressCallback>,
) -> Result<VerificationResult> {
    let total_files = par2_data.files.len() as u64;
    let mut verified_count = 0u64;

    let mut verified_files = HashMap::new();
    let mut missing_files = Vec::new();
    let mut damaged_files = Vec::new();
    let mut renamed_files = Vec::new();

    // First pass: Try to match files by name
    for (file_id, file_info) in &par2_data.files {
        if let Some(ref cb) = progress_callback {
            cb(Par2Operation::Verifying, verified_count, total_files);
        }

        let expected_path = base_path.join(&file_info.name);

        if expected_path.exists() {
            match verify_file(&expected_path, file_info) {
                Ok(true) => {
                    verified_files.insert(*file_id, expected_path);
                    verified_count += 1;
                    continue;
                }
                Ok(false) => {
                    // File exists but is damaged
                    damaged_files.push(*file_id);
                    verified_count += 1;
                    continue;
                }
                Err(_) => {
                    // Error reading file
                }
            }
        }

        // File not found or couldn't be read - will try hash matching in second pass
    }

    // Second pass: Try to match remaining files by hash (for obfuscated names)
    let unmatched_file_ids: Vec<_> = par2_data
        .files
        .keys()
        .filter(|&&id| !verified_files.contains_key(&id) && !damaged_files.iter().any(|d| d == &id))
        .copied()
        .collect();

    if !unmatched_file_ids.is_empty() {
        for extra_path in extra_files {
            // Skip files we already matched
            if verified_files.values().any(|p| p == extra_path) {
                continue;
            }

            // Try to match this file against unmatched file_ids by hash
            for file_id in &unmatched_file_ids {
                if let Some(file_info) = par2_data.files.get(file_id) {
                    match verify_file(extra_path, file_info) {
                        Ok(true) => {
                            // Found a match! This file should be renamed
                            verified_files.insert(*file_id, extra_path.clone());
                            renamed_files.push((extra_path.clone(), file_info.name.clone()));
                            verified_count += 1;
                            break;
                        }
                        Ok(false) => {
                            // Hash matches but file is damaged
                            damaged_files.push(*file_id);
                            verified_count += 1;
                            break;
                        }
                        Err(_) => {
                            // Doesn't match this file_info, try next
                            continue;
                        }
                    }
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

    let all_verified = damaged_files.is_empty() && missing_files.is_empty();

    if let Some(ref cb) = progress_callback {
        cb(Par2Operation::Verifying, total_files, total_files);
    }

    Ok(VerificationResult {
        all_verified,
        verified_files,
        missing_files,
        damaged_files,
        renamed_files,
    })
}

/// Verify a single file against expected hashes
///
/// Returns:
/// - Ok(true) if file is valid
/// - Ok(false) if file exists but is damaged
/// - Err if file size doesn't match or can't be read
fn verify_file(path: &Path, file_info: &FileInfo) -> Result<bool> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;

    // Check file size first (quick check)
    if metadata.len() != file_info.length {
        return Err(Par2Error::RepairFailed(format!(
            "File size mismatch: {} (expected {}, got {})",
            path.display(),
            file_info.length,
            metadata.len()
        ))
        );
    }

    // For small files (< 16KB), just do full hash
    if file_info.length < 16384 {
        return Ok(verify_full_hash(&mut file, &file_info.hash)?);
    }

    // For larger files, first check 16K hash (quick check)
    let hash_16k_matches = verify_16k_hash(&mut file, &file_info.hash_16k)?;

    if !hash_16k_matches {
        return Ok(false); // File is damaged
    }

    // 16K hash matches, now verify full file hash
    file.seek(SeekFrom::Start(0))?;
    Ok(verify_full_hash(&mut file, &file_info.hash)?)
}

/// Verify first 16KB of file
fn verify_16k_hash(file: &mut File, expected_hash: &FileHash) -> Result<bool> {
    file.seek(SeekFrom::Start(0))?;

    let mut context = md5::Context::new();
    let mut buffer = vec![0u8; 16384];
    let bytes_read = file.read(&mut buffer)?;

    context.consume(&buffer[..bytes_read]);
    let hash = context.compute();

    Ok(hash.0 == *expected_hash)
}

/// Verify full file hash
fn verify_full_hash(file: &mut File, expected_hash: &FileHash) -> Result<bool> {
    file.seek(SeekFrom::Start(0))?;

    let mut context = md5::Context::new();
    let mut buffer = vec![0u8; 65536]; // 64KB buffer

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        context.consume(&buffer[..bytes_read]);
    }

    let hash = context.compute();
    Ok(hash.0 == *expected_hash)
}

/// Compute MD5 hash of a file
pub fn hash_file(path: &Path) -> Result<FileHash> {
    let mut file = File::open(path)?;
    let mut context = md5::Context::new();
    let mut buffer = vec![0u8; 65536];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        context.consume(&buffer[..bytes_read]);
    }

    let hash = context.compute();
    Ok(hash.0)
}
