//! PAR2 multi-volume file splitting
//!
//! This module handles splitting recovery blocks across multiple PAR2 volume files.
//! Volume splitting allows users to download only the recovery data they need.
//!
//! # Volume Schemes
//!
//! - [`VolumeScheme::Single`]: All recovery blocks in one file
//! - [`VolumeScheme::Exponential`]: Blocks distributed as 1, 2, 4, 8... per file
//!
//! # Filename Format
//!
//! Volume files follow the PAR2 naming convention:
//! `basename.volSTART+COUNT.par2` (e.g., `archive.vol00+02.par2`)
//!
//! # Related Modules
//!
//! - [`crate::creator`]: Uses this module during PAR2 creation
//! - [`crate::parser`]: Reads volumes during loading

use crate::error::Result;
use crate::parser::RecoveryBlock;
use std::path::{Path, PathBuf};

/// Volume distribution scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeScheme {
    /// All recovery blocks in a single .par2 file
    Single,
    /// Exponential distribution: 1, 2, 4, 8, 16... blocks per file
    Exponential,
}

/// Information about a PAR2 volume file
#[derive(Debug, Clone)]
pub struct VolumeInfo {
    /// Full path to the volume file
    pub path: PathBuf,
    /// Starting exponent (inclusive)
    pub exponent_start: u32,
    /// Ending exponent (inclusive)
    pub exponent_end: u32,
    /// Recovery blocks in this volume
    pub recovery_blocks: Vec<RecoveryBlock>,
}

/// Calculate zero-padding width for volume filenames
/// Based on the maximum exponent value
fn calculate_padding_width(max_exponent: u32) -> usize {
    if max_exponent == 0 {
        return 1;
    }
    // Calculate number of digits needed
    let mut width = 0;
    let mut n = max_exponent;
    while n > 0 {
        width += 1;
        n /= 10;
    }
    width
}

/// Generate volume filename in PAR2 format
/// Format: basename.vol000+001.par2 or basename.vol000-001.par2
fn generate_volume_filename(
    base_name: &str,
    start_exp: u32,
    end_exp: u32,
    padding_width: usize,
) -> String {
    let count = end_exp - start_exp + 1;
    format!(
        "{}.vol{:0width$}+{:0width$}.par2",
        base_name,
        start_exp,
        count,
        width = padding_width
    )
}

/// Split recovery blocks into volumes according to scheme
///
/// # Arguments
/// * `recovery_blocks` - All recovery blocks (exponents 0..N)
/// * `scheme` - Volume distribution scheme
/// * `base_path` - Base path/name for output files (e.g., "archive" or "/path/to/archive")
///
/// # Returns
/// Vector of volume information, each containing recovery blocks and filename
pub fn split_into_volumes(
    recovery_blocks: Vec<(u32, Vec<u8>)>,
    scheme: VolumeScheme,
    base_path: &Path,
) -> Result<Vec<VolumeInfo>> {
    if recovery_blocks.is_empty() {
        return Ok(Vec::new());
    }

    // Extract base name from path
    let base_name = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("archive");

    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));

    match scheme {
        VolumeScheme::Single => {
            // All recovery blocks in one file: basename.vol000-NNN.par2
            let max_exponent = recovery_blocks.last().map(|(e, _)| *e).unwrap_or(0);
            let padding_width = calculate_padding_width(max_exponent);

            let filename = generate_volume_filename(base_name, 0, max_exponent, padding_width);

            let blocks: Vec<RecoveryBlock> = recovery_blocks
                .into_iter()
                .map(|(exp, data)| RecoveryBlock::from_memory(exp, data))
                .collect();

            let volume_info = VolumeInfo {
                path: parent.join(filename),
                exponent_start: 0,
                exponent_end: max_exponent,
                recovery_blocks: blocks,
            };
            // Validate exponent range
            debug_assert!(
                volume_info.exponent_start <= volume_info.exponent_end,
                "Volume exponent_start must be <= exponent_end"
            );
            Ok(vec![volume_info])
        }

        VolumeScheme::Exponential => {
            // Exponential scheme: 1, 2, 4, 8, 16... blocks per file
            let max_exponent = recovery_blocks.last().map(|(e, _)| *e).unwrap_or(0);
            let padding_width = calculate_padding_width(max_exponent);

            let mut volumes = Vec::new();
            let mut blocks_in_current = 0usize;
            let mut target_count = 1usize;

            let mut current_volume_blocks = Vec::new();
            let mut volume_start = 0u32;
            let mut volume_end = 0u32;

            for (exponent, data) in recovery_blocks {
                // Start new volume if this is the first block
                if blocks_in_current == 0 {
                    volume_start = exponent;
                }

                current_volume_blocks.push(RecoveryBlock::from_memory(exponent, data));
                blocks_in_current += 1;
                volume_end = exponent;

                // Check if we've reached the target count for this volume
                if blocks_in_current >= target_count {
                    let filename = generate_volume_filename(
                        base_name,
                        volume_start,
                        volume_end,
                        padding_width,
                    );

                    let volume_info = VolumeInfo {
                        path: parent.join(filename),
                        exponent_start: volume_start,
                        exponent_end: volume_end,
                        recovery_blocks: current_volume_blocks,
                    };
                    // Validate exponent range consistency
                    debug_assert!(
                        volume_info.exponent_start <= volume_info.exponent_end,
                        "Volume exponent_start must be <= exponent_end"
                    );
                    volumes.push(volume_info);

                    // Prepare for next volume
                    current_volume_blocks = Vec::new();
                    blocks_in_current = 0;
                    target_count = std::cmp::min(target_count * 2, 1024); // Cap at 1024 blocks per volume
                }
            }

            // Handle any remaining blocks
            if !current_volume_blocks.is_empty() {
                let filename =
                    generate_volume_filename(base_name, volume_start, volume_end, padding_width);

                let volume_info = VolumeInfo {
                    path: parent.join(filename),
                    exponent_start: volume_start,
                    exponent_end: volume_end,
                    recovery_blocks: current_volume_blocks,
                };
                // Validate exponent range consistency
                debug_assert!(
                    volume_info.exponent_start <= volume_info.exponent_end,
                    "Volume exponent_start must be <= exponent_end"
                );
                volumes.push(volume_info);
            }

            Ok(volumes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_padding_width() {
        assert_eq!(calculate_padding_width(0), 1);
        assert_eq!(calculate_padding_width(9), 1);
        assert_eq!(calculate_padding_width(10), 2);
        assert_eq!(calculate_padding_width(99), 2);
        assert_eq!(calculate_padding_width(100), 3);
        assert_eq!(calculate_padding_width(999), 3);
        assert_eq!(calculate_padding_width(1000), 4);
    }

    #[test]
    fn test_generate_volume_filename() {
        assert_eq!(
            generate_volume_filename("test", 0, 0, 1),
            "test.vol0+1.par2"
        );
        assert_eq!(
            generate_volume_filename("test", 0, 1, 1),
            "test.vol0+2.par2"
        );
        assert_eq!(
            generate_volume_filename("test", 0, 9, 2),
            "test.vol00+10.par2"
        );
        assert_eq!(
            generate_volume_filename("test", 10, 19, 2),
            "test.vol10+10.par2"
        );
        assert_eq!(
            generate_volume_filename("archive", 0, 99, 3),
            "archive.vol000+100.par2"
        );
    }

    #[test]
    fn test_split_single_volume() {
        let recovery_blocks = vec![
            (0, vec![1, 2, 3, 4]),
            (1, vec![5, 6, 7, 8]),
            (2, vec![9, 10, 11, 12]),
        ];

        let volumes =
            split_into_volumes(recovery_blocks, VolumeScheme::Single, Path::new("test")).unwrap();

        assert_eq!(volumes.len(), 1, "Single scheme should create 1 volume");
        assert_eq!(volumes[0].exponent_start, 0);
        assert_eq!(volumes[0].exponent_end, 2);
        assert_eq!(volumes[0].recovery_blocks.len(), 3);
        assert!(volumes[0].path.to_str().unwrap().contains("vol0+3.par2"));
    }

    #[test]
    fn test_split_exponential_small() {
        // Test with 7 blocks: should create 4 volumes (1, 2, 4, and remaining)
        let recovery_blocks: Vec<(u32, Vec<u8>)> = (0..7).map(|i| (i, vec![i as u8; 4])).collect();

        let volumes = split_into_volumes(
            recovery_blocks,
            VolumeScheme::Exponential,
            Path::new("test"),
        )
        .unwrap();

        // Exponential: 1 block, 2 blocks, 4 blocks
        // 1 + 2 + 4 = 7 blocks total
        assert_eq!(volumes.len(), 3, "Should create 3 volumes");

        // First volume: 1 block (exponent 0)
        assert_eq!(volumes[0].exponent_start, 0);
        assert_eq!(volumes[0].exponent_end, 0);
        assert_eq!(volumes[0].recovery_blocks.len(), 1);
        assert!(volumes[0].path.to_str().unwrap().contains("vol0+1.par2"));

        // Second volume: 2 blocks (exponents 1-2)
        assert_eq!(volumes[1].exponent_start, 1);
        assert_eq!(volumes[1].exponent_end, 2);
        assert_eq!(volumes[1].recovery_blocks.len(), 2);
        assert!(volumes[1].path.to_str().unwrap().contains("vol1+2.par2"));

        // Third volume: 4 blocks (exponents 3-6)
        assert_eq!(volumes[2].exponent_start, 3);
        assert_eq!(volumes[2].exponent_end, 6);
        assert_eq!(volumes[2].recovery_blocks.len(), 4);
        assert!(volumes[2].path.to_str().unwrap().contains("vol3+4.par2"));
    }

    #[test]
    fn test_split_exponential_padding() {
        // Test padding with 100 blocks (0-99, requires 2-digit padding)
        let recovery_blocks: Vec<(u32, Vec<u8>)> =
            (0..100).map(|i| (i, vec![i as u8; 4])).collect();

        let volumes = split_into_volumes(
            recovery_blocks,
            VolumeScheme::Exponential,
            Path::new("test"),
        )
        .unwrap();

        // With max exponent 99, padding width should be 2
        // Check that filenames have proper padding
        for volume in &volumes {
            let filename = volume.path.file_name().unwrap().to_str().unwrap();
            // Should have format like "test.vol00+01.par2" with 2-digit padding
            assert!(filename.contains("vol"), "Filename should contain 'vol'");

            // Extract the numbers and verify padding
            if let Some(vol_part) = filename.strip_prefix("test.vol") {
                if let Some(end) = vol_part.find('+') {
                    let start_str = &vol_part[..end];
                    // Should be 2 digits (since max exp is 99)
                    assert_eq!(
                        start_str.len(),
                        2,
                        "Start exponent should be 2 digits for max exp 99"
                    );
                }
            }
        }
    }

    #[test]
    fn test_empty_recovery_blocks() {
        let recovery_blocks = vec![];
        let volumes =
            split_into_volumes(recovery_blocks, VolumeScheme::Single, Path::new("test")).unwrap();

        assert_eq!(
            volumes.len(),
            0,
            "No recovery blocks should create no volumes"
        );
    }

    #[test]
    fn test_volume_with_parent_path() {
        let recovery_blocks = vec![(0, vec![1, 2, 3, 4])];

        let volumes = split_into_volumes(
            recovery_blocks,
            VolumeScheme::Single,
            Path::new("/tmp/test"),
        )
        .unwrap();

        assert_eq!(volumes.len(), 1);
        assert!(volumes[0].path.starts_with("/tmp"));
        assert!(volumes[0].path.to_str().unwrap().contains("test.vol"));
    }
}
