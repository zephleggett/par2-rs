//! Read-only PAR2 metadata inspection.
//!
//! This module provides [`Par2Info`], a lightweight, read-only view of a PAR2
//! recovery set. It parses the index (or any volume) file plus its sibling
//! `.vol*.par2` files and exposes the protected file list and recovery capacity
//! without performing any verification or repair (and without touching the
//! protected data files at all).
//!
//! # Example
//!
//! ```no_run
//! use par2_rs::Par2Info;
//! use std::path::Path;
//!
//! let info = Par2Info::load(Path::new("archive.par2")).unwrap();
//! println!("block size: {}", info.block_size);
//! println!("files: {}", info.files.len());
//! println!("usable recovery blocks: {}", info.distinct_recovery_blocks);
//! // Can we repair if 3 distinct data blocks are damaged?
//! assert_eq!(info.can_repair(3), 3 <= info.distinct_recovery_blocks);
//! ```

use crate::error::Result;
use crate::parser::Par2File;
use std::collections::HashSet;
use std::path::Path;

/// Metadata about a single protected file within a PAR2 recovery set.
#[derive(Debug, Clone)]
pub struct Par2FileEntry {
    /// File name as recorded in the PAR2 set.
    pub name: String,
    /// MD5 of the first 16 KiB (or the whole file if it is smaller).
    pub hash_16k: [u8; 16],
    /// Whole-file MD5.
    pub hash: [u8; 16],
    /// File length in bytes.
    pub length: u64,
}

/// Read-only metadata for a PAR2 recovery set.
///
/// Construct with [`Par2Info::load`]. This never reads or modifies the protected
/// data files; it only parses PAR2 metadata.
#[derive(Debug, Clone)]
pub struct Par2Info {
    /// Recovery block size in bytes.
    pub block_size: u64,
    /// Protected files (deduplicated by file id).
    pub files: Vec<Par2FileEntry>,
    /// Raw count of recovery blocks found across all loaded volumes.
    ///
    /// This is an upper bound on capacity: a set may contain duplicate
    /// recovery blocks (same exponent) which do not add usable capacity.
    pub recovery_block_count: usize,
    /// Number of distinct recovery-block exponents = the true number of
    /// data blocks that can actually be reconstructed.
    pub distinct_recovery_blocks: usize,
}

impl Par2Info {
    /// Parse a PAR2 set starting from `par2_file` (the index file or any
    /// `.vol*.par2` file). Sibling volume files in the same directory that share
    /// the same recovery-set id are discovered and their recovery blocks counted
    /// automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a valid PAR2 file.
    pub fn load(par2_file: &Path) -> Result<Self> {
        let base_dir = par2_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        // Par2File::load(path, base_path, progress_callback)
        let p = Par2File::load(par2_file, &base_dir, None)?;

        // Build file list from the file_id-deduped HashMap, NOT files_in_order
        // (which may contain duplicate descriptors across volumes).
        let files: Vec<Par2FileEntry> = p
            .files
            .values()
            .map(|fi| Par2FileEntry {
                name: fi.name.clone(),
                hash_16k: fi.hash_16k,
                hash: fi.hash,
                length: fi.length,
            })
            .collect();

        let recovery_block_count = p.recovery_blocks.len();
        let distinct_recovery_blocks = p
            .recovery_blocks
            .iter()
            .map(|b| b.exponent)
            .collect::<HashSet<_>>()
            .len();

        Ok(Self {
            block_size: p.block_size,
            files,
            recovery_block_count,
            distinct_recovery_blocks,
        })
    }

    /// Total number of data blocks across all protected files:
    /// `sum(ceil(len / block_size))`.
    pub fn total_data_blocks(&self) -> u64 {
        let bs = self.block_size.max(1);
        self.files.iter().map(|f| f.length.div_ceil(bs)).sum()
    }

    /// Returns `true` if `damaged_blocks` distinct data blocks can be
    /// reconstructed from the available recovery capacity.
    pub fn can_repair(&self, damaged_blocks: usize) -> bool {
        damaged_blocks <= self.distinct_recovery_blocks
    }
}
