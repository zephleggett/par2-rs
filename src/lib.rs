//! # par2-rs
//!
//! A pure Rust implementation of the PAR2 (Parchive 2.0) file verification and repair format.
//!
//! ## What is PAR2?
//!
//! PAR2 files allow you to verify file integrity and repair corrupted or missing data using
//! Reed-Solomon error correction. This is commonly used for:
//!
//! - Verifying and repairing downloads (especially from Usenet)
//! - Protecting archived files from bit rot
//! - Recovering from damaged storage media
//!
//! ## Quick Example
//!
//! ```no_run
//! use par2_rs::{Par2Repairer, Result};
//! use std::path::Path;
//!
//! fn main() -> Result<()> {
//!     // Create a repairer for a PAR2 file
//!     let repairer = Par2Repairer::new(Path::new("myfiles.par2"))?;
//!
//!     // Verify files (no repair)
//!     repairer.repair(false)?;
//!
//!     // Or verify and repair if damaged
//!     repairer.repair(true)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - Fast parallel processing using rayon
//! - Multi-volume PAR2 support (automatically loads `.vol*.par2` files)
//! - Content-based file matching (works with renamed files)
//! - Reed-Solomon GF(2^16) error correction
//! - Memory efficient streaming for large files
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`galois`] - Galois field GF(2^16) arithmetic for Reed-Solomon codes
//! - [`error`] - Error types and results
//! - Internal modules for parsing, verification, and repair
//!
//! Based on the [PAR 2.0 specification](https://parchive.sourceforge.net/docs/specifications/parity-volume-spec/article-spec.html).

mod creator;
pub mod galois;
pub mod hash;
mod par2_rs;
mod parser;
mod repair;
mod verify;
mod volumes;
mod writer;

pub mod error;

pub use creator::Par2Creator;
pub use error::{Par2Error, Result};
pub use volumes::VolumeScheme;

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// PAR2 operation type for progress tracking
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Par2Operation {
    Scanning = 0,
    Loading = 1,
    Verifying = 2,
    Repairing = 3,
}

/// Progress callback: (operation, current, total)
pub type ProgressCallback = Arc<dyn Fn(Par2Operation, u64, u64) + Send + Sync>;

/// Main PAR2 repair and verification interface
pub struct Par2Repairer {
    par2_file: PathBuf,
    base_path: PathBuf,
}

impl Par2Repairer {
    /// Create a new PAR2 repairer for the given PAR2 file
    pub fn new(par2_file: &Path) -> Result<Self> {
        if !par2_file.exists() {
            return Err(Par2Error::NotFound);
        }

        let base_path = par2_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        Ok(Self {
            par2_file: par2_file.to_path_buf(),
            base_path,
        })
    }

    /// Perform PAR2 verification and optional repair
    ///
    /// # Arguments
    /// * `do_repair` - If true, perform repair; if false, only verify
    /// * `purge_files` - If true, delete PAR2 files after successful repair
    /// * `progress_callback` - Optional callback for progress updates
    ///
    /// # Returns
    /// * `Ok(())` - Files were correct or successfully repaired
    /// * `Err(Par2Error)` - Verification/repair failed
    pub fn repair_with_progress(
        &self,
        do_repair: bool,
        purge_files: bool,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<()> {
        use std::time::Instant;

        // Step 1: Load and parse PAR2 file
        if let Some(ref cb) = progress_callback {
            cb(Par2Operation::Loading, 0, 100);
        }

        let load_start = Instant::now();
        let par2_data =
            parser::Par2File::load(&self.par2_file, &self.base_path, progress_callback.clone())?;
        tracing::debug!(
            elapsed_secs = load_start.elapsed().as_secs_f64(),
            "PAR2 data loaded"
        );

        // Step 2: Scan for files mentioned in PAR2 metadata
        if let Some(ref cb) = progress_callback {
            cb(Par2Operation::Scanning, 0, 100);
        }

        let scan_start = Instant::now();
        let extra_files = self.scan_directory(&par2_data.files)?;
        tracing::debug!(
            elapsed_secs = scan_start.elapsed().as_secs_f64(),
            "File scan completed"
        );

        // Step 3: Verify files
        if let Some(ref cb) = progress_callback {
            cb(Par2Operation::Verifying, 0, 100);
        }

        let verify_start = Instant::now();
        let verification_result = verify::verify_files(
            &par2_data,
            &extra_files,
            &self.base_path,
            progress_callback.clone(),
        )?;
        tracing::debug!(
            elapsed_secs = verify_start.elapsed().as_secs_f64(),
            "Verification completed"
        );

        // Step 4: Repair if needed and requested
        if !verification_result.all_verified && do_repair {
            if let Some(ref cb) = progress_callback {
                cb(Par2Operation::Repairing, 0, 100);
            }

            let repair_start = Instant::now();
            repair::repair_files_parallel(
                &par2_data,
                &verification_result,
                &self.base_path,
                progress_callback.clone(),
            )?;
            tracing::debug!(
                elapsed_secs = repair_start.elapsed().as_secs_f64(),
                "Repair completed"
            );
        } else if !verification_result.all_verified {
            return Err(Par2Error::RepairFailed(
                "Files are damaged and repair was not requested".to_string(),
            ));
        }

        // Step 5: Purge PAR2 files if requested
        if purge_files && verification_result.all_verified {
            self.purge_par2_files()?;
        }

        Ok(())
    }

    /// Simplified interface without progress callback
    pub fn repair(&self, do_repair: bool) -> Result<()> {
        self.repair_with_progress(do_repair, false, None)
    }

    /// Scan directory for all files to enable renamed file detection
    /// Returns all regular files in the base directory (excluding PAR2 files)
    fn scan_directory(
        &self,
        _par2_files: &std::collections::HashMap<parser::FileHash, parser::FileInfo>,
    ) -> Result<Vec<PathBuf>> {
        use std::fs;

        // Scan all files in the directory to enable renamed file detection
        let mut files = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.base_path) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        let path = entry.path();
                        // Skip PAR2 files
                        if let Some(ext) = path.extension() {
                            if ext == "par2" {
                                continue;
                            }
                        }
                        files.push(path);
                    }
                }
            }
        }

        tracing::debug!(
            file_count = files.len(),
            "Scanned all files in directory for verification"
        );

        Ok(files)
    }

    /// Delete PAR2 files after successful verification/repair
    /// Only deletes files belonging to the same recovery set to prevent data loss
    fn purge_par2_files(&self) -> Result<()> {
        // Get our recovery set ID first
        let our_recovery_set_id = parser::get_recovery_set_id(&self.par2_file)?;

        tracing::info!(
            "Purging PAR2 files for recovery set {:?}",
            our_recovery_set_id
        );

        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();

            let is_par2 = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_lowercase() == "par2")
                .unwrap_or(false);

            if is_par2 {
                // Check if this PAR2 file belongs to our recovery set
                match parser::get_recovery_set_id(&path) {
                    Ok(recovery_set_id) => {
                        if recovery_set_id == our_recovery_set_id {
                            tracing::info!("Deleting PAR2 file: {}", path.display());
                            if let Err(e) = std::fs::remove_file(&path) {
                                tracing::warn!(
                                    "Failed to delete PAR2 file {}: {}",
                                    path.display(),
                                    e
                                );
                            }
                        } else {
                            tracing::debug!(
                                "Skipping PAR2 file with different recovery set: {}",
                                path.display()
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Could not read recovery_set_id from {}: {}",
                            path.display(),
                            e
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
