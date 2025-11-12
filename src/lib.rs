// Pure Rust PAR2 implementation for file verification and repair
// Based on PAR2 specification: https://parchive.sourceforge.net/docs/specifications/parity-volume-spec/article-spec.html

mod galois;
mod par2_rs;
mod parser;
mod repair;
mod verify;

pub mod error;

#[cfg(test)]
mod tests;

pub use error::{Par2Error, Result};

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
    /// * `Err(DlNzbError)` - Verification/repair failed
    pub fn repair_with_progress(
        &self,
        do_repair: bool,
        purge_files: bool,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<()> {
        // Step 1: Scan for all files in directory
        if let Some(ref cb) = progress_callback {
            cb(Par2Operation::Scanning, 0, 100);
        }

        let extra_files = self.scan_directory()?;

        // Step 2: Load and parse PAR2 file
        if let Some(ref cb) = progress_callback {
            cb(Par2Operation::Loading, 0, 100);
        }

        let par2_data = parser::Par2File::load(&self.par2_file, &self.base_path, progress_callback.clone())?;

        // Step 3: Verify files
        if let Some(ref cb) = progress_callback {
            cb(Par2Operation::Verifying, 0, 100);
        }

        let verification_result = verify::verify_files(
            &par2_data,
            &extra_files,
            &self.base_path,
            progress_callback.clone(),
        )?;

        // Step 4: Repair if needed and requested
        if !verification_result.all_verified && do_repair {
            if let Some(ref cb) = progress_callback {
                cb(Par2Operation::Repairing, 0, 100);
            }

            repair::repair_files(
                &par2_data,
                &verification_result,
                &self.base_path,
                progress_callback.clone(),
            )?;
        } else if !verification_result.all_verified {
            return Err(Par2Error::RepairFailed(
                "Files are damaged and repair was not requested".to_string(),
            )
            );
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

    /// Scan directory for all non-PAR2 files (for hash-based matching)
    fn scan_directory(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();

            // Skip directories and PAR2 files
            if path.is_file() {
                let is_par2 = path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.to_lowercase() == "par2")
                    .unwrap_or(false);

                if !is_par2 {
                    files.push(path);
                }
            }
        }

        Ok(files)
    }

    /// Delete PAR2 files after successful verification/repair
    fn purge_par2_files(&self) -> Result<()> {
        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();

            let is_par2 = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_lowercase() == "par2")
                .unwrap_or(false);

            if is_par2 {
                if let Err(e) = std::fs::remove_file(&path) {
                    tracing::warn!("Failed to delete PAR2 file {}: {}", path.display(), e);
                }
            }
        }

        Ok(())
    }
}
