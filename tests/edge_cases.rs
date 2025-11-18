//! Edge case tests - invalid inputs, error conditions, boundary cases

use par2_rs::{Par2Creator, Par2Repairer};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

mod common;
use common::{corrupt_file, create_pattern_file};

// ============================================================================
// Invalid Inputs
// ============================================================================

#[test]
fn test_invalid_par2_file() {
    let temp = TempDir::new().unwrap();
    let fake_par2 = temp.path().join("fake.par2");
    fs::write(&fake_par2, b"INVALID_MAGIC_HEADER").unwrap();

    let repairer = Par2Repairer::new(&fake_par2).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should reject invalid PAR2 file"
    );
}

#[test]
fn test_nonexistent_par2_file() {
    let result = Par2Repairer::new(Path::new("/nonexistent/file.par2"));
    assert!(result.is_err(), "Should error on nonexistent file");
}

#[test]
fn test_empty_input_files() {
    let result = Par2Creator::new(vec![]);
    assert!(result.is_err(), "Should reject empty file list");
}

#[test]
fn test_nonexistent_input_file() {
    let result = Par2Creator::new(vec![Path::new("/nonexistent/file.bin").to_path_buf()]);
    assert!(result.is_err(), "Should reject nonexistent input file");
}

// ============================================================================
// Block Size Validation
// ============================================================================

#[test]
fn test_block_size_not_multiple_of_4() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 1000).unwrap();

    let creator = Par2Creator::new(vec![file]).unwrap();
    let result = creator.with_block_size(1001);

    assert!(result.is_err(), "Should reject block size not multiple of 4");
}

#[test]
fn test_block_size_zero() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 1000).unwrap();

    let creator = Par2Creator::new(vec![file]).unwrap();
    let result = creator.with_block_size(0);

    assert!(result.is_err(), "Should reject zero block size");
}

#[test]
fn test_block_size_too_small() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 10000).unwrap();

    let creator = Par2Creator::new(vec![file]).unwrap();
    let result = creator.with_block_size(4); // Less than MIN_BLOCK_SIZE (2048)

    assert!(result.is_err(), "Should reject block size less than 2KB");
}

// ============================================================================
// File Damage Detection
// ============================================================================

#[test]
fn test_truncated_file_detection() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"ORIGINAL", 10000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Truncate file
    let data = fs::read(&file).unwrap();
    fs::write(&file, &data[..5000]).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect truncated file"
    );
}

#[test]
fn test_extended_file_detection() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"ORIGINAL", 5000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Extend file with extra data
    let mut data = fs::read(&file).unwrap();
    data.extend_from_slice(b"EXTRA_DATA");
    fs::write(&file, data).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect extended file"
    );
}

#[test]
fn test_block_level_damage() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"TESTDATA", 10000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Corrupt specific blocks
    corrupt_file(&file, 2048, &[0xFF; 100]).unwrap();
    corrupt_file(&file, 6144, &[0xFF; 100]).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect block-level damage"
    );
}

// ============================================================================
// Redundancy Edge Cases
// ============================================================================

#[test]
fn test_zero_redundancy() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 5000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.with_redundancy(0.0).create().unwrap();

    // Should create PAR2 with zero redundancy
    assert!(!par2_files.is_empty());

    // Should verify intact file even with no recovery blocks
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(repairer.repair(false).is_ok());

    // But can't repair damage with zero redundancy
    corrupt_file(&file, 100, &[0xFF; 100]).unwrap();
    assert!(repairer.repair(false).is_err());
    assert!(repairer.repair(true).is_err(), "Can't repair with 0% redundancy");
}

#[test]
fn test_insufficient_redundancy() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 10000).unwrap();

    // Create with only 5% redundancy
    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(5.0);
    let par2_files = creator.create().unwrap();

    // Damage 50% of file (way more than 5% redundancy)
    corrupt_file(&file, 0, &[0xFF; 5000]).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(true).is_err(),
        "Should fail to repair when damage exceeds redundancy"
    );
}

// ============================================================================
// Multiple File Edge Cases
// ============================================================================

#[test]
fn test_missing_one_of_multiple_files() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("file1.bin");
    let file2 = temp.path().join("file2.bin");

    create_pattern_file(&file1, b"FILE1", 5000).unwrap();
    create_pattern_file(&file2, b"FILE2", 5000).unwrap();

    let creator = Par2Creator::new(vec![file1.clone(), file2.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Delete one file
    fs::remove_file(&file1).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect missing file"
    );
}

#[test]
fn test_all_files_missing() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("file1.bin");
    let file2 = temp.path().join("file2.bin");

    create_pattern_file(&file1, b"FILE1", 5000).unwrap();
    create_pattern_file(&file2, b"FILE2", 5000).unwrap();

    let creator = Par2Creator::new(vec![file1.clone(), file2.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Delete all data files
    fs::remove_file(&file1).unwrap();
    fs::remove_file(&file2).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect all files missing"
    );
}
