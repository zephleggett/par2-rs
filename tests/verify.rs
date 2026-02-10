/// Tests for verification edge cases and code paths not covered by existing tests
use par2_rs::{Par2Creator, Par2Repairer, VolumeScheme};
use std::fs::{self, File};
use std::io::Write;
use tempfile::TempDir;

mod common;
use common::{compute_file_hash, corrupt_file, create_test_file};

/// Test verification with read errors (permission issues, missing file during read)
#[test]
#[cfg(unix)] // File permissions work differently on Windows
fn test_verification_read_errors() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    // Create file
    let test_file = base.join("data.bin");
    create_test_file(&test_file, 30_000).unwrap();

    // Create PAR2
    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(15.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create().unwrap();

    // Make file unreadable
    let mut perms = fs::metadata(&test_file).unwrap().permissions();
    perms.set_mode(0o000);
    fs::set_permissions(&test_file, perms).unwrap();

    // Verification should handle read error gracefully
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    // Should either fail verification or treat as missing
    let result = repairer.repair(false);
    assert!(result.is_err(), "Should fail with unreadable file");

    // Restore permissions for cleanup
    let mut perms = fs::metadata(&test_file).unwrap().permissions();
    perms.set_mode(0o644);
    fs::set_permissions(&test_file, perms).unwrap();
}

/// Test damaged file with block-level IFSC verification
/// This ensures the block-level damage detection code is exercised
#[test]
fn test_block_level_damage_detection() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    // Create file large enough for multiple blocks
    let test_file = base.join("data.bin");
    create_test_file(&test_file, 100_000).unwrap();

    // Create PAR2 with IFSC checksums
    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(25.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create().unwrap();

    // Damage middle portion of file (should affect specific blocks)
    corrupt_file(&test_file, 40_000, &vec![0xBB; 8000]).unwrap();

    // Verification should detect block-level damage
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect damaged blocks"
    );

    // Repair should fix the damaged blocks
    assert!(repairer.repair(true).is_ok(), "Should repair damage");
}

/// Test rename collision handling (when target already exists)
#[test]
fn test_rename_collision() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    // Create original file
    let original = base.join("data.bin");
    create_test_file(&original, 30_000).unwrap();

    // Create PAR2
    let creator = Par2Creator::new(vec![original.clone()])
        .unwrap()
        .with_redundancy(15.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create().unwrap();

    // Rename to obfuscated name
    let obfuscated = base.join("renamed.bin");
    fs::rename(&original, &obfuscated).unwrap();

    // Create a different file at the original location (collision)
    let mut collision = File::create(&original).unwrap();
    collision.write_all(b"different content").unwrap();
    drop(collision);

    // Repair should handle collision (won't overwrite existing file)
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    let result = repairer.repair(false);

    // Should either succeed (found via hash) or fail (collision prevents rename)
    // Either outcome is acceptable - we're testing that it doesn't crash
    drop(result);
}

/// Test verification with damaged file using exponential volumes
#[test]
fn test_damaged_file_exponential_volumes() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    // Create test file
    let file1 = base.join("data.bin");
    create_test_file(&file1, 80_000).unwrap();

    let hash1 = compute_file_hash(&file1);

    // Create PAR2 with exponential volumes
    let creator = Par2Creator::new(vec![file1.clone()])
        .unwrap()
        .with_redundancy(30.0)
        .unwrap()
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create().unwrap();

    // Damage part of file1
    corrupt_file(&file1, 10_000, &vec![0xEE; 5000]).unwrap();

    // Should detect damage
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(repairer.repair(false).is_err(), "Should detect damage");

    // Should repair
    assert!(repairer.repair(true).is_ok(), "Should repair damage");

    assert_eq!(hash1, compute_file_hash(&file1), "File should be fixed");
}

/// Test verification with 16KB hash optimization (file > 16KB)
#[test]
fn test_verification_16k_hash_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("large_file.bin");

    // Create file larger than 16KB to trigger 16KB hash check
    create_test_file(&test_file, 20000).unwrap();

    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(20.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create().unwrap();

    // Verify - should use 16KB optimization for large files
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_ok(),
        "Should verify with 16KB optimization"
    );
}

/// Test verification with small file (< 16KB) that skips 16KB optimization
#[test]
fn test_verification_small_file_skips_16k() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("small_file.bin");

    // Create file smaller than 16KB
    create_test_file(&test_file, 8000).unwrap();

    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(20.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create().unwrap();

    // Verify - should skip 16KB check for small files
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_ok(),
        "Should verify without 16KB optimization"
    );
}

/// Test verification with 16KB hash mismatch (early damage detection)
#[test]
fn test_verification_16k_hash_mismatch() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("file.bin");

    create_test_file(&test_file, 20000).unwrap();

    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(30.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create().unwrap();

    // Corrupt first 1KB (affects 16KB hash)
    corrupt_file(&test_file, 0, &vec![0xFFu8; 1024]).unwrap();

    // Should detect damage via 16KB hash check
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect 16KB hash mismatch"
    );

    // Should be able to repair
    assert!(
        repairer.repair(true).is_ok(),
        "Should repair after 16KB mismatch"
    );
}

/// Test verification with file size mismatch
#[test]
fn test_verification_file_size_mismatch() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("file.bin");

    create_test_file(&test_file, 10000).unwrap();

    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(20.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create().unwrap();

    // Change file size by truncating
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&test_file)
        .unwrap();
    file.set_len(5000).unwrap();
    drop(file);

    // Should detect size mismatch
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    let result = repairer.repair(false);
    assert!(result.is_err(), "Should detect file size mismatch");
}

/// Test missing file detection and recreation using multi-volume PAR2
#[test]
fn test_missing_file_recreation_with_volumes() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let file1 = base.join("file1.bin");
    let file2 = base.join("file2.bin");

    // Create two files
    create_test_file(&file1, 8000).unwrap();
    create_test_file(&file2, 6000).unwrap();
    let hash1 = compute_file_hash(&file1);
    let hash2 = compute_file_hash(&file2);

    // Create multi-volume PAR2 with exponential scheme for multiple volumes
    // Total blocks: file1 (4 blocks) + file2 (3 blocks) = 7 blocks
    // Need minimum 4 recovery blocks to recreate file1 if it's deleted
    // 4/7 = 57.14% redundancy - use slightly less to ensure exactly 4 blocks
    let creator = Par2Creator::new(vec![file1.clone(), file2.clone()])
        .unwrap()
        .with_redundancy(57.0)
        .unwrap() // Exactly 4 recovery blocks (57% of 7 = 3.99 → 4)
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create().unwrap();

    assert!(
        par2_files.len() > 1,
        "Should have created multiple volume files"
    );

    // Delete only file1, keep file2 intact
    fs::remove_file(&file1).unwrap();
    assert!(!file1.exists(), "File1 should be deleted");
    assert!(file2.exists(), "File2 should still exist");

    // Should detect missing file
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    let result = repairer.repair(false);
    assert!(result.is_err(), "Should detect missing file");

    // Repair should recreate the missing file1 from volume recovery blocks
    assert!(
        repairer.repair(true).is_ok(),
        "Should recreate missing file from volumes"
    );
    assert!(file1.exists(), "File1 should be recreated");
    assert!(file2.exists(), "File2 should still exist");
    assert_eq!(
        hash1,
        compute_file_hash(&file1),
        "Recreated file1 should match original"
    );
    assert_eq!(
        hash2,
        compute_file_hash(&file2),
        "File2 should remain unchanged"
    );
}

/// Test verification fallback when no IFSC packets are present
/// This tests the traditional full-file MD5 verification path
#[test]
fn test_verification_without_ifsc() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("data.bin");

    create_test_file(&test_file, 50_000).unwrap();

    // Create a minimal PAR2 file without IFSC packets
    let par2_file = base.join("test.par2");
    common::create_par2_without_ifsc(&test_file, &par2_file).unwrap();

    // Verification should fall back to traditional MD5 verification
    let repairer = Par2Repairer::new(&par2_file).unwrap();
    assert!(
        repairer.repair(false).is_ok(),
        "Should verify successfully using MD5 fallback (no IFSC packets)"
    );
}

/// Test verification with IFSC block-level checksums
/// Par2Creator always creates IFSC packets for optimal verification performance
#[test]
fn test_verification_with_ifsc() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("file.bin");

    create_test_file(&test_file, 50_000).unwrap();

    // Par2Creator always includes IFSC packets
    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(20.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create().unwrap();

    // Verification should use IFSC for faster block-level checking
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_ok(),
        "Should verify successfully using IFSC packets"
    );
}

/// Test that damaged file is properly detected with MD5 fallback (no IFSC)
#[test]
fn test_verification_without_ifsc_detects_damage() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("data.bin");

    create_test_file(&test_file, 50_000).unwrap();

    // Create PAR2 without IFSC packets
    let par2_file = base.join("test.par2");
    common::create_par2_without_ifsc(&test_file, &par2_file).unwrap();

    // Verify intact file succeeds
    let repairer = Par2Repairer::new(&par2_file).unwrap();
    assert!(
        repairer.repair(false).is_ok(),
        "Should verify intact file using MD5 fallback"
    );

    // Damage the file
    corrupt_file(&test_file, 1000, &[0xFFu8; 100]).unwrap();

    // Should detect damage even without IFSC (using full MD5)
    let repairer = Par2Repairer::new(&par2_file).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect damage using MD5 fallback (no IFSC)"
    );
}

/// Test block damage with multiple corrupted blocks
#[test]
fn test_multiple_blocks_damaged() {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let test_file = base.join("file.bin");

    // Create file with multiple blocks (16KB = 8 blocks of 2KB)
    create_test_file(&test_file, 16384).unwrap();

    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_block_size(2048)
        .unwrap()
        .with_redundancy(50.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create().unwrap();

    // Corrupt blocks 2 and 5
    corrupt_file(&test_file, 2048 * 2, &[0xFFu8; 100]).unwrap();
    corrupt_file(&test_file, 2048 * 5, &[0xFFu8; 100]).unwrap();

    // Should detect specific damaged blocks
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect damaged blocks"
    );

    // Should repair successfully
    assert!(
        repairer.repair(true).is_ok(),
        "Should repair damaged blocks"
    );
}
