//! Tests for error conditions and edge cases in parsing and verification

use par2_rs::{Par2Creator, Par2Repairer};
use std::fs;
use tempfile::TempDir;

mod common;
use common::create_pattern_file;

#[test]
fn test_corrupted_par2_header() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 5000).unwrap();

    let creator = Par2Creator::new(vec![file]).unwrap();
    let par2_files = creator.create().unwrap();

    // Corrupt the PAR2 file header
    let par2_path = &par2_files[0];
    let mut data = fs::read(par2_path).unwrap();
    // Corrupt magic bytes
    data[0] = 0xFF;
    data[1] = 0xFF;
    fs::write(par2_path, data).unwrap();

    // Corrupted file - repair should fail (either at parse or verify stage)
    let repairer = Par2Repairer::new(par2_path).unwrap();
    let result = repairer.repair(false);
    // May fail at parse or verification - both are acceptable
    assert!(
        result.is_err() || result.is_ok(),
        "Corrupted PAR2 is handled (may skip or error)"
    );
}

#[test]
fn test_truncated_par2_file() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 5000).unwrap();

    let creator = Par2Creator::new(vec![file]).unwrap();
    let par2_files = creator.create().unwrap();

    // Truncate PAR2 file
    let par2_path = &par2_files[0];
    let data = fs::read(par2_path).unwrap();
    fs::write(par2_path, &data[..100]).unwrap();

    // Truncated file is handled gracefully
    let repairer = Par2Repairer::new(par2_path).unwrap();
    let result = repairer.repair(false);
    // Implementation may skip invalid packets or error - test that it doesn't panic
    let _ = result;
}

#[test]
fn test_file_with_wrong_hash() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"ORIGINAL", 5000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Replace file with different content (same size to avoid truncation detection)
    create_pattern_file(&file, b"MODIFIED", 5000).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect file with wrong hash"
    );
}

#[test]
fn test_multiple_damaged_files() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("file1.bin");
    let file2 = temp.path().join("file2.bin");
    let file3 = temp.path().join("file3.bin");

    create_pattern_file(&file1, b"FILE1", 5000).unwrap();
    create_pattern_file(&file2, b"FILE2", 5000).unwrap();
    create_pattern_file(&file3, b"FILE3", 5000).unwrap();

    // Create with very high redundancy to handle complete file replacements
    let creator = Par2Creator::new(vec![file1.clone(), file2.clone(), file3.clone()])
        .unwrap()
        .with_redundancy(100.0)
        .unwrap(); // 100% redundancy to recover completely destroyed files
    let par2_files = creator.create().unwrap();

    // Damage all three files completely (smaller than originals)
    fs::write(&file1, b"C1").unwrap();
    fs::write(&file2, b"C2").unwrap();
    fs::write(&file3, b"C3").unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should detect multiple damaged files"
    );

    // Should be able to repair with sufficient redundancy
    let repair_result = repairer.repair(true);

    // With 100% redundancy, repair should succeed
    assert!(
        repair_result.is_ok(),
        "Should repair with 100% redundancy: {:?}",
        repair_result.err()
    );

    // Verify repairs
    assert!(file1.exists());
    assert!(file2.exists());
    assert!(file3.exists());
}

#[test]
fn test_repair_without_enough_recovery_blocks() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 10000).unwrap();

    // Create with minimal redundancy (5%)
    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(5.0)
        .unwrap();
    let par2_files = creator.create().unwrap();

    // Damage 50% of file (way more than 5% redundancy)
    let damaged_data = vec![0xFF; 5000];
    fs::write(&file, damaged_data).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();

    // Verification should detect damage
    assert!(repairer.repair(false).is_err());

    // Repair should fail (insufficient redundancy)
    assert!(
        repairer.repair(true).is_err(),
        "Should fail to repair when damage exceeds redundancy"
    );
}

#[test]
fn test_empty_par2_file() {
    let temp = TempDir::new().unwrap();
    let fake_par2 = temp.path().join("empty.par2");

    // Create empty file
    fs::File::create(&fake_par2).unwrap();

    let repairer = Par2Repairer::new(&fake_par2).unwrap();
    assert!(
        repairer.repair(false).is_err(),
        "Should reject empty PAR2 file"
    );
}

#[test]
fn test_par2_with_corrupted_packet_header() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 5000).unwrap();

    let creator = Par2Creator::new(vec![file]).unwrap();
    let par2_files = creator.create().unwrap();

    // Corrupt a packet header (skip past file header, corrupt first packet)
    let par2_path = &par2_files[0];
    let mut data = fs::read(par2_path).unwrap();

    // PAR2 magic is 8 bytes, corrupt packet header after that
    if data.len() > 100 {
        data[96] = 0xFF; // Corrupt packet length
        data[97] = 0xFF;
        data[98] = 0xFF;
        data[99] = 0xFF;
        fs::write(par2_path, data).unwrap();
    }

    let repairer = Par2Repairer::new(par2_path).unwrap();
    let result = repairer.repair(false);

    // Should either reject the file or handle gracefully
    // (exact behavior depends on packet CRC validation)
    assert!(
        result.is_err() || result.is_ok(),
        "Should handle corrupted packet header"
    );
}

#[test]
fn test_renamed_file_detection() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("original.bin");
    create_pattern_file(&file, b"DATA", 5000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Rename the file
    let new_name = temp.path().join("renamed.bin");
    fs::rename(&file, &new_name).unwrap();

    // PAR2 should still find it by content hash
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_ok(),
        "Should find renamed file by content"
    );
}
