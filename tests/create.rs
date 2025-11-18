use par2_rs::{Par2Creator, Par2Repairer, VolumeScheme};
use std::fs;
use tempfile::TempDir;

mod common;
use common::create_pattern_file;

#[test]
fn test_create_and_verify_single_file() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.bin");

    create_pattern_file(
        &test_file,
        b"Hello, PAR2! This is a test file for creation.",
        500,
    )
    .unwrap();

    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(10.0)
        .with_output_path(temp_dir.path().join("test.par2"));

    let par2_files = creator.create().unwrap();

    assert!(
        !par2_files.is_empty(),
        "Should create at least one PAR2 file"
    );

    for par2_file in &par2_files {
        assert!(
            par2_file.exists(),
            "PAR2 file should exist: {:?}",
            par2_file
        );
    }

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(
        repairer.repair(false).is_ok(),
        "Verification should succeed"
    );
}

#[test]
fn test_create_and_verify_multiple_files() {
    let temp_dir = TempDir::new().unwrap();

    let test_files: Vec<_> = (0..3)
        .map(|i| {
            let path = temp_dir.path().join(format!("file{}.bin", i));
            create_pattern_file(&path, format!("File {} ", i).as_bytes(), 2000).unwrap();
            path
        })
        .collect();

    let creator = Par2Creator::new(test_files.clone())
        .unwrap()
        .with_redundancy(5.0)
        .with_output_path(temp_dir.path().join("archive.par2"));

    let par2_files = creator.create().unwrap();
    assert!(!par2_files.is_empty());

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(repairer.repair(false).is_ok());
}

#[test]
fn test_create_with_single_volume() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.bin");

    create_pattern_file(&test_file, b"Single volume test", 100).unwrap();

    let creator = Par2Creator::new(vec![test_file])
        .unwrap()
        .with_volume_scheme(VolumeScheme::Single)
        .with_output_path(temp_dir.path().join("test.par2"));

    let par2_files = creator.create().unwrap();
    assert_eq!(
        par2_files.len(),
        1,
        "Single volume should create one PAR2 file"
    );
}

#[test]
fn test_create_with_exponential_volumes() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("large.bin");

    create_pattern_file(&test_file, &[0x42u8], 100_000).unwrap();

    let creator = Par2Creator::new(vec![test_file])
        .unwrap()
        .with_redundancy(20.0)
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(temp_dir.path().join("large.par2"));

    let par2_files = creator.create().unwrap();
    assert!(!par2_files.is_empty(), "Should create at least one volume");
}

#[test]
fn test_create_and_repair_deleted_file() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("file1.bin");
    let file2 = temp_dir.path().join("file2.bin");

    create_pattern_file(&file1, b"Original file 1 content", 500).unwrap();
    create_pattern_file(&file2, b"Original file 2 content", 500).unwrap();

    let original_content = fs::read(&file1).unwrap();

    let creator = Par2Creator::new(vec![file1.clone(), file2.clone()])
        .unwrap()
        .with_redundancy(50.0)
        .with_output_path(temp_dir.path().join("archive.par2"));

    let par2_files = creator.create().unwrap();
    assert!(!par2_files.is_empty());

    fs::remove_file(&file1).unwrap();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(repairer.repair(true).is_ok(), "Repair should succeed");
    assert!(file1.exists(), "Deleted file should be restored");

    let restored_content = fs::read(&file1).unwrap();
    assert_eq!(
        original_content, restored_content,
        "Content should match original"
    );
}

#[test]
fn test_create_with_custom_block_size() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.bin");

    create_pattern_file(&test_file, &[0x42], 10_000).unwrap();

    let creator = Par2Creator::new(vec![test_file])
        .unwrap()
        .with_block_size(2048)
        .unwrap()
        .with_output_path(temp_dir.path().join("test.par2"));

    assert!(
        creator.create().is_ok(),
        "Creation with custom block size should succeed"
    );
}

#[test]
fn test_create_invalid_block_size() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.bin");

    create_pattern_file(&test_file, b"test", 100).unwrap();

    let creator = Par2Creator::new(vec![test_file]).unwrap();
    let result = creator.with_block_size(2047); // Not multiple of 4

    assert!(result.is_err(), "Should reject invalid block size");
}
