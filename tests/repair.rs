use par2_rs::{Par2Creator, Par2Repairer, Result, VolumeScheme};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

mod common;
use common::{
    compute_file_hash, corrupt_file, create_pattern_file, create_test_file, test_data_dir,
};

/// Test verification of intact files using existing test data
#[test]
fn test_verify_intact_files() -> Result<()> {
    let test_data = test_data_dir();
    let par2_file = test_data.join("testdata.par2");

    assert!(
        par2_file.exists(),
        "Test data not found at tests/data/testdata.par2"
    );

    let repairer = Par2Repairer::new(&par2_file)?;
    repairer.repair(false)?;

    Ok(())
}

/// Test repair when a single file is completely deleted
#[test]
fn test_repair_single_deleted_file() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    // Create two files
    let file1 = base.join("file1.bin");
    let file2 = base.join("file2.bin");
    create_pattern_file(&file1, b"FILE1_CONTENT_", 5000)?;
    create_pattern_file(&file2, b"FILE2_CONTENT_", 5000)?;

    let hash1 = compute_file_hash(&file1);

    // Create PAR2 with enough redundancy to recover one file
    let creator = Par2Creator::new(vec![file1.clone(), file2.clone()])?
        .with_redundancy(60.0)
        .unwrap() // 50% of total data
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create()?;

    // Verify files are intact
    let repairer = Par2Repairer::new(&par2_files[0])?;
    assert!(
        repairer.repair(false).is_ok(),
        "Pre-damage verification should pass"
    );

    // Delete file1
    fs::remove_file(&file1)?;
    assert!(!file1.exists(), "File should be deleted");

    // Verification should fail
    assert!(
        repairer.repair(false).is_err(),
        "Should detect missing file"
    );

    // Repair should succeed
    repairer.repair(true)?;

    // Verify file was restored correctly
    assert!(file1.exists(), "File should be restored");
    let restored_hash = compute_file_hash(&file1);
    assert_eq!(hash1, restored_hash, "Restored file should match original");

    Ok(())
}

/// Test repair when a file is corrupted (not deleted, but damaged)
#[test]
fn test_repair_single_damaged_file() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    let test_file = base.join("test.bin");
    create_test_file(&test_file, 50_000)?;
    let original_hash = compute_file_hash(&test_file);

    // Create PAR2 with 20% redundancy
    let creator = Par2Creator::new(vec![test_file.clone()])?
        .with_redundancy(20.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create()?;

    let repairer = Par2Repairer::new(&par2_files[0])?;
    assert!(repairer.repair(false).is_ok());

    // Corrupt 10% of the file
    let file_size = test_file.metadata()?.len();
    let corrupt_size = (file_size as f64 * 0.10) as usize;
    corrupt_file(&test_file, 1000, &vec![0xFF; corrupt_size])?;

    // Should detect corruption
    assert!(repairer.repair(false).is_err(), "Should detect corruption");

    // Should repair successfully
    repairer.repair(true)?;

    let repaired_hash = compute_file_hash(&test_file);
    assert_eq!(
        original_hash, repaired_hash,
        "Repaired file should match original"
    );

    Ok(())
}

/// Test repair when file is missing (different from deleted - tests discovery)
#[test]
fn test_repair_missing_file() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    let file1 = base.join("data.bin");
    create_pattern_file(&file1, b"IMPORTANT_DATA_", 8000)?;
    let hash1 = compute_file_hash(&file1);

    // Create PAR2 with 100% redundancy (can recover entire file)
    let creator = Par2Creator::new(vec![file1.clone()])?
        .with_redundancy(100.0)
        .unwrap()
        .with_output_path(base.join("data.par2"));
    let par2_files = creator.create()?;

    // Remove file before repair attempt
    fs::remove_file(&file1)?;

    // Repair should recreate the file
    let repairer = Par2Repairer::new(&par2_files[0])?;
    repairer.repair(true)?;

    assert!(file1.exists(), "Missing file should be recreated");
    assert_eq!(
        hash1,
        compute_file_hash(&file1),
        "Recreated file should match"
    );

    Ok(())
}

/// Test repair with multiple damaged files
#[test]
fn test_repair_multiple_damaged_files() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    let files: Vec<PathBuf> = (0..3)
        .map(|i| {
            let path = base.join(format!("file{}.bin", i));
            create_pattern_file(&path, format!("FILE_{}_DATA_", i).as_bytes(), 10_000).unwrap();
            path
        })
        .collect();

    let hashes: Vec<_> = files.iter().map(|f| compute_file_hash(f)).collect();

    // Create PAR2 with 40% redundancy
    let creator = Par2Creator::new(files.clone())?
        .with_redundancy(40.0)
        .unwrap()
        .with_output_path(base.join("archive.par2"));
    let par2_files = creator.create()?;

    // Damage 2 out of 3 files (< 40% of total)
    corrupt_file(&files[0], 100, &vec![0xAA; 3000])?;
    corrupt_file(&files[1], 200, &vec![0xBB; 3000])?;

    let repairer = Par2Repairer::new(&par2_files[0])?;

    // Should detect damage
    assert!(repairer.repair(false).is_err());

    // Should repair both files
    repairer.repair(true)?;

    for (file, original_hash) in files.iter().zip(hashes.iter()) {
        let repaired_hash = compute_file_hash(file);
        assert_eq!(
            *original_hash, repaired_hash,
            "File {:?} should be repaired",
            file
        );
    }

    Ok(())
}

/// Test scenario where there's insufficient recovery data
#[test]
fn test_insufficient_recovery_blocks() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    let test_file = base.join("test.bin");
    create_test_file(&test_file, 20_000)?;

    // Create PAR2 with only 10% redundancy
    let creator = Par2Creator::new(vec![test_file.clone()])?
        .with_redundancy(10.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));
    let par2_files = creator.create()?;

    // Corrupt 20% of the file (more than available redundancy)
    let file_size = test_file.metadata()?.len();
    let corrupt_size = (file_size as f64 * 0.20) as usize;
    corrupt_file(&test_file, 0, &vec![0xFF; corrupt_size])?;

    let repairer = Par2Repairer::new(&par2_files[0])?;

    // Should detect corruption
    assert!(repairer.repair(false).is_err());

    // Repair should fail due to insufficient recovery blocks
    let result = repairer.repair(true);
    assert!(
        result.is_err(),
        "Repair should fail when damage exceeds redundancy"
    );

    Ok(())
}

/// Test that parser correctly loads multiple volume files
#[test]
fn test_parser_loads_multiple_volumes() -> Result<()> {
    let test_data = test_data_dir();
    let par2_file = test_data.join("testdata.par2");

    // Check that volume files exist
    let vol_pattern = test_data.join("testdata.vol*.par2");
    let vol_files: Vec<_> = glob::glob(vol_pattern.to_str().unwrap())
        .expect("Failed to read glob pattern")
        .filter_map(|e| e.ok())
        .collect();

    assert!(!vol_files.is_empty(), "Should have volume files");

    let repairer = Par2Repairer::new(&par2_file)?;

    // If this succeeds, it means all volumes were loaded and verification passed
    repairer.repair(false)?;

    Ok(())
}

/// Test volume scheme: Single volume
#[test]
fn test_single_volume_scheme() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    let test_file = base.join("test.bin");
    create_pattern_file(&test_file, b"TEST_", 1000)?;

    let creator = Par2Creator::new(vec![test_file])?
        .with_volume_scheme(VolumeScheme::Single)
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create()?;

    assert_eq!(
        par2_files.len(),
        1,
        "Single scheme should create exactly one file"
    );
    assert!(par2_files[0].exists());

    Ok(())
}

/// Test volume scheme: Exponential volumes
#[test]
fn test_exponential_volume_scheme() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();

    let test_file = base.join("large.bin");
    create_test_file(&test_file, 100_000)?;

    let creator = Par2Creator::new(vec![test_file])?
        .with_redundancy(30.0)
        .unwrap()
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create()?;

    // With exponential scheme and sufficient blocks, should create multiple volumes
    assert!(!par2_files.is_empty());

    // Verify all created volumes exist
    for file in &par2_files {
        assert!(file.exists(), "Volume should exist: {:?}", file);
    }

    Ok(())
}
