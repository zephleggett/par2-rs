/// Fast end-to-end tests with realistic but small file sizes
/// These tests use 100-200KB files instead of 2MB+ for faster execution
use par2_rs::{Par2Creator, Par2Repairer, VolumeScheme};
use std::fs;
use std::time::Instant;
use tempfile::TempDir;

mod common;
use common::{compute_file_hash, corrupt_file, create_test_file};

/// Comprehensive test combining multiple scenarios with enough recovery blocks
/// This single test validates: creation, verification, corruption, deletion, and repair
#[test]
fn test_comprehensive_workflow() {
    println!("\n=== Comprehensive PAR2 Workflow Test ===\n");

    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let start_total = Instant::now();

    // Step 1: Create test files (200KB total - much smaller than original 2MB+)
    println!("Step 1: Creating test files (100KB + 100KB)...");
    let start = Instant::now();

    let file1 = base.join("file1.bin");
    let file2 = base.join("file2.bin");
    create_test_file(&file1, 100_000).unwrap();
    create_test_file(&file2, 100_000).unwrap();

    let hash1 = compute_file_hash(&file1);
    let hash2 = compute_file_hash(&file2);

    println!("  ✓ Created files in {:.2}s", start.elapsed().as_secs_f64());

    // Step 2: Create PAR2 with 60% redundancy (enough to recover one full file)
    println!("\nStep 2: Creating PAR2 files (60% redundancy)...");
    let start = Instant::now();

    let creator = Par2Creator::new(vec![file1.clone(), file2.clone()])
        .unwrap()
        .with_redundancy(60.0)
        .unwrap()
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create().unwrap();
    println!(
        "  ✓ Created {} volume(s) in {:.2}s",
        par2_files.len(),
        start.elapsed().as_secs_f64()
    );

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();

    // Step 3: Verify intact files
    println!("\nStep 3: Verifying intact files...");
    let start = Instant::now();
    assert!(
        repairer.repair(false).is_ok(),
        "Pre-damage verification should pass"
    );
    println!("  ✓ Verified in {:.2}s", start.elapsed().as_secs_f64());

    // Step 4: Test corruption repair (10% of file1)
    println!("\nStep 4: Testing corruption repair (10% damage)...");
    let start = Instant::now();

    let corrupt_size = 10_000; // 10% of 100KB
    corrupt_file(&file1, 5000, &vec![0xFF; corrupt_size]).unwrap();

    assert!(repairer.repair(false).is_err(), "Should detect corruption");
    assert!(repairer.repair(true).is_ok(), "Should repair corruption");
    assert_eq!(hash1, compute_file_hash(&file1), "File1 should be restored");

    println!(
        "  ✓ Corruption detected and repaired in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    // Step 5: Test deletion recovery (delete entire file2)
    println!("\nStep 5: Testing file deletion recovery...");
    let start = Instant::now();

    fs::remove_file(&file2).unwrap();
    assert!(!file2.exists(), "File should be deleted");

    assert!(
        repairer.repair(false).is_err(),
        "Should detect missing file"
    );
    assert!(repairer.repair(true).is_ok(), "Should recover deleted file");
    assert!(file2.exists(), "File should be restored");
    assert_eq!(
        hash2,
        compute_file_hash(&file2),
        "File2 should match original"
    );

    println!(
        "  ✓ Deleted file recovered in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    // Step 6: Final verification
    println!("\nStep 6: Final verification...");
    let start = Instant::now();
    assert!(
        repairer.repair(false).is_ok(),
        "Final verification should pass"
    );
    println!(
        "  ✓ Final verification in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    println!("\n=== Summary ===");
    println!("Total time: {:.2}s", start_total.elapsed().as_secs_f64());
    println!("✓ Tested: creation, verification, corruption repair, deletion recovery");
}

/// Test block-level repair with smaller files for speed
#[test]
fn test_block_level_repair() {
    println!("\n=== Block-Level Repair Test (Fast) ===\n");

    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let start_total = Instant::now();

    // Use 100KB file instead of 2MB
    println!("Step 1: Creating test file (100KB)...");
    let test_file = base.join("test.bin");
    create_test_file(&test_file, 100_000).unwrap();
    let original_hash = compute_file_hash(&test_file);

    // Create PAR2 with 15% redundancy
    println!("\nStep 2: Creating PAR2 files (15% redundancy)...");
    let start = Instant::now();

    let creator = Par2Creator::new(vec![test_file.clone()])
        .unwrap()
        .with_redundancy(15.0)
        .unwrap()
        .with_output_path(base.join("test.par2"));

    let par2_files = creator.create().unwrap();
    println!("  ✓ Created in {:.2}s", start.elapsed().as_secs_f64());

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(repairer.repair(false).is_ok());

    // Corrupt 5% of file (within 15% redundancy)
    println!("\nStep 3: Damaging 5% of file...");
    let damage_size = 5_000; // 5% of 100KB
    let damage_offset = 50_000 - damage_size / 2; // Center

    corrupt_file(&test_file, damage_offset as u64, &vec![0xAA; damage_size]).unwrap();

    // Should detect and repair
    println!("\nStep 4: Detecting and repairing...");
    let start = Instant::now();

    assert!(repairer.repair(false).is_err(), "Should detect corruption");
    match repairer.repair(true) {
        Ok(_) => {}
        Err(e) => panic!("Repair failed: {:?}", e),
    }
    assert_eq!(
        original_hash,
        compute_file_hash(&test_file),
        "Should match original"
    );

    println!("  ✓ Repaired in {:.2}s", start.elapsed().as_secs_f64());

    println!("\n=== Summary ===");
    println!("Total time: {:.2}s", start_total.elapsed().as_secs_f64());
    println!("✓ Block-level repair with IFSC validated");
}

/// Test multiple files with minimal overhead
#[test]
fn test_multiple_files_workflow() {
    println!("\n=== Multiple Files Test (Fast) ===\n");

    let temp_dir = TempDir::new().unwrap();
    let base = temp_dir.path();
    let start_total = Instant::now();

    // Create 3 files: 80KB, 80KB, 40KB (200KB total vs 5MB original)
    println!("Step 1: Creating 3 files (200KB total)...");
    let files: Vec<_> = [80_000, 80_000, 40_000]
        .iter()
        .enumerate()
        .map(|(i, &size)| {
            let path = base.join(format!("file{}.bin", i));
            create_test_file(&path, size).unwrap();
            path
        })
        .collect();

    let hashes: Vec<_> = files.iter().map(|f| compute_file_hash(f)).collect();

    // Create PAR2 with 25% redundancy (enough to recover smallest file)
    println!("\nStep 2: Creating PAR2 files (25% redundancy)...");
    let start = Instant::now();

    let creator = Par2Creator::new(files.clone())
        .unwrap()
        .with_redundancy(25.0)
        .unwrap()
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(base.join("archive.par2"));

    let par2_files = creator.create().unwrap();
    println!("  ✓ Created in {:.2}s", start.elapsed().as_secs_f64());

    // Delete smallest file (20% of total data)
    println!("\nStep 3: Deleting smallest file...");
    fs::remove_file(&files[2]).unwrap();

    // Repair
    println!("\nStep 4: Repairing...");
    let start = Instant::now();

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    assert!(repairer.repair(true).is_ok(), "Repair should succeed");

    println!("  ✓ Repaired in {:.2}s", start.elapsed().as_secs_f64());

    // Verify all files
    for (file, original_hash) in files.iter().zip(hashes.iter()) {
        assert!(file.exists());
        assert_eq!(*original_hash, compute_file_hash(file));
    }

    println!("\n=== Summary ===");
    println!("Total time: {:.2}s", start_total.elapsed().as_secs_f64());
    println!("✓ Multiple file recovery validated");
}
