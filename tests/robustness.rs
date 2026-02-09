//! Robustness tests for production readiness
//!
//! Tests large files, many small files, progress accuracy, and edge cases
//! that matter for real-world use as a library.

use par2_rs::{MessageLevel, Par2Creator, Par2Operation, Par2Repairer, VolumeScheme};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

mod common;
use common::{compute_file_hash, corrupt_file, create_pattern_file};

// ============================================================================
// Large file tests
// ============================================================================

/// Test create → corrupt → repair cycle with a 10MB file
#[test]
fn test_large_single_file_10mb() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("large.bin");

    // Create a 10MB file with varied content
    create_pattern_file(&file, b"ABCDEFGHIJKLMNOP", 10 * 1024 * 1024).unwrap();
    let original_hash = compute_file_hash(&file);

    // Create PAR2 with 10% redundancy
    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(10.0)
        .with_output_path(temp.path().join("large.par2"));
    let par2_files = creator.create().unwrap();

    // Verify intact
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer.repair(false).unwrap();

    // Corrupt 5% scattered across multiple blocks
    corrupt_file(&file, 100_000, &[0xFF; 50_000]).unwrap();
    corrupt_file(&file, 5_000_000, &[0xAA; 50_000]).unwrap();
    corrupt_file(&file, 9_000_000, &[0x55; 50_000]).unwrap();

    // Repair
    repairer.repair(true).unwrap();
    assert_eq!(
        original_hash,
        compute_file_hash(&file),
        "10MB file should be fully restored after repair"
    );
}

/// Test with a deleted large file (full reconstruction)
#[test]
fn test_large_file_full_reconstruction() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");

    create_pattern_file(&file, b"RECONSTRUCT_ME", 5 * 1024 * 1024).unwrap();
    let original_hash = compute_file_hash(&file);

    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(100.0) // 100% redundancy to cover full file loss
        .with_output_path(temp.path().join("data.par2"));
    let par2_files = creator.create().unwrap();

    // Delete the file entirely
    fs::remove_file(&file).unwrap();
    assert!(!file.exists());

    // Reconstruct
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer.repair(true).unwrap();

    assert!(file.exists(), "File should be reconstructed");
    assert_eq!(
        original_hash,
        compute_file_hash(&file),
        "Reconstructed file should match original"
    );
}

// ============================================================================
// Many small files tests
// ============================================================================

/// Test with 50 small files (1KB each)
#[test]
fn test_many_small_files_50() {
    let temp = TempDir::new().unwrap();

    // Create 50 small files with distinct content
    let files: Vec<_> = (0..50)
        .map(|i| {
            let path = temp.path().join(format!("file_{:03}.bin", i));
            let pattern = format!("FILE_{:03}_DATA_", i);
            create_pattern_file(&path, pattern.as_bytes(), 1024).unwrap();
            path
        })
        .collect();

    let hashes: Vec<_> = files.iter().map(|f| compute_file_hash(f)).collect();

    // Create PAR2 with 20% redundancy
    let creator = Par2Creator::new(files.clone())
        .unwrap()
        .with_redundancy(20.0)
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(temp.path().join("archive.par2"));
    let par2_files = creator.create().unwrap();

    // Verify all intact
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer.repair(false).unwrap();

    // Delete a few files (within redundancy budget)
    let files_to_delete = [5, 15, 25, 35, 45];
    for &idx in &files_to_delete {
        fs::remove_file(&files[idx]).unwrap();
    }

    // Repair
    repairer.repair(true).unwrap();

    // Verify all restored
    for (i, (file, hash)) in files.iter().zip(hashes.iter()).enumerate() {
        assert!(file.exists(), "File {} should exist after repair", i);
        assert_eq!(
            *hash,
            compute_file_hash(file),
            "File {} should match original",
            i
        );
    }
}

/// Test with 20 files of varying sizes
#[test]
fn test_mixed_file_sizes() {
    let temp = TempDir::new().unwrap();

    // Mix of tiny to medium files
    let sizes = [
        100, 500, 1024, 2048, 4096, 8192, 16384, 32768, 50000, 100_000, 200_000, 500_000, 100, 500,
        1024, 2048, 4096, 8192, 16384, 32768,
    ];

    let files: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| {
            let path = temp.path().join(format!("mixed_{:02}.bin", i));
            let pattern = format!("MIXED_{:02}_", i);
            create_pattern_file(&path, pattern.as_bytes(), size).unwrap();
            path
        })
        .collect();

    let hashes: Vec<_> = files.iter().map(|f| compute_file_hash(f)).collect();

    let creator = Par2Creator::new(files.clone())
        .unwrap()
        .with_redundancy(30.0)
        .with_output_path(temp.path().join("mixed.par2"));
    let par2_files = creator.create().unwrap();

    // Corrupt a few small files (within 30% redundancy budget)
    corrupt_file(&files[0], 0, &[0xFF; 50]).unwrap(); // 100 byte file
    corrupt_file(&files[4], 0, &[0xFF; 100]).unwrap(); // 4096 byte file
    corrupt_file(&files[6], 0, &[0xFF; 500]).unwrap(); // 16384 byte file

    // Repair
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer.repair(true).unwrap();

    // Verify all restored
    for (i, (file, hash)) in files.iter().zip(hashes.iter()).enumerate() {
        assert!(file.exists(), "File {} should exist after repair", i);
        assert_eq!(
            *hash,
            compute_file_hash(file),
            "File {} should match original after repair",
            i
        );
    }
}

// ============================================================================
// Progress reporting accuracy tests
// ============================================================================

/// Verify progress never exceeds total and reaches 100%
#[test]
fn test_progress_never_exceeds_total() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"PROGRESS_TEST", 100_000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(20.0);
    let par2_files = creator.create().unwrap();

    // Corrupt the file
    corrupt_file(&file, 5000, &[0xFF; 10_000]).unwrap();

    // Track all progress callbacks
    let progress_log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = progress_log.clone();

    let progress_cb = move |op: Par2Operation, current: u64, total: u64| {
        log_clone.lock().unwrap().push((op, current, total));
    };

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_progress(true, false, Some(Arc::new(progress_cb)))
        .unwrap();

    let log = progress_log.lock().unwrap();

    // Verify no progress exceeds total
    for (op, current, total) in log.iter() {
        assert!(
            *current <= *total,
            "Progress exceeded total: {:?} {}/{}",
            op,
            current,
            total
        );
        assert!(*total > 0, "Total should never be zero for {:?}", op);
    }

    // Verify each operation type reaches its total
    for target_op in [
        Par2Operation::Loading,
        Par2Operation::Verifying,
        Par2Operation::Repairing,
    ] {
        let op_entries: Vec<_> = log.iter().filter(|(op, _, _)| *op == target_op).collect();
        if let Some(last) = op_entries.last() {
            assert_eq!(
                last.1, last.2,
                "{:?} should reach 100% (got {}/{})",
                target_op, last.1, last.2
            );
        }
    }
}

/// Verify progress is monotonically increasing per operation
#[test]
fn test_progress_monotonic_per_operation() {
    let temp = TempDir::new().unwrap();

    // Use multiple files so verification has more progress points
    let file1 = temp.path().join("a.bin");
    let file2 = temp.path().join("b.bin");
    create_pattern_file(&file1, b"AAAA", 50_000).unwrap();
    create_pattern_file(&file2, b"BBBB", 50_000).unwrap();

    let creator = Par2Creator::new(vec![file1.clone(), file2.clone()])
        .unwrap()
        .with_redundancy(20.0);
    let par2_files = creator.create().unwrap();

    corrupt_file(&file1, 1000, &[0xFF; 5000]).unwrap();

    let progress_log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = progress_log.clone();

    let progress_cb = move |op: Par2Operation, current: u64, total: u64| {
        log_clone.lock().unwrap().push((op, current, total));
    };

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_progress(true, false, Some(Arc::new(progress_cb)))
        .unwrap();

    let log = progress_log.lock().unwrap();

    // Check monotonicity for Loading progress (single-threaded, should be strictly monotonic)
    let loading: Vec<u64> = log
        .iter()
        .filter(|(op, _, _)| *op == Par2Operation::Loading)
        .map(|(_, c, _)| *c)
        .collect();
    for i in 1..loading.len() {
        assert!(
            loading[i] >= loading[i - 1],
            "Loading progress should be monotonic: {} -> {}",
            loading[i - 1],
            loading[i]
        );
    }

    // Repair progress should be monotonic
    let repairing: Vec<u64> = log
        .iter()
        .filter(|(op, _, _)| *op == Par2Operation::Repairing)
        .map(|(_, c, _)| *c)
        .collect();
    for i in 1..repairing.len() {
        assert!(
            repairing[i] >= repairing[i - 1],
            "Repair progress should be monotonic: {} -> {}",
            repairing[i - 1],
            repairing[i]
        );
    }
}

/// Test creator progress callbacks
#[test]
fn test_creator_progress_callbacks() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"CREATE_PROGRESS", 200_000).unwrap();

    let progress_log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = progress_log.clone();

    let progress_cb = move |op: Par2Operation, current: u64, total: u64| {
        log_clone.lock().unwrap().push((op, current, total));
    };

    let creator = Par2Creator::new(vec![file])
        .unwrap()
        .with_redundancy(10.0)
        .with_progress_callback(Arc::new(progress_cb));
    creator.create().unwrap();

    let log = progress_log.lock().unwrap();
    assert!(
        !log.is_empty(),
        "Should receive progress callbacks during creation"
    );

    // Verify we see encoding progress
    let encoding: Vec<_> = log
        .iter()
        .filter(|(op, _, _)| *op == Par2Operation::Repairing)
        .collect();
    assert!(
        !encoding.is_empty(),
        "Should see encoding progress during creation"
    );

    // Verify it reaches 100%
    if let Some(last) = encoding.last() {
        assert_eq!(last.1, last.2, "Encoding progress should reach 100%");
    }
}

// ============================================================================
// Post-repair verification test
// ============================================================================

/// Verify that repair automatically verifies after completion
#[test]
fn test_repair_includes_post_verification() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"VERIFY_AFTER_REPAIR", 50_000).unwrap();
    let original_hash = compute_file_hash(&file);

    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(20.0);
    let par2_files = creator.create().unwrap();

    // Corrupt the file
    corrupt_file(&file, 10_000, &[0xFF; 5000]).unwrap();

    // Track messages to see post-repair verification message
    let messages = Arc::new(Mutex::new(Vec::new()));
    let msg_clone = messages.clone();

    let msg_cb = move |level: MessageLevel, msg: &str| {
        msg_clone.lock().unwrap().push((level, msg.to_string()));
    };

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_callbacks(true, false, None, Some(Arc::new(msg_cb)))
        .unwrap();

    // Verify the file was actually repaired
    assert_eq!(original_hash, compute_file_hash(&file));

    // Check that we got the post-repair verification message
    let msgs = messages.lock().unwrap();
    let has_verify_msg = msgs
        .iter()
        .any(|(_, msg)| msg.contains("Repair successful"));
    assert!(
        has_verify_msg,
        "Should see 'Repair successful' message after post-repair verification"
    );
}

// ============================================================================
// Purge after repair test
// ============================================================================

/// Verify purge works after successful repair (not just after clean verification)
#[test]
fn test_purge_after_successful_repair() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"PURGE_AFTER_REPAIR", 50_000).unwrap();
    let original_hash = compute_file_hash(&file);

    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(20.0);
    let par2_files = creator.create().unwrap();

    // Corrupt the file
    corrupt_file(&file, 5000, &[0xFF; 5000]).unwrap();

    // Repair with purge
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer.repair_with_progress(true, true, None).unwrap();

    // Verify file repaired
    assert_eq!(original_hash, compute_file_hash(&file));

    // Verify PAR2 files were purged
    let remaining_par2: Vec<_> = par2_files.iter().filter(|p| p.exists()).collect();
    assert_eq!(
        remaining_par2.len(),
        0,
        "All PAR2 files should be purged after successful repair"
    );
}

// ============================================================================
// Message callback tests
// ============================================================================

/// Verify message callbacks report useful information
#[test]
fn test_message_callbacks_during_repair() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"MSG_TEST", 50_000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(20.0);
    let par2_files = creator.create().unwrap();

    // Corrupt the file
    corrupt_file(&file, 5000, &[0xFF; 5000]).unwrap();

    let messages = Arc::new(Mutex::new(Vec::new()));
    let msg_clone = messages.clone();

    let msg_cb = move |level: MessageLevel, msg: &str| {
        msg_clone.lock().unwrap().push((level, msg.to_string()));
    };

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_callbacks(true, false, None, Some(Arc::new(msg_cb)))
        .unwrap();

    let msgs = messages.lock().unwrap();

    // Should have damage detection messages
    let has_damage_msg = msgs
        .iter()
        .any(|(level, _)| *level == MessageLevel::Warning);
    assert!(has_damage_msg, "Should see warning about file damage");

    // Should have repair info messages
    let has_repair_msg = msgs
        .iter()
        .any(|(level, msg)| *level == MessageLevel::Info && msg.contains("Repair"));
    assert!(has_repair_msg, "Should see repair-related info messages");
}

/// Test with missing file - should report it via message callback
#[test]
fn test_message_callbacks_missing_file() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"MISSING_TEST", 10_000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(100.0);
    let par2_files = creator.create().unwrap();

    // Delete the file
    fs::remove_file(&file).unwrap();

    let messages = Arc::new(Mutex::new(Vec::new()));
    let msg_clone = messages.clone();

    let msg_cb = move |level: MessageLevel, msg: &str| {
        msg_clone.lock().unwrap().push((level, msg.to_string()));
    };

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_callbacks(true, false, None, Some(Arc::new(msg_cb)))
        .unwrap();

    let msgs = messages.lock().unwrap();

    let has_missing_msg = msgs
        .iter()
        .any(|(level, msg)| *level == MessageLevel::Error && msg.contains("Missing"));
    assert!(
        has_missing_msg,
        "Should see error about missing file. Messages: {:?}",
        *msgs
    );
}

// ============================================================================
// Volume scheme tests with larger files
// ============================================================================

/// Test exponential volumes with a file large enough to produce many volumes
#[test]
fn test_exponential_volumes_many_blocks() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"VOLUMES_TEST", 1_000_000).unwrap();
    let original_hash = compute_file_hash(&file);

    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(50.0)
        .with_volume_scheme(VolumeScheme::Exponential)
        .with_output_path(temp.path().join("volumes.par2"));
    let par2_files = creator.create().unwrap();

    // Should have created multiple volume files
    assert!(
        par2_files.len() >= 3,
        "Should create multiple volume files, got {}",
        par2_files.len()
    );

    // Corrupt part of the file (within 50% redundancy)
    corrupt_file(&file, 100_000, &[0xFF; 200_000]).unwrap();

    // Repair from volumes
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer.repair(true).unwrap();

    assert_eq!(
        original_hash,
        compute_file_hash(&file),
        "File should be repaired from exponential volumes"
    );
}

// ============================================================================
// Concurrent progress safety test
// ============================================================================

/// Verify atomic progress counter works correctly with parallel verification
#[test]
fn test_parallel_verification_progress() {
    let temp = TempDir::new().unwrap();

    // Create 10 files to exercise parallel verification
    let files: Vec<_> = (0..10)
        .map(|i| {
            let path = temp.path().join(format!("para_{:02}.bin", i));
            create_pattern_file(&path, format!("PARA_{:02}", i).as_bytes(), 20_000).unwrap();
            path
        })
        .collect();

    let creator = Par2Creator::new(files.clone())
        .unwrap()
        .with_redundancy(10.0)
        .with_output_path(temp.path().join("parallel.par2"));
    let par2_files = creator.create().unwrap();

    let max_progress = Arc::new(AtomicU64::new(0));
    let max_clone = max_progress.clone();
    let total_seen = Arc::new(AtomicU64::new(0));
    let total_clone = total_seen.clone();

    let progress_cb = move |op: Par2Operation, current: u64, total: u64| {
        if op == Par2Operation::Verifying {
            max_clone.fetch_max(current, Ordering::Relaxed);
            total_clone.store(total, Ordering::Relaxed);
        }
    };

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_progress(false, false, Some(Arc::new(progress_cb)))
        .unwrap();

    let max = max_progress.load(Ordering::Relaxed);
    let total = total_seen.load(Ordering::Relaxed);

    assert!(total > 0, "Should see verification progress total");
    assert!(
        max <= total,
        "Progress max ({}) should not exceed total ({})",
        max,
        total
    );
    // Final progress should equal total (100%)
    assert_eq!(max, total, "Final progress should reach total");
}
