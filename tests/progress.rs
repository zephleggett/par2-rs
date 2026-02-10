//! Progress callback tests

use par2_rs::{Par2Creator, Par2Operation, Par2Repairer};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

mod common;
use common::{corrupt_file, create_pattern_file};

#[test]
fn test_progress_callbacks_during_repair() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"TEST", 50000).unwrap();

    // Create PAR2 with 20% redundancy
    let creator = Par2Creator::new(vec![file.clone()])
        .unwrap()
        .with_redundancy(20.0)
        .unwrap();
    let par2_files = creator.create().unwrap();

    // Damage the file
    corrupt_file(&file, 1000, &[0xFF; 5000]).unwrap();

    // Track progress callbacks
    let operations_seen = Arc::new(Mutex::new(Vec::new()));
    let ops_clone = operations_seen.clone();

    let progress_cb = move |operation: Par2Operation, current: u64, total: u64| {
        ops_clone.lock().unwrap().push((operation, current, total));
    };

    // Repair with progress callback
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_progress(true, false, Some(Arc::new(progress_cb)))
        .unwrap();

    // Verify we received progress callbacks for key operations
    let ops = operations_seen.lock().unwrap();
    let operations: Vec<_> = ops.iter().map(|(op, _, _)| *op).collect();

    assert!(
        operations.contains(&Par2Operation::Scanning),
        "Should see Scanning operation"
    );
    assert!(
        operations.contains(&Par2Operation::Loading),
        "Should see Loading operation"
    );
    assert!(
        operations.contains(&Par2Operation::Verifying),
        "Should see Verifying operation"
    );
    assert!(
        operations.contains(&Par2Operation::Repairing),
        "Should see Repairing operation"
    );

    // Verify progress values are reasonable
    for (_op, current, total) in ops.iter() {
        assert!(*total > 0, "Total should be non-zero");
        assert!(*current <= *total, "Current should not exceed total");
    }
}

#[test]
fn test_progress_callbacks_verify_only() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"TEST", 30000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Track operations
    let operations_seen = Arc::new(Mutex::new(Vec::new()));
    let ops_clone = operations_seen.clone();

    let progress_cb = move |operation: Par2Operation, _current: u64, _total: u64| {
        ops_clone.lock().unwrap().push(operation);
    };

    // Verify without repair
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_progress(false, false, Some(Arc::new(progress_cb)))
        .unwrap();

    let ops = operations_seen.lock().unwrap();

    // Should see scanning, loading, verifying - but NOT repairing
    assert!(ops.contains(&Par2Operation::Scanning));
    assert!(ops.contains(&Par2Operation::Loading));
    assert!(ops.contains(&Par2Operation::Verifying));
    assert!(
        !ops.contains(&Par2Operation::Repairing),
        "Should not repair when verify-only"
    );
}

#[test]
fn test_progress_callbacks_with_purge() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("data.bin");
    create_pattern_file(&file, b"DATA", 20000).unwrap();

    let creator = Par2Creator::new(vec![file.clone()]).unwrap();
    let par2_files = creator.create().unwrap();

    // Track operations
    let operations_seen = Arc::new(Mutex::new(Vec::new()));
    let ops_clone = operations_seen.clone();

    let progress_cb = move |operation: Par2Operation, _current: u64, _total: u64| {
        ops_clone.lock().unwrap().push(operation);
    };

    // Count PAR2 files before purge
    let par2_count_before = par2_files.len();
    assert!(par2_count_before > 0);

    // Verify with purge enabled
    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_progress(false, true, Some(Arc::new(progress_cb)))
        .unwrap();

    // Check that PAR2 files were purged
    let remaining_par2: Vec<_> = par2_files.iter().filter(|p| p.exists()).collect();
    assert_eq!(
        remaining_par2.len(),
        0,
        "All PAR2 files should be purged after successful verification"
    );

    let ops = operations_seen.lock().unwrap();
    assert!(ops.contains(&Par2Operation::Verifying));
}

#[test]
fn test_progress_increments_during_loading() {
    let temp = TempDir::new().unwrap();
    let file1 = temp.path().join("file1.bin");
    let file2 = temp.path().join("file2.bin");
    create_pattern_file(&file1, b"FILE1", 25000).unwrap();
    create_pattern_file(&file2, b"FILE2", 25000).unwrap();

    let creator = Par2Creator::new(vec![file1, file2]).unwrap();
    let par2_files = creator.create().unwrap();

    // Track loading progress
    let loading_progress = Arc::new(Mutex::new(Vec::new()));
    let progress_clone = loading_progress.clone();

    let progress_cb = move |operation: Par2Operation, current: u64, total: u64| {
        if operation == Par2Operation::Loading {
            progress_clone.lock().unwrap().push((current, total));
        }
    };

    let repairer = Par2Repairer::new(&par2_files[0]).unwrap();
    repairer
        .repair_with_progress(false, false, Some(Arc::new(progress_cb)))
        .unwrap();

    let progress = loading_progress.lock().unwrap();

    // Should have seen at least one loading progress update
    assert!(!progress.is_empty(), "Should see loading progress updates");

    // Progress should be monotonically increasing or constant
    for i in 1..progress.len() {
        assert!(
            progress[i].0 >= progress[i - 1].0,
            "Loading progress should not decrease"
        );
    }
}
