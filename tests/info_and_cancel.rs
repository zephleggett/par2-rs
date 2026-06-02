//! Tests for the read-only `Par2Info` metadata API and cancellation support
//! (verify + temp-file-safe repair).
//!
//! These use the REAL fixtures under `tests/data/` (`testdata.par2`,
//! `testdata.vol*.par2`, and `test-*.data`). For any test that mutates files,
//! the fixtures are first copied into a `TempDir` so the read-only originals are
//! never touched.

use par2_rs::{Par2Error, Par2Info, Par2Repairer};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

mod common;
use common::test_data_dir;

/// Copy the entire PAR2 fixture set (par2 files + data files) into a fresh temp
/// dir and return (tempdir, path-to-index-par2).
fn copy_fixture_set() -> (TempDir, PathBuf) {
    let src = test_data_dir();
    let temp = TempDir::new().unwrap();
    let dst = temp.path();

    for entry in fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().unwrap();
        let name_str = name.to_string_lossy();
        // Copy all par2 files and all data files.
        if name_str.ends_with(".par2") || name_str.ends_with(".data") {
            fs::copy(&path, dst.join(name)).unwrap();
        }
    }

    let index = dst.join("testdata.par2");
    assert!(index.exists(), "fixture index par2 must exist after copy");
    (temp, index)
}

/// Snapshot every `.data` file in `dir` as name -> bytes.
fn snapshot_data_files(dir: &Path) -> HashMap<String, Vec<u8>> {
    let mut map = HashMap::new();
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("data") {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            map.insert(name, fs::read(&path).unwrap());
        }
    }
    map
}

/// Assert that the on-disk `.data` files exactly match a prior snapshot, and that
/// no stray `.par2tmp` files were left behind.
fn assert_data_files_unchanged(dir: &Path, before: &HashMap<String, Vec<u8>>) {
    let after = snapshot_data_files(dir);
    assert_eq!(
        after.len(),
        before.len(),
        "number of data files changed (before={}, after={})",
        before.len(),
        after.len()
    );
    for (name, bytes) in before {
        let now = after
            .get(name)
            .unwrap_or_else(|| panic!("data file {} disappeared", name));
        assert_eq!(
            now, bytes,
            "data file {} was modified by a cancelled call",
            name
        );
    }

    // No leftover temp files from a cancelled/failed repair.
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            !name.ends_with(".par2tmp"),
            "leftover temp file found after cancellation: {}",
            name
        );
    }
}

// ---------------------------------------------------------------------------
// Part A: Par2Info read-only metadata API
// ---------------------------------------------------------------------------

#[test]
fn test_info_load_basic_fields() {
    let par2 = test_data_dir().join("testdata.par2");
    let info = Par2Info::load(&par2).expect("Par2Info::load should succeed on fixture");

    assert!(info.block_size > 0, "block_size must be positive");
    assert!(!info.files.is_empty(), "files list must be non-empty");

    for f in &info.files {
        assert!(!f.name.is_empty(), "file name must be non-empty");
        assert!(f.length > 0, "file length must be plausible (>0)");
    }

    assert!(
        info.distinct_recovery_blocks <= info.recovery_block_count,
        "distinct ({}) must be <= raw count ({})",
        info.distinct_recovery_blocks,
        info.recovery_block_count
    );
    assert!(
        info.distinct_recovery_blocks > 0,
        "fixture set should have usable recovery capacity"
    );
}

#[test]
fn test_info_can_repair_boundary() {
    let par2 = test_data_dir().join("testdata.par2");
    let info = Par2Info::load(&par2).unwrap();

    let cap = info.distinct_recovery_blocks;
    assert!(info.can_repair(cap), "can_repair(distinct) must be true");
    assert!(
        !info.can_repair(cap + 1),
        "can_repair(distinct + 1) must be false"
    );
    assert!(info.can_repair(0), "can_repair(0) must be true");
}

#[test]
fn test_info_total_data_blocks_matches_manual_sum() {
    let par2 = test_data_dir().join("testdata.par2");
    let info = Par2Info::load(&par2).unwrap();

    let bs = info.block_size.max(1);
    let manual: u64 = info.files.iter().map(|f| f.length.div_ceil(bs)).sum();

    assert_eq!(
        info.total_data_blocks(),
        manual,
        "total_data_blocks() must equal manual div_ceil sum"
    );
}

#[test]
fn test_info_load_from_volume_file_discovers_siblings() {
    // Loading from a vol file (not the index) should still discover the rest of
    // the set via recovery_set_id and report the same protected files.
    let dir = test_data_dir();
    let index = dir.join("testdata.par2");
    let vol = dir.join("testdata.vol00+01.par2");
    assert!(vol.exists(), "fixture vol file must exist");

    let from_index = Par2Info::load(&index).unwrap();
    let from_vol = Par2Info::load(&vol).unwrap();

    assert_eq!(
        from_index.files.len(),
        from_vol.files.len(),
        "file count must match whether loaded from index or vol"
    );
    assert_eq!(
        from_index.block_size, from_vol.block_size,
        "block_size must match whether loaded from index or vol"
    );
}

// ---------------------------------------------------------------------------
// Part B: cancellation + temp-file-safe repair
// ---------------------------------------------------------------------------

#[test]
fn test_old_repair_with_callbacks_still_works() {
    // The legacy 4-arg signature must still perform a normal repair (delegates
    // with cancel = None).
    let (temp, index) = copy_fixture_set();
    let dir = temp.path();

    // Corrupt one data file so repair is required.
    let target = dir.join("test-3.data");
    let before = fs::read(&target).unwrap();
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = fs::OpenOptions::new().write(true).open(&target).unwrap();
        f.seek(SeekFrom::Start(1024)).unwrap();
        f.write_all(&[0xFFu8; 4096]).unwrap();
    }
    assert_ne!(fs::read(&target).unwrap(), before, "corruption must take");

    let repairer = Par2Repairer::new(&index).unwrap();
    // Verify-only should detect damage.
    assert!(repairer.repair(false).is_err(), "damage should be detected");

    // Legacy signature repair should succeed and restore the file.
    repairer
        .repair_with_callbacks(true, false, None, None)
        .expect("legacy repair_with_callbacks must succeed");

    assert_eq!(
        fs::read(&target).unwrap(),
        before,
        "repaired file must match original bytes"
    );
}

#[test]
fn test_repair_cancellation_preset_flag_is_safe() {
    // STRONGEST temp-file-safety guard: set the cancel flag BEFORE the call so
    // cancellation is observed deterministically. Regardless of where it trips,
    // the on-disk data files must be byte-identical afterwards.
    let (temp, index) = copy_fixture_set();
    let dir = temp.path();

    // Corrupt a couple of files (within recovery capacity) so repair is needed.
    {
        use std::io::{Seek, SeekFrom, Write};
        for name in ["test-1.data", "test-6.data"] {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .open(dir.join(name))
                .unwrap();
            f.seek(SeekFrom::Start(2048)).unwrap();
            f.write_all(&[0xABu8; 8192]).unwrap();
        }
    }

    let before = snapshot_data_files(dir);

    let cancel = Arc::new(AtomicBool::new(true)); // already cancelled
    let repairer = Par2Repairer::new(&index).unwrap();
    let result =
        repairer.repair_with_callbacks_cancellable(true, true, None, None, Some(cancel.clone()));

    assert!(
        matches!(result, Err(Par2Error::Cancelled)),
        "expected Cancelled, got {:?}",
        result
    );

    // Temp-file safety regression guard: corrupted data files remain exactly as
    // they were before the cancelled call (originals never mutated).
    assert_data_files_unchanged(dir, &before);
}

#[test]
fn test_repair_cancellation_from_thread_is_safe() {
    // Set the flag from another thread immediately. This may win the race at any
    // stage; whether it returns Cancelled or completes, the invariant we assert
    // is that IF it returns Cancelled, data files are byte-identical.
    let (temp, index) = copy_fixture_set();
    let dir = temp.path();

    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = fs::OpenOptions::new()
            .write(true)
            .open(dir.join("test-4.data"))
            .unwrap();
        f.seek(SeekFrom::Start(512)).unwrap();
        f.write_all(&[0xCDu8; 16384]).unwrap();
    }

    let before = snapshot_data_files(dir);

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_setter = cancel.clone();
    let handle = thread::spawn(move || {
        // Fire essentially immediately.
        cancel_setter.store(true, Ordering::Relaxed);
    });

    let repairer = Par2Repairer::new(&index).unwrap();
    let result =
        repairer.repair_with_callbacks_cancellable(true, false, None, None, Some(cancel.clone()));
    handle.join().unwrap();

    match result {
        Err(Par2Error::Cancelled) => {
            // If cancelled, originals must be untouched (temp-file safety).
            assert_data_files_unchanged(dir, &before);
        }
        Ok(()) => {
            // Cancellation lost the race and the repair completed: the corrupted
            // file must now be repaired (different from the corrupted snapshot).
            let repaired = fs::read(dir.join("test-4.data")).unwrap();
            assert_ne!(
                &repaired,
                before.get("test-4.data").unwrap(),
                "if repair completed, the file should have been fixed"
            );
            // And no temp files left behind.
            for entry in fs::read_dir(dir).unwrap() {
                let name = entry.unwrap().file_name().to_string_lossy().to_string();
                assert!(!name.ends_with(".par2tmp"), "stray temp file: {}", name);
            }
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn test_verify_cancellation_preset_flag() {
    // Verify-only with a pre-set cancel flag must return Cancelled and never
    // touch the data files.
    let (temp, index) = copy_fixture_set();
    let dir = temp.path();

    let before = snapshot_data_files(dir);

    let cancel = Arc::new(AtomicBool::new(true));
    let repairer = Par2Repairer::new(&index).unwrap();
    let result = repairer.repair_with_callbacks_cancellable(false, false, None, None, Some(cancel));

    assert!(
        matches!(result, Err(Par2Error::Cancelled)),
        "verify with preset cancel must return Cancelled, got {:?}",
        result
    );
    assert_data_files_unchanged(dir, &before);
}

#[test]
fn test_verify_cancellation_from_thread() {
    // Verify with the flag set from another thread. Intact fixtures verify fast,
    // so this races; we just assert the result is either Ok or Cancelled (never
    // a spurious other error) and files are untouched.
    let (temp, index) = copy_fixture_set();
    let dir = temp.path();

    let before = snapshot_data_files(dir);

    let cancel = Arc::new(AtomicBool::new(false));
    let setter = cancel.clone();
    let handle = thread::spawn(move || {
        thread::sleep(Duration::from_micros(1));
        setter.store(true, Ordering::Relaxed);
    });

    let repairer = Par2Repairer::new(&index).unwrap();
    let result = repairer.repair_with_callbacks_cancellable(false, false, None, None, Some(cancel));
    handle.join().unwrap();

    match result {
        Ok(()) | Err(Par2Error::Cancelled) => {}
        Err(e) => panic!("unexpected error from verify: {:?}", e),
    }
    // Verify never writes data files.
    assert_data_files_unchanged(dir, &before);
}

#[test]
fn test_normal_cancellable_repair_succeeds_with_flag_unset() {
    // Sanity: the cancellable path with a never-set flag behaves like a normal
    // successful repair. We corrupt a single block (the fixture's block_size is
    // 5376 bytes) so the damage is well within recovery capacity.
    let (temp, index) = copy_fixture_set();
    let dir = temp.path();

    let target = dir.join("test-2.data");
    let before = fs::read(&target).unwrap();
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = fs::OpenOptions::new().write(true).open(&target).unwrap();
        // Stay within a single block (offset + len < block_size).
        f.seek(SeekFrom::Start(100)).unwrap();
        f.write_all(&[0x00u8; 1000]).unwrap();
    }
    assert_ne!(fs::read(&target).unwrap(), before, "corruption must take");

    let cancel = Arc::new(AtomicBool::new(false));
    let repairer = Par2Repairer::new(&index).unwrap();
    repairer
        .repair_with_callbacks_cancellable(true, false, None, None, Some(cancel))
        .expect("repair with unset cancel flag must succeed");

    assert_eq!(
        fs::read(&target).unwrap(),
        before,
        "repaired file must match original"
    );
}
