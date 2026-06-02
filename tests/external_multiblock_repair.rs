//! Regression tests for multi-block repair of REAL (par2cmdline-generated) sets.
//!
//! These guard against the FileID-ordering bug where repairs that needed a
//! recovery block with exponent >= 1 reconstructed WRONG bytes. Exponent-0
//! recovery is plain XOR parity (independent of the RS constants), so single-block
//! repairs hid the bug; anything needing 2+ recovery blocks exposed it.
//!
//! Every test asserts that `repair(true)` returns `Ok` AND that the repaired
//! file(s) are byte-identical to pristine copies of the bundled fixtures.

use par2_rs::{Par2Info, Par2Repairer};
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

mod common;
use common::test_data_dir;

/// Copy the entire bundled fixture set (real par2cmdline output) into a fresh
/// temp dir so tests can mutate copies without touching the read-only fixtures.
fn setup_fixture(tag: &str) -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::TempDir::new().unwrap();
    let base = temp.path();
    for entry in fs::read_dir(test_data_dir()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".data") || name.ends_with(".par2") {
            fs::copy(entry.path(), base.join(&name)).unwrap();
        }
    }
    let idx = base.join("testdata.par2");
    assert!(idx.exists(), "fixture index missing for tag {tag}");
    (temp, idx)
}

/// Overwrite `len` bytes at `off` with a fixed corruption pattern.
fn corrupt_at(path: &Path, off: u64, len: usize) {
    let mut f = fs::OpenOptions::new().write(true).open(path).unwrap();
    f.seek(SeekFrom::Start(off)).unwrap();
    f.write_all(&vec![0xABu8; len]).unwrap();
}

fn block_size(idx: &Path) -> u64 {
    Par2Info::load(idx).unwrap().block_size
}

/// (a) Two separate blocks corrupted in a single file -> repairs byte-exact.
/// Needs >=2 recovery blocks, so it exercises exponent >= 1.
#[test]
fn two_blocks_in_one_file_repairs_byte_exact() {
    let (_tmp, idx) = setup_fixture("two_in_one");
    let base = idx.parent().unwrap();
    let target = base.join("test-1.data");
    let pristine = fs::read(test_data_dir().join("test-1.data")).unwrap();
    let bs = block_size(&idx);

    // Corrupt two non-adjacent blocks.
    corrupt_at(&target, 10, 64);
    corrupt_at(&target, bs * 3 + 10, 64);

    let repairer = Par2Repairer::new(&idx).unwrap();
    repairer
        .repair(true)
        .expect("multi-block repair (exp>=1) must succeed");

    let after = fs::read(&target).unwrap();
    assert_eq!(
        after, pristine,
        "two-block repair must reconstruct byte-identical original"
    );
}

/// (a') Three separate blocks corrupted in a single file -> repairs byte-exact.
/// Forces recovery exponents 0,1,2 together.
#[test]
fn three_blocks_in_one_file_repairs_byte_exact() {
    let (_tmp, idx) = setup_fixture("three_in_one");
    let base = idx.parent().unwrap();
    let target = base.join("test-1.data");
    let pristine = fs::read(test_data_dir().join("test-1.data")).unwrap();
    let bs = block_size(&idx);

    corrupt_at(&target, 10, 50);
    corrupt_at(&target, bs * 2 + 10, 50);
    corrupt_at(&target, bs * 5 + 10, 50);

    let repairer = Par2Repairer::new(&idx).unwrap();
    repairer
        .repair(true)
        .expect("three-block repair (exp 0,1,2) must succeed");

    let after = fs::read(&target).unwrap();
    assert_eq!(after, pristine, "three-block repair must be byte-identical");
}

/// (b) Corrupt blocks in TWO different files -> both byte-exact.
#[test]
fn blocks_in_two_files_repair_byte_exact() {
    let (_tmp, idx) = setup_fixture("two_files");
    let base = idx.parent().unwrap();
    let bs = block_size(&idx);

    let t1 = base.join("test-1.data");
    let t4 = base.join("test-4.data");
    let pristine1 = fs::read(test_data_dir().join("test-1.data")).unwrap();
    let pristine4 = fs::read(test_data_dir().join("test-4.data")).unwrap();

    corrupt_at(&t1, bs + 5, 40);
    corrupt_at(&t4, bs * 2 + 5, 40);

    let repairer = Par2Repairer::new(&idx).unwrap();
    repairer
        .repair(true)
        .expect("cross-file multi-block repair must succeed");

    assert_eq!(
        fs::read(&t1).unwrap(),
        pristine1,
        "test-1 must be byte-exact"
    );
    assert_eq!(
        fs::read(&t4).unwrap(),
        pristine4,
        "test-4 must be byte-exact"
    );
}

/// (c) Delete a whole multi-block file -> recreated byte-exact.
#[test]
fn deleted_multiblock_file_recreated_byte_exact() {
    let (_tmp, idx) = setup_fixture("deleted");
    let base = idx.parent().unwrap();

    // test-1.data is the largest file (33 blocks), so this needs many recovery
    // blocks with exponents spanning 0..32.
    let target = base.join("test-1.data");
    let pristine = fs::read(test_data_dir().join("test-1.data")).unwrap();
    fs::remove_file(&target).unwrap();

    let repairer = Par2Repairer::new(&idx).unwrap();
    repairer
        .repair(true)
        .expect("recreating a deleted multi-block file must succeed");

    assert!(target.exists(), "deleted file must be recreated");
    assert_eq!(
        fs::read(&target).unwrap(),
        pristine,
        "recreated file must be byte-identical to the original"
    );
}

/// (d) Single damaged block (exponent-0 only) keeps working.
#[test]
fn single_block_repairs_byte_exact() {
    let (_tmp, idx) = setup_fixture("single");
    let base = idx.parent().unwrap();
    let target = base.join("test-1.data");
    let pristine = fs::read(test_data_dir().join("test-1.data")).unwrap();

    corrupt_at(&target, 10, 32);

    let repairer = Par2Repairer::new(&idx).unwrap();
    repairer
        .repair(true)
        .expect("single-block repair must succeed");

    assert_eq!(
        fs::read(&target).unwrap(),
        pristine,
        "single-block repair must be byte-identical"
    );
}

/// Pre-rename verification gate: if reconstruction would produce WRONG bytes,
/// repair must fail and leave the user's (damaged) original byte-identical rather
/// than clobbering it. We force a wrong reconstruction by tampering with the
/// recovery DATA in a volume file so the RS math yields incorrect output.
#[test]
fn bad_reconstruction_does_not_clobber_original() {
    let (_tmp, idx) = setup_fixture("bad_recovery");
    let base = idx.parent().unwrap();
    let bs = block_size(&idx);

    // Corrupt the recovery DATA inside the exponent-0 volume so reconstruction is
    // wrong but the available-block count still appears sufficient. A single
    // damaged block uses the lowest-exponent recovery block (exp 0), so tampering
    // here guarantees the reconstruction will be incorrect.
    //
    // The RecoverySlice packet body is: exponent (4 bytes) then `block_size` bytes
    // of recovery payload. Locate the packet by its type marker and flip bytes in
    // the payload region (after the 64-byte header + 4-byte exponent).
    let vol = base.join("testdata.vol00+01.par2");
    let mut data = fs::read(&vol).unwrap();
    let marker = b"PAR 2.0\x00RecvSlic";
    let pkt = data
        .windows(marker.len())
        .position(|w| w == marker)
        .expect("RecvSlice packet must exist in vol00+01");
    // The type marker sits at header offset 48, so the packet starts 48 bytes back
    // and the payload begins 16 (rest of header) + 4 (exponent) bytes after it.
    let payload_start = pkt + 16 + 4;
    for b in data
        .iter_mut()
        .skip(payload_start + 100)
        .take((bs as usize) / 2)
    {
        *b ^= 0xFF;
    }
    fs::write(&vol, &data).unwrap();

    // Damage one block in the target file; repair will try to use the (tampered)
    // recovery data and reconstruct wrong bytes.
    let target = base.join("test-1.data");
    corrupt_at(&target, bs + 5, 64);
    let damaged_snapshot = fs::read(&target).unwrap();

    let repairer = Par2Repairer::new(&idx).unwrap();
    let result = repairer.repair(true);
    assert!(
        result.is_err(),
        "repair must fail when reconstruction can't be verified"
    );

    // The pre-rename gate must have refused to commit: the original file must be
    // exactly as it was before repair (still the damaged bytes, NOT garbage from a
    // bad reconstruction), and no temp file may be left behind.
    let after = fs::read(&target).unwrap();
    assert_eq!(
        after, damaged_snapshot,
        "failed repair must leave the original byte-identical (no clobber)"
    );
    assert!(
        !base.join("test-1.data.par2tmp").exists(),
        "temp file must be cleaned up on verification failure"
    );
}

/// Intact fixtures must verify cleanly (sanity guard on the ordering change).
#[test]
fn intact_fixtures_verify_ok() {
    let (_tmp, idx) = setup_fixture("intact");
    let repairer = Par2Repairer::new(&idx).unwrap();
    repairer
        .repair(false)
        .expect("intact real fixtures must verify");
}
