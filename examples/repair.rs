use par2_rs::{Par2Repairer, Result};
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

fn main() -> Result<()> {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    println!("Copying test files to temp directory...");
    for entry in fs::read_dir(&source).expect("Failed to read source dir") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.is_file() {
            let filename = path.file_name().expect("No filename");
            fs::copy(&path, temp_dir.path().join(filename)).expect("Failed to copy");
        }
    }

    // Damage a test file
    let damaged_file = temp_dir.path().join("test-0.data");
    println!("Damaging file: {}", damaged_file.display());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(&damaged_file)
        .expect("Failed to open file");
    file.write_all(b"CORRUPTED").expect("Failed to write");
    drop(file);

    // Try to repair
    let par2_file = temp_dir.path().join("testdata.par2");
    println!("Attempting repair with: {}", par2_file.display());

    let repairer = Par2Repairer::new(&par2_file)?;

    match repairer.repair(true) {
        Ok(()) => {
            println!("✓ Repair successful!");
            Ok(())
        }
        Err(e) => {
            println!("✗ Repair failed: {}", e);
            Err(e)
        }
    }
}
