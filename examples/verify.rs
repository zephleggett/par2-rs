use par2_rs::{Par2Repairer, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let test_data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let par2_file = test_data.join("testdata.par2");

    println!("Verifying files with: {}", par2_file.display());

    let repairer = Par2Repairer::new(&par2_file)?;

    match repairer.repair(false) {
        Ok(()) => {
            println!("✓ All files verified successfully!");
            Ok(())
        }
        Err(e) => {
            println!("✗ Verification failed: {}", e);
            Err(e)
        }
    }
}
