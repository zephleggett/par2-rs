use par2_rs::Result;
use std::path::Path;

fn main() -> Result<()> {
    let par2_file = Path::new("/Users/zeph/Downloads/A.Charlie.Brown.Christmas.1965.2160p.BDRip.AAC.5.1.HDR10.x265.10bit-MarkII-xpost/015c154f2d9c4dc9bfcaddbfb42318fd.par2");

    println!("Testing PAR2 repair for part53");
    println!("==============================\n");

    // First, just try to verify (no repair)
    println!("Step 1: Verifying without repair...");
    let repairer = par2_rs::Par2Repairer::new(par2_file)?;
    match repairer.repair(false) {
        Ok(()) => println!("✓ All files are OK - no repair needed"),
        Err(e) => println!("✗ Verification failed: {}", e),
    }

    println!("\nStep 2: Attempting repair...");
    let repairer2 = par2_rs::Par2Repairer::new(par2_file)?;
    match repairer2.repair(true) {
        Ok(()) => {
            println!("✓ Repair completed successfully!");

            // Verify again after repair
            println!("\nStep 3: Re-verifying after repair...");
            let repairer3 = par2_rs::Par2Repairer::new(par2_file)?;
            match repairer3.repair(false) {
                Ok(()) => println!("✓ All files are now OK!"),
                Err(e) => println!("✗ Files still damaged after repair: {}", e),
            }

            Ok(())
        }
        Err(e) => {
            println!("✗ Repair failed: {}", e);
            Err(e)
        }
    }
}
