use par2_rs::Result;
use std::path::Path;
use std::fs::OpenOptions;
use std::io::Write as _;

fn main() -> Result<()> {
    let base_path = Path::new("/Users/zeph/Downloads/A.Charlie.Brown.Christmas.1965.2160p.BDRip.AAC.5.1.HDR10.x265.10bit-MarkII-xpost");
    let par2_file = base_path.join("015c154f2d9c4dc9bfcaddbfb42318fd.par2");
    let target_file = base_path.join("015c154f2d9c4dc9bfcaddbfb42318fd.part53.rar");

    println!("PAR2 Repair Demo");
    println!("================\n");

    // Step 1: Verify current state
    println!("Step 1: Verify all files are OK");
    let repairer = par2_rs::Par2Repairer::new(&par2_file)?;
    match repairer.repair(false) {
        Ok(()) => println!("✓ All files verified OK\n"),
        Err(e) => {
            println!("✗ Files are already damaged: {}\n", e);
            println!("Repairing before demo...");
            repairer.repair(true)?;
            println!("✓ Repaired\n");
        }
    }

    // Step 2: Damage the file
    println!("Step 2: Damaging part53.rar (overwriting 50KB with zeros)");
    {
        let mut file = OpenOptions::new()
            .write(true)
            .open(&target_file)?;

        let zeros = vec![0u8; 50 * 1024];
        file.write_all(&zeros)?;
    }
    println!("✓ File damaged\n");

    // Step 3: Verify damage detected
    println!("Step 3: Verify damage is detected");
    let repairer = par2_rs::Par2Repairer::new(&par2_file)?;
    match repairer.repair(false) {
        Ok(()) => {
            println!("✗ ERROR: Damage not detected!");
            return Ok(());
        }
        Err(_) => {
            println!("✓ Damage detected correctly\n");
        }
    }

    // Step 4: Repair the file
    println!("Step 4: Repairing damaged file with par2-rs");
    let repairer = par2_rs::Par2Repairer::new(&par2_file)?;
    match repairer.repair(true) {
        Ok(()) => println!("✓ Repair completed\n"),
        Err(e) => {
            println!("✗ Repair failed: {}", e);
            return Err(e);
        }
    }

    // Step 5: Verify repair was successful
    println!("Step 5: Verify repaired file is now OK");
    let repairer = par2_rs::Par2Repairer::new(&par2_file)?;
    match repairer.repair(false) {
        Ok(()) => {
            println!("✓ All files verified OK after repair\n");
            println!("SUCCESS! The file was damaged and successfully repaired.");
        }
        Err(e) => {
            println!("✗ File still damaged after repair: {}", e);
            return Err(e);
        }
    }

    Ok(())
}
