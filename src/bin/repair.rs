use par2_rs::{Par2Repairer, Result};
use std::env;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize tracing subscriber to enable RUST_LOG output
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <par2_file> [verify|repair]", args[0]);
        eprintln!("\nCommands:");
        eprintln!("  verify  - Check file integrity only");
        eprintln!("  repair  - Verify and repair if needed (default)");
        std::process::exit(1);
    }

    let par2_path = Path::new(&args[1]);
    let verify_only = args.get(2).map(|s| s.as_str()) == Some("verify");

    // Canonicalize the path to handle relative paths correctly
    let par2_path = par2_path
        .canonicalize()
        .unwrap_or_else(|_| par2_path.to_path_buf());

    let total_start = Instant::now();

    let init_start = Instant::now();
    let repairer = Par2Repairer::new(&par2_path)?;
    eprintln!(
        "[TIMING] Initialization: {:.3}s",
        init_start.elapsed().as_secs_f64()
    );

    if verify_only {
        println!("Verifying files...");
        let verify_start = Instant::now();
        repairer.repair(false)?;
        eprintln!(
            "[TIMING] Verification: {:.3}s",
            verify_start.elapsed().as_secs_f64()
        );
        println!("All files verified successfully");
    } else {
        println!("Verifying and repairing files...");
        let repair_start = Instant::now();
        repairer.repair(true)?;
        eprintln!(
            "[TIMING] Repair: {:.3}s",
            repair_start.elapsed().as_secs_f64()
        );
        println!("Complete");
    }

    eprintln!(
        "[TIMING] Total: {:.3}s",
        total_start.elapsed().as_secs_f64()
    );
    Ok(())
}
