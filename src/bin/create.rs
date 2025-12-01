// PAR2 file creation CLI tool

use par2_rs::{Par2Creator, VolumeScheme};
use std::path::PathBuf;
use std::process;

fn main() {
    // Initialize tracing for library logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    // Parse command line arguments
    let mut input_files = Vec::new();
    let mut redundancy: Option<f32> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut block_size: Option<u64> = None;
    let mut single_volume = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage(&args[0]);
                process::exit(0);
            }
            "-r" | "--redundancy" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: {} requires an argument", args[i]);
                    process::exit(1);
                }
                i += 1;
                let redundancy_str = &args[i];
                redundancy = Some(parse_redundancy(redundancy_str));
            }
            "-o" | "--output" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: {} requires an argument", args[i]);
                    process::exit(1);
                }
                i += 1;
                output_path = Some(PathBuf::from(&args[i]));
            }
            "-b" | "--block-size" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: {} requires an argument", args[i]);
                    process::exit(1);
                }
                i += 1;
                block_size = Some(parse_size(&args[i]));
            }
            "-s" | "--single" => {
                single_volume = true;
            }
            arg if arg.starts_with('-') => {
                eprintln!("Error: Unknown option: {}", arg);
                print_usage(&args[0]);
                process::exit(1);
            }
            _ => {
                let path = PathBuf::from(&args[i]);
                if !path.exists() {
                    eprintln!("Error: File not found: {}", path.display());
                    process::exit(1);
                }
                if !path.is_file() {
                    eprintln!("Error: Not a file: {}", path.display());
                    process::exit(1);
                }
                input_files.push(path);
            }
        }
        i += 1;
    }

    if input_files.is_empty() {
        eprintln!("Error: No input files specified");
        print_usage(&args[0]);
        process::exit(1);
    }

    // Create PAR2 files
    println!("Creating PAR2 files for {} input files", input_files.len());
    if let Some(r) = redundancy {
        println!("Redundancy: {}%", r);
    } else {
        println!("Redundancy: 5% (default)");
    }

    let mut creator = match Par2Creator::new(input_files) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    if let Some(r) = redundancy {
        creator = creator.with_redundancy(r);
    }

    if let Some(path) = output_path {
        creator = creator.with_output_path(path);
    }

    if let Some(size) = block_size {
        creator = match creator.with_block_size(size) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        };
    }

    if single_volume {
        creator = creator.with_volume_scheme(VolumeScheme::Single);
    }

    match creator.create() {
        Ok(files) => {
            println!("\nSuccess! Created {} PAR2 file(s):", files.len());
            for file in files {
                println!("  {}", file.display());
            }
        }
        Err(e) => {
            eprintln!("Error creating PAR2 files: {}", e);
            process::exit(1);
        }
    }
}

fn print_usage(program: &str) {
    println!("Usage: {} [OPTIONS] <input files...>", program);
    println!();
    println!("Create PAR2 recovery files for the specified input files.");
    println!();
    println!("Options:");
    println!("  -r, --redundancy <percent>  Redundancy percentage (default: 5)");
    println!("                              Examples: -r 10, -r 5.5");
    println!("  -o, --output <path>         Output PAR2 file path (default: first_file.par2)");
    println!("  -b, --block-size <bytes>    Block size in bytes (default: auto-calculated)");
    println!("                              Supports: 2048, 1K, 1M, etc.");
    println!("  -s, --single                Create single volume (default: exponential)");
    println!("  -h, --help                  Show this help message");
    println!();
    println!("Examples:");
    println!("  {} file1.bin file2.bin file3.bin", program);
    println!("  {} -r 10 -o backup.par2 *.bin", program);
    println!("  {} -b 2M -s archive.tar", program);
}

fn parse_redundancy(s: &str) -> f32 {
    // Remove % if present
    let s = s.trim_end_matches('%');

    match s.parse::<f32>() {
        Ok(v) if v > 0.0 && v <= 100.0 => v,
        _ => {
            eprintln!("Error: Invalid redundancy value: {}", s);
            eprintln!("Must be between 0 and 100");
            process::exit(1);
        }
    }
}

fn parse_size(s: &str) -> u64 {
    let s = s.trim().to_uppercase();

    // Try to parse as plain number first
    if let Ok(size) = s.parse::<u64>() {
        return size;
    }

    // Parse with suffix (K, M, G)
    let (num_str, multiplier) = if s.ends_with('K') {
        (&s[..s.len() - 1], 1024u64)
    } else if s.ends_with('M') {
        (&s[..s.len() - 1], 1024u64 * 1024)
    } else if s.ends_with('G') {
        (&s[..s.len() - 1], 1024u64 * 1024 * 1024)
    } else {
        eprintln!("Error: Invalid size format: {}", s);
        eprintln!("Use plain numbers or suffixes: K, M, G (e.g., 2M, 512K)");
        process::exit(1);
    };

    match num_str.parse::<u64>() {
        Ok(num) => num * multiplier,
        Err(_) => {
            eprintln!("Error: Invalid size format: {}", s);
            process::exit(1);
        }
    }
}
