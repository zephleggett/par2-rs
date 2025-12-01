//! Display information about available SIMD strategies

use par2_rs::galois;

fn main() {
    // Initialize the SIMD registry
    galois::configure_simd();

    println!("Available SIMD Strategies:");
    println!("{}", "=".repeat(50));

    let strategies = galois::list_simd_strategies();
    for (name, priority) in &strategies {
        let marker = if Some(*name) == galois::get_selected_simd_strategy() {
            "→"
        } else {
            " "
        };
        println!("{} {} (priority: {:?})", marker, name, priority);
    }

    println!(
        "\nSelected strategy: {}",
        galois::get_selected_simd_strategy().unwrap_or("None")
    );

    println!("\nThis strategy will be used for:");
    println!("  • gf_mul_slice() - Multiply slice by scalar");
    println!("  • gf_muladd() - Multiply-accumulate");
    println!("  • gf_muladd_region() - Region-based processing (Reed-Solomon reconstruction)");
}
