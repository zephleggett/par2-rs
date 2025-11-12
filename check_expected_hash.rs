use par2_rs::Result;
use std::path::Path;

// We need to access internal parse functions
fn main() -> Result<()> {
    let par2_file = Path::new("/Users/zeph/Downloads/A.Charlie.Brown.Christmas.1965.2160p.BDRip.AAC.5.1.HDR10.x265.10bit-MarkII-xpost/015c154f2d9c4dc9bfcaddbfb42318fd.par2");
    let base_path = par2_file.parent().unwrap();

    // This won't work because Par2File::load is not accessible
    // Let me try a different approach
    println!("Can't access internal structs");
    Ok(())
}
