/// Common test utilities and fixtures for PAR2 testing
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Get path to read-only test data directory
#[allow(dead_code)]
pub fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// Create test file with deterministic content from existing test data files
///
/// This concatenates all .data files from tests/data repeatedly until target size is reached.
/// Useful for creating realistic test data that's reproducible.
#[allow(dead_code)]
pub fn create_test_file(target_path: &Path, target_size: usize) -> std::io::Result<usize> {
    let data_dir = test_data_dir();

    // Collect all .data files
    let mut data_files: Vec<PathBuf> = fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "data"))
        .collect();
    data_files.sort(); // Deterministic order

    if data_files.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No .data files found in tests/data",
        ));
    }

    // Read all data files into memory once
    let mut all_data = Vec::new();
    for file_path in &data_files {
        let mut data = Vec::new();
        File::open(file_path)?.read_to_end(&mut data)?;
        all_data.extend(data);
    }

    if all_data.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Data files are empty",
        ));
    }

    // Write data repeatedly until we reach target size
    let mut output = File::create(target_path)?;
    let mut written = 0;

    while written < target_size {
        let remaining = target_size - written;
        let chunk_size = remaining.min(all_data.len());
        output.write_all(&all_data[..chunk_size])?;
        written += chunk_size;
    }

    Ok(written)
}

/// Create test file with simple repeating pattern (faster for small tests)
///
/// Example: `create_pattern_file(path, b"ABCD", 1000)` creates a 1000-byte file
/// with "ABCDABCDABCD..." pattern.
pub fn create_pattern_file(path: &Path, pattern: &[u8], size: usize) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let mut written = 0;
    while written < size {
        let chunk = (size - written).min(pattern.len());
        file.write_all(&pattern[..chunk])?;
        written += chunk;
    }
    Ok(())
}

/// Compute MD5 hash of file (PAR2 uses MD5)
pub fn compute_file_hash(path: &Path) -> [u8; 16] {
    use md5::{Digest, Md5};

    let mut file = File::open(path).expect("Failed to open file for hashing");
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 65536];

    loop {
        let bytes_read = file.read(&mut buffer).expect("Failed to read file");
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    hasher.finalize().into()
}

/// Corrupt file at specified offset with given data
pub fn corrupt_file(path: &Path, offset: u64, data: &[u8]) -> std::io::Result<()> {
    let mut file = File::options().write(true).open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(data)?;
    Ok(())
}

/// Create a minimal PAR2 file without IFSC packets for testing fallback path
/// This creates a valid PAR2 file with main and file descriptor packets only
#[allow(dead_code)]
pub fn create_par2_without_ifsc(data_file: &Path, output_path: &Path) -> std::io::Result<()> {
    use md5::{Digest, Md5};
    use std::io::Write;

    // Read the data file
    let mut file = File::open(data_file)?;
    let file_size = file.metadata()?.len();

    // Compute file hash
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 65536];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let file_hash: [u8; 16] = hasher.finalize().into();

    // Compute 16KB hash
    let mut file = File::open(data_file)?;
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 16384];
    let bytes_read = file.read(&mut buffer)?;
    hasher.update(&buffer[..bytes_read]);
    let hash_16k: [u8; 16] = hasher.finalize().into();

    // Generate file ID (MD5 of hash_16k + length + filename) - per PAR2 spec
    let filename = data_file.file_name().unwrap().to_str().unwrap();
    let mut hasher = Md5::new();
    hasher.update(hash_16k);
    hasher.update(file_size.to_le_bytes());
    hasher.update(filename.as_bytes());
    let file_id: [u8; 16] = hasher.finalize().into();

    // Build main packet body first (needed to compute recovery set ID)
    let block_size: u64 = 2048;
    let num_files: u64 = 1;
    let file_ids_hash = {
        let mut hasher = Md5::new();
        hasher.update(file_id);
        let result: [u8; 16] = hasher.finalize().into();
        result
    };

    let mut main_body = Vec::new();
    main_body.extend_from_slice(&block_size.to_le_bytes());
    main_body.extend_from_slice(&num_files.to_le_bytes());
    main_body.extend_from_slice(&file_ids_hash);

    // Compute recovery set ID from main packet body (per PAR2 spec)
    let recovery_set_id: [u8; 16] = {
        let mut hasher = Md5::new();
        hasher.update(&main_body);
        hasher.finalize().into()
    };

    let mut output = File::create(output_path)?;

    // Helper to write a complete packet
    let write_packet =
        |output: &mut File, packet_type: &[u8; 16], body: &[u8]| -> std::io::Result<()> {
            // Pad body to 4-byte alignment
            let mut aligned_body = body.to_vec();
            while aligned_body.len() % 4 != 0 {
                aligned_body.push(0);
            }

            let packet_length = 64u64 + aligned_body.len() as u64;

            // Compute packet MD5: hash(recovery_set_id + packet_type + body)
            let packet_hash: [u8; 16] = {
                let mut hasher = Md5::new();
                hasher.update(recovery_set_id);
                hasher.update(packet_type);
                hasher.update(&aligned_body);
                hasher.finalize().into()
            };

            // Write packet header
            output.write_all(b"PAR2\x00PKT")?; // Magic
            output.write_all(&packet_length.to_le_bytes())?; // Length
            output.write_all(&packet_hash)?; // MD5 hash
            output.write_all(&recovery_set_id)?; // Recovery set ID
            output.write_all(packet_type)?; // Packet type
            output.write_all(&aligned_body)?; // Body

            Ok(())
        };

    // Write Main packet
    let main_packet_type: [u8; 16] = [
        b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'M', b'a', b'i', b'n', 0, 0, 0, 0,
    ];
    write_packet(&mut output, &main_packet_type, &main_body)?;

    // Build File Descriptor packet body
    let mut file_desc_body = Vec::new();
    file_desc_body.extend_from_slice(&file_id);
    file_desc_body.extend_from_slice(&file_hash);
    file_desc_body.extend_from_slice(&hash_16k);
    file_desc_body.extend_from_slice(&file_size.to_le_bytes());
    file_desc_body.extend_from_slice(filename.as_bytes());

    // Write File Descriptor packet
    let file_desc_packet_type: [u8; 16] = [
        b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'F', b'i', b'l', b'e', b'D', b'e', b's', b'c',
    ];
    write_packet(&mut output, &file_desc_packet_type, &file_desc_body)?;

    // Note: We intentionally skip IFSC packets to test the fallback path
    // Note: We also skip recovery packets since we're just testing verification

    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_pattern_file() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("pattern.bin");

        create_pattern_file(&path, b"ABC", 10).unwrap();

        let content = fs::read(&path).unwrap();
        assert_eq!(content, b"ABCABCABCA");
    }

    #[test]
    fn test_corrupt_file() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("corrupt.bin");

        create_pattern_file(&path, b"ORIGINAL", 16).unwrap();
        corrupt_file(&path, 4, b"XXXX").unwrap();

        let content = fs::read(&path).unwrap();
        assert_eq!(&content[0..4], b"ORIG");
        assert_eq!(&content[4..8], b"XXXX");
    }

    #[test]
    fn test_compute_file_hash_deterministic() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("hash.bin");

        create_pattern_file(&path, b"HASH_TEST", 100).unwrap();

        let hash1 = compute_file_hash(&path);
        let hash2 = compute_file_hash(&path);

        assert_eq!(hash1, hash2, "Hash should be deterministic");
    }
}
