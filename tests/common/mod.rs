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
