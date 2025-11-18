// PAR2 file format writer
// Creates PAR2 packets for file protection and recovery

use crate::error::Result;
use crate::parser::{FileHash, FileInfo, RecoveryBlock, SliceChecksum};
use std::io::Write;

/// PAR2 magic packet header (ASCII "PAR2\0PKT")
const PAR2_MAGIC: [u8; 8] = [b'P', b'A', b'R', b'2', 0, b'P', b'K', b'T'];

/// PAR2 packet type identifiers (16 bytes each)
pub mod packet_types {
    // Main packet: "PAR 2.0\0Main\0\0\0\0"
    pub const MAIN: [u8; 16] = [
        b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'M', b'a', b'i', b'n', 0, 0, 0, 0,
    ];

    // File description: "PAR 2.0\0FileDesc"
    pub const FILE_DESC: [u8; 16] = [
        b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'F', b'i', b'l', b'e', b'D', b'e', b's', b'c',
    ];

    // Input file slice checksum: "PAR 2.0\0IFSC\0\0\0\0"
    pub const IFSC: [u8; 16] = [
        b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'I', b'F', b'S', b'C', 0, 0, 0, 0,
    ];

    // Recovery slice: "PAR 2.0\0RecvSlic"
    pub const RECV_SLICE: [u8; 16] = [
        b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'R', b'e', b'c', b'v', b'S', b'l', b'i', b'c',
    ];

    // Creator: "PAR 2.0\0Creator\0"
    pub const CREATOR: [u8; 16] = [
        b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'C', b'r', b'e', b'a', b't', b'o', b'r', 0,
    ];
}

/// Compute File ID as specified in PAR2 spec
/// File ID = MD5(Hash16k + Length + Filename)
pub fn compute_file_id(hash_16k: &[u8; 16], length: u64, filename: &str) -> FileHash {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(hash_16k);
    hasher.update(length.to_le_bytes());
    hasher.update(filename.as_bytes());
    hasher.finalize().into()
}

/// Compute Recovery Set ID from Main packet body
/// Recovery Set ID = MD5(Main packet body)
pub fn compute_recovery_set_id(main_packet_body: &[u8]) -> FileHash {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(main_packet_body);
    hasher.finalize().into()
}

/// Pad data to 4-byte alignment with zeros
fn pad_to_alignment(data: &[u8]) -> Vec<u8> {
    let mut padded = data.to_vec();
    while padded.len() % 4 != 0 {
        padded.push(0);
    }
    padded
}

/// Write a complete PAR2 packet with header and body
///
/// Packet structure:
/// - Magic (8 bytes): "PAR2\0PKT"
/// - Length (8 bytes): total packet length including header
/// - MD5 Hash (16 bytes): MD5 of (recovery_set_id + packet_type + body)
/// - Recovery Set ID (16 bytes)
/// - Packet Type (16 bytes)
/// - Body (variable, must be multiple of 4 bytes)
pub fn write_packet(
    writer: &mut impl Write,
    packet_type: &[u8; 16],
    recovery_set_id: &FileHash,
    body: &[u8],
) -> Result<()> {
    // Ensure body is aligned to 4 bytes
    let aligned_body = pad_to_alignment(body);

    // Calculate total packet length (header 64 bytes + body)
    let packet_length = 64u64 + aligned_body.len() as u64;

    // Compute MD5 hash of (recovery_set_id + packet_type + body)
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(recovery_set_id);
    hasher.update(packet_type);
    hasher.update(&aligned_body);
    let packet_hash: [u8; 16] = hasher.finalize().into();

    // Write packet header
    writer.write_all(&PAR2_MAGIC)?; // Magic (8 bytes)
    writer.write_all(&packet_length.to_le_bytes())?; // Length (8 bytes)
    writer.write_all(&packet_hash)?; // MD5 hash (16 bytes)
    writer.write_all(recovery_set_id)?; // Recovery Set ID (16 bytes)
    writer.write_all(packet_type)?; // Packet type (16 bytes)

    // Write packet body
    writer.write_all(&aligned_body)?;

    Ok(())
}

/// Write a Creator packet
/// Body: ASCII text identifying the client (e.g., "par2-rs v0.1.0 https://github.com/...")
pub fn write_creator_packet(
    writer: &mut impl Write,
    recovery_set_id: &FileHash,
    creator_text: &str,
) -> Result<()> {
    let body = creator_text.as_bytes();
    write_packet(writer, &packet_types::CREATOR, recovery_set_id, body)
}

/// Write a Main packet
/// Body:
/// - Block size (8 bytes, u64)
/// - File count (4 bytes, u32)
/// - File IDs array (16 bytes × file_count, sorted numerically)
pub fn write_main_packet(
    writer: &mut impl Write,
    recovery_set_id: &FileHash,
    block_size: u64,
    file_ids: &[FileHash],
) -> Result<()> {
    let mut body = Vec::new();

    // Write block size
    body.extend_from_slice(&block_size.to_le_bytes());

    // Write file count
    let file_count = file_ids.len() as u32;
    body.extend_from_slice(&file_count.to_le_bytes());

    // Write sorted file IDs
    let mut sorted_ids = file_ids.to_vec();
    sorted_ids.sort_unstable();
    for file_id in &sorted_ids {
        body.extend_from_slice(file_id);
    }

    write_packet(writer, &packet_types::MAIN, recovery_set_id, &body)
}

/// Write a File Description packet
/// Body:
/// - File ID (16 bytes)
/// - Full file MD5 (16 bytes)
/// - First 16KB MD5 (16 bytes)
/// - File length (8 bytes, u64)
/// - Filename (ASCII, null-terminated, padded to 4-byte alignment)
pub fn write_file_desc_packet(
    writer: &mut impl Write,
    recovery_set_id: &FileHash,
    file_info: &FileInfo,
) -> Result<()> {
    let mut body = Vec::new();

    // File ID
    body.extend_from_slice(&file_info.file_id);

    // Full file hash
    body.extend_from_slice(&file_info.hash);

    // 16KB hash
    body.extend_from_slice(&file_info.hash_16k);

    // File length
    body.extend_from_slice(&file_info.length.to_le_bytes());

    // Filename (null-terminated)
    body.extend_from_slice(file_info.name.as_bytes());
    body.push(0); // null terminator

    write_packet(writer, &packet_types::FILE_DESC, recovery_set_id, &body)
}

/// Write an Input File Slice Checksum (IFSC) packet
/// Body:
/// - File ID (16 bytes)
/// - Array of slice checksums (20 bytes each: 16-byte MD5 + 4-byte CRC32)
pub fn write_ifsc_packet(
    writer: &mut impl Write,
    recovery_set_id: &FileHash,
    file_id: &FileHash,
    checksums: &[SliceChecksum],
) -> Result<()> {
    let mut body = Vec::new();

    // File ID
    body.extend_from_slice(file_id);

    // Slice checksums
    for checksum in checksums {
        body.extend_from_slice(&checksum.md5);
        body.extend_from_slice(&checksum.crc32.to_le_bytes());
    }

    write_packet(writer, &packet_types::IFSC, recovery_set_id, &body)
}

/// Write a Recovery Slice packet
/// Body:
/// - Exponent (4 bytes, u32)
/// - Recovery data (block_size bytes)
pub fn write_recovery_slice_packet(
    writer: &mut impl Write,
    recovery_set_id: &FileHash,
    recovery_block: &RecoveryBlock,
) -> Result<()> {
    let mut body = Vec::new();

    // Exponent
    body.extend_from_slice(&recovery_block.exponent.to_le_bytes());

    // Recovery data
    let data = recovery_block
        .read_all()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    body.extend_from_slice(&data);

    write_packet(writer, &packet_types::RECV_SLICE, recovery_set_id, &body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_file_id() {
        // Test vector: create a file ID and verify it's deterministic
        let hash_16k = [1u8; 16];
        let length = 12345u64;
        let filename = "test.txt";

        let file_id1 = compute_file_id(&hash_16k, length, filename);
        let file_id2 = compute_file_id(&hash_16k, length, filename);

        assert_eq!(file_id1, file_id2, "File ID should be deterministic");
        assert_ne!(file_id1, [0u8; 16], "File ID should not be all zeros");

        // Different inputs should produce different IDs
        let file_id3 = compute_file_id(&hash_16k, length, "different.txt");
        assert_ne!(
            file_id1, file_id3,
            "Different filenames should produce different IDs"
        );
    }

    #[test]
    fn test_compute_recovery_set_id() {
        let body = b"test body data";

        let id1 = compute_recovery_set_id(body);
        let id2 = compute_recovery_set_id(body);

        assert_eq!(id1, id2, "Recovery set ID should be deterministic");
        assert_ne!(id1, [0u8; 16], "Recovery set ID should not be all zeros");

        // Different body should produce different ID
        let id3 = compute_recovery_set_id(b"different body");
        assert_ne!(id1, id3, "Different bodies should produce different IDs");
    }

    #[test]
    fn test_pad_to_alignment() {
        assert_eq!(pad_to_alignment(&[1, 2, 3, 4]), vec![1, 2, 3, 4]);
        assert_eq!(pad_to_alignment(&[1, 2, 3]), vec![1, 2, 3, 0]);
        assert_eq!(pad_to_alignment(&[1, 2]), vec![1, 2, 0, 0]);
        assert_eq!(pad_to_alignment(&[1]), vec![1, 0, 0, 0]);
        assert_eq!(pad_to_alignment(&[]), vec![]);
    }

    #[test]
    fn test_write_creator_packet() {
        let mut output = Vec::new();
        let recovery_set_id = [42u8; 16];
        let creator_text = "par2-rs v0.1.0 https://github.com/zephleggett/par2-rs";

        write_creator_packet(&mut output, &recovery_set_id, creator_text).unwrap();

        // Verify magic
        assert_eq!(&output[0..8], &PAR2_MAGIC);

        // Verify packet length
        let packet_length = u64::from_le_bytes([
            output[8], output[9], output[10], output[11], output[12], output[13], output[14],
            output[15],
        ]);
        assert_eq!(
            packet_length as usize,
            output.len(),
            "Packet length should match actual length"
        );

        // Verify length is multiple of 4
        assert_eq!(packet_length % 4, 0, "Packet length must be multiple of 4");

        // Verify recovery set ID is present
        let mut stored_id = [0u8; 16];
        stored_id.copy_from_slice(&output[32..48]);
        assert_eq!(stored_id, recovery_set_id);

        // Verify packet type
        let mut packet_type = [0u8; 16];
        packet_type.copy_from_slice(&output[48..64]);
        assert_eq!(packet_type, packet_types::CREATOR);

        // Verify body contains creator text
        let body_start = 64;
        let body = &output[body_start..];
        assert!(body.starts_with(creator_text.as_bytes()));
    }

    #[test]
    fn test_write_main_packet() {
        let mut output = Vec::new();
        let recovery_set_id = [42u8; 16];
        let block_size = 2048u64;
        let file_ids = vec![[1u8; 16], [2u8; 16], [3u8; 16]];

        write_main_packet(&mut output, &recovery_set_id, block_size, &file_ids).unwrap();

        // Verify magic
        assert_eq!(&output[0..8], &PAR2_MAGIC);

        // Verify packet type
        let mut packet_type = [0u8; 16];
        packet_type.copy_from_slice(&output[48..64]);
        assert_eq!(packet_type, packet_types::MAIN);

        // Verify body content
        let body_start = 64;

        // Block size at offset 0
        let stored_block_size = u64::from_le_bytes([
            output[body_start],
            output[body_start + 1],
            output[body_start + 2],
            output[body_start + 3],
            output[body_start + 4],
            output[body_start + 5],
            output[body_start + 6],
            output[body_start + 7],
        ]);
        assert_eq!(stored_block_size, block_size);

        // File count at offset 8
        let stored_file_count = u32::from_le_bytes([
            output[body_start + 8],
            output[body_start + 9],
            output[body_start + 10],
            output[body_start + 11],
        ]);
        assert_eq!(stored_file_count, 3);

        // File IDs should be present and sorted
        let mut stored_id1 = [0u8; 16];
        stored_id1.copy_from_slice(&output[body_start + 12..body_start + 28]);
        let mut stored_id2 = [0u8; 16];
        stored_id2.copy_from_slice(&output[body_start + 28..body_start + 44]);
        let mut stored_id3 = [0u8; 16];
        stored_id3.copy_from_slice(&output[body_start + 44..body_start + 60]);

        // Verify they're sorted
        assert!(stored_id1 <= stored_id2);
        assert!(stored_id2 <= stored_id3);
    }

    #[test]
    fn test_write_file_desc_packet() {
        let mut output = Vec::new();
        let recovery_set_id = [42u8; 16];
        let file_info = FileInfo {
            file_id: [10u8; 16],
            hash: [20u8; 16],
            hash_16k: [30u8; 16],
            length: 54321,
            name: "testfile.bin".to_string(),
        };

        write_file_desc_packet(&mut output, &recovery_set_id, &file_info).unwrap();

        // Verify magic and packet type
        assert_eq!(&output[0..8], &PAR2_MAGIC);
        let mut packet_type = [0u8; 16];
        packet_type.copy_from_slice(&output[48..64]);
        assert_eq!(packet_type, packet_types::FILE_DESC);

        // Verify body
        let body_start = 64;

        // File ID
        let mut stored_file_id = [0u8; 16];
        stored_file_id.copy_from_slice(&output[body_start..body_start + 16]);
        assert_eq!(stored_file_id, file_info.file_id);

        // Full hash
        let mut stored_hash = [0u8; 16];
        stored_hash.copy_from_slice(&output[body_start + 16..body_start + 32]);
        assert_eq!(stored_hash, file_info.hash);

        // 16K hash
        let mut stored_hash_16k = [0u8; 16];
        stored_hash_16k.copy_from_slice(&output[body_start + 32..body_start + 48]);
        assert_eq!(stored_hash_16k, file_info.hash_16k);

        // Length
        let stored_length = u64::from_le_bytes([
            output[body_start + 48],
            output[body_start + 49],
            output[body_start + 50],
            output[body_start + 51],
            output[body_start + 52],
            output[body_start + 53],
            output[body_start + 54],
            output[body_start + 55],
        ]);
        assert_eq!(stored_length, file_info.length);

        // Filename (null-terminated)
        let filename_bytes = &output[body_start + 56..];
        assert!(filename_bytes.starts_with(file_info.name.as_bytes()));
        // Find null terminator
        let null_pos = filename_bytes.iter().position(|&b| b == 0).unwrap();
        assert_eq!(&filename_bytes[..null_pos], file_info.name.as_bytes());
    }

    #[test]
    fn test_write_recovery_slice_packet() {
        let mut output = Vec::new();
        let recovery_set_id = [42u8; 16];
        let test_data = vec![0x11, 0x22, 0x33, 0x44];
        let recovery_block = RecoveryBlock::from_memory(5, test_data.clone());

        write_recovery_slice_packet(&mut output, &recovery_set_id, &recovery_block).unwrap();

        // Verify magic and packet type
        assert_eq!(&output[0..8], &PAR2_MAGIC);
        let mut packet_type = [0u8; 16];
        packet_type.copy_from_slice(&output[48..64]);
        assert_eq!(packet_type, packet_types::RECV_SLICE);

        // Verify body
        let body_start = 64;

        // Exponent
        let stored_exponent = u32::from_le_bytes([
            output[body_start],
            output[body_start + 1],
            output[body_start + 2],
            output[body_start + 3],
        ]);
        assert_eq!(stored_exponent, recovery_block.exponent);

        // Recovery data
        if let Some(ref data) = recovery_block.data {
            assert_eq!(&output[body_start + 4..body_start + 8], &data[..]);
        }
    }
}
