// PAR2 file format parser
// Based on specification: https://parchive.sourceforge.net/docs/specifications/parity-volume-spec/article-spec.html

use crate::error::{Par2Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// PAR2 magic packet header (ASCII "PAR2\0PKT")
const PAR2_MAGIC: [u8; 8] = [b'P', b'A', b'R', b'2', 0, b'P', b'K', b'T'];

/// PAR2 packet types (16-byte type identifiers)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    Main,
    FileDescription,
    InputFileSlice,
    RecoverySlice,
    Creator,
    Unknown,
}

impl PacketType {
    fn from_bytes(bytes: &[u8; 16]) -> Self {
        // Main packet: "PAR 2.0\0Main\0\0\0\0"
        const MAIN: [u8; 16] = [
            b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'M', b'a', b'i', b'n', 0, 0, 0, 0,
        ];
        // File description: "PAR 2.0\0FileDesc"
        const FILE_DESC: [u8; 16] = [
            b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'F', b'i', b'l', b'e', b'D', b'e', b's',
            b'c',
        ];
        // Input file slice: "PAR 2.0\0IFSC\0\0\0\0"
        const IFSC: [u8; 16] = [
            b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'I', b'F', b'S', b'C', 0, 0, 0, 0,
        ];
        // Recovery slice: "PAR 2.0\0RecvSlic"
        const RECV_SLICE: [u8; 16] = [
            b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'R', b'e', b'c', b'v', b'S', b'l', b'i',
            b'c',
        ];
        // Creator: "PAR 2.0\0Creator\0"
        const CREATOR: [u8; 16] = [
            b'P', b'A', b'R', b' ', b'2', b'.', b'0', 0, b'C', b'r', b'e', b'a', b't', b'o', b'r',
            0,
        ];

        match *bytes {
            MAIN => PacketType::Main,
            FILE_DESC => PacketType::FileDescription,
            IFSC => PacketType::InputFileSlice,
            RECV_SLICE => PacketType::RecoverySlice,
            CREATOR => PacketType::Creator,
            _ => PacketType::Unknown,
        }
    }
}

/// PAR2 file hash (16-byte MD5)
pub type FileHash = [u8; 16];

/// Information about a protected file
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub file_id: FileHash,
    pub hash: FileHash,
    pub hash_16k: FileHash,
    pub length: u64,
    pub name: String,
}

/// Recovery block information
///
/// Supports two modes:
/// 1. File-backed (for reading PAR2 files): data is read on-demand from file
/// 2. Memory-backed (for creating PAR2 files): data is stored in memory
#[derive(Debug, Clone)]
pub struct RecoveryBlock {
    pub exponent: u32,
    /// In-memory data (used during PAR2 creation)
    /// If Some, this takes precedence over file-backed reading
    pub data: Option<Vec<u8>>,
    /// File path containing this recovery block (used when reading PAR2 files)
    pub file_path: PathBuf,
    /// Byte offset in the file where recovery data starts (after packet header + exponent)
    pub data_offset: u64,
    /// Length of recovery data
    pub data_length: u64,
}

impl RecoveryBlock {
    /// Create a memory-backed recovery block (for PAR2 creation)
    pub fn from_memory(exponent: u32, data: Vec<u8>) -> Self {
        let data_length = data.len() as u64;
        Self {
            exponent,
            data: Some(data),
            file_path: PathBuf::new(),
            data_offset: 0,
            data_length,
        }
    }

    /// Read a chunk of recovery data on-demand
    ///
    /// # Arguments
    /// * `chunk_offset` - Offset within the recovery block (0 to data_length)
    /// * `chunk_size` - Number of bytes to read
    ///
    /// # Returns
    /// The requested chunk of data, or an error if the read fails
    pub fn read_chunk(&self, chunk_offset: usize, chunk_size: usize) -> Result<Vec<u8>> {
        // If we have in-memory data, use it
        if let Some(ref data) = self.data {
            let end = (chunk_offset + chunk_size).min(data.len());
            let start = chunk_offset.min(data.len());
            return Ok(data[start..end].to_vec());
        }

        // Otherwise read from file
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let mut file = File::open(&self.file_path)?;
        let read_offset = self.data_offset + chunk_offset as u64;
        let bytes_to_read =
            chunk_size.min((self.data_length as usize).saturating_sub(chunk_offset));

        file.seek(SeekFrom::Start(read_offset))?;
        let mut buffer = vec![0u8; bytes_to_read];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    /// Read the entire recovery block data
    pub fn read_all(&self) -> Result<Vec<u8>> {
        if let Some(ref data) = self.data {
            return Ok(data.clone());
        }
        self.read_chunk(0, self.data_length as usize)
    }
}

/// Per-slice checksum info
#[derive(Debug, Clone)]
pub struct SliceChecksum {
    pub md5: [u8; 16],
    pub crc32: u32,
}

/// Parsed PAR2 file data
#[derive(Debug)]
pub struct Par2File {
    pub block_size: u64,
    pub files: HashMap<FileHash, FileInfo>,
    pub files_in_order: Vec<FileInfo>, // Preserve order from PAR2 file
    pub recovery_blocks: Vec<RecoveryBlock>,
    pub slice_checksums: HashMap<FileHash, Vec<SliceChecksum>>, // per file
}

impl Par2File {
    /// Load and parse a PAR2 file
    pub fn load(
        path: &Path,
        _base_path: &Path,
        progress_callback: Option<super::ProgressCallback>,
    ) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();

        let mut recovery_set_id = [0u8; 16];
        let mut block_size = 0u64;
        let mut files = HashMap::new();
        let mut files_in_order = Vec::new();
        let mut recovery_blocks = Vec::new();
        let mut slice_checksums: HashMap<FileHash, Vec<SliceChecksum>> = HashMap::new();

        let mut position = 0u64;
        let _packet_count = 0u64;

        // Read packets until end of file
        while position < file_size {
            file.seek(SeekFrom::Start(position))?;

            // Update progress
            if let Some(ref cb) = progress_callback {
                let progress = (position * 100 / file_size).min(100);
                cb(super::Par2Operation::Loading, progress, 100);
            }

            // Read packet header (64 bytes minimum)
            let mut header = [0u8; 64];
            if file.read(&mut header)? < 64 {
                break; // End of file or truncated packet
            }

            // Verify magic
            if header[0..8] != PAR2_MAGIC {
                // Skip this packet and try next alignment
                position += 1;
                continue;
            }

            // Extract recovery_set_id from first valid packet (needed for volume discovery)
            if recovery_set_id == [0u8; 16] {
                recovery_set_id.copy_from_slice(&header[32..48]);
            }

            // Read packet length (little-endian u64 at offset 8)
            let packet_length = u64::from_le_bytes([
                header[8], header[9], header[10], header[11], header[12], header[13], header[14],
                header[15],
            ]);

            // Read packet type (16 bytes at offset 48)
            let mut packet_type_bytes = [0u8; 16];
            packet_type_bytes.copy_from_slice(&header[48..64]);
            let packet_type = PacketType::from_bytes(&packet_type_bytes);

            // Validate packet length
            if packet_length > 100_000_000 {
                // Skip suspiciously large packets (>100MB)
                position += packet_length;
                continue;
            }

            // Optimize memory: For RecoverySlice packets, only read exponent (4 bytes)
            // The actual recovery data will be read on-demand during repair
            if packet_type == PacketType::RecoverySlice {
                // Read only the exponent (first 4 bytes of body)
                let mut exponent_bytes = [0u8; 4];
                if file.read(&mut exponent_bytes)? == 4 {
                    let data_file_offset = position + 64;
                    if let Some(recovery) = parse_recovery_from_body_minimal(
                        &exponent_bytes,
                        block_size,
                        path,
                        data_file_offset,
                    ) {
                        recovery_blocks.push(recovery);
                    }
                }
                // Skip to next packet
                position += packet_length;
                if packet_length % 4 != 0 {
                    position += 4 - (packet_length % 4);
                }
                continue;
            }

            // For other packet types, read the full body
            let body_size = (packet_length.saturating_sub(64)) as usize;
            let mut packet_body = vec![0u8; body_size];
            if file.read(&mut packet_body)? < body_size {
                break; // Truncated packet
            }

            // Verify packet MD5 integrity (bytes 16-31 of header)
            // MD5 is computed over: recovery_set_id + packet_type + body
            use md5::{Digest, Md5};
            let stored_md5 = &header[16..32];
            let mut hasher = Md5::new();
            hasher.update(&header[32..48]); // recovery_set_id
            hasher.update(&header[48..64]); // packet_type
            hasher.update(&packet_body); // body
            let computed_md5: [u8; 16] = hasher.finalize().into();

            if computed_md5 != *stored_md5 {
                tracing::warn!(
                    "Packet MD5 mismatch at position {}, skipping (possibly corrupted)",
                    position
                );
                position += packet_length;
                if packet_length % 4 != 0 {
                    position += 4 - (packet_length % 4);
                }
                continue;
            }

            // Parse packet based on type (from in-memory buffer)
            // Note: RecoverySlice packets are handled above for memory efficiency
            match packet_type {
                PacketType::Main => {
                    if let Some(partial) = parse_main_packet_from_header(&header) {
                        if let Some(main_packet) = complete_main_packet(partial, &packet_body) {
                            block_size = main_packet.block_size;
                        }
                    }
                }
                PacketType::FileDescription => {
                    if let Some(file_info) = parse_file_desc_from_body(&packet_body) {
                        files.insert(file_info.file_id, file_info.clone());
                        files_in_order.push(file_info);
                    }
                }
                PacketType::RecoverySlice => {
                    // This should not happen - RecoverySlice is handled above
                    // but keep this for safety
                }
                PacketType::InputFileSlice => {
                    if let Ok((file_id, checksums)) = parse_ifsc_from_body(&packet_body) {
                        slice_checksums.insert(file_id, checksums);
                    }
                }
                _ => {
                    // Skip unknown packet types
                }
            }

            // Move to next packet (packets are aligned on even boundaries)
            position += packet_length;
            if packet_length % 4 != 0 {
                position += 4 - (packet_length % 4);
            }
        }

        if recovery_set_id == [0u8; 16] {
            return Err(Par2Error::RepairFailed(
                "Invalid PAR2 file: no main packet".to_string(),
            ));
        }

        // Load additional PAR2 volumes if they exist
        // IMPORTANT: Match by recovery_set_id, not filename (for obfuscated Usenet files!)
        tracing::debug!("Scanning for volume files in {:?}", path.parent());
        if let Some(parent) = path.parent() {
            // Look for ALL .par2 files with matching recovery_set_id
            if let Ok(entries) = std::fs::read_dir(parent) {
                let mut vol_count = 0;
                let mut par2_files_found = 0;
                for entry in entries.filter_map(|e| e.ok()) {
                    let vol_path = entry.path();

                    // Skip the main file we just parsed
                    if vol_path == path {
                        continue;
                    }

                    // Only check .par2 files (case-insensitive)
                    let is_par2 = vol_path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_lowercase() == "par2")
                        .unwrap_or(false);

                    if is_par2 {
                        par2_files_found += 1;
                        eprintln!("[VOLUME SCAN] Found PAR2 file: {}", vol_path.display());
                        tracing::debug!("Found PAR2 file: {}", vol_path.display());
                    } else {
                        continue;
                    }

                    // Check if this PAR2 file has the same recovery_set_id
                    match get_recovery_set_id(&vol_path) {
                        Ok(vol_set_id) => {
                            tracing::debug!(
                                "Recovery set ID match: {:?} vs {:?}",
                                vol_set_id,
                                recovery_set_id
                            );
                            if vol_set_id == recovery_set_id {
                                // Parse recovery packets from volume file
                                tracing::debug!(
                                    file = %vol_path.display(),
                                    "Loading recovery blocks from volume file"
                                );
                                match parse_volume_file(
                                    &vol_path,
                                    block_size,
                                    progress_callback.clone(),
                                ) {
                                    Ok(vol_recovery) => {
                                        tracing::debug!(
                                            count = vol_recovery.len(),
                                            "Loaded recovery blocks"
                                        );
                                        recovery_blocks.extend(vol_recovery);
                                        vol_count += 1;
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Failed to parse volume file {}: {}",
                                            vol_path.display(),
                                            e
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::debug!(
                                "Failed to get recovery_set_id from {}: {}",
                                vol_path.display(),
                                e
                            );
                        }
                    }
                }
                eprintln!("[VOLUME SCAN] Found {} PAR2 files, loaded {} volumes, total recovery blocks: {}",
                    par2_files_found, vol_count, recovery_blocks.len());
                tracing::info!(
                    par2_files_found = par2_files_found,
                    volumes = vol_count,
                    total_recovery_blocks = recovery_blocks.len(),
                    "Loaded volume files"
                );
            }
        }

        Ok(Self {
            block_size,
            files,
            files_in_order,
            recovery_blocks,
            slice_checksums,
        })
    }
}

/// Main packet data
struct MainPacket {
    block_size: u64,
}

/// Quickly read just the recovery_set_id from a PAR2 file (for volume discovery)
pub(crate) fn get_recovery_set_id(path: &Path) -> Result<FileHash> {
    let mut file = File::open(path)?;

    // Read first packet header
    let mut header = [0u8; 64];
    file.read_exact(&mut header)?;

    // Check PAR2 magic
    if header[0..8] != PAR2_MAGIC {
        return Err(Par2Error::RepairFailed("Invalid PAR2 file".to_string()));
    }

    // Recovery set ID is at offset 32-48 in packet header
    let mut recovery_set_id = [0u8; 16];
    recovery_set_id.copy_from_slice(&header[32..48]);

    Ok(recovery_set_id)
}

/// Parse main packet from header + body buffer (no file seeks needed)
fn parse_main_packet_from_header(_header: &[u8; 64]) -> Option<MainPacket> {
    // Body parsing moved to caller since it needs the packet_body
    // Return a partial result that caller will complete
    Some(MainPacket {
        block_size: 0, // Will be filled from body
    })
}

/// Complete main packet parsing from body
fn complete_main_packet(mut packet: MainPacket, body: &[u8]) -> Option<MainPacket> {
    if body.len() < 8 {
        return None;
    }

    // Read block size (u64 at body offset 0)
    packet.block_size = u64::from_le_bytes([
        body[0], body[1], body[2], body[3], body[4], body[5], body[6], body[7],
    ]);

    Some(packet)
}

/// Parse file description from in-memory body
fn parse_file_desc_from_body(body: &[u8]) -> Option<FileInfo> {
    if body.len() < 56 {
        return None;
    }

    // Read file ID (16 bytes at offset 0)
    let mut file_id = [0u8; 16];
    file_id.copy_from_slice(&body[0..16]);

    // Read file hash (16 bytes at offset 16)
    let mut hash = [0u8; 16];
    hash.copy_from_slice(&body[16..32]);

    // Read 16k hash (16 bytes at offset 32)
    let mut hash_16k = [0u8; 16];
    hash_16k.copy_from_slice(&body[32..48]);

    // Read file length (u64 at offset 48)
    let length = u64::from_le_bytes([
        body[48], body[49], body[50], body[51], body[52], body[53], body[54], body[55],
    ]);

    // Read filename (null-terminated UTF-8 string starting at offset 56)
    let name_end = body[56..]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(body.len() - 56);
    let name = String::from_utf8_lossy(&body[56..56 + name_end]).to_string();

    Some(FileInfo {
        file_id,
        hash,
        hash_16k,
        length,
        name,
    })
}

/// Parse recovery slice metadata from just the exponent bytes (memory-efficient)
///
/// # Arguments
/// * `exponent_bytes` - The 4-byte exponent from the packet body
/// * `block_size` - Expected size of recovery data
/// * `file_path` - Path to the PAR2 file containing this packet
/// * `data_file_offset` - Absolute byte offset in the file where packet body starts
///
/// # Returns
/// RecoveryBlock with metadata (no data is copied into memory)
fn parse_recovery_from_body_minimal(
    exponent_bytes: &[u8; 4],
    block_size: u64,
    file_path: &Path,
    data_file_offset: u64,
) -> Option<RecoveryBlock> {
    // Read exponent (u32)
    let exponent = u32::from_le_bytes(*exponent_bytes);

    // Recovery data starts at offset 4 (after exponent) within the packet body
    // Actual file offset = data_file_offset + 4
    let data_offset = data_file_offset + 4;
    let data_length = block_size;

    Some(RecoveryBlock {
        exponent,
        data: None, // File-backed: no in-memory data
        file_path: file_path.to_path_buf(),
        data_offset,
        data_length,
    })
}

/// Parse IFSC packet from in-memory body
fn parse_ifsc_from_body(body: &[u8]) -> Result<(FileHash, Vec<SliceChecksum>)> {
    if body.len() < 16 {
        return Err(Par2Error::RepairFailed("IFSC packet too small".to_string()));
    }

    // First 16 bytes: File ID
    let mut file_id = [0u8; 16];
    file_id.copy_from_slice(&body[0..16]);

    // The rest is an array of (MD5 16 bytes, CRC32 4 bytes) per slice
    let remaining = body.len() - 16;
    let slice_entry_size = 20; // md5 (16) + crc32 (4)

    if remaining % slice_entry_size != 0 {
        // Not fatal; parse what we can
    }

    let count = remaining / slice_entry_size;
    let mut checksums = Vec::with_capacity(count);

    for i in 0..count {
        let offset = 16 + i * slice_entry_size;
        if offset + slice_entry_size > body.len() {
            break;
        }

        let mut md5 = [0u8; 16];
        md5.copy_from_slice(&body[offset..offset + 16]);

        let crc32 = u32::from_le_bytes([
            body[offset + 16],
            body[offset + 17],
            body[offset + 18],
            body[offset + 19],
        ]);

        checksums.push(SliceChecksum { md5, crc32 });
    }

    Ok((file_id, checksums))
}

fn parse_volume_file(
    path: &Path,
    block_size: u64,
    _progress_callback: Option<super::ProgressCallback>,
) -> Result<Vec<RecoveryBlock>> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut recovery_blocks = Vec::new();
    let mut position = 0u64;

    while position < file_size {
        file.seek(SeekFrom::Start(position))?;

        let mut header = [0u8; 64];
        if file.read(&mut header)? < 64 {
            break;
        }

        if header[0..8] != PAR2_MAGIC {
            position += 1;
            continue;
        }

        let packet_length = u64::from_le_bytes([
            header[8], header[9], header[10], header[11], header[12], header[13], header[14],
            header[15],
        ]);

        let mut packet_type_bytes = [0u8; 16];
        packet_type_bytes.copy_from_slice(&header[48..64]);
        let packet_type = PacketType::from_bytes(&packet_type_bytes);

        if packet_type == PacketType::RecoverySlice {
            tracing::debug!(
                "Found RecoverySlice packet, packet_length={}",
                packet_length
            );
            // Memory-efficient: Only read exponent (4 bytes), not the full recovery data
            if packet_length <= 100_000_000 {
                let mut exponent_bytes = [0u8; 4];
                if file.read(&mut exponent_bytes)? == 4 {
                    let data_file_offset = position + 64;
                    if let Some(recovery) = parse_recovery_from_body_minimal(
                        &exponent_bytes,
                        block_size,
                        path,
                        data_file_offset,
                    ) {
                        tracing::debug!("Parsed recovery block, exponent={}", recovery.exponent);
                        recovery_blocks.push(recovery);
                    }
                } else {
                    tracing::warn!("Failed to read exponent");
                }
            } else {
                tracing::warn!("Packet length too large: {}", packet_length);
            }
        }

        position += packet_length;
        if packet_length % 4 != 0 {
            position += 4 - (packet_length % 4);
        }
    }

    Ok(recovery_blocks)
}
