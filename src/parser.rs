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
#[derive(Debug, Clone)]
pub struct RecoveryBlock {
    pub exponent: u32,
    pub data: Vec<u8>,
}

/// Parsed PAR2 file data
#[derive(Debug)]
pub struct Par2File {
    pub recovery_set_id: FileHash,
    pub block_size: u64,
    pub file_count: u32,
    pub files: HashMap<FileHash, FileInfo>,
    pub files_in_order: Vec<FileInfo>, // Preserve order from PAR2 file
    pub recovery_blocks: Vec<RecoveryBlock>,
}

impl Par2File {
    /// Load and parse a PAR2 file
    pub fn load(
        path: &Path,
        base_path: &Path,
        progress_callback: Option<super::ProgressCallback>,
    ) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();

        let mut recovery_set_id = [0u8; 16];
        let mut block_size = 0u64;
        let mut file_count = 0u32;
        let mut files = HashMap::new();
        let mut files_in_order = Vec::new();
        let mut recovery_blocks = Vec::new();

        let mut position = 0u64;
        let mut packet_count = 0u64;

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

            // Read packet length (little-endian u64 at offset 8)
            let packet_length = u64::from_le_bytes([
                header[8], header[9], header[10], header[11], header[12], header[13], header[14],
                header[15],
            ]);

            // Read packet type (16 bytes at offset 48)
            let mut packet_type_bytes = [0u8; 16];
            packet_type_bytes.copy_from_slice(&header[48..64]);
            let packet_type = PacketType::from_bytes(&packet_type_bytes);

            // Parse packet based on type
            match packet_type {
                PacketType::Main => {
                    let main_packet = self::parse_main_packet(&mut file, position)?;
                    recovery_set_id = main_packet.recovery_set_id;
                    block_size = main_packet.block_size;
                    file_count = main_packet.file_count;
                }
                PacketType::FileDescription => {
                    let file_info = self::parse_file_desc_packet(&mut file, position)?;
                    files.insert(file_info.file_id, file_info.clone());
                    files_in_order.push(file_info);
                }
                PacketType::RecoverySlice => {
                    let recovery = self::parse_recovery_packet(&mut file, position, block_size)?;
                    recovery_blocks.push(recovery);
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

            packet_count += 1;
        }

        if recovery_set_id == [0u8; 16] {
            return Err(
                Par2Error::RepairFailed("Invalid PAR2 file: no main packet".to_string())
                    .into(),
            );
        }

        // Load additional PAR2 volumes if they exist
        // IMPORTANT: Match by recovery_set_id, not filename (for obfuscated Usenet files!)
        if let Some(parent) = path.parent() {
            // Look for ALL .par2 files with matching recovery_set_id
            if let Ok(entries) = std::fs::read_dir(parent) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let vol_path = entry.path();

                    // Skip the main file we just parsed
                    if vol_path == path {
                        continue;
                    }

                    // Only check .par2 files
                    if !vol_path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s == "par2")
                        .unwrap_or(false)
                    {
                        continue;
                    }

                    // Check if this PAR2 file has the same recovery_set_id
                    if let Ok(vol_set_id) = get_recovery_set_id(&vol_path) {
                        if vol_set_id == recovery_set_id {
                            // Parse recovery packets from volume file
                            if let Ok(vol_recovery) =
                                self::parse_volume_file(&vol_path, block_size, progress_callback.clone())
                            {
                                recovery_blocks.extend(vol_recovery);
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            recovery_set_id,
            block_size,
            file_count,
            files,
            files_in_order,
            recovery_blocks,
        })
    }
}

/// Main packet data
struct MainPacket {
    recovery_set_id: FileHash,
    block_size: u64,
    file_count: u32,
}

/// Quickly read just the recovery_set_id from a PAR2 file (for volume discovery)
fn get_recovery_set_id(path: &Path) -> Result<FileHash> {
    let mut file = File::open(path)?;

    // Read first packet header
    let mut header = [0u8; 64];
    file.read_exact(&mut header)?;

    // Check PAR2 magic
    if &header[0..8] != &PAR2_MAGIC {
        return Err(Par2Error::RepairFailed("Invalid PAR2 file".to_string()));
    }

    // Recovery set ID is at offset 32-48 in packet header
    let mut recovery_set_id = [0u8; 16];
    recovery_set_id.copy_from_slice(&header[32..48]);

    Ok(recovery_set_id)
}

fn parse_main_packet(file: &mut File, position: u64) -> Result<MainPacket> {
    // Main packet header is at position, body starts at position + 64
    // Recovery set ID is in the header at offset 32 (after magic + length + hash)
    file.seek(SeekFrom::Start(position + 32))?;
    let mut recovery_set_id = [0u8; 16];
    file.read_exact(&mut recovery_set_id)?;

    // Now read body (starts at position + 64)
    file.seek(SeekFrom::Start(position + 64))?;

    // Read block size (u64 at body offset 0)
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    let block_size = u64::from_le_bytes(buf);

    // Read file count (u32 at body offset 8, right after block_size)
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let file_count = u32::from_le_bytes(buf);

    Ok(MainPacket {
        recovery_set_id,
        block_size,
        file_count,
    })
}

fn parse_file_desc_packet(file: &mut File, position: u64) -> Result<FileInfo> {
    file.seek(SeekFrom::Start(position + 64))?;

    // Read file ID (16 bytes)
    let mut file_id = [0u8; 16];
    file.read_exact(&mut file_id)?;

    // Read file hash (16 bytes)
    let mut hash = [0u8; 16];
    file.read_exact(&mut hash)?;

    // Read 16k hash (16 bytes)
    let mut hash_16k = [0u8; 16];
    file.read_exact(&mut hash_16k)?;

    // Read file length (u64)
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    let length = u64::from_le_bytes(buf);

    // Read filename (null-terminated UTF-8 string)
    let mut name_bytes = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        if file.read(&mut byte)? == 0 {
            break;
        }
        if byte[0] == 0 {
            break;
        }
        name_bytes.push(byte[0]);
    }

    let name = String::from_utf8_lossy(&name_bytes).to_string();

    Ok(FileInfo {
        file_id,
        hash,
        hash_16k,
        length,
        name,
    })
}

fn parse_recovery_packet(file: &mut File, position: u64, block_size: u64) -> Result<RecoveryBlock> {
    file.seek(SeekFrom::Start(position + 64))?;

    // Read exponent (u32)
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let exponent = u32::from_le_bytes(buf);

    // Skip to recovery data (after 16-byte recovery set ID)
    file.seek(SeekFrom::Start(position + 64 + 4 + 16))?;

    // Read recovery block data
    let mut data = vec![0u8; block_size as usize];
    file.read_exact(&mut data)?;

    Ok(RecoveryBlock { exponent, data })
}

fn parse_volume_file(
    path: &Path,
    block_size: u64,
    progress_callback: Option<super::ProgressCallback>,
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
            if let Ok(recovery) = parse_recovery_packet(&mut file, position, block_size) {
                recovery_blocks.push(recovery);
            }
        }

        position += packet_length;
        if packet_length % 4 != 0 {
            position += 4 - (packet_length % 4);
        }
    }

    Ok(recovery_blocks)
}
