//! Parallel PAR2 file repair using Reed-Solomon reconstruction
//!
//! This module implements high-performance parallel repair by:
//!
//! 1. **Streaming chunks**: Processing blocks in 64KB-512KB chunks to limit memory
//! 2. **Precomputed transform**: Computing Gaussian elimination once, reusing for all chunks
//! 3. **Adaptive parallelism**: Scaling parallel chunks based on CPU cores and available RAM
//! 4. **Platform-optimized I/O**: Using `pread`/`read_at` on Unix for concurrent file access
//!
//! # Performance Tuning
//!
//! Environment variables for tuning (usually not needed):
//! - `PAR2_PARALLELISM`: Multiplier for parallel chunk count (default: auto)
//! - `PAR2_MAX_PARALLEL_CHUNKS`: Hard cap on parallel chunks
//! - `PAR2_MIN_CHUNK`: Minimum chunk size in bytes (default: 64KB)
//!
//! # Related Modules
//!
//! - [`crate::verify`]: Identifies damaged blocks before repair
//! - [`crate::par2_rs`]: Reed-Solomon codec implementation
//! - [`crate::galois`]: GF(2^16) arithmetic for reconstruction

use super::parser::{FileHash, Par2File};
use super::verify::VerificationResult;
use super::{Par2Operation, ProgressCallback};
use crate::error::{Par2Error, Result};
use crate::par2_rs::Par2ReedSolomon;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
#[cfg(not(unix))]
use std::io::Read;
use std::io::{Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Thread-local file handle cache for Windows
// Each rayon worker thread gets its own cache to avoid contention
#[cfg(not(unix))]
use std::cell::RefCell;

use std::cell::UnsafeCell;

/// Type alias for chunk write operations: (block_idx, chunk_offset, data)
type ChunkWrites = Vec<(usize, u64, Vec<u8>)>;

#[cfg(not(unix))]
thread_local! {
    static FILE_HANDLE_CACHE: RefCell<HashMap<PathBuf, File>> = RefCell::new(HashMap::new());
}

// Thread-local buffer pool to reduce allocations in hot paths
// Each thread gets its own pool to avoid contention
thread_local! {
    static BUFFER_POOL: UnsafeCell<Vec<Vec<u8>>> = const { UnsafeCell::new(Vec::new()) };
}

/// Get a buffer from the thread-local pool, or allocate a new one
/// The buffer is guaranteed to have at least `size` capacity
fn get_pooled_buffer(size: usize) -> Vec<u8> {
    BUFFER_POOL.with(|pool| {
        // SAFETY: We're the only thread accessing this pool
        let pool = unsafe { &mut *pool.get() };
        if let Some(mut buf) = pool.pop() {
            if buf.capacity() >= size {
                buf.clear();
                buf.resize(size, 0);
                return buf;
            }
            // Buffer too small, drop it and allocate new
        }
        vec![0u8; size]
    })
}

/// Process a single chunk of repair data
/// Extracted for use with static thread pool
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    chunk_idx: usize,
    chunk_size: usize,
    block_size: usize,
    total_blocks: usize,
    block_to_file: &HashMap<usize, (FileHash, usize)>,
    _file_paths: &HashMap<FileHash, PathBuf>,
    input_files: &HashMap<FileHash, File>,
    recovery_blocks: &[crate::parser::RecoveryBlock],
    recovery_file_handles: &HashMap<PathBuf, File>,
    rs: &Par2ReedSolomon,
    damaged_block_indices: &[usize],
    good_indices: &[usize],
    present_recovery_indices: &[usize],
    transform: &crate::par2_rs::ReconstructionTransform,
) -> Result<Vec<(usize, u64, Vec<u8>)>> {
    let chunk_offset = chunk_idx * chunk_size;
    let this_chunk_size = (block_size - chunk_offset).min(chunk_size);

    // Storage for reconstructed chunks
    let mut reconstructed_chunks: HashMap<usize, Vec<u8>> = HashMap::new();

    // Callback to read a block chunk on-demand
    let mut read_block = |block_idx: usize,
                          offset: usize,
                          size: usize|
     -> std::result::Result<Vec<u8>, String> {
        if block_idx < total_blocks {
            // Data block - read from file
            if let Some(&(file_id, block_in_file)) = block_to_file.get(&block_idx) {
                let block_byte_offset = (block_in_file * block_size) as u64;
                let chunk_byte_offset = block_byte_offset + offset as u64;
                let mut buffer = get_pooled_buffer(size);

                #[cfg(unix)]
                {
                    let file = input_files
                        .get(&file_id)
                        .ok_or_else(|| "Input file handle not found".to_string())?;
                    match FileExt::read_at(file, &mut buffer, chunk_byte_offset) {
                        Ok(bytes_read) => {
                            if bytes_read < size {
                                buffer[bytes_read..].fill(0);
                            }
                            Ok(buffer)
                        }
                        Err(err) => Err(format!("Read failed: {}", err)),
                    }
                }
                #[cfg(not(unix))]
                {
                    let file_path = file_paths
                        .get(&file_id)
                        .ok_or_else(|| "File ID not found in file_paths map".to_string())?;

                    FILE_HANDLE_CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();
                        if !cache.contains_key(file_path) {
                            match File::open(file_path) {
                                Ok(f) => {
                                    cache.insert(file_path.clone(), f);
                                }
                                Err(e) => return Err(format!("File open failed: {}", e)),
                            }
                        }
                        let file = cache.get_mut(file_path).unwrap();
                        file.seek(SeekFrom::Start(chunk_byte_offset))
                            .map_err(|err| format!("Seek failed: {}", err))?;
                        let bytes_read = file
                            .read(&mut buffer)
                            .map_err(|err| format!("Read failed: {}", err))?;
                        if bytes_read < size {
                            buffer[bytes_read..].fill(0);
                        }
                        Ok(buffer)
                    })
                }
            } else {
                Err(format!(
                    "Block {} not found in block_to_file map",
                    block_idx
                ))
            }
        } else {
            // Recovery block
            let rec_idx = block_idx - total_blocks;
            if rec_idx < recovery_blocks.len() {
                let rec_block = &recovery_blocks[rec_idx];

                if let Some(ref data) = rec_block.data {
                    let start = offset.min(data.len());
                    let end = (offset + size).min(data.len());
                    let mut chunk = data[start..end].to_vec();
                    chunk.resize(size, 0);
                    return Ok(chunk);
                }

                let read_offset = rec_block.data_offset + offset as u64;
                let bytes_to_read =
                    size.min((rec_block.data_length as usize).saturating_sub(offset));
                let mut buffer = get_pooled_buffer(size);

                #[cfg(unix)]
                {
                    if let Some(file) = recovery_file_handles.get(&rec_block.file_path) {
                        match FileExt::read_at(file, &mut buffer[..bytes_to_read], read_offset) {
                            Ok(bytes_read) => {
                                if bytes_read < bytes_to_read {
                                    buffer[bytes_read..].fill(0);
                                }
                            }
                            Err(err) => return Err(format!("Recovery read_at failed: {}", err)),
                        }
                        Ok(buffer)
                    } else {
                        let mut chunk = rec_block
                            .read_chunk(offset, size)
                            .map_err(|err| format!("Recovery block read failed: {}", err))?;
                        chunk.resize(size, 0);
                        Ok(chunk)
                    }
                }
                #[cfg(not(unix))]
                {
                    FILE_HANDLE_CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();
                        if !cache.contains_key(&rec_block.file_path) {
                            match File::open(&rec_block.file_path) {
                                Ok(f) => {
                                    cache.insert(rec_block.file_path.clone(), f);
                                }
                                Err(e) => return Err(format!("Recovery file open failed: {}", e)),
                            }
                        }
                        let file = cache.get_mut(&rec_block.file_path).unwrap();
                        file.seek(SeekFrom::Start(read_offset))
                            .map_err(|err| format!("Recovery seek failed: {}", err))?;
                        let bytes_read = file
                            .read(&mut buffer[..bytes_to_read])
                            .map_err(|err| format!("Recovery read failed: {}", err))?;
                        if bytes_read < bytes_to_read {
                            buffer[bytes_read..].fill(0);
                        }
                        Ok(buffer)
                    })
                }
            } else {
                Err(format!("Recovery block {} out of range", rec_idx))
            }
        }
    };

    // Callback to write reconstructed chunk
    let mut write_result = |block_idx: usize, data: Vec<u8>| -> std::result::Result<(), String> {
        reconstructed_chunks.insert(block_idx, data);
        Ok(())
    };

    // Call streaming reconstruction with precomputed transformation
    rs.reconstruct_streaming_chunk(
        damaged_block_indices,
        good_indices,
        present_recovery_indices,
        transform,
        chunk_offset,
        this_chunk_size,
        &mut read_block,
        &mut write_result,
    )
    .map_err(|e| Par2Error::RepairFailed(format!("Streaming RS reconstruction failed: {}", e)))?;

    // Collect reconstructed chunks to return (use remove to avoid clone)
    let mut writes = Vec::with_capacity(damaged_block_indices.len());
    for &damaged_idx in damaged_block_indices {
        if let Some(reconstructed_chunk) = reconstructed_chunks.remove(&damaged_idx) {
            writes.push((damaged_idx, chunk_offset as u64, reconstructed_chunk));
        } else {
            return Err(Par2Error::RepairFailed(format!(
                "BUG: Block {} was not reconstructed (chunk_offset={}, chunk_idx={})",
                damaged_idx, chunk_offset, chunk_idx
            )));
        }
    }

    Ok(writes)
}

/// Parallel repair with optimal CPU utilization
pub fn repair_files_parallel(
    par2_data: &Par2File,
    verification_result: &VerificationResult,
    base_path: &Path,
    progress_callback: Option<ProgressCallback>,
) -> Result<()> {
    let block_size = par2_data.block_size as usize;
    let file_map = &par2_data.files;

    // Build file_id -> (start_block, num_blocks) map
    let mut total_blocks = 0usize;
    let mut file_block_map: HashMap<FileHash, (usize, usize)> = HashMap::new();

    for file_info in &par2_data.files_in_order {
        let num_blocks = file_info.length.div_ceil(par2_data.block_size) as usize;
        file_block_map.insert(file_info.file_id, (total_blocks, num_blocks));
        total_blocks += num_blocks;
    }

    // Identify damaged blocks
    let mut damaged_block_indices: Vec<usize> = Vec::new();

    for file_id in &verification_result.missing_files {
        if let Some(&(start_block, num_blocks)) = file_block_map.get(file_id) {
            for block_idx in 0..num_blocks {
                damaged_block_indices.push(start_block + block_idx);
            }
        }
    }

    for file_id in &verification_result.damaged_files {
        if let Some(block_damage) = verification_result.block_damages.get(file_id) {
            if let Some(&(start_block, _num_blocks)) = file_block_map.get(file_id) {
                for &block_idx in &block_damage.damaged_block_indices {
                    damaged_block_indices.push(start_block + block_idx);
                }
            }
        } else if let Some(&(start_block, num_blocks)) = file_block_map.get(file_id) {
            for block_idx in 0..num_blocks {
                damaged_block_indices.push(start_block + block_idx);
            }
        }
    }

    if damaged_block_indices.is_empty() {
        return Ok(());
    }

    tracing::info!(
        damaged_blocks = damaged_block_indices.len(),
        total_blocks,
        "Starting parallel repair"
    );

    if let Some(ref cb) = progress_callback {
        cb(Par2Operation::Repairing, 0, total_blocks as u64);
    }

    // Sort recovery blocks by exponent
    let mut recovery_indices: Vec<usize> = (0..par2_data.recovery_blocks.len()).collect();
    recovery_indices.sort_by_key(|&i| par2_data.recovery_blocks[i].exponent);

    // Only use as many recovery blocks as we have damaged blocks
    let num_recovery_needed = damaged_block_indices.len().min(recovery_indices.len());
    let recovery_exponents: Vec<u32> = recovery_indices
        .iter()
        .take(num_recovery_needed)
        .map(|&i| par2_data.recovery_blocks[i].exponent)
        .collect();

    // Store metadata for recovery blocks we need (no data pre-loading!)
    // We'll read the data on-demand during reconstruction
    let recovery_blocks: Arc<Vec<crate::parser::RecoveryBlock>> = Arc::new(
        recovery_indices
            .iter()
            .take(num_recovery_needed)
            .map(|&idx| par2_data.recovery_blocks[idx].clone())
            .collect(),
    );

    // Pre-open file handles for recovery blocks (eliminates 20K+ file opens during repair)
    // Group recovery blocks by file path and open one handle per unique file
    let mut recovery_file_handles: HashMap<PathBuf, File> = HashMap::new();
    for rec_block in recovery_blocks.iter() {
        if !rec_block.file_path.as_os_str().is_empty()
            && !recovery_file_handles.contains_key(&rec_block.file_path)
        {
            match OpenOptions::new().read(true).open(&rec_block.file_path) {
                Ok(file) => {
                    recovery_file_handles.insert(rec_block.file_path.clone(), file);
                }
                Err(e) => {
                    tracing::warn!(
                        path = ?rec_block.file_path,
                        error = %e,
                        "Failed to pre-open recovery file, will fall back to per-read open"
                    );
                }
            }
        }
    }
    let recovery_file_handles = Arc::new(recovery_file_handles);
    tracing::debug!(
        num_handles = recovery_file_handles.len(),
        "Pre-opened recovery file handles"
    );

    // Create Reed-Solomon codec with full recovery count for proper matrix dimensions
    let total_recovery_count = par2_data.recovery_blocks.len();
    let rs = Arc::new(Par2ReedSolomon::new(total_blocks, total_recovery_count));

    // Chunk size for optimal balance between syscalls and parallelism
    // Use available_parallelism() instead of num_cpus::get() for accurate container detection
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Target chunk size to ensure good parallelism
    // Smaller chunks = more parallelism, but more overhead
    // Target ~4x CPU cores worth of chunks minimum for good load balancing
    let target_chunks = (num_cpus * 4).max(16);
    // Allow override via env, default to 64KB minimum to reduce syscall overhead
    let min_chunk_env = std::env::var("PAR2_MIN_CHUNK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64 * 1024);
    let mut chunk_size = (block_size / target_chunks).max(min_chunk_env);
    if chunk_size > block_size {
        chunk_size = block_size; // never exceed block size
    }
    let chunk_size = (chunk_size / 2) * 2; // Ensure even for u16 alignment
    let num_chunks = block_size.div_ceil(chunk_size);

    let num_damaged = damaged_block_indices.len();
    tracing::info!(
        chunk_size,
        chunk_size_kb = chunk_size / 1000,
        num_chunks,
        num_cpus,
        num_damaged,
        "Optimized chunk size for performance"
    );

    // Identify ALL good block indices
    // Reed-Solomon reconstruction needs ALL present blocks for correct math
    let damaged_set: std::collections::HashSet<usize> =
        damaged_block_indices.iter().copied().collect();
    let good_indices: Vec<usize> = (0..total_blocks)
        .filter(|idx| !damaged_set.contains(idx))
        .collect();

    tracing::debug!(
        good_blocks = good_indices.len(),
        damaged_blocks = damaged_block_indices.len(),
        recovery_blocks = num_recovery_needed,
        "Block allocation"
    );

    // Build reverse map: block_idx -> (file_id, block_in_file)
    let mut block_to_file: HashMap<usize, (FileHash, usize)> = HashMap::new();
    for (file_id, &(start_block, num_blocks)) in &file_block_map {
        for local_idx in 0..num_blocks {
            block_to_file.insert(start_block + local_idx, (*file_id, local_idx));
        }
    }

    // Build file paths map
    // Include both verified files AND damaged files (we need to read good blocks from damaged files!)
    let mut file_paths: HashMap<FileHash, PathBuf> = verification_result
        .verified_files
        .iter()
        .map(|(id, path)| (*id, path.clone()))
        .collect();

    // Add damaged files (they're on disk, just have some bad blocks)
    for file_id in &verification_result.damaged_files {
        if let Some(file_info) = file_map.get(file_id) {
            let file_path = base_path.join(&file_info.name);
            file_paths.insert(*file_id, file_path);
        }
    }

    // Open file handles once for writing (reduce file open/close overhead)
    let mut output_files: HashMap<FileHash, File> = HashMap::new();
    // Open file handles once for reading (avoid per-read open/seek/close)
    let mut input_files: HashMap<FileHash, File> = HashMap::new();

    // Missing files: create new, truncate is OK since file doesn't exist
    for file_id in &verification_result.missing_files {
        if let Some(file_info) = file_map.get(file_id) {
            let file_path = base_path.join(&file_info.name);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&file_path)?;

            // Pre-allocate file size to ensure proper file growth
            // This prevents issues with sparse files and ensures the file has the correct size
            file.set_len(file_info.length)?;

            output_files.insert(*file_id, file);
            // Also open a read-only handle for efficient read_at during reconstruction
            // (separate handle avoids cursor contention with writer)
            #[cfg(unix)]
            {
                let ro = OpenOptions::new().read(true).open(&file_path)?;
                input_files.insert(*file_id, ro);
            }
        }
    }

    // Damaged files: DON'T truncate! We need to read good blocks first
    for file_id in &verification_result.damaged_files {
        if let Some(file_info) = file_map.get(file_id) {
            let file_path = base_path.join(&file_info.name);
            let file = OpenOptions::new().read(true).write(true).open(&file_path)?;
            output_files.insert(*file_id, file);
            // Open separate read-only handle to enable read_at in parallel
            #[cfg(unix)]
            {
                let ro = OpenOptions::new().read(true).open(&file_path)?;
                input_files.insert(*file_id, ro);
            }
        }
    }

    // For verified (good) files, open read-only handles
    for (file_id, path) in &file_paths {
        if !input_files.contains_key(file_id) {
            if let Ok(ro) = OpenOptions::new().read(true).open(path) {
                input_files.insert(*file_id, ro);
            }
        }
    }

    // Log system info for debugging (no longer used for batch sizing)
    tracing::info!(
        num_cpus,
        num_chunks,
        chunk_size,
        damaged_blocks = damaged_block_indices.len(),
        recovery_blocks = recovery_blocks.len(),
        "Parallel repair configuration"
    );

    // Prepare indices for streaming API
    let present_recovery_indices: Vec<usize> = (0..recovery_blocks.len())
        .map(|i| total_blocks + i)
        .collect();

    // CRITICAL: Compute Gaussian elimination transformation ONCE for all chunks
    // This is the key optimization - avoid O(m³) work per chunk!
    let transform = rs
        .compute_reconstruction_transform(
            &damaged_block_indices,
            &good_indices,
            &present_recovery_indices,
            &recovery_exponents,
        )
        .map_err(|e| Par2Error::RepairFailed(format!("Failed to compute transformation: {}", e)))?;

    tracing::info!(
        "Gaussian elimination computed once (will be reused for all {} chunks)",
        num_chunks
    );

    // Hybrid approach: mega-batches with rayon::scope inside each
    // This enables I/O pipelining (write batch N while computing batch N+1)
    // while keeping synchronization overhead low (~10 sync points vs hundreds)
    let num_threads = rayon::current_num_threads();

    // Mega-batch size: large enough to amortize sync overhead, small enough for I/O pipelining
    // Target: ~10-20 batches total for good pipelining without excessive sync
    let mega_batch_size = (num_chunks / 10).max(num_threads * 4).min(num_chunks);

    tracing::debug!(
        num_chunks,
        num_threads,
        mega_batch_size,
        num_batches = num_chunks.div_ceil(mega_batch_size),
        "Mega-batch configuration for parallel repair"
    );

    let mut total_writes = 0;

    for batch_start in (0..num_chunks).step_by(mega_batch_size) {
        let batch_end = (batch_start + mega_batch_size).min(num_chunks);
        let batch_len = batch_end - batch_start;

        // Pre-allocate result storage for this batch only (bounded memory)
        let results: Vec<UnsafeCell<Option<Result<ChunkWrites>>>> =
            (0..batch_len).map(|_| UnsafeCell::new(None)).collect();

        // SAFETY: Each thread writes to disjoint indices within this batch
        struct ResultsWrapper<'a>(&'a [UnsafeCell<Option<Result<ChunkWrites>>>]);
        unsafe impl<'a> Sync for ResultsWrapper<'a> {}
        let results_wrapper = ResultsWrapper(&results);

        // Static distribution within mega-batch
        let chunks_per_thread = batch_len.div_ceil(num_threads);

        rayon::scope(|s| {
            for thread_id in 0..num_threads {
                let start = thread_id * chunks_per_thread;
                let end = (start + chunks_per_thread).min(batch_len);

                if start >= batch_len {
                    break;
                }

                // Borrow all the data we need (rayon::scope allows this)
                let block_to_file = &block_to_file;
                let file_paths = &file_paths;
                let input_files = &input_files;
                let recovery_blocks = &recovery_blocks;
                let recovery_file_handles = &recovery_file_handles;
                let rs = &rs;
                let damaged_block_indices = &damaged_block_indices;
                let good_indices = &good_indices;
                let present_recovery_indices = &present_recovery_indices;
                let transform = &transform;
                let results_wrapper = &results_wrapper;

                s.spawn(move |_| {
                    for local_idx in start..end {
                        let chunk_idx = batch_start + local_idx;
                        let result = process_chunk(
                            chunk_idx,
                            chunk_size,
                            block_size,
                            total_blocks,
                            block_to_file,
                            file_paths,
                            input_files,
                            recovery_blocks,
                            recovery_file_handles,
                            rs,
                            damaged_block_indices,
                            good_indices,
                            present_recovery_indices,
                            transform,
                        );
                        // SAFETY: Each thread writes to disjoint indices
                        unsafe {
                            *results_wrapper.0[local_idx].get() = Some(result);
                        }
                    }
                });
            }
        }); // Sync point per mega-batch (only ~10 total)

        // Write results from this batch immediately (enables I/O pipelining)
        for (local_idx, result_cell) in results.iter().enumerate() {
            let chunk_result = unsafe { (*result_cell.get()).take() }.ok_or_else(|| {
                Par2Error::RepairFailed(format!(
                    "Chunk {} was not processed",
                    batch_start + local_idx
                ))
            })?;

            for (block_idx, chunk_offset, data) in chunk_result? {
                if let Some(&(file_id, block_in_file)) = block_to_file.get(&block_idx) {
                    if let Some(file) = output_files.get_mut(&file_id) {
                        if let Some(file_info) = file_map.get(&file_id) {
                            let block_byte_offset = (block_in_file * block_size) as u64;
                            let write_offset = block_byte_offset + chunk_offset;

                            file.seek(SeekFrom::Start(write_offset))?;

                            let file_remaining = file_info.length.saturating_sub(write_offset);
                            let bytes_to_write = (data.len() as u64).min(file_remaining) as usize;

                            if bytes_to_write > 0 {
                                file.write_all(&data[..bytes_to_write])?;
                                total_writes += 1;
                            }
                        }
                    }
                }
            }
        }

        // Progress tracking per mega-batch
        tracing::debug!(
            batch_start,
            batch_end,
            total = num_chunks,
            progress_pct = (batch_end * 100) / num_chunks,
            total_writes,
            "Completed mega-batch"
        );
    }

    // Verify chunks were written (note: total_writes may be less than theoretical max due to partial last blocks)
    let theoretical_max_writes = num_chunks * damaged_block_indices.len();
    tracing::info!(
        total_writes,
        theoretical_max = theoretical_max_writes,
        num_chunks,
        damaged_blocks = damaged_block_indices.len(),
        "Chunk writes completed"
    );

    if total_writes < theoretical_max_writes {
        tracing::debug!(
            diff = theoretical_max_writes - total_writes,
            "Some chunks were skipped (likely due to partial last blocks)"
        );
    }

    // Flush all output files to disk
    // sync_all (fsync) is disabled by default for performance
    // OS will handle flushing to disk in the background
    for (_file_id, file) in output_files.iter_mut() {
        file.flush()?;
        // Note: sync_all disabled by default - OS will flush to disk asynchronously
    }

    tracing::info!(
        repaired_blocks = damaged_block_indices.len(),
        "Parallel repair complete"
    );

    if let Some(ref cb) = progress_callback {
        cb(
            Par2Operation::Repairing,
            damaged_block_indices.len() as u64,
            damaged_block_indices.len() as u64,
        );
    }

    Ok(())
}
