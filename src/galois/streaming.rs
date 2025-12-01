//! Streaming Reed-Solomon reconstruction with cache optimization
//!
//! This module implements a three-level hierarchy for efficient reconstruction:
//!
//! 1. **Streaming Level** (I/O optimization)
//!    - Reads large chunks (1-10MB) to minimize disk I/O syscalls
//!    - Buffers data in memory for batch processing
//!
//! 2. **Region Level** (Cache optimization)
//!    - Subdivides chunks into 128KB regions that fit in L2 cache
//!    - Processes multiple sources together for better cache reuse
//!
//! 3. **SIMD Level** (Instruction optimization)
//!    - Uses platform-specific SIMD (AVX2, PMULL, etc.)
//!    - Automatically selected via strategy registry
//!
//! # Example
//!
//! ```ignore
//! let processor = StreamingProcessor::new(destinations, sources, coefficients);
//!
//! // Process entire block in streaming fashion
//! processor.process_block(block_size, |offset, size| {
//!     // Read chunk from disk
//!     read_chunk(offset, size)
//! })?;
//! ```

use super::region::{gf_muladd_region, gf_muladd_region_shuffle2x, region_size_bytes};
use super::simd;
use crate::error::Result;

/// Configuration for streaming reconstruction
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of I/O chunks to read at once (default: 4MB)
    pub chunk_size: usize,

    /// Size of cache-optimized regions (default: 128KB)
    pub region_size: usize,

    /// Number of sources to batch together (default: 8)
    pub source_batch_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 4 * 1024 * 1024,      // 4MB for I/O
            region_size: region_size_bytes(), // 128KB for cache
            source_batch_size: 8,             // Tuned for register pressure
        }
    }
}

impl StreamingConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.chunk_size < self.region_size {
            return Err(crate::error::Par2Error::InvalidFormat(format!(
                "chunk_size ({}) must be >= region_size ({})",
                self.chunk_size, self.region_size
            )));
        }

        if self.region_size % 2 != 0 {
            return Err(crate::error::Par2Error::InvalidFormat(
                "region_size must be even (u16 alignment)".to_string(),
            ));
        }

        Ok(())
    }
}

/// Streaming Reed-Solomon processor
///
/// Handles the three-level hierarchy: Streaming → Region → SIMD
pub struct StreamingProcessor {
    config: StreamingConfig,
    /// Whether to use shuffle2x format for better performance
    use_shuffle2x: bool,
}

impl StreamingProcessor {
    /// Create a new streaming processor with default configuration
    pub fn new() -> Self {
        // Check if current SIMD strategy supports shuffle2x
        let use_shuffle2x = simd::get_registry()
            .get_selected()
            .map(|s| s.supports_shuffle2x())
            .unwrap_or(false);

        Self {
            config: StreamingConfig::default(),
            use_shuffle2x,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StreamingConfig) -> Result<Self> {
        config.validate()?;

        let use_shuffle2x = simd::get_registry()
            .get_selected()
            .map(|s| s.supports_shuffle2x())
            .unwrap_or(false);

        Ok(Self {
            config,
            use_shuffle2x,
        })
    }

    /// Process a complete block using streaming I/O and cache-optimized regions
    ///
    /// # Arguments
    ///
    /// * `destinations` - Output blocks to reconstruct
    /// * `sources` - Available source blocks (will be read via callback)
    /// * `coefficients` - Reed-Solomon coefficient matrix
    /// * `block_size` - Total size of block in bytes
    /// * `read_source` - Callback to read a source block chunk
    ///   - Parameters: (source_index, chunk_offset, chunk_size)
    ///   - Returns: bytes for that chunk
    ///
    /// # Three-Level Processing
    ///
    /// ```text
    /// Block (100MB)
    ///  ├─ Chunk 0-4MB    ← Level 1: Read from disk
    ///  │  ├─ Region 0-128KB    ← Level 2: Cache optimization
    ///  │  │  └─ SIMD vectors   ← Level 3: AVX2/PMULL
    ///  │  ├─ Region 128-256KB
    ///  │  └─ ...
    ///  ├─ Chunk 4-8MB
    ///  └─ ...
    /// ```
    pub fn process_block_streaming<F>(
        &self,
        destinations: &mut [&mut [u16]],
        source_indices: &[usize],
        coefficients: &[&[u16]],
        block_size: usize,
        mut read_source: F,
    ) -> Result<()>
    where
        F: FnMut(usize, usize, usize) -> Result<Vec<u8>>,
    {
        let num_dsts = destinations.len();
        let num_srcs = source_indices.len();

        if num_dsts == 0 || num_srcs == 0 {
            return Ok(());
        }

        // Get strategy for shuffle2x conversions
        let strategy = if self.use_shuffle2x {
            simd::get_registry().get_selected()
        } else {
            None
        };

        // Convert destinations to shuffle2x format at the start
        if let Some(strat) = strategy {
            for dst in destinations.iter_mut() {
                unsafe {
                    strat.prepare_shuffle2x(dst);
                }
            }
        }

        // Level 1: Stream through block in large I/O chunks
        let mut chunk_offset = 0;
        while chunk_offset < block_size {
            let chunk_size = (block_size - chunk_offset).min(self.config.chunk_size);

            // Process this chunk in batches for memory efficiency
            self.process_chunk_batched(
                destinations,
                source_indices,
                coefficients,
                chunk_offset,
                chunk_size,
                &mut read_source,
            )?;

            chunk_offset += chunk_size;
        }

        // Convert destinations back to interleaved format at the end
        if let Some(strat) = strategy {
            for dst in destinations.iter_mut() {
                unsafe {
                    strat.finish_shuffle2x(dst);
                }
            }
        }

        Ok(())
    }

    /// Process a single chunk in source batches
    fn process_chunk_batched<F>(
        &self,
        destinations: &mut [&mut [u16]],
        source_indices: &[usize],
        coefficients: &[&[u16]],
        chunk_offset: usize,
        chunk_size: usize,
        read_source: &mut F,
    ) -> Result<()>
    where
        F: FnMut(usize, usize, usize) -> Result<Vec<u8>>,
    {
        let chunk_u16s = chunk_size / 2;
        let num_srcs = source_indices.len();

        // Get strategy for shuffle2x conversion
        let strategy = if self.use_shuffle2x {
            simd::get_registry().get_selected()
        } else {
            None
        };

        // Pre-allocate buffer pool for source chunks (reused across batches)
        // This reduces allocation pressure during processing
        let batch_capacity = self.config.source_batch_size;
        let mut source_buffers: Vec<Vec<u16>> = (0..batch_capacity)
            .map(|_| vec![0u16; chunk_u16s])
            .collect();

        // Process sources in batches to control memory usage
        let mut batch_start = 0;
        while batch_start < num_srcs {
            let batch_end = (batch_start + self.config.source_batch_size).min(num_srcs);
            let batch_len = batch_end - batch_start;

            // Read this batch of sources into pre-allocated buffers
            for (i, &src_idx) in source_indices[batch_start..batch_end].iter().enumerate() {
                let bytes = read_source(src_idx, chunk_offset, chunk_size)?;
                crate::galois::bytes_to_u16_simd(&bytes, &mut source_buffers[i]);

                // Convert to shuffle2x format if supported
                if let Some(strat) = strategy {
                    unsafe {
                        strat.prepare_shuffle2x(&mut source_buffers[i]);
                    }
                }
            }

            // Build source slice references from pre-allocated buffers
            let sources: Vec<&[u16]> = source_buffers[..batch_len]
                .iter()
                .map(|v| v.as_slice())
                .collect();

            // Build coefficient slices for this batch
            let batch_coeffs: Vec<&[u16]> = coefficients
                .iter()
                .map(|row| &row[batch_start..batch_end])
                .collect();

            // Level 2: Subdivide chunk into cache-optimized regions
            self.process_chunk_regions(
                destinations,
                &sources,
                &batch_coeffs,
                chunk_offset,
                chunk_u16s,
            );

            batch_start = batch_end;
        }

        Ok(())
    }

    /// Process a chunk by subdividing into cache-optimized regions
    ///
    /// This is Level 2 of the hierarchy - ensures working set fits in L2 cache.
    fn process_chunk_regions(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        chunk_offset: usize,
        chunk_u16s: usize,
    ) {
        let region_u16s = self.config.region_size / 2;

        let mut region_offset = 0;
        while region_offset < chunk_u16s {
            let region_size = (chunk_u16s - region_offset).min(region_u16s);

            // Level 3: SIMD processing
            // Use shuffle2x region if data is in shuffle2x format
            if self.use_shuffle2x {
                gf_muladd_region_shuffle2x(
                    destinations,
                    sources,
                    coefficients,
                    chunk_offset + region_offset,
                    region_size,
                );
            } else {
                gf_muladd_region(
                    destinations,
                    sources,
                    coefficients,
                    chunk_offset + region_offset,
                    region_size,
                );
            }

            region_offset += region_size;
        }
    }
}

impl Default for StreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::galois::init_tables;

    #[test]
    fn test_config_validation() {
        let mut config = StreamingConfig::default();
        assert!(config.validate().is_ok());

        // Chunk must be >= region
        config.chunk_size = 64 * 1024;
        config.region_size = 128 * 1024;
        assert!(config.validate().is_err());

        // Region must be even
        config.region_size = 127 * 1024;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_streaming_basic() {
        init_tables();

        // Simple test: reconstruct 1 block from 2 sources
        let block_size = 1024; // 1KB

        let src1 = vec![1u16; 512];
        let src2 = vec![2u16; 512];

        let mut dst = vec![0u16; 512];
        let mut destinations = vec![dst.as_mut_slice()];

        let source_indices = vec![0, 1];
        let coeffs = vec![3u16, 5u16];
        let coefficients = vec![coeffs.as_slice()];

        let processor = StreamingProcessor::new();

        // Mock read function
        let sources = [src1, src2];
        let read_fn = |src_idx: usize, offset: usize, size: usize| {
            let start = offset / 2;
            let count = size / 2;
            let data = &sources[src_idx][start..start + count];

            let mut bytes = vec![0u8; size];
            for (i, &val) in data.iter().enumerate() {
                bytes[i * 2] = val as u8;
                bytes[i * 2 + 1] = (val >> 8) as u8;
            }
            Ok(bytes)
        };

        processor
            .process_block_streaming(
                &mut destinations,
                &source_indices,
                &coefficients,
                block_size,
                read_fn,
            )
            .unwrap();

        // Verify result: dst = src1 * 3 + src2 * 5 (in GF)
        let expected = crate::galois::gf_mul(1, 3) ^ crate::galois::gf_mul(2, 5);
        assert_eq!(dst[0], expected);
    }
}
