//! SIMD optimizations for Galois field operations
//!
//! This module provides a unified interface for platform-specific SIMD
//! implementations of GF(2^16) operations. The strategy pattern allows
//! runtime selection of the optimal implementation based on CPU features.

use std::fmt;

#[cfg(target_arch = "x86_64")]
pub mod x86;

/// Priority levels for SIMD strategy selection
/// Higher values indicate preferred strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Fallback = 0,
    Basic = 10,    // SSE2, NEON
    Enhanced = 20, // SSSE3, SSE4.1
    Advanced = 30, // AVX2, PMULL
    Optimal = 40,  // AVX512, GFNI
}

/// Trait for SIMD implementations of Galois field operations
pub trait GaloisSimdStrategy: Send + Sync {
    /// Name of this strategy for debugging and selection
    fn name(&self) -> &'static str;

    /// Check if this strategy is available on the current CPU
    fn is_available(&self) -> bool;

    /// Priority for strategy selection (higher = preferred)
    fn priority(&self) -> Priority;

    /// Multiply a slice by a scalar in GF(2^16)
    ///
    /// # Safety
    /// Implementations may use unsafe SIMD intrinsics
    unsafe fn mul_slice(&self, scalar: u16, data: &mut [u16]);

    /// Multiply-accumulate: dst[i] ^= src[i] * scalar
    ///
    /// # Safety
    /// Implementations may use unsafe SIMD intrinsics
    unsafe fn muladd(&self, dst: &mut [u16], src: &[u16], scalar: u16);

    /// Region-based multiply-accumulate for multiple destinations
    ///
    /// # Safety
    /// Implementations may use unsafe SIMD intrinsics
    unsafe fn muladd_region(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    );

    // === Shuffle2x format support for high-performance x86 ===
    //
    // Shuffle2x format stores data with all low bytes in one half and all high
    // bytes in the other half. This allows 4 table lookups per 256-bit vector
    // instead of 8, nearly doubling throughput on AVX2.
    //
    // Strategies that don't support shuffle2x use the default implementations
    // which fall back to the regular interleaved format operations.

    /// Returns true if this strategy supports native shuffle2x format operations
    fn supports_shuffle2x(&self) -> bool {
        false
    }

    /// Convert interleaved u16 data to shuffle2x format (in-place)
    ///
    /// Shuffle2x format: [lo_bytes... | hi_bytes...] within each 256-bit block
    ///
    /// # Safety
    /// Implementations may use unsafe SIMD intrinsics
    unsafe fn prepare_shuffle2x(&self, _data: &mut [u16]) {
        // Default: no-op for strategies that don't support shuffle2x
    }

    /// Convert shuffle2x format back to interleaved u16 data (in-place)
    ///
    /// # Safety
    /// Implementations may use unsafe SIMD intrinsics
    unsafe fn finish_shuffle2x(&self, _data: &mut [u16]) {
        // Default: no-op for strategies that don't support shuffle2x
    }

    /// Multiply-accumulate on shuffle2x format data (no format conversion)
    ///
    /// Both dst and src must be in shuffle2x format. This is the high-performance
    /// path that avoids per-operation format conversion overhead.
    ///
    /// # Safety
    /// - Data must be in shuffle2x format
    /// - Implementations may use unsafe SIMD intrinsics
    unsafe fn muladd_shuffle2x(&self, dst: &mut [u16], src: &[u16], scalar: u16) {
        // Default: convert, operate, convert back (for non-shuffle2x strategies)
        self.muladd(dst, src, scalar);
    }

    /// Region-based multiply-accumulate on shuffle2x format data
    ///
    /// All destinations and sources must be in shuffle2x format.
    ///
    /// # Safety
    /// - Data must be in shuffle2x format
    /// - Implementations may use unsafe SIMD intrinsics
    unsafe fn muladd_region_shuffle2x(
        &self,
        destinations: &mut [&mut [u16]],
        sources: &[&[u16]],
        coefficients: &[&[u16]],
        region_offset: usize,
        region_size: usize,
    ) {
        // Default: use regular region operation
        self.muladd_region(
            destinations,
            sources,
            coefficients,
            region_offset,
            region_size,
        );
    }
}

/// SIMD strategy registry for runtime selection
pub struct SimdRegistry {
    strategies: Vec<Box<dyn GaloisSimdStrategy>>,
    selected: Option<usize>,
}

impl Default for SimdRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdRegistry {
    /// Create a new registry and detect available strategies
    pub fn new() -> Self {
        let mut registry = SimdRegistry {
            strategies: Vec::new(),
            selected: None,
        };

        // Register platform-specific strategies
        #[cfg(target_arch = "x86_64")]
        x86::register_strategies(&mut registry);

        // Note: ARM uses direct dispatch in galois/mod.rs for better performance
        // (avoids trait object overhead in hot path)

        // Select the best available strategy
        registry.select_best();

        registry
    }

    /// Register a new strategy
    pub fn register(&mut self, strategy: Box<dyn GaloisSimdStrategy>) {
        if strategy.is_available() {
            self.strategies.push(strategy);
        }
    }

    /// Select the best available strategy based on priority
    fn select_best(&mut self) {
        if self.strategies.is_empty() {
            return;
        }

        // Find the strategy with highest priority
        let mut best_idx = 0;
        let mut best_priority = self.strategies[0].priority();

        for (idx, strategy) in self.strategies.iter().enumerate().skip(1) {
            let priority = strategy.priority();
            if priority > best_priority {
                best_priority = priority;
                best_idx = idx;
            }
        }

        self.selected = Some(best_idx);
    }

    /// Get the currently selected strategy
    pub fn get_selected(&self) -> Option<&dyn GaloisSimdStrategy> {
        self.selected
            .and_then(|idx| self.strategies.get(idx).map(|s| s.as_ref()))
    }

    /// Override strategy selection by name
    pub fn select_by_name(&mut self, name: &str) -> bool {
        for (idx, strategy) in self.strategies.iter().enumerate() {
            if strategy.name() == name {
                self.selected = Some(idx);
                return true;
            }
        }
        false
    }

    /// List all available strategies
    pub fn list_available(&self) -> Vec<(&'static str, Priority)> {
        self.strategies
            .iter()
            .map(|s| (s.name(), s.priority()))
            .collect()
    }
}

impl fmt::Display for SimdRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SIMD Strategy Registry:")?;
        for (i, strategy) in self.strategies.iter().enumerate() {
            let selected = self.selected == Some(i);
            writeln!(
                f,
                "  {} {} (priority: {:?}){}",
                if selected { "→" } else { " " },
                strategy.name(),
                strategy.priority(),
                if selected { " [selected]" } else { "" }
            )?;
        }
        Ok(())
    }
}

// Thread-safe global registry
use std::sync::OnceLock;
static SIMD_REGISTRY: OnceLock<SimdRegistry> = OnceLock::new();

/// Get or initialize the global SIMD registry
pub fn get_registry() -> &'static SimdRegistry {
    SIMD_REGISTRY.get_or_init(SimdRegistry::new)
}

/// Initialize the SIMD registry with custom configuration
pub fn init_registry_with_config(force_strategy: Option<&str>) {
    let _ = SIMD_REGISTRY.get_or_init(|| {
        let mut registry = SimdRegistry::new();
        if let Some(name) = force_strategy {
            registry.select_by_name(name);
        }
        registry
    });
}
