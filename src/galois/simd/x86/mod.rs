//! x86-64 SIMD implementations for Galois field operations
//!
//! This module provides various SIMD strategies optimized for different
//! x86-64 CPU features, from basic SSE2 to AVX2.

use super::SimdRegistry;

pub mod avx2;
pub mod pclmul;
pub mod sse;

/// Register all available x86 strategies
pub fn register_strategies(registry: &mut SimdRegistry) {
    // Register in order of preference (will be sorted by priority anyway)

    // Basic SSE2/SSSE3 implementations
    registry.register(Box::new(sse::Sse2Strategy));
    registry.register(Box::new(sse::Ssse3Strategy));

    // AVX2 shuffle-based implementation (fastest)
    registry.register(Box::new(avx2::Avx2ShuffleStrategy));

    // PCLMUL-based implementations (functional but slower than shuffle)
    registry.register(Box::new(pclmul::PclmulStrategy));
    registry.register(Box::new(pclmul::Avx2PclmulStrategy));
}

/// Runtime CPU feature detection for x86
#[derive(Clone, Copy)]
pub struct CpuFeatures {
    pub sse2: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub avx2: bool,
    pub pclmulqdq: bool,
}

impl CpuFeatures {
    /// Detect available CPU features at runtime
    pub fn detect() -> Self {
        CpuFeatures {
            sse2: is_x86_feature_detected!("sse2"),
            ssse3: is_x86_feature_detected!("ssse3"),
            sse41: is_x86_feature_detected!("sse4.1"),
            avx2: is_x86_feature_detected!("avx2"),
            pclmulqdq: is_x86_feature_detected!("pclmulqdq"),
        }
    }
}

// Thread-local cache for CPU features
thread_local! {
    static CPU_FEATURES: CpuFeatures = CpuFeatures::detect();
}

/// Get cached CPU features for the current thread
pub fn cpu_features() -> CpuFeatures {
    CPU_FEATURES.with(|f| *f)
}

/// Check if a specific CPU feature is available
#[inline]
pub fn has_feature(feature: &str) -> bool {
    CPU_FEATURES.with(|f| match feature {
        "sse2" => f.sse2,
        "ssse3" => f.ssse3,
        "sse4.1" => f.sse41,
        "avx2" => f.avx2,
        "pclmulqdq" => f.pclmulqdq,
        _ => false,
    })
}
