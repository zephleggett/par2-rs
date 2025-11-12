use thiserror::Error;

/// PAR2 operation errors
#[derive(Error, Debug)]
pub enum Par2Error {
    #[error("PAR2 file not found")]
    NotFound,

    #[error("Invalid PAR2 file format: {0}")]
    InvalidFormat(String),

    #[error("PAR2 parsing failed: {0}")]
    ParseError(String),

    #[error("File verification failed: {0}")]
    VerificationFailed(String),

    #[error("PAR2 repair failed: {0}")]
    RepairFailed(String),

    #[error("Insufficient recovery blocks: need {needed}, have {available}")]
    InsufficientRecovery { needed: usize, available: usize },

    #[error("File size mismatch for {file}: expected {expected}, got {actual}")]
    SizeMismatch {
        file: String,
        expected: u64,
        actual: u64,
    },

    #[error("Hash mismatch for file {0}")]
    HashMismatch(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Par2Error>;
