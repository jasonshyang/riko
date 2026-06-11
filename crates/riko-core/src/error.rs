use thiserror::Error;

/// Crate-wide `Result` bound to [`RicoError`].
pub type Result<T> = std::result::Result<T, RicoError>;

/// Top-level error returned by fallible engine APIs.
#[derive(Debug, Error)]
pub enum RicoError {
    /// Tool execution failed.
    #[error("tool error: {0}")]
    Tool(String),

    /// Filesystem or other std::io failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
