use thiserror::Error;

/// Crate-wide `Result` bound to [`RikoError`].
pub type Result<T> = std::result::Result<T, RikoError>;

/// Top-level error returned by fallible engine APIs.
#[derive(Debug, Error)]
pub enum RikoError {
    /// Settings malformed, missing a required value, or otherwise invalid.
    #[error("configuration error: {0}")]
    Config(String),

    /// Caller supplied an argument the callee cannot accept.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Lookup of a named resource (model, tool, item, …) failed.
    #[error("not found: {0}")]
    NotFound(String),

    /// Tool execution failed.
    #[error("tool error: {0}")]
    Tool(String),

    /// Filesystem or other std::io failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON encoding or decoding failure.
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    /// The operation observed its cancellation token and stopped.
    #[error("cancelled")]
    Cancelled,

    /// LLM provider rejected the request or returned an error stream.
    #[error("provider error: {0}")]
    Provider(String),
}
