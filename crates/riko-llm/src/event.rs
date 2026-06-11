use futures::Stream;
use riko_core::{Message, StopReason, ToolCallId, Usage};
use smol_str::SmolStr;
use std::pin::Pin;

/// Boxed stream returned by [`crate::Provider::stream`].
pub type ProviderStream = Pin<Box<dyn Stream<Item = ProviderEvent> + Send>>;

/// Normalized event emitted by a provider's response stream.
#[derive(Debug, Clone)]
pub enum ProviderEvent {
    /// Stream accepted by the provider.
    Start,

    /// Text block opened at `idx`.
    TextStart { idx: u32 },
    /// Append `text` to the text block at `idx`.
    TextDelta { idx: u32, text: String },
    /// Text block at `idx` closed.
    TextEnd { idx: u32 },

    /// Thinking block opened at `idx`.
    ThinkingStart { idx: u32 },
    /// Append to the thinking block at `idx`; `signature` carries the opaque replay token
    /// when the provider emits one.
    ThinkingDelta { idx: u32, text: String, signature: Option<String> },
    /// Thinking block at `idx` closed.
    ThinkingEnd { idx: u32 },

    /// Tool call opened at `idx`.
    ToolCallStart { idx: u32, call_id: ToolCallId, name: SmolStr },
    /// Raw partial-JSON fragment for the tool-call arguments at `idx` (for live display).
    ToolCallArgsDelta { idx: u32, fragment: String },
    /// Tool call at `idx` closed with final, parsed `args`.
    ToolCallEnd { idx: u32, args: serde_json::Value },

    /// Cumulative usage for this turn.
    UsageUpdate(Usage),

    /// Terminal success: the assembled assistant turn.
    Done { stop: StopReason, message: Box<Message> },
    /// Terminal failure: the stream aborted before completion.
    Error { reason: ErrorReason, message: String },

    /// Other unsupported events
    Other(String),
}

/// Why a provider stream ended in [`ProviderEvent::Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorReason {
    /// Cancellation token fired.
    Cancelled,
    /// HTTP / network failure.
    Transport,
    /// Provider returned a 5xx, rate-limit, or other retryable status.
    Retryable,
    /// Provider returned a 4xx or unparseable payload — do not retry.
    Permanent,
    /// Authentication failed (missing or invalid credentials).
    Auth,
}
