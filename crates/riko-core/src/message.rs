use smol_str::SmolStr;

use crate::{Content, ModelRef, ToolCallId};

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

#[derive(Debug, Clone)]
pub enum Role {
    /// Human input.
    User,
    /// Model output, with the model that produced it and per-turn bookkeeping.
    Assistant { model: ModelRef, usage: Usage, stop: StopReason },
    /// Result of a tool execution, paired with the call that triggered it.
    ToolResult { call_id: ToolCallId, tool: SmolStr, is_error: bool },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Prompt tokens charged at full price.
    pub input_tokens: u32,
    /// Generated output tokens.
    pub output_tokens: u32,
    /// Prompt tokens served from the provider's cache.
    pub cache_read_tokens: u32,
    /// Prompt tokens written into the provider's cache for future reuse.
    pub cache_write_tokens: u32,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens as u64
            + self.output_tokens as u64
            + self.cache_read_tokens as u64
            + self.cache_write_tokens as u64
    }

    pub fn merge(&mut self, other: Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_read_tokens = self.cache_read_tokens.saturating_add(other.cache_read_tokens);
        self.cache_write_tokens = self.cache_write_tokens.saturating_add(other.cache_write_tokens);
    }
}

/// Why an assistant turn stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Model produced an end-of-turn signal naturally.
    Stop,
    /// Output truncated at the max-tokens cap.
    Length,
    /// Model emitted tool calls and is awaiting results.
    ToolUse,
    /// Provider returned an error mid-stream.
    Error,
    /// Cancellation fired and the stream was aborted.
    Aborted,
}
