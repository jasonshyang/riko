use bytes::Bytes;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::ToolCallId;

/// One block inside a [`crate::Message`]'s content array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Content {
    Text(String),
    Image(Image),
    Thinking(Thinking),
    ToolCall(ToolCall),
}

impl Content {
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }
}

/// Inline image content: raw bytes plus a MIME type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub data: Bytes,
    pub mime: SmolStr,
}

/// Extended-thinking block emitted by reasoning-capable models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thinking {
    pub text: String,
}

/// One tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: SmolStr,
    pub arguments: serde_json::Value,
}
