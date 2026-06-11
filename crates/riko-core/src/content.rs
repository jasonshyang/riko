use bytes::Bytes;
use smol_str::SmolStr;

use crate::ToolCallId;

/// One block inside a [`crate::Message`]'s content array.
#[derive(Debug, Clone)]
pub enum Content {
    Text(Text),
    Image(Image),
    Thinking(Thinking),
    ToolCall(ToolCall),
}

impl Content {
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(Text { text: content.into() })
    }
}

#[derive(Debug, Clone)]
pub struct Text {
    pub text: String,
}

/// Inline image content: raw bytes plus a MIME type.
#[derive(Debug, Clone)]
pub struct Image {
    pub data: Bytes,
    pub mime: SmolStr,
}

/// Extended-thinking block emitted by reasoning-capable models.
#[derive(Debug, Clone)]
pub struct Thinking {
    pub text: String,
}

/// One tool call requested by the assistant.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: SmolStr,
    pub arguments: serde_json::Value,
}
