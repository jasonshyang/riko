use bytes::Bytes;
use smol_str::SmolStr;

use crate::ToolCallId;

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

#[derive(Debug, Clone)]
pub struct Image {
    pub data: Bytes,
    pub mime: SmolStr,
}

#[derive(Debug, Clone)]
pub struct Thinking {
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: SmolStr,
    pub arguments: serde_json::Value,
}
