use bytes::Bytes;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::ToolCallId;

/// One block inside a [`crate::Message`]'s content array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    #[serde(with = "bytes_serde")]
    pub data: Bytes,
    pub mime: SmolStr,
}

/// Extended-thinking block emitted by reasoning-capable models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thinking {
    pub text: String,
    /// Provider-specific opaque signature for verbatim replay back to the same model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// True when the provider returned redacted thinking (signature only, no text).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
}

/// One tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Identifier matching the eventual tool-result turn.
    pub id: ToolCallId,
    /// Tool name (looked up in the active tool registry).
    pub name: SmolStr,
    /// Parsed arguments, validated against the tool's schema before execution.
    pub arguments: serde_json::Value,
}

mod bytes_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Bytes, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_bytes(value)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Bytes, D::Error> {
        let buf = <Vec<u8>>::deserialize(de)?;
        Ok(Bytes::from(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_roundtrip() {
        let c = Content::text("hello");
        let back: Content = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn thinking_skips_defaults() {
        let c =
            Content::Thinking(Thinking { text: "...".into(), signature: None, redacted: false });
        let json = serde_json::to_string(&c).unwrap();
        assert!(!json.contains("redacted"));
        assert!(!json.contains("signature"));
    }

    #[test]
    fn tool_call_roundtrip() {
        let c = Content::ToolCall(ToolCall {
            id: ToolCallId::new("tc_abc"),
            name: "read".into(),
            arguments: serde_json::json!({ "path": "/tmp/x" }),
        });
        let back: Content = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(c, back);
    }
}
