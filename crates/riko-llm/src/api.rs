use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smol_str::SmolStr;

/// Discriminant for a provider's wire-protocol family. One `Api` may have many concrete
/// endpoints behind it (every OpenAI-compatible server shares `OpenAiCompletions`). The
/// [`crate::ProviderRegistry`] keys on this value.
///
/// Serializes as a plain string (e.g. `"anthropic-messages"`) so it reads cleanly in
/// config and logs; unknown strings round-trip through [`Api::Custom`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Api {
    /// Anthropic Messages API (and Anthropic-compatible endpoints).
    AnthropicMessages,
    /// OpenAI Chat Completions and every compatible endpoint (xAI, Groq, Together,
    /// DeepSeek, OpenRouter, Ollama, vLLM, llama.cpp, …).
    OpenAiCompletions,
    /// Escape hatch for a downstream-registered provider.
    Custom(SmolStr),
}

impl Api {
    /// Stable string identifier used in config, logs, and the registry.
    pub fn as_str(&self) -> &str {
        match self {
            Api::AnthropicMessages => "anthropic-messages",
            Api::OpenAiCompletions => "openai-completions",
            Api::Custom(name) => name.as_str(),
        }
    }

    fn from_wire(value: &str) -> Self {
        match value {
            "anthropic-messages" => Api::AnthropicMessages,
            "openai-completions" => Api::OpenAiCompletions,
            other => Api::Custom(SmolStr::from(other)),
        }
    }
}

impl std::fmt::Display for Api {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Api {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Api {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = SmolStr::deserialize(deserializer)?;
        Ok(Api::from_wire(&raw))
    }
}
