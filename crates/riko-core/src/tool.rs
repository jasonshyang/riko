use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// Provider-facing description of one tool the model may invoke. Holds only what a
/// provider needs to expose the tool — the implementation and execution machinery live
/// in the crates that own that behavior.
#[derive(Debug, Clone)]
pub struct ToolDescriptor {
    /// Stable identifier the model emits when it calls the tool.
    pub name: SmolStr,
    /// Free-form description shown to the model.
    pub description: String,
    /// JSON Schema for the tool's input arguments.
    pub schema: ToolSchema,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolSchema(serde_json::Value);

impl ToolSchema {
    pub fn from_value(value: serde_json::Value) -> Self {
        Self(value)
    }

    pub fn as_value(&self) -> &serde_json::Value {
        &self.0
    }
}
