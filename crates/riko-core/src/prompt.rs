use crate::{Message, ToolDescriptor};

/// The assembled input to a model for one turn: the system prompt, the conversation in
/// turn order, and the tools the model may call.
///
/// A workspace render projects to this; a provider consumes it. It is provider-neutral —
/// adapting it to a vendor's wire format is the provider's job, and any compaction or
/// redaction has already happened upstream.
#[derive(Debug, Clone, Default)]
pub struct Prompt {
    /// Fully composed system prompt.
    pub system: String,
    /// Conversation turns in order.
    pub messages: Vec<Message>,
    /// Tool schemas the model may invoke this turn.
    pub tools: Vec<ToolDescriptor>,
}
