use riko_context::ItemId;
use riko_llm::ProviderEvent;

/// Run-lifecycle and ephemeral streaming events emitted by the agent. Durable item changes
/// (the assistant turn landing, tool results appended) arrive on the workspace's event stream;
/// a frontend watches both.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A run has begun.
    RunStarted,
    /// A turn within the run has begun.
    TurnStarted,
    /// A provider streaming delta — ephemeral, for live display; never logged.
    Streaming(ProviderEvent),
    /// An item the agent just finalized into the workspace (assistant turn or tool result).
    Settled { id: ItemId },
    /// A turn finished.
    TurnEnded,
    /// The run finished.
    RunEnded { reason: EndReason },
    /// A non-fatal error surfaced mid-run (e.g. a provider error ending the turn).
    Error { message: String },
}

/// Why a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// The assistant produced a natural stop with no pending tool calls.
    Stopped,
    /// The run was aborted via the cancel token.
    Aborted,
    /// The provider returned an error, or setup failed.
    Failed,
}
