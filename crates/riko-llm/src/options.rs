use riko_core::ThinkingLevel;
use std::time::Duration;

/// Per-request knobs handed alongside a [`riko_core::Prompt`] to a provider.
#[derive(Debug, Clone)]
pub struct StreamOptions {
    /// Cap on tokens the model may produce this turn.
    pub max_output_tokens: u32,
    /// Sampling temperature (provider default when `None`).
    pub temperature: Option<f32>,
    /// Requested thinking intensity; providers clamp to what the model supports.
    pub thinking: ThinkingLevel,
    /// Provider-cache retention hint.
    pub cache: CacheRetention,
    /// Per-attempt timeout for the underlying HTTP request.
    pub timeout: Duration,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            max_output_tokens: 4_096,
            temperature: None,
            thinking: ThinkingLevel::Off,
            cache: CacheRetention::Off,
            timeout: Duration::from_secs(600),
        }
    }
}

/// Three-tier abstraction over per-provider prompt caching; providers map it onto their
/// native cache controls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CacheRetention {
    /// Do not request caching.
    #[default]
    Off,
    /// Short-lived cache (provider default, typically minutes).
    Brief,
    /// Long-lived cache (typically an hour or more).
    Persistent,
}
