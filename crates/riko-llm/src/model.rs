use riko_core::{ModelRef, Result, RikoError, ThinkingLevel};
use smol_str::SmolStr;
use std::collections::HashMap;

use crate::Api;

/// Static description of one model: its id, capabilities, costs, and how to reach it.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Canonical `provider:model-id` reference.
    pub id: ModelRef,
    /// Human-visible display name.
    pub name: SmolStr,
    /// Wire-protocol family this model speaks.
    pub api: Api,
    /// Endpoint base URL; `None` uses the provider's default.
    pub base_url: Option<String>,
    /// Maximum prompt tokens the model accepts.
    pub context_window: u32,
    /// Maximum tokens the model produces in one response.
    pub max_output: u32,
    /// Per-million-token costs.
    pub cost: ModelCost,
    /// Thinking levels the model supports, mapped to the provider's raw parameter for each.
    /// A level's presence means "supported"; absent levels fall back via [`clamp_thinking`].
    pub thinking_levels: HashMap<ThinkingLevel, SmolStr>,
    /// True if the model accepts image input.
    pub supports_images: bool,
    /// True if the model supports tool calls.
    pub supports_tools: bool,
    /// Environment variable holding this model's API key; `None` for unauthenticated
    /// endpoints (e.g. a local Ollama).
    pub api_key_env: Option<SmolStr>,
}

impl ModelSpec {
    pub fn builder() -> ModelSpecBuilder {
        ModelSpecBuilder::default()
    }

    /// Clamp a requested thinking level down to the nearest level the model supports,
    /// returning `Off` if none qualifies.
    pub fn clamp_thinking(&self, requested: ThinkingLevel) -> ThinkingLevel {
        const DESCENDING: [ThinkingLevel; 5] = [
            ThinkingLevel::Max,
            ThinkingLevel::High,
            ThinkingLevel::Medium,
            ThinkingLevel::Low,
            ThinkingLevel::Minimal,
        ];
        match DESCENDING.iter().position(|&level| level == requested) {
            None => ThinkingLevel::Off,
            Some(start) => DESCENDING[start..]
                .iter()
                .copied()
                .find(|level| self.thinking_levels.contains_key(level))
                .unwrap_or(ThinkingLevel::Off),
        }
    }
}

/// Per-million-token costs in USD.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// Builder for [`ModelSpec`]. `id`, `name`, `api`, `context_window`, and `max_output` are
/// required; [`build`](Self::build) reports the first missing one.
#[derive(Default)]
pub struct ModelSpecBuilder {
    id: Option<ModelRef>,
    name: Option<SmolStr>,
    api: Option<Api>,
    base_url: Option<String>,
    context_window: Option<u32>,
    max_output: Option<u32>,
    cost: ModelCost,
    thinking_levels: HashMap<ThinkingLevel, SmolStr>,
    supports_images: bool,
    supports_tools: bool,
    api_key_env: Option<SmolStr>,
}

impl ModelSpecBuilder {
    pub fn id(mut self, id: impl Into<ModelRef>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn name(mut self, name: impl Into<SmolStr>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn api(mut self, api: Api) -> Self {
        self.api = Some(api);
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    pub fn max_output(mut self, tokens: u32) -> Self {
        self.max_output = Some(tokens);
        self
    }

    pub fn cost(mut self, cost: ModelCost) -> Self {
        self.cost = cost;
        self
    }

    /// Mark a thinking level supported, with the provider's raw parameter for it.
    pub fn thinking_level(mut self, level: ThinkingLevel, param: impl Into<SmolStr>) -> Self {
        self.thinking_levels.insert(level, param.into());
        self
    }

    pub fn supports_images(mut self, yes: bool) -> Self {
        self.supports_images = yes;
        self
    }

    pub fn supports_tools(mut self, yes: bool) -> Self {
        self.supports_tools = yes;
        self
    }

    pub fn api_key_env(mut self, var: impl Into<SmolStr>) -> Self {
        self.api_key_env = Some(var.into());
        self
    }

    pub fn build(self) -> Result<ModelSpec> {
        Ok(ModelSpec {
            id: Self::require(self.id, "id")?,
            name: Self::require(self.name, "name")?,
            api: Self::require(self.api, "api")?,
            base_url: self.base_url,
            context_window: Self::require(self.context_window, "context_window")?,
            max_output: Self::require(self.max_output, "max_output")?,
            cost: self.cost,
            thinking_levels: self.thinking_levels,
            supports_images: self.supports_images,
            supports_tools: self.supports_tools,
            api_key_env: self.api_key_env,
        })
    }

    fn require<T>(field: Option<T>, name: &str) -> Result<T> {
        field.ok_or_else(|| RikoError::Config(format!("model spec missing required field: {name}")))
    }
}
