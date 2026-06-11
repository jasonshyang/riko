use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// Resolved riko configuration. Construct via [`crate::load`] or assemble manually in tests
/// and downstream binaries.
///
/// Only the sections with consumers today exist: the default model and per-provider overrides.
/// The prototype's `[context]` block configured the old layer pipeline (disabled layers,
/// redaction patterns, token budget) — none of which the workspace model has — so it is
/// deliberately gone. New sections land as their consumers do; unknown ones fail loudly.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    #[serde(default)]
    pub model: ModelSettings,
    /// Provider-prefix → override block. Keyed on the prefix the user puts in
    /// `provider:model-id` (e.g. `"anthropic"`, `"openai"`, `"ollama"`).
    #[serde(default)]
    pub providers: BTreeMap<SmolStr, ProviderSettings>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSettings {
    /// Default model reference when the caller doesn't pass one. Form: `"provider:model-id"`.
    #[serde(default)]
    pub default: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSettings {
    /// Override the default base URL for this provider prefix.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Override the env var that supplies the API key. `None` keeps the binary default
    /// (e.g. `ANTHROPIC_API_KEY`).
    #[serde(default)]
    pub api_key_env: Option<SmolStr>,
}

impl Settings {
    /// Merge `overlay` on top of `self` — every field the overlay explicitly sets wins.
    pub(crate) fn merge(&mut self, overlay: Settings) {
        if overlay.model.default.is_some() {
            self.model.default = overlay.model.default;
        }
        for (prefix, provider) in overlay.providers {
            match self.providers.get_mut(&prefix) {
                Some(existing) => Self::merge_provider(existing, provider),
                None => {
                    self.providers.insert(prefix, provider);
                }
            }
        }
    }

    fn merge_provider(base: &mut ProviderSettings, overlay: ProviderSettings) {
        if overlay.base_url.is_some() {
            base.base_url = overlay.base_url;
        }
        if overlay.api_key_env.is_some() {
            base.api_key_env = overlay.api_key_env;
        }
    }
}
