use riko_config::Settings;
use riko_core::{ModelRef, Result, RikoError};
use riko_llm::{Api, ModelSpec};
use smol_str::SmolStr;

// Conservative, model-agnostic capability defaults. A per-model catalog can refine these later;
// today every assembled model advertises a large window and tool + image support.
const DEFAULT_CONTEXT_WINDOW: u32 = 200_000;
const DEFAULT_MAX_OUTPUT: u32 = 8_192;

/// How to reach a provider prefix: its wire API plus the endpoint and key-env defaults, after
/// any `[providers.<prefix>]` overrides are layered on.
pub struct ModelRoute {
    api: Api,
    base_url: Option<String>,
    api_key_env: Option<SmolStr>,
}

impl ModelRoute {
    /// Resolve the effective model.
    pub fn resolve(requested: Option<&str>, settings: &Settings) -> Result<ModelSpec> {
        let reference = requested
            .map(str::to_owned)
            .or_else(|| settings.model.default.clone())
            .ok_or(RikoError::Config("No model".to_string()))?;
        let route = Self::resolve_route(&reference, settings)?;
        route.build_spec(&reference)
    }

    fn resolve_route(reference: &str, settings: &Settings) -> Result<ModelRoute> {
        let (prefix, _) = reference.split_once(':').ok_or_else(|| {
            RikoError::Config(format!("model reference must be `provider:id`, got `{reference}`"))
        })?;
        let mut route = Self::default_route(prefix)?;
        if let Some(over) = settings.providers.get(prefix) {
            if let Some(url) = &over.base_url {
                route.base_url = Some(url.clone());
            }
            if let Some(env) = &over.api_key_env {
                route.api_key_env = Some(env.clone());
            }
        }
        Ok(route)
    }

    /// Built-in routes.
    fn default_route(prefix: &str) -> Result<Self> {
        match prefix {
            "anthropic" => Ok(ModelRoute {
                api: Api::AnthropicMessages,
                base_url: None,
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
            }),
            "openai" => Ok(ModelRoute {
                api: Api::OpenAiCompletions,
                base_url: Some("https://api.openai.com/v1".to_owned()),
                api_key_env: Some("OPENAI_API_KEY".into()),
            }),
            "ollama" => Ok(ModelRoute {
                api: Api::OpenAiCompletions,
                base_url: Some("http://localhost:11434/v1".to_owned()),
                api_key_env: None,
            }),
            other => Err(RikoError::Config(format!(
                "unsupported model prefix `{other}` — supported prefixes are `anthropic:`, \
                 `openai:`, and `ollama:`"
            ))),
        }
    }

    fn build_spec(self, reference: &str) -> Result<ModelSpec> {
        let model_id = reference.split_once(':').map_or(reference, |(_, rest)| rest);
        let mut builder = ModelSpec::builder()
            .id(ModelRef::new(reference))
            .name(model_id)
            .api(self.api)
            .context_window(DEFAULT_CONTEXT_WINDOW)
            .max_output(DEFAULT_MAX_OUTPUT)
            .supports_images(true)
            .supports_tools(true);
        if let Some(url) = self.base_url {
            builder = builder.base_url(url);
        }
        if let Some(env) = self.api_key_env {
            builder = builder.api_key_env(env);
        }
        builder.build()
    }
}
