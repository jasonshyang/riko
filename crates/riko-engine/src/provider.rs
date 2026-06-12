//! Constructing the concrete provider a model needs, for when the caller doesn't inject a
//! registry. Mirrors the model route the [`crate::model`] resolver produced: which API family,
//! which endpoint, and which environment variable holds the key.

use riko_core::{Result, RikoError};
use riko_llm::{
    Api, AuthSpec, ModelSpec, ProviderRegistry, anthropic::AnthropicProvider,
    openai::OpenAiCompletionsProvider, resolve_api_key,
};

/// Build a registry holding the single provider `model`'s API needs. The API key is resolved
/// from the environment up front, so a missing or empty key fails at build time rather than on
/// the first turn.
pub(crate) fn default_registry(model: &ModelSpec) -> Result<ProviderRegistry> {
    let mut registry = ProviderRegistry::new();
    match &model.api {
        Api::AnthropicMessages => registry.register(anthropic(model)?),
        Api::OpenAiCompletions => registry.register(openai(model)?),
        other => {
            return Err(RikoError::Config(format!(
                "no built-in provider for API `{other}`; inject a ProviderRegistry instead"
            )));
        }
    }
    Ok(registry)
}

/// Anthropic bakes the key into the client at construction, so it must be resolved here.
fn anthropic(model: &ModelSpec) -> Result<AnthropicProvider> {
    let mut provider = AnthropicProvider::new(required_key(model)?);
    if let Some(base_url) = &model.base_url {
        provider = provider.with_base_url(base_url.clone());
    }
    Ok(provider)
}

/// The OpenAI-completions provider reads the key from the model's `api_key_env` at call time and
/// the endpoint from its `base_url`, so it only needs registering — but pre-resolve a configured
/// key now so a missing one is caught at startup. A model with no `api_key_env` (e.g. a local
/// Ollama) needs no key at all.
fn openai(model: &ModelSpec) -> Result<OpenAiCompletionsProvider> {
    if model.api_key_env.is_some() {
        required_key(model)?;
    }
    Ok(OpenAiCompletionsProvider::new())
}

fn required_key(model: &ModelSpec) -> Result<String> {
    let var = model.api_key_env.as_ref().ok_or_else(|| {
        RikoError::Config(format!(
            "model `{}` has no api_key_env to authenticate with",
            model.id.as_str()
        ))
    })?;
    resolve_api_key(&AuthSpec::Env { var: var.clone() })?.ok_or_else(|| {
        RikoError::Config(format!("api key for model `{}` resolved to none", model.id.as_str()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(api: Api, api_key_env: Option<&str>) -> ModelSpec {
        let mut builder = ModelSpec::builder()
            .id("test:m")
            .name("m")
            .api(api)
            .context_window(1_000)
            .max_output(100);
        if let Some(var) = api_key_env {
            builder = builder.api_key_env(var);
        }
        builder.build().unwrap()
    }

    #[test]
    fn ollama_style_model_needs_no_key() {
        let registry = default_registry(&model(Api::OpenAiCompletions, None)).unwrap();
        assert!(registry.contains(&Api::OpenAiCompletions));
    }

    #[test]
    fn anthropic_without_its_key_fails_loudly() {
        let spec = model(Api::AnthropicMessages, Some("RIKO_ENGINE_GUARANTEED_UNSET_KEY_X9F2"));
        let err = default_registry(&spec).err().unwrap();
        assert!(matches!(err, RikoError::Config(_)));
        assert!(err.to_string().contains("RIKO_ENGINE_GUARANTEED_UNSET_KEY_X9F2"));
    }

    #[test]
    fn custom_api_has_no_builtin_provider() {
        let spec = model(Api::Custom("weird".into()), None);
        let err = default_registry(&spec).err().unwrap();
        assert!(matches!(err, RikoError::Config(_)));
        assert!(err.to_string().contains("inject a ProviderRegistry"));
    }
}
