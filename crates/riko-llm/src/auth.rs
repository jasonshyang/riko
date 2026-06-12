use riko_core::{Result, RikoError};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// How to source an API key for a provider.
///
/// Env-var sourcing is the default; an inline key is supported for tests but must never be
/// committed to a settings file. Files written via the CLI live under `~/.riko/auth/`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum AuthSpec {
    /// Read the key from an environment variable.
    Env { var: SmolStr },
    /// Inline key value — tests only.
    Inline { key: String },
    /// No authentication (local runtimes, etc.).
    None,
}

/// Resolve an [`AuthSpec`] into a concrete API key.
///
/// Returns `Ok(None)` for [`AuthSpec::None`]. Fails loudly when an env var is missing
/// rather than silently degrading — matches the "no silent fallbacks" principle.
pub fn resolve_api_key(spec: &AuthSpec) -> Result<Option<String>> {
    match spec {
        AuthSpec::Env { var } => match std::env::var(var.as_str()) {
            Ok(v) if v.is_empty() => {
                Err(RikoError::Config(format!("environment variable `{var}` is set but empty")))
            }
            Ok(v) => Ok(Some(v)),
            Err(_) => Err(RikoError::Config(format!("environment variable `{var}` is not set"))),
        },
        AuthSpec::Inline { key } => Ok(Some(key.clone())),
        AuthSpec::None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_resolves_to_none() {
        assert!(resolve_api_key(&AuthSpec::None).unwrap().is_none());
    }

    #[test]
    fn inline_returns_the_value() {
        let spec = AuthSpec::Inline { key: "sk-test".into() };
        assert_eq!(resolve_api_key(&spec).unwrap().as_deref(), Some("sk-test"));
    }

    #[test]
    fn missing_env_var_fails_loudly() {
        let spec = AuthSpec::Env { var: "RIKO_DEFINITELY_UNSET_KEY_X9F2".into() };
        let err = resolve_api_key(&spec).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("RIKO_DEFINITELY_UNSET_KEY_X9F2"));
        assert!(msg.contains("not set"));
    }

    #[test]
    fn present_env_var_resolves() {
        // `PATH` is always set in any environment that can run cargo test.
        let spec = AuthSpec::Env { var: "PATH".into() };
        let resolved = resolve_api_key(&spec).unwrap().expect("PATH should be set");
        assert!(!resolved.is_empty());
    }
}
