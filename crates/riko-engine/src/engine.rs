use riko_agent::Agent;
use riko_config::{LoadOptions, Settings, Skill, bootstrap_workspace};
use riko_context::Workspace;
use riko_core::{Result, RikoError};
use riko_llm::{ModelSpec, ProviderRegistry, StreamOptions};
use riko_tools::ToolRegistry;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{ModelRoute, provider::default_registry};

/// Wires every engine crate into one [`Engine`]. Everything has a sensible default: the current
/// directory as the root, an in-memory workspace, settings discovered from `~/.riko` and
/// `<root>/.riko`, and — when no registry is injected — the provider the resolved model needs,
/// built with its env-resolved API key.
///
/// Inject a [`ProviderRegistry`] via [`providers`](Self::providers) to supply a mock or a custom
/// provider; otherwise the builder constructs the one the chosen model's API requires and fails
/// loudly at build time if its key is missing.
pub struct Engine {
    workspace: Arc<Workspace>,
    agent: Arc<Agent>,
    root: PathBuf,
    model: ModelSpec,
    skills: Vec<Skill>,
}

impl Engine {
    /// Start assembling an engine. See [`EngineBuilder`] for the required inputs.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    /// Assemble from already-wired parts. `EngineBuilder::build` is the usual entry point; this
    /// exists so the builder is the only place that constructs the private fields.
    pub(crate) fn from_parts(
        workspace: Arc<Workspace>,
        agent: Arc<Agent>,
        root: PathBuf,
        model: ModelSpec,
        skills: Vec<Skill>,
    ) -> Self {
        Self { workspace, agent, root, model, skills }
    }

    /// The shared workspace — query it, submit operations, or subscribe to its change events.
    pub fn workspace(&self) -> &Arc<Workspace> {
        &self.workspace
    }

    /// The agent driving this workspace — prompt it, run, abort, steer, or queue a follow-up.
    pub fn agent(&self) -> &Arc<Agent> {
        &self.agent
    }

    /// The resolved workspace root that tools resolve relative paths against.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The resolved model spec the agent runs with.
    pub fn model(&self) -> &ModelSpec {
        &self.model
    }

    /// The skills discovered at startup (the same catalog seeded into the workspace).
    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }
}

/// Wires every engine crate into one [`Engine`].
#[derive(Default)]
pub struct EngineBuilder {
    workspace_root: Option<PathBuf>,
    model: Option<String>,
    session_path: Option<PathBuf>,
    settings: Option<LoadOptions>,
    skill_roots: Option<Vec<PathBuf>>,
    providers: Option<Arc<ProviderRegistry>>,
    options: Option<StreamOptions>,
}

impl EngineBuilder {
    /// Directory the session is rooted at; tools resolve relative paths against it. Defaults to
    /// the current directory. The path is canonicalized at build time.
    pub fn workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// `provider:model-id` reference to run. When unset, the settings default is used, then a
    /// built-in fallback.
    pub fn model(mut self, reference: impl Into<String>) -> Self {
        self.model = Some(reference.into());
        self
    }

    /// Path to the persistent operation log. When set, an existing log is replayed (and not
    /// re-seeded); a missing one is created and seeded. When unset, the workspace is in-memory.
    pub fn session_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.session_path = Some(path.into());
        self
    }

    /// Which settings files to load. When unset, the standard `~/.riko/settings.toml` (user) and
    /// `<root>/.riko/settings.toml` (workspace) paths are used. Pass [`LoadOptions::default`] to
    /// load nothing — useful for hermetic tests.
    pub fn settings(mut self, options: LoadOptions) -> Self {
        self.settings = Some(options);
        self
    }

    /// Directories to discover skills under. When unset, defaults to
    /// [`default_skill_roots`](riko_config::default_skill_roots) for the resolved root.
    pub fn skill_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.skill_roots = Some(roots);
        self
    }

    /// The provider registry the agent dispatches through. Optional: when unset, the builder
    /// constructs the provider the resolved model needs (resolving its API key from the
    /// environment). Inject one to use a mock or a custom provider.
    pub fn providers(mut self, providers: Arc<ProviderRegistry>) -> Self {
        self.providers = Some(providers);
        self
    }

    /// Per-request stream options (token cap, thinking, cache, timeout). Defaults to
    /// [`StreamOptions::default`].
    pub fn options(mut self, options: StreamOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Resolve, load, open, seed, and wire everything into an [`Engine`]. Fails loudly at the
    /// first problem — an inaccessible root, a missing provider, malformed settings — rather than
    /// deferring it to the first turn.
    pub fn build(self) -> Result<Engine> {
        let root = Self::resolve_root(self.workspace_root)?;
        let settings = Self::load_settings(self.settings, &root)?;
        let model = ModelRoute::resolve(self.model.as_deref(), &settings)?;

        let providers = match self.providers {
            Some(injected) => {
                Self::ensure_provider_available(&injected, &model)?;
                injected
            }
            None => Arc::new(default_registry(&model)?),
        };

        let tools = Arc::new(Self::default_tools());
        let skill_roots = self.skill_roots.unwrap_or_else(|| Self::default_skill_roots(&root));
        let (workspace, fresh) = Self::open_workspace(self.session_path.as_deref())?;
        let workspace = Arc::new(workspace);
        let skills = bootstrap_workspace(&workspace, &root, &skill_roots, fresh)?;

        let agent = Self::build_agent(
            Arc::clone(&workspace),
            providers,
            model.clone(),
            tools,
            &root,
            self.options,
        )?;

        Ok(Engine::from_parts(workspace, Arc::new(agent), root, model, skills))
    }

    fn build_agent(
        workspace: Arc<Workspace>,
        providers: Arc<ProviderRegistry>,
        model: ModelSpec,
        tools: Arc<ToolRegistry>,
        root: &Path,
        options: Option<StreamOptions>,
    ) -> Result<Agent> {
        let mut builder = Agent::builder()
            .workspace(workspace)
            .providers(providers)
            .model(model)
            .tools(tools)
            .root(root.to_path_buf());
        if let Some(options) = options {
            builder = builder.options(options);
        }
        builder.build()
    }

    fn default_tools() -> ToolRegistry {
        let mut tools = ToolRegistry::default();
        tools.register_defaults();
        tools
    }

    fn resolve_root(root: Option<PathBuf>) -> Result<PathBuf> {
        let raw = root.unwrap_or_else(|| PathBuf::from("."));
        std::fs::canonicalize(&raw).map_err(|e| {
            RikoError::Config(format!("workspace root {} is not accessible: {e}", raw.display()))
        })
    }

    fn ensure_provider_available(providers: &ProviderRegistry, model: &ModelSpec) -> Result<()> {
        if providers.contains(&model.api) {
            return Ok(());
        }
        Err(RikoError::Config(format!(
            "no provider registered for `{}`, which model `{}` requires",
            model.api,
            model.id.as_str()
        )))
    }

    fn load_settings(options: Option<LoadOptions>, root: &Path) -> Result<Settings> {
        Ok(options.unwrap_or_else(|| Self::default_settings_paths(root)).load()?.settings)
    }

    /// The standard discovery paths: user settings under `$HOME/.riko`, workspace settings under the
    /// session root. Either may be absent — [`load`] treats missing files as non-errors.
    fn default_settings_paths(root: &Path) -> LoadOptions {
        let user_file = std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".riko").join("settings.toml"));
        let workspace_file = Some(root.join(".riko").join("settings.toml"));
        LoadOptions { user_file, workspace_file }
    }

    /// Default skill roots: `~/.riko/skills` then `<workspace>/.riko/skills`.
    fn default_skill_roots(workspace: &Path) -> Vec<PathBuf> {
        let mut roots = Vec::with_capacity(2);
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            roots.push(home.join(".riko").join("skills"));
        }
        roots.push(workspace.join(".riko").join("skills"));
        roots
    }

    /// Open the persistent workspace (or an in-memory one), reporting whether it is fresh. A fresh
    /// workspace gets seeded; a replayed one already carries its startup items in the log.
    fn open_workspace(path: Option<&Path>) -> Result<(Workspace, bool)> {
        let Some(path) = path else {
            return Ok((Workspace::new(), true));
        };
        let fresh = !path.exists();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        Ok((Workspace::open(path)?, fresh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riko_llm::{Api, mock::MockProvider};

    fn registry_for(api: Api) -> Arc<ProviderRegistry> {
        let mut registry = ProviderRegistry::new();
        registry.register(MockProvider::for_api(api));
        Arc::new(registry)
    }

    #[test]
    fn auto_builds_a_provider_when_none_is_injected() {
        // An Ollama model needs no API key, so the builder can construct its provider with no
        // registry injected and no environment set up — exercising the auto-build path.
        let engine = Engine::builder()
            .workspace_root(".")
            .model("ollama:llama")
            .settings(LoadOptions::default())
            .skill_roots(Vec::new())
            .build()
            .unwrap();
        assert_eq!(engine.model().api, Api::OpenAiCompletions);
    }

    #[test]
    fn build_rejects_a_registry_that_cannot_serve_the_model() {
        // The model routes to Anthropic, but only an OpenAI provider is registered.
        let err = Engine::builder()
            .workspace_root(".")
            .model("anthropic:claude")
            .settings(LoadOptions::default())
            .providers(registry_for(Api::OpenAiCompletions))
            .build()
            .err()
            .unwrap();
        assert!(matches!(err, RikoError::Config(_)));
        let message = err.to_string();
        assert!(message.contains("no provider registered"));
        assert!(message.contains("anthropic-messages"));
    }
}
