use std::path::PathBuf;

use clap::Parser;

/// riko — a coding agent whose context is an editable workspace.
#[derive(Debug, Parser)]
#[command(name = "riko", version, about)]
pub struct Cli {
    /// Model to run, as `provider:model-id` (e.g. `anthropic:claude-opus-4-8`). Defaults to the
    /// settings `[model].default`, then a built-in fallback.
    #[arg(long)]
    pub model: Option<String>,

    /// Workspace root the agent operates in. Defaults to the current directory.
    #[arg(long)]
    pub workspace: Option<PathBuf>,

    /// Persist the session to this operation log (replayed on restart). In-memory if omitted.
    #[arg(long)]
    pub session: Option<PathBuf>,
}
