use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ignore::WalkBuilder;
use regex::Regex;
use riko_core::{RikoError, ToolCall, ToolDescriptor};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::fs;
use crate::{Tool, ToolContext, ToolFuture, ToolOutput};

const DEFAULT_MAX_RESULTS: u32 = 500;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindArgs {
    /// Regex matched against each candidate path (relative to the search root). E.g. `\.rs$`
    /// finds Rust files, `(?i)readme` finds READMEs case-insensitively.
    pub pattern: String,
    /// Optional search root. Defaults to the workspace root.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Cap on number of paths returned. Defaults to 500.
    #[serde(default)]
    pub max_results: Option<u32>,
}

/// List workspace paths matching a regex, honoring .gitignore.
pub struct FindTool;

impl FindTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FindTool {
    fn default() -> Self {
        Self::new()
    }
}

fn descriptor() -> &'static ToolDescriptor {
    static DESCRIPTOR: OnceLock<ToolDescriptor> = OnceLock::new();
    DESCRIPTOR.get_or_init(|| ToolDescriptor {
        name: "find".into(),
        description: "List workspace paths matching a regex. Honors .gitignore. Returns up to \
                      `max_results` paths (default 500), relative to the search root."
            .into(),
        schema: super::schema_for::<FindArgs>(),
    })
}

impl Tool for FindTool {
    fn descriptor(&self) -> &ToolDescriptor {
        descriptor()
    }

    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        cancel: CancellationToken,
    ) -> ToolFuture<'a> {
        Box::pin(async move {
            let args: FindArgs = serde_json::from_value(call.arguments)
                .map_err(|e| RikoError::InvalidArgument(format!("find tool args: {e}")))?;
            let regex = Regex::new(&args.pattern)
                .map_err(|e| RikoError::InvalidArgument(format!("find pattern: {e}")))?;
            let root = match args.path {
                Some(p) => fs::resolve_path(&ctx.root, &p),
                None => ctx.root.clone(),
            };
            let cap = args.max_results.unwrap_or(DEFAULT_MAX_RESULTS) as usize;

            let cancel_for_blocking = cancel.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                find_blocking(&root, &regex, cap, cancel_for_blocking)
            })
            .await
            .map_err(|e| RikoError::Tool(format!("find task join failed: {e}")))?;

            if cancel.is_cancelled() {
                return Err(RikoError::Cancelled);
            }

            let FindOutcome { paths, truncated } = outcome?;
            let body = if paths.is_empty() {
                "no matches".to_string()
            } else {
                let mut out = paths.join("\n");
                if truncated {
                    out.push_str(&format!("\n[truncated at {cap} paths]"));
                }
                out
            };
            Ok(ToolOutput::text(body))
        })
    }
}

struct FindOutcome {
    paths: Vec<String>,
    truncated: bool,
}

fn find_blocking(
    root: &Path,
    regex: &Regex,
    cap: usize,
    cancel: CancellationToken,
) -> std::result::Result<FindOutcome, RikoError> {
    let mut paths = Vec::with_capacity(cap.min(64));

    for entry in WalkBuilder::new(root).follow_links(false).build() {
        if cancel.is_cancelled() {
            return Err(RikoError::Cancelled);
        }
        let Ok(entry) = entry else { continue };
        let relative = entry.path().strip_prefix(root).unwrap_or(entry.path());
        let candidate = relative.to_string_lossy();
        if candidate.is_empty() {
            continue;
        }
        if regex.is_match(&candidate) {
            paths.push(candidate.into_owned());
            if paths.len() >= cap {
                return Ok(FindOutcome { paths, truncated: true });
            }
        }
    }
    Ok(FindOutcome { paths, truncated: false })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileAccess;
    use riko_core::{Content, ToolCallId};
    use serde_json::json;
    use tempfile::TempDir;

    async fn run(root: &Path, args: serde_json::Value) -> riko_core::Result<ToolOutput> {
        let ctx = ToolContext { root: root.to_path_buf(), file_access: FileAccess::new() };
        let call = ToolCall { id: ToolCallId::fresh(), name: "find".into(), arguments: args };
        FindTool::new().run(call, ctx, CancellationToken::new()).await
    }

    #[tokio::test]
    async fn matches_paths_by_regex() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("keep.rs"), "").unwrap();
        std::fs::write(temp.path().join("skip.txt"), "").unwrap();
        let out = run(temp.path(), json!({ "pattern": "\\.rs$" })).await.unwrap();
        match &out.content[0] {
            Content::Text(text) => {
                assert!(text.contains("keep.rs"));
                assert!(!text.contains("skip.txt"));
            }
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reports_no_matches() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("a.txt"), "").unwrap();
        let out = run(temp.path(), json!({ "pattern": "\\.rs$" })).await.unwrap();
        assert_eq!(out.content, vec![Content::text("no matches")]);
    }
}
