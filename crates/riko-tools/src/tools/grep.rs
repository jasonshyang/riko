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

const DEFAULT_MAX_RESULTS: u32 = 200;
const MAX_LINE_BYTES: usize = 4 * 1024;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrepArgs {
    /// Regex pattern, matched per line.
    pub pattern: String,
    /// Optional search root. Defaults to the workspace root.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Cap on the number of matching lines returned. Defaults to 200.
    #[serde(default)]
    pub max_results: Option<u32>,
}

/// Search workspace files for a regex pattern, honoring .gitignore.
pub struct GrepTool;

impl Default for GrepTool {
    fn default() -> Self {
        Self
    }
}

impl Tool for GrepTool {
    fn descriptor(&self) -> &ToolDescriptor {
        static DESCRIPTOR: OnceLock<ToolDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| ToolDescriptor {
            name: "grep".into(),
            description: "Search workspace files for a regex pattern. Honors .gitignore. Lines \
                          longer than 4 KiB are truncated; total results capped to `max_results` \
                          (default 200)."
                .into(),
            schema: super::schema_for::<GrepArgs>(),
        })
    }

    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        cancel: CancellationToken,
    ) -> ToolFuture<'a> {
        Box::pin(async move {
            let args: GrepArgs = serde_json::from_value(call.arguments)
                .map_err(|e| RikoError::InvalidArgument(format!("grep tool args: {e}")))?;
            let regex = Regex::new(&args.pattern)
                .map_err(|e| RikoError::InvalidArgument(format!("grep pattern: {e}")))?;
            let root = match args.path {
                Some(p) => fs::resolve_path(&ctx.root, &p),
                None => ctx.root.clone(),
            };
            let cap = args.max_results.unwrap_or(DEFAULT_MAX_RESULTS) as usize;

            // Walking + reading is blocking — push it off the async runtime.
            let cancel_for_blocking = cancel.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                grep_blocking(&root, &regex, cap, cancel_for_blocking)
            })
            .await
            .map_err(|e| RikoError::Tool(format!("grep task join failed: {e}")))?;

            if cancel.is_cancelled() {
                return Err(RikoError::Cancelled);
            }

            let GrepOutcome { hits, truncated } = outcome?;
            let body = if hits.is_empty() {
                "no matches".to_string()
            } else {
                let mut out = hits.join("\n");
                if truncated {
                    out.push_str(&format!("\n[truncated at {cap} matches]"));
                }
                out
            };
            Ok(ToolOutput::text(body))
        })
    }
}

struct GrepOutcome {
    hits: Vec<String>,
    truncated: bool,
}

fn grep_blocking(
    root: &Path,
    regex: &Regex,
    cap: usize,
    cancel: CancellationToken,
) -> std::result::Result<GrepOutcome, RikoError> {
    let mut hits = Vec::with_capacity(cap.min(64));

    for entry in WalkBuilder::new(root).follow_links(false).build() {
        if cancel.is_cancelled() {
            return Err(RikoError::Cancelled);
        }
        let Ok(entry) = entry else { continue };
        let Some(ft) = entry.file_type() else { continue };
        if !ft.is_file() {
            continue;
        }
        let path = entry.path();
        let Ok(content) = std::fs::read_to_string(path) else { continue };
        for (line_no, line) in content.lines().enumerate() {
            if regex.is_match(line) {
                let shown = if line.len() > MAX_LINE_BYTES {
                    format!("{}…", &line[..MAX_LINE_BYTES])
                } else {
                    line.to_string()
                };
                hits.push(format!("{}:{}: {}", path.display(), line_no + 1, shown));
                if hits.len() >= cap {
                    return Ok(GrepOutcome { hits, truncated: true });
                }
            }
        }
    }
    Ok(GrepOutcome { hits, truncated: false })
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
        let call = ToolCall { id: ToolCallId::fresh(), name: "grep".into(), arguments: args };
        GrepTool.run(call, ctx, CancellationToken::new()).await
    }

    #[tokio::test]
    async fn finds_matching_lines() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("a.txt"), "alpha\nbeta\ngamma").unwrap();
        let out = run(temp.path(), json!({ "pattern": "^beta$" })).await.unwrap();
        match &out.content[0] {
            Content::Text(text) => {
                assert!(text.contains("a.txt"));
                assert!(text.contains(":2: beta"), "line number and content: {text}");
            }
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reports_no_matches() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("a.txt"), "alpha").unwrap();
        let out = run(temp.path(), json!({ "pattern": "zzz" })).await.unwrap();
        assert_eq!(out.content, vec![Content::text("no matches")]);
    }

    #[tokio::test]
    async fn invalid_pattern_is_an_argument_error() {
        let temp = TempDir::new().unwrap();
        let err = run(temp.path(), json!({ "pattern": "(" })).await.unwrap_err();
        assert!(matches!(err, RikoError::InvalidArgument(_)));
    }
}
