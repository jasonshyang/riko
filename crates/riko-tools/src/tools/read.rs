use std::path::PathBuf;
use std::sync::OnceLock;

use riko_core::{Result, RikoError, ToolCall, ToolDescriptor};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::fs;
use crate::{Tool, ToolContext, ToolFuture, ToolOutput};

/// Maximum bytes returned in one `Read` call if neither `offset` nor `limit` is set.
const DEFAULT_MAX_BYTES: usize = 256 * 1024;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadArgs {
    /// File path to read. Workspace-relative — for example `Cargo.toml` or `src/main.rs`.
    /// Absolute paths starting with `/` are treated as filesystem-absolute, NOT as "from the
    /// workspace root."
    pub path: PathBuf,
    /// Optional 1-based line to start at.
    #[serde(default)]
    pub offset: Option<u32>,
    /// Optional number of lines to return after `offset`.
    #[serde(default)]
    pub limit: Option<u32>,
}

/// Read a UTF-8 file from the workspace, optionally selecting a line range.
pub struct ReadTool;

impl Default for ReadTool {
    fn default() -> Self {
        Self
    }
}

impl Tool for ReadTool {
    fn descriptor(&self) -> &ToolDescriptor {
        static DESCRIPTOR: OnceLock<ToolDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| ToolDescriptor {
            name: "read".into(),
            description:
                "Read a UTF-8 file from the workspace. Pass a workspace-relative path like \
                          `Cargo.toml` (NOT `/Cargo.toml`). Optional offset/limit select a line \
                          range."
                    .into(),
            schema: super::schema_for::<ReadArgs>(),
        })
    }

    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        _cancel: CancellationToken,
    ) -> ToolFuture<'a> {
        Box::pin(async move {
            let args: ReadArgs = parse_args(&call)?;
            let path = fs::resolve_path(&ctx.root, &args.path);
            let body = tokio::fs::read_to_string(&path).await.map_err(|e| {
                let hint = fs::describe_path_problem(&ctx.root, &args.path);
                RikoError::Tool(format!("read {}: {e}{hint}", path.display()))
            })?;
            let selected = select_lines(&body, args.offset, args.limit);
            ctx.file_access.record_read(path);
            Ok(ToolOutput::text(selected))
        })
    }
}

fn parse_args(call: &ToolCall) -> Result<ReadArgs> {
    serde_json::from_value(call.arguments.clone())
        .map_err(|e| RikoError::InvalidArgument(format!("read tool args: {e}")))
}

fn select_lines(body: &str, offset: Option<u32>, limit: Option<u32>) -> String {
    match (offset, limit) {
        (None, None) => {
            if body.len() <= DEFAULT_MAX_BYTES {
                body.to_string()
            } else {
                let truncated = &body[..DEFAULT_MAX_BYTES];
                format!("{truncated}\n[truncated to {DEFAULT_MAX_BYTES} bytes]")
            }
        }
        _ => {
            let lines: Vec<&str> = body.lines().collect();
            let start = offset.unwrap_or(1).saturating_sub(1) as usize;
            let end = limit.map(|l| start + l as usize).unwrap_or(lines.len());
            let start = start.min(lines.len());
            let end = end.min(lines.len());
            lines[start..end].join("\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileAccess;
    use riko_core::{Content, ToolCallId};
    use serde_json::json;
    use tempfile::TempDir;

    async fn run(
        root: &std::path::Path,
        access: &FileAccess,
        args: serde_json::Value,
    ) -> Result<ToolOutput> {
        let ctx = ToolContext { root: root.to_path_buf(), file_access: access.clone() };
        let call = ToolCall { id: ToolCallId::fresh(), name: "read".into(), arguments: args };
        ReadTool.run(call, ctx, CancellationToken::new()).await
    }

    #[test]
    fn select_lines_returns_full_body_when_no_range() {
        assert_eq!(select_lines("a\nb\nc", None, None), "a\nb\nc");
    }

    #[test]
    fn select_lines_honors_offset_and_limit() {
        assert_eq!(select_lines("a\nb\nc\nd", Some(2), Some(2)), "b\nc");
    }

    #[test]
    fn select_lines_clamps_to_end() {
        assert_eq!(select_lines("a\nb", Some(1), Some(100)), "a\nb");
    }

    #[tokio::test]
    async fn reads_a_file_and_records_the_read() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("hello.txt"), "hi there").unwrap();
        let access = FileAccess::new();
        let out = run(temp.path(), &access, json!({ "path": "hello.txt" })).await.unwrap();
        assert_eq!(out.content, vec![Content::text("hi there")]);
        assert!(access.has_read(&temp.path().join("hello.txt")), "read must be recorded");
    }

    #[tokio::test]
    async fn missing_file_is_a_tool_error() {
        let temp = TempDir::new().unwrap();
        let err =
            run(temp.path(), &FileAccess::new(), json!({ "path": "nope.txt" })).await.unwrap_err();
        assert!(matches!(err, RikoError::Tool(_)));
    }
}
