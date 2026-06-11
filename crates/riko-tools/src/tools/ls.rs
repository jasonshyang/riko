use std::path::PathBuf;
use std::sync::OnceLock;

use riko_core::{RikoError, ToolCall, ToolDescriptor};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::fs;
use crate::{Tool, ToolContext, ToolFuture, ToolOutput};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LsArgs {
    /// Directory to list. Workspace-relative — for example `src` — or omit to list the
    /// workspace root. Absolute paths starting with `/` are treated as filesystem-absolute and
    /// only work for real filesystem targets (e.g. `/etc`); they do NOT mean "from the
    /// workspace root".
    #[serde(default)]
    pub path: Option<PathBuf>,
}

/// List entries in a workspace directory with type and size annotations.
pub struct LsTool;

impl Default for LsTool {
    fn default() -> Self {
        Self
    }
}

impl Tool for LsTool {
    fn descriptor(&self) -> &ToolDescriptor {
        static DESCRIPTOR: OnceLock<ToolDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| ToolDescriptor {
            name: "ls".into(),
            description:
                "List entries in a workspace directory with type and size annotations. Omit \
                          `path` to list the workspace root; otherwise pass a workspace-relative \
                          path like `src` (NOT `/src`)."
                    .into(),
            schema: super::schema_for::<LsArgs>(),
        })
    }

    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        _cancel: CancellationToken,
    ) -> ToolFuture<'a> {
        Box::pin(async move {
            let args: LsArgs = serde_json::from_value(call.arguments)
                .map_err(|e| RikoError::InvalidArgument(format!("ls tool args: {e}")))?;
            let target = match args.path.as_deref() {
                Some(p) => fs::resolve_path(&ctx.root, p),
                None => ctx.root.clone(),
            };

            let mut entries = tokio::fs::read_dir(&target).await.map_err(|e| {
                let hint = match args.path.as_deref() {
                    Some(requested) => fs::describe_path_problem(&ctx.root, requested),
                    None => String::new(),
                };
                RikoError::Tool(format!("ls {}: {e}{hint}", target.display()))
            })?;

            let mut rows = Vec::with_capacity(32);
            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| RikoError::Tool(format!("ls iter {}: {e}", target.display())))?
            {
                let name = entry.file_name().to_string_lossy().into_owned();
                let metadata = entry.metadata().await.map_err(|e| {
                    RikoError::Tool(format!("ls stat {}: {e}", entry.path().display()))
                })?;
                let kind = if metadata.is_dir() {
                    "dir"
                } else if metadata.is_symlink() {
                    "link"
                } else {
                    "file"
                };
                let size = if metadata.is_file() { metadata.len() } else { 0 };
                rows.push((name, kind, size));
            }
            rows.sort_by(|a, b| a.0.cmp(&b.0));

            let body = rows
                .iter()
                .map(|(name, kind, size)| format!("{kind:>4} {size:>10}  {name}"))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(ToolOutput::text(body))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileAccess;
    use riko_core::ToolCallId;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn lists_workspace_root_sorted() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("b.txt"), "bb").unwrap();
        std::fs::create_dir(temp.path().join("a_dir")).unwrap();
        let ctx = ToolContext { root: temp.path().to_path_buf(), file_access: FileAccess::new() };
        let call = ToolCall { id: ToolCallId::fresh(), name: "ls".into(), arguments: json!({}) };
        let out = LsTool.run(call, ctx, CancellationToken::new()).await.unwrap();
        match &out.content[0] {
            riko_core::Content::Text(text) => {
                let a = text.find("a_dir").expect("a_dir listed");
                let b = text.find("b.txt").expect("b.txt listed");
                assert!(a < b, "entries are name-sorted");
                assert!(text.contains("dir"));
            }
            other => panic!("expected text, got {other:?}"),
        }
    }
}
