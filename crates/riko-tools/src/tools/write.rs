use riko_core::{RikoError, ToolCall, ToolDescriptor};
use schemars::JsonSchema;
use serde::Deserialize;
use std::{path::PathBuf, sync::OnceLock};
use tokio_util::sync::CancellationToken;

use crate::{Tool, ToolContext, ToolFuture, ToolOutput, fs};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriteArgs {
    /// File path to create or overwrite. Workspace-relative — for example `src/main.rs`.
    /// Absolute paths starting with `/` are treated as filesystem-absolute, NOT as "from the
    /// workspace root."
    pub path: PathBuf,
    /// File contents.
    pub contents: String,
}

/// Create or overwrite a UTF-8 file in the workspace, creating parent directories as needed.
pub struct WriteTool;

impl Tool for WriteTool {
    fn descriptor(&self) -> &ToolDescriptor {
        static DESCRIPTOR: OnceLock<ToolDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| ToolDescriptor {
            name: "write".into(),
            description: "Create or overwrite a UTF-8 file in the workspace. Pass a \
                          workspace-relative path like `src/main.rs` (NOT `/src/main.rs`)."
                .into(),
            schema: super::schema_for::<WriteArgs>(),
        })
    }

    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        _cancel: CancellationToken,
    ) -> ToolFuture<'a> {
        Box::pin(async move {
            let args: WriteArgs = serde_json::from_value(call.arguments)
                .map_err(|e| RikoError::InvalidArgument(format!("write tool args: {e}")))?;
            let path = fs::resolve_path(&ctx.root, &args.path);

            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    RikoError::Tool(format!("create parent {}: {e}", parent.display()))
                })?;
            }
            let bytes = args.contents.len();
            tokio::fs::write(&path, &args.contents).await.map_err(|e| {
                let hint = fs::describe_path_problem(&ctx.root, &args.path);
                RikoError::Tool(format!("write {}: {e}{hint}", path.display()))
            })?;

            Ok(ToolOutput::text(format!("wrote {bytes} bytes to {}", path.display())))
        })
    }
}

impl Default for WriteTool {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileAccess;
    use riko_core::ToolCallId;
    use serde_json::json;
    use tempfile::TempDir;

    async fn run(root: &std::path::Path, args: serde_json::Value) -> riko_core::Result<ToolOutput> {
        let ctx = ToolContext { root: root.to_path_buf(), file_access: FileAccess::new() };
        let call = ToolCall { id: ToolCallId::fresh(), name: "write".into(), arguments: args };
        WriteTool.run(call, ctx, CancellationToken::new()).await
    }

    #[tokio::test]
    async fn writes_a_file_creating_parent_dirs() {
        let temp = TempDir::new().unwrap();
        run(temp.path(), json!({ "path": "nested/dir/out.txt", "contents": "payload" }))
            .await
            .unwrap();
        let written = std::fs::read_to_string(temp.path().join("nested/dir/out.txt")).unwrap();
        assert_eq!(written, "payload");
    }

    #[tokio::test]
    async fn overwrites_an_existing_file() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("f.txt"), "old").unwrap();
        run(temp.path(), json!({ "path": "f.txt", "contents": "new" })).await.unwrap();
        assert_eq!(std::fs::read_to_string(temp.path().join("f.txt")).unwrap(), "new");
    }
}
