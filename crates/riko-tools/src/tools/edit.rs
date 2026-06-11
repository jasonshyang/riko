use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use riko_core::{RikoError, ToolCall, ToolDescriptor};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::fs;
use crate::{Tool, ToolContext, ToolFuture, ToolOutput};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditArgs {
    /// File to edit. Workspace-relative — for example `src/main.rs`. Absolute paths starting
    /// with `/` are treated as filesystem-absolute, NOT as "from the workspace root."
    pub path: PathBuf,
    /// Exact text to find. Must match a single occurrence unless `replace_all` is set.
    pub old_string: String,
    /// Replacement text.
    pub new_string: String,
    /// Replace every occurrence instead of requiring uniqueness.
    #[serde(default)]
    pub replace_all: bool,
}

/// Apply an exact string substitution to a workspace file the model has already read this run.
pub struct EditTool;

impl Default for EditTool {
    fn default() -> Self {
        Self
    }
}

impl Tool for EditTool {
    fn descriptor(&self) -> &ToolDescriptor {
        static DESCRIPTOR: OnceLock<ToolDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| ToolDescriptor {
            name: "edit".into(),
            description: "Apply an exact string substitution to a previously-read workspace file. \
                          old_string must match exactly once unless replace_all=true."
                .into(),
            schema: super::schema_for::<EditArgs>(),
        })
    }

    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        _cancel: CancellationToken,
    ) -> ToolFuture<'a> {
        Box::pin(async move {
            let args: EditArgs = serde_json::from_value(call.arguments)
                .map_err(|e| RikoError::InvalidArgument(format!("edit tool args: {e}")))?;
            let path = fs::resolve_path(&ctx.root, &args.path);

            if !ctx.file_access.has_read(&path) {
                return Err(RikoError::Tool(format!(
                    "refusing to edit {} — it has not been Read this run",
                    path.display()
                )));
            }

            let original = tokio::fs::read_to_string(&path).await.map_err(|e| {
                let hint = fs::describe_path_problem(&ctx.root, &args.path);
                RikoError::Tool(format!("edit read {}: {e}{hint}", path.display()))
            })?;

            let new_contents =
                apply_edit(&original, &args.old_string, &args.new_string, args.replace_all)
                    .map_err(|reason| {
                        RikoError::Tool(format!("edit {}: {reason}", path.display()))
                    })?;

            tokio::fs::write(&path, &new_contents)
                .await
                .map_err(|e| RikoError::Tool(format!("edit write {}: {e}", path.display())))?;

            Ok(ToolOutput::text(unified_diff(&path, &original, &new_contents)))
        })
    }
}

/// Build a small unified diff (three lines of context) from `before` to `after`. The header
/// carries the file path so the model knows which file changed.
fn unified_diff(path: &Path, before: &str, after: &str) -> String {
    let diff = similar::TextDiff::from_lines(before, after);
    let label = path.display().to_string();
    let mut udiff = diff
        .unified_diff()
        .header(&format!("{label} (before)"), &format!("{label} (after)"))
        .context_radius(3)
        .to_string();
    if udiff.is_empty() {
        udiff = format!("--- {label}\n+++ {label}\n(no textual change)");
    }
    udiff
}

/// Returns the rewritten file contents, or an explanatory message for the model on failure.
fn apply_edit(
    original: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> std::result::Result<String, String> {
    if old.is_empty() {
        return Err("old_string must not be empty".into());
    }
    let matches = original.matches(old).count();
    if matches == 0 {
        return Err("old_string not found".into());
    }
    if !replace_all && matches > 1 {
        return Err(format!(
            "old_string is not unique ({matches} matches); pass replace_all=true to replace \
             every match"
        ));
    }
    Ok(if replace_all { original.replace(old, new) } else { original.replacen(old, new, 1) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileAccess;
    use riko_core::ToolCallId;
    use serde_json::json;
    use tempfile::TempDir;

    async fn run(
        root: &Path,
        access: &FileAccess,
        args: serde_json::Value,
    ) -> riko_core::Result<ToolOutput> {
        let ctx = ToolContext { root: root.to_path_buf(), file_access: access.clone() };
        let call = ToolCall { id: ToolCallId::fresh(), name: "edit".into(), arguments: args };
        EditTool.run(call, ctx, CancellationToken::new()).await
    }

    #[test]
    fn unique_match_replaces_once() {
        assert_eq!(apply_edit("foo bar baz", "bar", "QUX", false).unwrap(), "foo QUX baz");
    }

    #[test]
    fn missing_match_errors() {
        assert!(apply_edit("foo", "missing", "x", false).unwrap_err().contains("not found"));
    }

    #[test]
    fn ambiguous_match_errors_without_replace_all() {
        assert!(apply_edit("aaa", "a", "b", false).unwrap_err().contains("not unique"));
    }

    #[test]
    fn replace_all_handles_ambiguous_match() {
        assert_eq!(apply_edit("aaa", "a", "b", true).unwrap(), "bbb");
    }

    #[test]
    fn empty_old_string_errors() {
        assert!(apply_edit("abc", "", "x", false).unwrap_err().contains("must not be empty"));
    }

    #[tokio::test]
    async fn edit_requires_a_prior_read() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("f.txt"), "hello world").unwrap();
        let access = FileAccess::new();
        // No read recorded yet → refused.
        let err = run(
            temp.path(),
            &access,
            json!({ "path": "f.txt", "old_string": "world", "new_string": "riko" }),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RikoError::Tool(_)));

        // Record the read, then the same edit applies.
        access.record_read(temp.path().join("f.txt"));
        run(
            temp.path(),
            &access,
            json!({ "path": "f.txt", "old_string": "world", "new_string": "riko" }),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(temp.path().join("f.txt")).unwrap(), "hello riko");
    }
}
