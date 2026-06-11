use std::sync::OnceLock;
use std::time::Duration;

use riko_core::{RikoError, ToolCall, ToolDescriptor};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::{Tool, ToolContext, ToolFuture, ToolOutput};

/// Wall-time cap when the model omits a timeout.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Hard ceiling — even an explicit `timeout_secs` is clamped to this.
const MAX_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BashArgs {
    /// Command line executed via `/bin/sh -c "..."` in the workspace cwd.
    pub command: String,
    /// Optional per-call wall-time cap, in seconds. Clamped to [1, 600].
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

/// Run a shell command in the workspace, capturing stdout/stderr and the exit code.
pub struct BashTool;

impl Default for BashTool {
    fn default() -> Self {
        Self
    }
}

impl Tool for BashTool {
    fn descriptor(&self) -> &ToolDescriptor {
        static DESCRIPTOR: OnceLock<ToolDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| ToolDescriptor {
            name: "bash".into(),
            description:
                "Run a shell command in the workspace cwd via /bin/sh -c. Captures stdout \
                          and stderr; per-call timeout (default 120s, max 600s)."
                    .into(),
            schema: super::schema_for::<BashArgs>(),
        })
    }

    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        cancel: CancellationToken,
    ) -> ToolFuture<'a> {
        Box::pin(async move {
            let args: BashArgs = serde_json::from_value(call.arguments)
                .map_err(|e| RikoError::InvalidArgument(format!("bash tool args: {e}")))?;
            let timeout = clamp_timeout(args.timeout_secs);

            let cmd_fut = Command::new("/bin/sh")
                .arg("-c")
                .arg(&args.command)
                .current_dir(&ctx.root)
                .output();

            let outcome = tokio::select! {
                res = tokio::time::timeout(timeout, cmd_fut) => res,
                _ = cancel.cancelled() => return Err(RikoError::Cancelled),
            };

            let output = match outcome {
                Ok(Ok(out)) => out,
                Ok(Err(e)) => return Err(RikoError::Tool(format!("bash spawn failed: {e}"))),
                Err(_) => {
                    return Err(RikoError::Tool(format!(
                        "bash timed out after {}s",
                        timeout.as_secs()
                    )));
                }
            };

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);
            Ok(ToolOutput::text(format_output(&stdout, &stderr, exit_code)))
        })
    }
}

fn clamp_timeout(secs: Option<u64>) -> Duration {
    let raw = secs.unwrap_or(DEFAULT_TIMEOUT_SECS);
    Duration::from_secs(raw.clamp(1, MAX_TIMEOUT_SECS))
}

fn format_output(stdout: &str, stderr: &str, exit_code: i32) -> String {
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => format!("[exit {exit_code}]"),
        (false, true) => format!("{stdout}\n[exit {exit_code}]"),
        (true, false) => format!("STDERR:\n{stderr}\n[exit {exit_code}]"),
        (false, false) => format!("{stdout}\nSTDERR:\n{stderr}\n[exit {exit_code}]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileAccess;
    use riko_core::ToolCallId;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn clamp_uses_default_when_unset() {
        assert_eq!(clamp_timeout(None), Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    }

    #[test]
    fn clamp_caps_at_max() {
        assert_eq!(clamp_timeout(Some(99_999)), Duration::from_secs(MAX_TIMEOUT_SECS));
    }

    #[test]
    fn clamp_lifts_zero_to_one() {
        assert_eq!(clamp_timeout(Some(0)), Duration::from_secs(1));
    }

    #[test]
    fn format_output_shapes_match_each_combination() {
        assert_eq!(format_output("", "", 0), "[exit 0]");
        assert_eq!(format_output("hi", "", 0), "hi\n[exit 0]");
        assert_eq!(format_output("", "warn", 1), "STDERR:\nwarn\n[exit 1]");
        assert_eq!(format_output("ok", "warn", 1), "ok\nSTDERR:\nwarn\n[exit 1]");
    }

    #[tokio::test]
    async fn runs_a_command_in_the_workspace_cwd() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("marker.txt"), "x").unwrap();
        let ctx = ToolContext { root: temp.path().to_path_buf(), file_access: FileAccess::new() };
        let call = ToolCall {
            id: ToolCallId::fresh(),
            name: "bash".into(),
            arguments: json!({ "command": "ls" }),
        };
        let out = BashTool.run(call, ctx, CancellationToken::new()).await.unwrap();
        match &out.content[0] {
            riko_core::Content::Text(text) => {
                assert!(text.contains("marker.txt"), "cwd listing should show the marker: {text}");
                assert!(text.contains("[exit 0]"));
            }
            other => panic!("expected text, got {other:?}"),
        }
    }
}
