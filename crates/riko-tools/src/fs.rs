use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Per-run record of which paths have been read, shared across the tool calls in a single
/// agent run. Cheap to clone — clones share one backing set. `read` records into it; `edit`
/// consults it to refuse editing a file the model hasn't read this run; the agent resets it at
/// the start of every run so a read from a prior run can't authorize an edit.
#[derive(Debug, Clone, Default)]
pub struct FileAccess {
    reads: Arc<parking_lot::Mutex<HashSet<PathBuf>>>,
}

impl FileAccess {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `path` was read. Idempotent.
    pub fn record_read(&self, path: PathBuf) {
        self.reads.lock().insert(path);
    }

    /// True if `path` has been read since the last [`reset`](Self::reset).
    pub fn has_read(&self, path: &Path) -> bool {
        self.reads.lock().contains(path)
    }

    /// Forget every recorded read — the agent calls this at the start of each run.
    pub fn reset(&self) {
        self.reads.lock().clear();
    }
}

/// Resolve `path` against `root` in a way that tolerates the most common path confusion small
/// models make: writing `/src` to mean "the `src/` dir inside the workspace" instead of the
/// filesystem-absolute `/src`.
///
/// Rules:
/// 1. Relative paths (`src`, `./src`, `tests/data`) → joined onto `root`.
/// 2. Absolute paths that exist as-is on disk → used unchanged (so `/etc/hosts` keeps working
///    for callers who genuinely mean it).
/// 3. Absolute paths that **don't** exist as-is, with their leading `/` stripped, matched
///    against `root` → use the workspace-relative version.
/// 4. Anything else → pass through the original; the caller surfaces the filesystem error with
///    [`describe_path_problem`] attached.
pub(crate) fn resolve_path(root: &Path, path: &Path) -> PathBuf {
    if !path.is_absolute() {
        return root.join(path);
    }
    if path.exists() {
        return path.to_path_buf();
    }
    if let Ok(stripped) = path.strip_prefix("/") {
        let candidate = root.join(stripped);
        if candidate.exists() {
            return candidate;
        }
    }
    path.to_path_buf()
}

/// Build a hint that tool error messages append when an absolute-looking path fails — gently
/// steering the model toward workspace-relative usage on the retry.
pub(crate) fn describe_path_problem(root: &Path, attempted: &Path) -> String {
    if !attempted.is_absolute() {
        return String::new();
    }
    if let Ok(stripped) = attempted.strip_prefix("/") {
        let candidate = root.join(stripped);
        if candidate.exists() {
            return format!(
                " — hint: `{}` is filesystem-absolute. Did you mean the workspace-relative \
                 `{}`?",
                attempted.display(),
                stripped.display()
            );
        }
    }
    format!(
        " — hint: paths are workspace-relative by default. The workspace is `{}`. Avoid \
         leading-slash paths unless you really mean a filesystem-absolute target.",
        root.display()
    )
}

#[cfg(test)]
mod tests {
    use riko_core::Content;
    use tempfile::TempDir;

    use crate::ToolOutput;

    use super::*;

    #[test]
    fn records_and_checks_reads() {
        let access = FileAccess::new();
        let path = PathBuf::from("/work/src/lib.rs");
        assert!(!access.has_read(&path));
        access.record_read(path.clone());
        assert!(access.has_read(&path));
    }

    #[test]
    fn clones_share_one_set() {
        let access = FileAccess::new();
        let other = access.clone();
        access.record_read(PathBuf::from("/a"));
        assert!(other.has_read(Path::new("/a")), "a clone observes the same reads");
    }

    #[test]
    fn reset_forgets_reads() {
        let access = FileAccess::new();
        access.record_read(PathBuf::from("/a"));
        access.reset();
        assert!(!access.has_read(Path::new("/a")));
    }

    #[test]
    fn text_output_is_one_text_block() {
        let out = ToolOutput::text("hi");
        assert_eq!(out.content, vec![Content::text("hi")]);
    }

    #[test]
    fn relative_path_joins_root() {
        let root = PathBuf::from("/work");
        assert_eq!(resolve_path(&root, Path::new("src/lib.rs")), PathBuf::from("/work/src/lib.rs"));
    }

    #[test]
    fn existing_absolute_path_passes_through() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("file.txt"), "x").unwrap();
        let root = PathBuf::from("/some/other/workspace");
        let abs = temp.path().join("file.txt");
        assert_eq!(resolve_path(&root, &abs), abs);
    }

    #[test]
    fn missing_absolute_falls_back_to_relative_when_that_exists() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        assert_eq!(resolve_path(temp.path(), Path::new("/src")), temp.path().join("src"));
    }

    #[test]
    fn missing_absolute_with_no_match_returns_original() {
        let temp = TempDir::new().unwrap();
        let abs = Path::new("/definitely/not/here");
        assert_eq!(resolve_path(temp.path(), abs), PathBuf::from("/definitely/not/here"));
    }

    #[test]
    fn describe_problem_is_empty_for_relative_path() {
        let root = PathBuf::from("/work");
        assert!(describe_path_problem(&root, Path::new("src/lib.rs")).is_empty());
    }

    #[test]
    fn describe_problem_suggests_relative_when_match_exists() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        let hint = describe_path_problem(temp.path(), Path::new("/src"));
        assert!(hint.contains("workspace-relative"), "got: {hint}");
    }
}
