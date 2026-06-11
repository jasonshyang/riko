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

#[cfg(test)]
mod tests {
    use riko_core::Content;

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
}
