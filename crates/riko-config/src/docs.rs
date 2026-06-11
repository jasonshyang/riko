use std::path::Path;

/// Filenames searched in each ancestor directory, in precedence order: when more than one
/// exists in a directory, the first listed wins.
const FILENAMES: &[&str] = &["AGENTS.md", "CLAUDE.md", "RIKO.md"];

/// Hard cap on ancestors walked — workspace + ~7 parents covers any realistic tree.
const MAX_DEPTH: usize = 8;

/// Walk `root` and its ancestors (stopping at `$HOME` or [`MAX_DEPTH`], whichever is shallower)
/// and return the concatenated contents of the project-doc files found, each prefixed with a
/// `--- <path> ---` header. Returns `None` when nothing matches.
pub fn discover_project_docs(root: &Path) -> Option<String> {
    let stop_at = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let mut combined = String::new();
    for ancestor in root.ancestors().take(MAX_DEPTH) {
        for name in FILENAMES {
            if let Ok(body) = std::fs::read_to_string(ancestor.join(name)) {
                if !combined.is_empty() {
                    combined.push_str("\n\n");
                }
                combined.push_str(&format!("--- {} ---\n", ancestor.join(name).display()));
                combined.push_str(body.trim_end());
                break; // first hit per directory wins
            }
        }
        if stop_at.as_deref() == Some(ancestor) {
            break;
        }
    }
    if combined.is_empty() { None } else { Some(combined) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn includes_matching_files_walking_upward() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join("a").join("AGENTS.md"), "parent agents").unwrap();
        std::fs::write(nested.join("RIKO.md"), "workspace riko").unwrap();
        let docs = discover_project_docs(&nested).unwrap();
        assert!(docs.contains("workspace riko"));
        assert!(docs.contains("parent agents"));
        assert!(docs.contains("RIKO.md"));
        assert!(docs.contains("AGENTS.md"));
    }

    #[test]
    fn no_docs_yields_none() {
        let dir = TempDir::new().unwrap();
        assert!(discover_project_docs(dir.path()).is_none());
    }

    #[test]
    fn first_match_per_directory_wins() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "agents body").unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "claude body").unwrap();
        let docs = discover_project_docs(dir.path()).unwrap();
        assert!(docs.contains("agents body"));
        assert!(!docs.contains("claude body"));
    }
}
