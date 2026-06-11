use std::path::{Path, PathBuf};

use riko_core::{Result, RikoError};
use smol_str::SmolStr;

const FRONTMATTER_DELIMITER: &str = "---";

/// One Agent Skill discovered on disk.
///
/// Only the `SKILL.md` frontmatter metadata is held; the body stays at `body_path` and the
/// model loads it on demand via the `read` tool. That matches progressive disclosure — the
/// catalog goes into the system prompt as a seed item, the body only enters context when the
/// model actually reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: SmolStr,
    pub description: String,
    pub body_path: PathBuf,
}

impl Skill {
    pub fn new(
        name: impl Into<SmolStr>,
        description: impl Into<String>,
        body_path: PathBuf,
    ) -> Self {
        Self { name: name.into(), description: description.into(), body_path }
    }

    /// Walk each directory in `roots` and return one [`Skill`] per `<dir>/SKILL.md` found. Roots
    /// that don't exist are skipped (callers commonly pass optional per-user / per-workspace
    /// paths). Returns the catalog in deterministic name order, deduplicated by name.
    pub fn discover_skills<P: AsRef<Path>>(roots: &[P]) -> Result<Vec<Skill>> {
        let mut found = Vec::with_capacity(8);
        for root in roots {
            Self::collect_under(root.as_ref(), &mut found)?;
        }
        found.sort_by(|a, b| a.name.cmp(&b.name));
        found.dedup_by(|a, b| a.name == b.name);
        Ok(found)
    }

    fn collect_under(root: &Path, into: &mut Vec<Skill>) -> Result<()> {
        if !root.exists() {
            return Ok(());
        }
        let entries = std::fs::read_dir(root).map_err(|e| {
            RikoError::Config(format!("reading skills root {}: {e}", root.display()))
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                RikoError::Config(format!("walking skills root {}: {e}", root.display()))
            })?;
            let path = entry.path();
            let skill_file = path.join("SKILL.md");
            if path.is_dir() && skill_file.is_file() {
                into.push(Self::parse_skill_file(&skill_file)?);
            }
        }
        Ok(())
    }

    fn parse_skill_file(path: &Path) -> Result<Skill> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| RikoError::Config(format!("reading {}: {e}", path.display())))?;
        let (name, description) = Self::parse_frontmatter(&raw)
            .map_err(|e| RikoError::Config(format!("parsing {}: {e}", path.display())))?;
        Ok(Skill::new(name, description, path.to_path_buf()))
    }

    /// Pull `name:` and `description:` from a markdown file whose frontmatter is delimited by `---`
    /// lines. The body after the frontmatter is ignored (loaded later on demand). Strict: missing
    /// delimiters or required keys are errors.
    fn parse_frontmatter(raw: &str) -> std::result::Result<(String, String), String> {
        let lines: Vec<&str> = raw.trim_start_matches('\u{feff}').lines().collect();
        if lines.first().map(|l| l.trim()) != Some(FRONTMATTER_DELIMITER) {
            return Err("missing opening `---` frontmatter delimiter".into());
        }
        let end = lines[1..]
            .iter()
            .position(|l| l.trim() == FRONTMATTER_DELIMITER)
            .ok_or_else(|| "missing closing `---` frontmatter delimiter".to_string())?
            + 1;

        let mut name = None;
        let mut description = None;
        for line in &lines[1..end] {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let (key, value) = trimmed.split_once(':').ok_or_else(|| {
                format!("invalid frontmatter line `{trimmed}` — expected `key: value`")
            })?;
            let value = value.trim().trim_matches('"').trim_matches('\'').to_string();
            match key.trim().to_ascii_lowercase().as_str() {
                "name" => name = Some(value),
                "description" => description = Some(value),
                _ => {}
            }
        }
        Ok((
            name.ok_or_else(|| "missing required `name:` field".to_string())?,
            description.ok_or_else(|| "missing required `description:` field".to_string())?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_skill(dir: &Path, subdir: &str, body: &str) {
        let skill_dir = dir.join(subdir);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), body).unwrap();
    }

    #[test]
    fn parse_frontmatter_extracts_name_and_description() {
        let (name, description) =
            Skill::parse_frontmatter("---\nname: foo\ndescription: bar baz\n---\n\nbody\n")
                .unwrap();
        assert_eq!(name, "foo");
        assert_eq!(description, "bar baz");
    }

    #[test]
    fn parse_frontmatter_strips_quotes() {
        let (name, description) =
            Skill::parse_frontmatter("---\nname: \"foo\"\ndescription: 'bar'\n---\n").unwrap();
        assert_eq!(name, "foo");
        assert_eq!(description, "bar");
    }

    #[test]
    fn missing_delimiter_and_missing_field_error() {
        assert!(Skill::parse_frontmatter("name: foo\n").is_err());
        assert!(
            Skill::parse_frontmatter("---\nname: foo\n---\n").unwrap_err().contains("description")
        );
    }

    #[test]
    fn discover_walks_subdirectories_in_name_order() {
        let temp = TempDir::new().unwrap();
        write_skill(temp.path(), "beta", "---\nname: beta\ndescription: second\n---\n");
        write_skill(temp.path(), "alpha", "---\nname: alpha\ndescription: first\n---\n");
        let skills = Skill::discover_skills(&[temp.path()]).unwrap();
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name.as_str(), "alpha");
        assert_eq!(skills[1].name.as_str(), "beta");
    }

    #[test]
    fn missing_root_is_not_an_error() {
        assert!(
            Skill::discover_skills(&[PathBuf::from("/definitely/not/here")]).unwrap().is_empty()
        );
    }

    #[test]
    fn duplicate_names_are_deduplicated() {
        let temp = TempDir::new().unwrap();
        let (a, b) = (temp.path().join("a"), temp.path().join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        write_skill(&a, "same", "---\nname: same\ndescription: one\n---\n");
        write_skill(&b, "same", "---\nname: same\ndescription: two\n---\n");
        assert_eq!(Skill::discover_skills(&[a, b]).unwrap().len(), 1);
    }
}
