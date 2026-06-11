use std::path::{Path, PathBuf};

use riko_context::{ItemPayload, Workspace};
use riko_core::Result;
use smol_str::SmolStr;

use crate::docs::discover_project_docs;
use crate::skills::Skill;

/// Base agent instructions, seeded as the lead (untagged) System item.
const BASE_INSTRUCTIONS: &str = "You are riko, a focused coding agent working in the user's \
    workspace. Use the built-in tools to inspect and modify files. Paths are workspace-relative \
    by default. Keep responses concise and act directly rather than narrating each step.";

pub fn bootstrap_workspace(
    workspace: &Workspace,
    root: &Path,
    skill_roots: &[PathBuf],
    fresh: bool,
) -> Result<Vec<Skill>> {
    let skills = Skill::discover_skills(skill_roots)?;

    if fresh {
        let docs = discover_project_docs(root);
        for item in startup_items(root, docs.as_deref(), &skills) {
            workspace.add(item)?;
        }
    }
    Ok(skills)
}

fn startup_items(root: &Path, docs: Option<&str>, skills: &[Skill]) -> Vec<ItemPayload> {
    let mut items = Vec::with_capacity(4);
    items.push(system(None, BASE_INSTRUCTIONS.to_string()));
    if let Some(docs) = docs {
        items.push(system(Some("project_context"), docs.to_string()));
    }
    items.push(system(Some("env"), environment(root)));
    if let Some(catalog) = skills_catalog(skills) {
        items.push(system(Some("skills"), catalog));
    }
    items
}

fn system(tag: Option<&str>, text: String) -> ItemPayload {
    ItemPayload::System { tag: tag.map(SmolStr::from), text }
}

fn environment(root: &Path) -> String {
    let date = riko_utils::Timestamp::now().format("%Y-%m-%d").to_string();
    format!("date: {date}\ncwd: {}", root.display())
}

fn skills_catalog(skills: &[Skill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut body = String::with_capacity(skills.len() * 80);
    body.push_str(
        "Available skills (names + descriptions). Read the SKILL.md at the listed path to load a \
             skill's body before using it.",
    );
    for skill in skills {
        body.push_str(&format!(
            "\n  - {} — {} ({})",
            skill.name,
            skill.description,
            skill.body_path.display()
        ));
    }
    Some(body)
}
