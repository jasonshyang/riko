use std::path::{Path, PathBuf};

use riko_core::{Result, RikoError};

use crate::settings::Settings;

/// Where [`load`] looks for settings files. Either path may be `None` (skipped). The
/// workspace file's settings deep-merge on top of the user file's.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub user_file: Option<PathBuf>,
    pub workspace_file: Option<PathBuf>,
}

/// Outcome of a [`load`] call. `settings` are always returned (possibly default); the per-file
/// flags let callers report what was actually found.
#[derive(Debug, Clone)]
pub struct LoadOutcome {
    pub settings: Settings,
    pub user_loaded: bool,
    pub workspace_loaded: bool,
}

impl LoadOptions {
    /// Read and merge the configured layers. Missing files are NOT errors — they're reported via
    /// the per-file flags. Parse errors and env-interpolation failures ARE surfaced loudly so
    /// callers can fail at startup.
    pub fn load(&self) -> Result<LoadOutcome> {
        let mut settings = Settings::default();
        let user_loaded = Self::read_into(&mut settings, self.user_file.as_deref())?;
        let workspace_loaded = Self::read_into(&mut settings, self.workspace_file.as_deref())?;
        Ok(LoadOutcome { settings, user_loaded, workspace_loaded })
    }

    fn read_into(base: &mut Settings, path: Option<&Path>) -> Result<bool> {
        let Some(path) = path else { return Ok(false) };
        let raw = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(RikoError::Config(format!("reading {}: {e}", path.display()))),
        };
        let interpolated = Self::interpolate_env(&raw)
            .map_err(|e| RikoError::Config(format!("interpolating {}: {e}", path.display())))?;
        let overlay: Settings = toml::from_str(&interpolated)
            .map_err(|e| RikoError::Config(format!("parsing {}: {e}", path.display())))?;
        base.merge(overlay);
        Ok(true)
    }

    /// Replace every `${VAR}` with the value of that environment variable.
    fn interpolate_env(input: &str) -> std::result::Result<String, String> {
        let mut out = String::with_capacity(input.len());
        let mut rest = input;
        while let Some(idx) = rest.find("${") {
            out.push_str(&rest[..idx]);
            let after = &rest[idx + 2..];
            let end = after.find('}').ok_or_else(|| {
                "unterminated `${` reference — every interpolation must end with `}`".to_string()
            })?;
            let var = &after[..end];
            if var.is_empty() {
                return Err("empty `${}` reference".into());
            }
            let value = std::env::var(var)
                .map_err(|_| format!("environment variable `{var}` is not set"))?;
            out.push_str(&value);
            rest = &after[end + 1..];
        }
        out.push_str(rest);
        Ok(out)
    }
}
