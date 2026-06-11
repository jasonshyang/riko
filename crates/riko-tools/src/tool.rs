use riko_core::{Content, Result, ToolCall, ToolDescriptor};
use smol_str::SmolStr;
use std::{collections::HashMap, path::PathBuf, pin::Pin, sync::Arc};
use tokio_util::sync::CancellationToken;

use crate::FileAccess;

/// Boxed future returned by [`Tool::run`].
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + 'a>>;

/// A tool the agent can dispatch on the model's behalf. Implemented by `riko-tools`; the trait
/// is held as a registry trait object, so it stays dyn-compatible (hence the boxed future).
pub trait Tool: Send + Sync + 'static {
    /// What the model sees: name, description, and input schema.
    fn descriptor(&self) -> &ToolDescriptor;

    /// Execute one call. Returning `Err` becomes an error tool-result the model can read and
    /// react to; `cancel` must be respected for long-running work.
    fn run<'a>(
        &'a self,
        call: ToolCall,
        ctx: ToolContext,
        cancel: CancellationToken,
    ) -> ToolFuture<'a>;
}

/// Per-call execution context.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Directory that tools resolve relative paths against.
    pub root: PathBuf,
    /// Per-run record of reads, so `edit` can enforce read-before-edit.
    pub file_access: FileAccess,
}

/// A tool's successful output.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    pub content: Vec<Content>,
}

impl ToolOutput {
    /// Output consisting of a single text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self { content: vec![Content::text(text)] }
    }
}

/// The tools available to the agent, keyed by name. Built at startup, then shared immutably.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<SmolStr, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Register a tool under its declared name, replacing any prior one.
    pub fn register<T: Tool>(&mut self, tool: T) {
        let name = tool.descriptor().name.clone();
        self.tools.insert(name, Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Descriptors for every tool, sorted by name for a deterministic prompt.
    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        let mut descriptors: Vec<ToolDescriptor> =
            self.tools.values().map(|tool| tool.descriptor().clone()).collect();
        descriptors.sort_by(|a, b| a.name.cmp(&b.name));
        descriptors
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}
