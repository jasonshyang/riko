use std::sync::Arc;

use smol_str::SmolStr;

use crate::{Item, ItemId};

riko_utils::id_type!(
    /// Identifier for a workspace branch.
    BranchId, "br"
);

/// One branch's ordered items. Items are held behind `Arc` so edits are copy-on-write and
/// forks share untouched items. The branch tree and parent links arrive with branch operations.
pub struct Branch {
    pub id: BranchId,
    pub parent: Option<BranchId>,
    pub label: Option<SmolStr>,
    pub items: Vec<Arc<Item>>,
}

impl Branch {
    pub(crate) fn root(id: BranchId) -> Self {
        Self { id, parent: None, label: Some(SmolStr::new_static("main")), items: Vec::new() }
    }

    pub(crate) fn fork(
        id: BranchId,
        parent: BranchId,
        label: Option<SmolStr>,
        items: Vec<Arc<Item>>,
    ) -> Self {
        Self { id, parent: Some(parent), label, items }
    }

    /// Index of the item with this id, if present.
    pub fn position(&self, id: &ItemId) -> Option<usize> {
        self.items.iter().position(|item| &item.id == id)
    }

    pub(crate) fn info(&self, active: bool) -> BranchInfo {
        BranchInfo {
            id: self.id.clone(),
            parent: self.parent.clone(),
            label: self.label.clone(),
            active,
        }
    }
}

/// Read-only view of a branch's place in the tree, for listing branches.
#[derive(Debug, Clone)]
pub struct BranchInfo {
    pub id: BranchId,
    pub parent: Option<BranchId>,
    pub label: Option<SmolStr>,
    pub active: bool,
}
