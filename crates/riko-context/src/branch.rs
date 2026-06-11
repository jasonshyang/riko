use std::sync::Arc;

use crate::{Item, ItemId};

riko_utils::id_type!(
    /// Identifier for a workspace branch.
    BranchId, "br"
);

/// One branch's ordered items. Items are held behind `Arc` so edits are copy-on-write and
/// forks share untouched items. The branch tree and parent links arrive with branch operations.
pub struct Branch {
    pub id: BranchId,
    pub items: Vec<Arc<Item>>,
}

impl Branch {
    pub fn root(id: BranchId) -> Self {
        Self { id, items: Vec::new() }
    }

    /// Index of the item with this id, if present.
    pub fn position(&self, id: &ItemId) -> Option<usize> {
        self.items.iter().position(|item| &item.id == id)
    }
}
