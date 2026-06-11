use std::sync::Arc;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::{BranchId, Item, ItemId, ItemMeta, ItemPayload};

/// The only way to durably mutate a workspace.
///
/// Every operation is fully resolved — ids and timestamps are minted by the caller-facing
/// helpers *before* the op is built — so the operation log replays deterministically and
/// [`Workspace::apply`](crate::Workspace::apply) never mints anything itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Append an item to the active branch.
    Add { item: Arc<Item> },
    /// Replace the payload of an existing item, keeping its id and creation time.
    Edit { id: ItemId, payload: ItemPayload },
    /// Remove an item from the active branch.
    Drop { id: ItemId },
    /// Replace an item's metadata.
    SetMeta { id: ItemId, meta: ItemMeta },
    /// Create a new branch off the active one, sharing its items.
    Fork { new: BranchId, label: Option<SmolStr> },
    /// Check out another branch.
    Switch { target: BranchId },
    /// Append a copy of one item from another branch onto the active branch.
    Merge { from: BranchId, item: ItemId },
    /// Delete an inactive branch.
    DeleteBranch { target: BranchId },
    /// Replace a set of items with a single summary item, inserted where the range began.
    Summarize { replace: Vec<ItemId>, summary: Arc<Item> },
}
