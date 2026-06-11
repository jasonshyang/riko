use std::sync::Arc;

use crate::{Item, ItemId, ItemMeta, ItemPayload};

/// The only way to durably mutate a workspace.
///
/// Every operation is fully resolved — ids and timestamps are minted by the caller-facing
/// helpers *before* the op is built — so the operation log replays deterministically and
/// [`Workspace::apply`](crate::Workspace::apply) never mints anything itself.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Append an item to the active branch.
    Add { item: Arc<Item> },
    /// Replace the payload of an existing item, keeping its id and creation time.
    Edit { id: ItemId, payload: ItemPayload },
    /// Remove an item from the active branch.
    Drop { id: ItemId },
    /// Replace an item's metadata.
    SetMeta { id: ItemId, meta: ItemMeta },
}
