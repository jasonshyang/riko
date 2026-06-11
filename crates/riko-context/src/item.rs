use riko_core::Message;
use riko_utils::Timestamp;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

riko_utils::id_type!(
    /// Identifier for a workspace item.
    ItemId, "itm"
);

/// A unit of content in the workspace that may flow into the next model payload. Identity and
/// creation time are immutable; [`meta`](Self::meta) carries the mutable annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: ItemId,
    pub created: Timestamp,
    pub payload: ItemPayload,
    pub meta: ItemMeta,
}

/// What an [`Item`] carries. Start small; new kinds (file, summary, annotation) get added
/// when their feature lands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemPayload {
    /// A conversation turn (user, assistant, or tool result).
    Message(Message),
    /// Text folded into the system prompt, optionally wrapped in a named section.
    System { tag: Option<SmolStr>, text: String },
    /// A condensed summary that replaced a range of items during compaction.
    Summary { text: String },
}

/// Mutable per-item annotations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ItemMeta {
    /// Always include this item verbatim when rendering.
    pub pinned: bool,
    /// Never include this item when rendering.
    pub hidden: bool,
    /// User-facing label for navigation and search.
    pub label: Option<SmolStr>,
}
