use riko_core::{Prompt, Result, RikoError, ToolDescriptor};
use riko_utils::Timestamp;
use smol_str::SmolStr;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::branch::Branch;
use crate::store::OperationSink;
use crate::{BranchId, BranchInfo, Item, ItemId, ItemMeta, ItemPayload, Operation};

const EVENT_CAPACITY: usize = 256;

/// The workspace. Holds the branch state behind a sync mutex (mutations are short and never
/// await) and a broadcast channel for change events. Shared as `Arc<Workspace>`; the user and
/// the agent are peer mutators through [`apply`](Self::apply).
pub struct Workspace {
    state: parking_lot::Mutex<State>,
    events: broadcast::Sender<WorkspaceEvent>,
    sink: Option<parking_lot::Mutex<OperationSink>>,
}

/// Durable change broadcast after an [`Operation`](crate::Operation) settles. Carries ids
/// only — subscribers re-read the workspace for detail, and re-sync fully on a lagged channel.
///
/// Ephemeral, in-flight updates (streaming an assistant turn) are a separate concern handled
/// where the agent drives the stream.
#[derive(Debug, Clone)]
pub enum WorkspaceEvent {
    ItemAdded { branch: BranchId, id: ItemId },
    ItemEdited { branch: BranchId, id: ItemId },
    ItemDropped { branch: BranchId, id: ItemId },
    ItemMetaChanged { branch: BranchId, id: ItemId },
    BranchForked { branch: BranchId, parent: BranchId },
    BranchSwitched { from: BranchId, to: BranchId },
    BranchMerged { into: BranchId, from: BranchId, item: ItemId },
    BranchDeleted { branch: BranchId },
    ItemSummarized { branch: BranchId, summary: ItemId, replaced: Vec<ItemId> },
}

impl Workspace {
    pub fn new() -> Self {
        Self::from_parts(State::with_root(BranchId::fresh()), None)
    }

    /// Open a persistent workspace at `path`, replaying an existing log or starting a fresh
    /// one. Subsequent operations append to the log.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            let (root, ops) = crate::store::read_log(path)?;
            let mut state = State::with_root(root);
            for op in &ops {
                state.apply(op)?;
            }
            Ok(Self::from_parts(state, Some(OperationSink::open_existing(path)?)))
        } else {
            let root = BranchId::fresh();
            let sink = OperationSink::create(path, root.clone())?;
            Ok(Self::from_parts(State::with_root(root), Some(sink)))
        }
    }
    fn from_parts(state: State, sink: Option<OperationSink>) -> Self {
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        Self {
            state: parking_lot::Mutex::new(state),
            events,
            sink: sink.map(parking_lot::Mutex::new),
        }
    }

    /// Receive change events as the workspace mutates.
    pub fn subscribe(&self) -> broadcast::Receiver<WorkspaceEvent> {
        self.events.subscribe()
    }

    /// Id of the active (checked-out) branch.
    pub fn active_branch(&self) -> BranchId {
        self.state.lock().active.id.clone()
    }

    /// Snapshot of the active branch's items. Cheap — clones `Arc`s, not item contents.
    pub fn items(&self) -> Vec<Arc<Item>> {
        self.state.lock().active.items.clone()
    }

    /// Fetch one item from the active branch by id.
    pub fn item(&self, id: &ItemId) -> Option<Arc<Item>> {
        let state = self.state.lock();
        state.active.position(id).map(|pos| state.active.items[pos].clone())
    }

    /// Add an item to the active branch, returning its freshly minted id.
    pub fn add(&self, payload: ItemPayload) -> Result<ItemId> {
        let item = Item {
            id: ItemId::fresh(),
            created: Timestamp::now(),
            payload,
            meta: ItemMeta::default(),
        };
        let id = item.id.clone();
        self.apply(Operation::Add { item: Arc::new(item) })?;
        Ok(id)
    }

    /// Replace an item's payload, keeping its id and creation time.
    pub fn edit(&self, id: &ItemId, payload: ItemPayload) -> Result<()> {
        self.apply(Operation::Edit { id: id.clone(), payload })
    }

    /// Remove an item from the active branch.
    pub fn drop_item(&self, id: &ItemId) -> Result<()> {
        self.apply(Operation::Drop { id: id.clone() })
    }

    /// Replace an item's metadata.
    pub fn set_meta(&self, id: &ItemId, meta: ItemMeta) -> Result<()> {
        self.apply(Operation::SetMeta { id: id.clone(), meta })
    }

    /// Replace a set of items with one summary item (inserted where the earliest replaced item
    /// was). The summary text is produced upstream — e.g. by an LLM-backed summarizer — and
    /// this records the result as one durable operation.
    pub fn summarize(&self, replace: &[ItemId], text: String) -> Result<ItemId> {
        let summary = Item {
            id: ItemId::fresh(),
            created: Timestamp::now(),
            payload: ItemPayload::Summary { text },
            meta: ItemMeta::default(),
        };
        let id = summary.id.clone();
        self.apply(Operation::Summarize { replace: replace.to_vec(), summary: Arc::new(summary) })?;
        Ok(id)
    }

    pub fn apply(&self, op: Operation) -> Result<()> {
        let event = self.state.lock().apply(&op)?;
        if let Some(sink) = &self.sink {
            sink.lock().append(&op)?;
        }
        let _ = self.events.send(event);
        Ok(())
    }

    /// Render the active branch into the wire payload for the given tools.
    pub fn render(&self, tools: Vec<ToolDescriptor>) -> Prompt {
        crate::render::render(&self.items(), tools)
    }

    /// Fork the active branch into a new one (sharing its items) without switching to it.
    pub fn fork(&self, label: Option<SmolStr>) -> Result<BranchId> {
        let new = BranchId::fresh();
        self.apply(Operation::Fork { new: new.clone(), label })?;
        Ok(new)
    }

    /// Check out another branch.
    pub fn switch(&self, target: &BranchId) -> Result<()> {
        self.apply(Operation::Switch { target: target.clone() })
    }

    /// Append a copy of one item from another branch onto the active branch.
    pub fn merge(&self, from: &BranchId, item: &ItemId) -> Result<()> {
        self.apply(Operation::Merge { from: from.clone(), item: item.clone() })
    }

    /// Delete an inactive branch.
    pub fn delete_branch(&self, target: &BranchId) -> Result<()> {
        self.apply(Operation::DeleteBranch { target: target.clone() })
    }

    /// List every branch, with the active one flagged.
    pub fn branches(&self) -> Vec<BranchInfo> {
        let state = self.state.lock();
        let mut out = Vec::with_capacity(state.inactive.len() + 1);
        out.push(state.active.info(true));
        out.extend(state.inactive.values().map(|branch| branch.info(false)));
        out
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

struct State {
    active: Branch,
    inactive: HashMap<BranchId, Branch>,
}

impl State {
    fn with_root(root: BranchId) -> Self {
        Self { active: Branch::root(root), inactive: HashMap::new() }
    }

    fn apply(&mut self, op: &Operation) -> Result<WorkspaceEvent> {
        let event = match op {
            Operation::Add { item } => {
                let id = item.id.clone();
                self.active.items.push(item.clone());
                WorkspaceEvent::ItemAdded { branch: self.active.id.clone(), id }
            }
            Operation::Edit { id, payload } => {
                let pos = self.active.position(id).ok_or_else(|| Self::item_not_found(id))?;
                self.active.items[pos] =
                    Self::revise(&self.active.items[pos], |item| item.payload = payload.clone());
                WorkspaceEvent::ItemEdited { branch: self.active.id.clone(), id: id.clone() }
            }
            Operation::Drop { id } => {
                let pos = self.active.position(id).ok_or_else(|| Self::item_not_found(id))?;
                self.active.items.remove(pos);
                WorkspaceEvent::ItemDropped { branch: self.active.id.clone(), id: id.clone() }
            }
            Operation::SetMeta { id, meta } => {
                let pos = self.active.position(id).ok_or_else(|| Self::item_not_found(id))?;
                self.active.items[pos] =
                    Self::revise(&self.active.items[pos], |item| item.meta = meta.clone());
                WorkspaceEvent::ItemMetaChanged { branch: self.active.id.clone(), id: id.clone() }
            }
            Operation::Fork { new, label } => {
                let parent = self.active.id.clone();
                let forked = Branch::fork(
                    new.clone(),
                    parent.clone(),
                    label.clone(),
                    self.active.items.clone(),
                );
                self.inactive.insert(new.clone(), forked);
                WorkspaceEvent::BranchForked { branch: new.clone(), parent }
            }
            Operation::Switch { target } => {
                let from = self.active.id.clone();
                if *target != from {
                    let next = self
                        .inactive
                        .remove(target)
                        .ok_or_else(|| Self::branch_not_found(target))?;
                    let prev = std::mem::replace(&mut self.active, next);
                    self.inactive.insert(prev.id.clone(), prev);
                }
                WorkspaceEvent::BranchSwitched { from, to: target.clone() }
            }
            Operation::Merge { from, item } => {
                let merged = {
                    let source = self.branch(from).ok_or_else(|| Self::branch_not_found(from))?;
                    source
                        .items
                        .iter()
                        .find(|candidate| &candidate.id == item)
                        .cloned()
                        .ok_or_else(|| Self::item_not_found(item))?
                };
                self.active.items.push(merged);
                WorkspaceEvent::BranchMerged {
                    into: self.active.id.clone(),
                    from: from.clone(),
                    item: item.clone(),
                }
            }
            Operation::DeleteBranch { target } => {
                if *target == self.active.id {
                    return Err(RikoError::InvalidArgument(
                        "cannot delete the active branch".into(),
                    ));
                }
                self.inactive.remove(target).ok_or_else(|| Self::branch_not_found(target))?;
                WorkspaceEvent::BranchDeleted { branch: target.clone() }
            }
            Operation::Summarize { replace, summary } => {
                let mut positions = Vec::with_capacity(replace.len());
                for id in replace {
                    positions
                        .push(self.active.position(id).ok_or_else(|| Self::item_not_found(id))?);
                }
                let insert_at = positions.iter().copied().min().unwrap_or(self.active.items.len());
                positions.sort_unstable();
                for &pos in positions.iter().rev() {
                    self.active.items.remove(pos);
                }
                self.active.items.insert(insert_at, summary.clone());
                WorkspaceEvent::ItemSummarized {
                    branch: self.active.id.clone(),
                    summary: summary.id.clone(),
                    replaced: replace.clone(),
                }
            }
        };
        Ok(event)
    }

    /// The branch with this id, active or inactive.
    fn branch(&self, id: &BranchId) -> Option<&Branch> {
        if self.active.id == *id { Some(&self.active) } else { self.inactive.get(id) }
    }

    /// Copy-on-write: clone the item, apply the change, and re-wrap. Other branches holding the
    /// old `Arc` are untouched.
    fn revise(item: &Arc<Item>, change: impl FnOnce(&mut Item)) -> Arc<Item> {
        let mut next = Item::clone(item);
        change(&mut next);
        Arc::new(next)
    }

    fn item_not_found(id: &ItemId) -> RikoError {
        RikoError::NotFound(format!("item {id} not found"))
    }

    fn branch_not_found(id: &BranchId) -> RikoError {
        RikoError::NotFound(format!("branch {id} not found"))
    }
}

#[cfg(test)]
mod tests {
    use riko_core::{Content, Message, Role};

    use super::*;

    fn sys(text: &str) -> ItemPayload {
        ItemPayload::System { tag: None, text: text.into() }
    }

    fn msg(text: &str) -> ItemPayload {
        ItemPayload::Message(Message { role: Role::User, content: vec![Content::text(text)] })
    }

    fn text_of(item: &Item) -> &str {
        match &item.payload {
            ItemPayload::System { text, .. } => text,
            other => panic!("expected system item, got {other:?}"),
        }
    }

    #[test]
    fn add_then_read() {
        let ws = Workspace::new();
        let id = ws.add(sys("hello")).unwrap();
        let items = ws.items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, id);
        assert_eq!(text_of(&ws.item(&id).unwrap()), "hello");
    }

    #[test]
    fn edit_is_copy_on_write() {
        let ws = Workspace::new();
        let id = ws.add(sys("old")).unwrap();
        let before = ws.items();
        ws.edit(&id, sys("new")).unwrap();
        let after = ws.items();
        assert_eq!(text_of(&before[0]), "old");
        assert_eq!(text_of(&after[0]), "new");
        assert!(!Arc::ptr_eq(&before[0], &after[0]));
        assert_eq!(after[0].id, id);
    }

    #[test]
    fn drop_removes() {
        let ws = Workspace::new();
        let id = ws.add(sys("x")).unwrap();
        ws.drop_item(&id).unwrap();
        assert!(ws.items().is_empty());
        assert!(ws.item(&id).is_none());
    }

    #[test]
    fn set_meta_updates_flags() {
        let ws = Workspace::new();
        let id = ws.add(sys("x")).unwrap();
        ws.set_meta(&id, ItemMeta { pinned: true, ..Default::default() }).unwrap();
        assert!(ws.item(&id).unwrap().meta.pinned);
    }

    #[test]
    fn mutating_missing_item_is_not_found() {
        let ws = Workspace::new();
        let ghost = ItemId::fresh();
        assert!(matches!(ws.edit(&ghost, sys("x")), Err(RikoError::NotFound(_))));
        assert!(matches!(ws.drop_item(&ghost), Err(RikoError::NotFound(_))));
        assert!(matches!(ws.set_meta(&ghost, ItemMeta::default()), Err(RikoError::NotFound(_))));
    }

    #[test]
    fn add_broadcasts_event() {
        let ws = Workspace::new();
        let mut rx = ws.subscribe();
        let id = ws.add(sys("x")).unwrap();
        match rx.try_recv() {
            Ok(WorkspaceEvent::ItemAdded { id: got, .. }) => assert_eq!(got, id),
            other => panic!("expected ItemAdded, got {other:?}"),
        }
    }

    #[test]
    fn render_projects_active_branch() {
        let ws = Workspace::new();
        ws.add(sys("you are riko")).unwrap();
        ws.add(msg("hi")).unwrap();
        let prompt = ws.render(Vec::new());
        assert_eq!(prompt.system, "you are riko");
        assert_eq!(prompt.messages.len(), 1);
    }

    #[test]
    fn summarize_replaces_a_range_with_one_summary() {
        let ws = Workspace::new();
        let a = ws.add(msg("turn one")).unwrap();
        let b = ws.add(msg("turn two")).unwrap();
        let keep = ws.add(msg("turn three")).unwrap();
        let summary = ws.summarize(&[a, b], "earlier: two turns".into()).unwrap();
        let items = ws.items();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, summary);
        assert_eq!(items[1].id, keep);
        let prompt = ws.render(Vec::new());
        assert!(prompt.system.contains("<summary>"));
        assert!(prompt.system.contains("earlier: two turns"));
        assert_eq!(prompt.messages.len(), 1);
    }

    #[test]
    fn fork_creates_a_child_sharing_items() {
        let ws = Workspace::new();
        ws.add(sys("shared")).unwrap();
        let main = ws.active_branch();
        let fork = ws.fork(Some("exp".into())).unwrap();
        let branches = ws.branches();
        assert_eq!(branches.len(), 2);
        let child = branches.iter().find(|b| b.id == fork).unwrap();
        assert_eq!(child.parent.as_ref(), Some(&main));
        ws.switch(&fork).unwrap();
        assert_eq!(ws.items().len(), 1);
    }

    #[test]
    fn edits_on_a_fork_leave_the_parent_untouched() {
        let ws = Workspace::new();
        let id = ws.add(sys("original")).unwrap();
        let main = ws.active_branch();
        let fork = ws.fork(None).unwrap();
        ws.switch(&fork).unwrap();
        ws.edit(&id, sys("changed")).unwrap();
        assert_eq!(text_of(&ws.item(&id).unwrap()), "changed");
        ws.switch(&main).unwrap();
        assert_eq!(text_of(&ws.item(&id).unwrap()), "original");
    }

    #[test]
    fn merge_copies_one_item_across_branches() {
        let ws = Workspace::new();
        let main = ws.active_branch();
        let fork = ws.fork(None).unwrap();
        ws.switch(&fork).unwrap();
        let summary = ws.add(sys("sub-agent result")).unwrap();
        ws.switch(&main).unwrap();
        assert!(ws.item(&summary).is_none());
        ws.merge(&fork, &summary).unwrap();
        assert_eq!(text_of(&ws.item(&summary).unwrap()), "sub-agent result");
    }

    #[test]
    fn cannot_delete_the_active_branch() {
        let ws = Workspace::new();
        let main = ws.active_branch();
        let fork = ws.fork(None).unwrap();
        assert!(matches!(ws.delete_branch(&main), Err(RikoError::InvalidArgument(_))));
        ws.delete_branch(&fork).unwrap();
        assert_eq!(ws.branches().len(), 1);
    }

    #[test]
    fn switch_to_missing_branch_errors() {
        let ws = Workspace::new();
        assert!(matches!(ws.switch(&BranchId::fresh()), Err(RikoError::NotFound(_))));
    }
}
