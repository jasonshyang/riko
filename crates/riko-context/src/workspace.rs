use riko_core::{Result, RikoError};
use riko_utils::Timestamp;
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::branch::Branch;
use crate::{BranchId, Item, ItemId, ItemMeta, ItemPayload, Operation};

const EVENT_CAPACITY: usize = 256;

/// The workspace. Holds the branch state behind a sync mutex (mutations are short and never
/// await) and a broadcast channel for change events. Shared as `Arc<Workspace>`; the user and
/// the agent are peer mutators through [`apply`](Self::apply).
pub struct Workspace {
    state: parking_lot::Mutex<State>,
    events: broadcast::Sender<WorkspaceEvent>,
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
}

impl Workspace {
    pub fn new() -> Self {
        let active = Branch::root(BranchId::fresh());
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        Self { state: parking_lot::Mutex::new(State { active, log: Vec::new() }), events }
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

    /// The single funnel for durable mutation: apply the op, log it, broadcast the change.
    pub fn apply(&self, op: Operation) -> Result<()> {
        let event = self.state.lock().apply(op)?;
        let _ = self.events.send(event);
        Ok(())
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

struct State {
    active: Branch,
    log: Vec<Operation>,
}

impl State {
    fn apply(&mut self, op: Operation) -> Result<WorkspaceEvent> {
        let branch = self.active.id.clone();
        let event = match &op {
            Operation::Add { item } => {
                let id = item.id.clone();
                self.active.items.push(item.clone());
                WorkspaceEvent::ItemAdded { branch, id }
            }
            Operation::Edit { id, payload } => {
                let pos = self.active.position(id).ok_or_else(|| Self::not_found(id))?;
                self.active.items[pos] = Self::revise(&self.active.items[pos], |item| {
                    item.payload = payload.clone();
                });
                WorkspaceEvent::ItemEdited { branch, id: id.clone() }
            }
            Operation::Drop { id } => {
                let pos = self.active.position(id).ok_or_else(|| Self::not_found(id))?;
                self.active.items.remove(pos);
                WorkspaceEvent::ItemDropped { branch, id: id.clone() }
            }
            Operation::SetMeta { id, meta } => {
                let pos = self.active.position(id).ok_or_else(|| Self::not_found(id))?;
                self.active.items[pos] = Self::revise(&self.active.items[pos], |item| {
                    item.meta = meta.clone();
                });
                WorkspaceEvent::ItemMetaChanged { branch, id: id.clone() }
            }
        };
        self.log.push(op);
        Ok(event)
    }

    /// Copy-on-write: clone the item, apply the change, and re-wrap. Other branches holding the
    /// old `Arc` are untouched.
    fn revise(item: &Arc<Item>, change: impl FnOnce(&mut Item)) -> Arc<Item> {
        let mut next = Item::clone(item);
        change(&mut next);
        Arc::new(next)
    }

    fn not_found(id: &ItemId) -> RikoError {
        RikoError::NotFound(format!("item {id} not on active branch"))
    }
}
