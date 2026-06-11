use std::collections::VecDeque;

use riko_core::Message;

/// FIFO queue of pending user messages the run loop drains at a turn boundary.
///
/// Backs both steering (drained before the next LLM call, to course-correct a run already in
/// progress) and follow-up (drained after a natural stop, to keep the run going). Pushes
/// arrive from another task while the loop runs, so the backing store sits behind a short
/// sync mutex; the guard is never held across an `.await` — [`drain`](Self::drain) moves the
/// backlog out and the caller adds to the workspace after the lock is released.
///
/// Both queues drain their whole backlog at once. A finer-grained policy (one message per
/// turn) can be added behind this type if a caller ever needs it.
#[derive(Default)]
pub struct PendingQueue {
    inner: parking_lot::Mutex<VecDeque<Message>>,
}

impl PendingQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a message for the next drain.
    pub fn push(&self, message: Message) {
        self.inner.lock().push_back(message);
    }

    /// Take every queued message, oldest first, leaving the queue empty.
    pub fn drain(&self) -> Vec<Message> {
        self.inner.lock().drain(..).collect()
    }

    /// Drop every queued message without delivering it.
    pub fn clear(&self) {
        self.inner.lock().clear();
    }
}
