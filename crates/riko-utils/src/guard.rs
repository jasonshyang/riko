use std::sync::atomic::{AtomicBool, Ordering};

pub struct RunGuard<'a>(pub &'a AtomicBool);

impl<'a> From<&'a AtomicBool> for RunGuard<'a> {
    fn from(value: &'a AtomicBool) -> Self {
        RunGuard(value)
    }
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}
