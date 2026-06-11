use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// UTC-anchored timestamp used across the engine.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Timestamp(DateTime<Utc>);

impl Timestamp {
    pub fn now() -> Self {
        Self(Utc::now())
    }
}

impl std::ops::Deref for Timestamp {
    type Target = DateTime<Utc>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Timestamp {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
