use riko_core::{Result, RikoError};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Api, Provider};

/// Maps each [`Api`] to the provider that speaks it. Built at startup, then shared
/// immutably (`Arc<ProviderRegistry>`) for lookups.
#[derive(Default)]
pub struct ProviderRegistry {
    providers: HashMap<Api, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider under its declared [`Api`], replacing any prior one.
    pub fn register<P: Provider>(&mut self, provider: P) {
        self.providers.insert(provider.api(), Arc::new(provider));
    }

    /// Look up the provider for an API family.
    pub fn get(&self, api: &Api) -> Result<Arc<dyn Provider>> {
        self.providers
            .get(api)
            .cloned()
            .ok_or_else(|| RikoError::NotFound(format!("no provider registered for `{api}`")))
    }

    pub fn contains(&self, api: &Api) -> bool {
        self.providers.contains_key(api)
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}
