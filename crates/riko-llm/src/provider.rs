use riko_core::{Prompt, Result};
use std::future::Future;
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

use crate::{Api, ModelSpec, ProviderStream, StreamOptions};

/// Boxed future returned by [`Provider::stream`].
pub type ProviderFuture<'a> = Pin<Box<dyn Future<Output = Result<ProviderStream>> + Send + 'a>>;

/// Wire-protocol adapter for one LLM API family.
///
/// An implementation translates a provider-neutral [`Prompt`] plus [`StreamOptions`] into the
/// vendor's request, performs the call, and yields a normalized [`ProviderStream`]. The trait
/// is held as a registry trait object, so it must stay dyn-compatible — hence the boxed
/// future and stream rather than `async fn`.
pub trait Provider: Send + Sync + 'static {
    /// Which API family this implements; the registry's lookup key.
    fn api(&self) -> Api;

    /// Open a streaming response. The stream ends with exactly one terminal
    /// [`crate::ProviderEvent::Done`] or [`crate::ProviderEvent::Error`]. Firing `cancel`
    /// aborts in-flight work and yields a clean `Error { reason: Cancelled, .. }`.
    fn stream<'a>(
        &'a self,
        model: &'a ModelSpec,
        prompt: Prompt,
        options: StreamOptions,
        cancel: CancellationToken,
    ) -> ProviderFuture<'a>;
}
