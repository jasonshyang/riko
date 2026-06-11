use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use futures::StreamExt;
use riko_context::{ItemPayload, Workspace};
use riko_core::{Content, Message, Result, RikoError, Role};
use riko_llm::{
    ErrorReason, ModelSpec, ProviderEvent, ProviderRegistry, ProviderStream, StreamOptions,
};
use riko_utils::RunGuard;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::{AgentEvent, EndReason};

const EVENT_CAPACITY: usize = 256;

/// Drives turns against a shared [`Workspace`]. The user and the agent are peer mutators of the
/// same workspace; the agent's job is to render it, stream a response, and record the result as
/// operations. One run at a time.
pub struct Agent {
    workspace: Arc<Workspace>,
    providers: Arc<ProviderRegistry>,
    model: ModelSpec,
    options: StreamOptions,
    events: broadcast::Sender<AgentEvent>,
    running: AtomicBool,
    cancel: parking_lot::Mutex<CancellationToken>,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Subscribe to the agent's run-lifecycle and streaming events.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Add a user turn to the workspace, then run.
    pub async fn prompt(&self, text: impl Into<String>) -> Result<()> {
        let message = Message { role: Role::User, content: vec![Content::text(text)] };
        self.workspace.add(ItemPayload::Message(message))?;
        self.run().await
    }

    /// Abort the current run, if any. A later [`run`](Self::run) gets a fresh token.
    pub fn abort(&self) {
        self.cancel.lock().cancel();
    }

    /// Run the loop on the current workspace state until the assistant stops. Returns `Err`
    /// only when a run cannot start (already running, or provider setup fails); a provider
    /// error or an abort is a normal end, reported via [`AgentEvent::RunEnded`].
    pub async fn run(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(RikoError::InvalidArgument("agent is already running".into()));
        }
        let _guard = RunGuard::from(&self.running);
        let cancel = self.reset_cancel();
        let _ = self.events.send(AgentEvent::RunStarted);
        let outcome = self.run_turn(&cancel).await;
        let reason = match &outcome {
            Ok(reason) => *reason,
            Err(_) => EndReason::Failed,
        };
        let _ = self.events.send(AgentEvent::RunEnded { reason });
        outcome.map(|_| ())
    }

    fn reset_cancel(&self) -> CancellationToken {
        let token = CancellationToken::new();
        *self.cancel.lock() = token.clone();
        token
    }

    async fn run_turn(&self, cancel: &CancellationToken) -> Result<EndReason> {
        let _ = self.events.send(AgentEvent::TurnStarted);
        let prompt = self.workspace.render(Vec::new());
        let provider = self.providers.get(&self.model.api)?;
        let mut stream =
            provider.stream(&self.model, prompt, self.options.clone(), cancel.clone()).await?;
        let reason = self.consume(&mut stream).await?;
        let _ = self.events.send(AgentEvent::TurnEnded);
        Ok(reason)
    }

    async fn consume(&self, stream: &mut ProviderStream) -> Result<EndReason> {
        while let Some(event) = stream.next().await {
            match event {
                ProviderEvent::Done { message, .. } => {
                    let id = self.workspace.add(ItemPayload::Message(*message))?;
                    let _ = self.events.send(AgentEvent::Settled { id });
                    return Ok(EndReason::Stopped);
                }
                ProviderEvent::Error { reason, message } => {
                    let _ = self.events.send(AgentEvent::Error { message });
                    return Ok(Self::end_reason_for(reason));
                }
                delta => {
                    let _ = self.events.send(AgentEvent::Streaming(delta));
                }
            }
        }
        Ok(EndReason::Stopped)
    }

    fn end_reason_for(reason: ErrorReason) -> EndReason {
        match reason {
            ErrorReason::Cancelled => EndReason::Aborted,
            _ => EndReason::Failed,
        }
    }
}

#[derive(Default)]
pub struct AgentBuilder {
    workspace: Option<Arc<Workspace>>,
    providers: Option<Arc<ProviderRegistry>>,
    model: Option<ModelSpec>,
    options: Option<StreamOptions>,
}

impl AgentBuilder {
    pub fn workspace(mut self, workspace: Arc<Workspace>) -> Self {
        self.workspace = Some(workspace);
        self
    }

    pub fn providers(mut self, providers: Arc<ProviderRegistry>) -> Self {
        self.providers = Some(providers);
        self
    }

    pub fn model(mut self, model: ModelSpec) -> Self {
        self.model = Some(model);
        self
    }

    pub fn options(mut self, options: StreamOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn build(self) -> Result<Agent> {
        let workspace = self.workspace.ok_or_else(|| Self::missing("workspace"))?;
        let providers = self.providers.ok_or_else(|| Self::missing("providers"))?;
        let model = self.model.ok_or_else(|| Self::missing("model"))?;
        let (events, _) = broadcast::channel(EVENT_CAPACITY);
        Ok(Agent {
            workspace,
            providers,
            model,
            options: self.options.unwrap_or_default(),
            events,
            running: AtomicBool::new(false),
            cancel: parking_lot::Mutex::new(CancellationToken::new()),
        })
    }

    fn missing(field: &str) -> RikoError {
        RikoError::Config(format!("agent builder missing required field: {field}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riko_core::StopReason;
    use riko_llm::{
        Api,
        mock::{MockProvider, MockScript, MockStep},
    };
    use std::time::Duration;

    fn test_model() -> ModelSpec {
        ModelSpec::builder()
            .id("mock:m")
            .name("m")
            .api(Api::AnthropicMessages)
            .context_window(8_192)
            .max_output(1_024)
            .build()
            .unwrap()
    }

    fn user(text: &str) -> ItemPayload {
        ItemPayload::Message(Message { role: Role::User, content: vec![Content::text(text)] })
    }

    fn agent_with(mock: MockProvider, workspace: Arc<Workspace>) -> Agent {
        let mut providers = ProviderRegistry::new();
        providers.register(mock);
        Agent::builder()
            .workspace(workspace)
            .providers(Arc::new(providers))
            .model(test_model())
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn runs_a_text_turn_and_adds_the_assistant_item() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append_text("hello from the model");
        let agent = agent_with(mock, workspace.clone());
        workspace.add(user("hi")).unwrap();

        agent.run().await.unwrap();

        let items = workspace.items();
        assert_eq!(items.len(), 2);
        match &items[1].payload {
            ItemPayload::Message(message) => {
                assert!(matches!(message.role, Role::Assistant { .. }));
                assert_eq!(message.content, vec![Content::text("hello from the model")]);
            }
            other => panic!("expected assistant message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn prompt_adds_a_user_turn_then_runs() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append_text("ok");
        let agent = agent_with(mock, workspace.clone());

        agent.prompt("do a thing").await.unwrap();

        let items = workspace.items();
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0].payload,
            ItemPayload::Message(Message { role: Role::User, .. })
        ));
    }

    #[tokio::test]
    async fn emits_run_lifecycle_events() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append_text("hi");
        let agent = agent_with(mock, workspace.clone());
        let mut events = agent.subscribe();

        agent.prompt("hello").await.unwrap();

        let mut drained = Vec::new();
        while let Ok(event) = events.try_recv() {
            drained.push(event);
        }
        assert!(matches!(drained.first(), Some(AgentEvent::RunStarted)));
        assert!(matches!(
            drained.last(),
            Some(AgentEvent::RunEnded { reason: EndReason::Stopped })
        ));
        assert!(drained.iter().any(|event| matches!(event, AgentEvent::Settled { .. })));
    }

    #[tokio::test]
    async fn already_running_is_rejected() {
        // A direct re-entrant call: hold the flag, then attempt a run.
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append_text("x");
        let agent = agent_with(mock, workspace);
        agent.running.store(true, Ordering::SeqCst);
        assert!(matches!(agent.run().await, Err(RikoError::InvalidArgument(_))));
    }

    #[tokio::test]
    async fn abort_ends_the_run_without_an_assistant_item() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages)
            .with_step_delay(Duration::from_millis(20));
        let steps = (0..20).map(|_| MockStep::Text("x".into())).collect();
        mock.append(MockScript { steps, stop: StopReason::Stop });
        let agent = Arc::new(agent_with(mock, workspace.clone()));
        workspace.add(user("hi")).unwrap();
        let mut events = agent.subscribe();

        let runner = {
            let agent = Arc::clone(&agent);
            tokio::spawn(async move { agent.run().await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        agent.abort();
        runner.await.unwrap().unwrap();

        assert_eq!(workspace.items().len(), 1);
        let mut reason = None;
        while let Ok(event) = events.try_recv() {
            if let AgentEvent::RunEnded { reason: ended } = event {
                reason = Some(ended);
            }
        }
        assert_eq!(reason, Some(EndReason::Aborted));
    }
}
