use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use futures::StreamExt;
use riko_context::{ItemPayload, Workspace};
use riko_core::{Content, Message, Result, RikoError, Role, ToolCall};
use riko_llm::{
    ErrorReason, ModelSpec, ProviderEvent, ProviderRegistry, ProviderStream, StreamOptions,
};
use riko_tools::{FileAccess, ToolContext, ToolOutput, ToolRegistry};
use riko_utils::RunGuard;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::{AgentEvent, EndReason, PendingQueue, QueueKind};

const EVENT_CAPACITY: usize = 256;

/// What a turn produced: a terminal reason, or tool calls to dispatch before the next turn.
enum TurnOutcome {
    Done(EndReason),
    Tools(Vec<ToolCall>),
}

/// What consuming a provider stream produced.
enum StreamOutcome {
    Completed { tool_calls: Vec<ToolCall> },
    Aborted,
    Failed,
}

/// Drives turns against a shared [`Workspace`]. The user and the agent are peer mutators of the
/// same workspace; the agent's job is to render it, stream a response, and record the result as
/// operations. One run at a time.
pub struct Agent {
    workspace: Arc<Workspace>,
    providers: Arc<ProviderRegistry>,
    model: ModelSpec,
    options: StreamOptions,
    tools: Arc<ToolRegistry>,
    root: PathBuf,
    events: broadcast::Sender<AgentEvent>,
    running: AtomicBool,
    cancel: parking_lot::Mutex<CancellationToken>,
    /// Per-run read-set shared with each tool call; reset at the start of every run.
    file_access: FileAccess,
    /// Messages to merge in at the next turn boundary (mid-run course correction).
    steering: PendingQueue,
    /// Messages to deliver after a natural stop, continuing the run instead of ending it.
    follow_up: PendingQueue,
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
        self.workspace.add(ItemPayload::Message(Self::user_message(text)))?;
        self.run().await
    }

    /// Queue a message to merge onto the workspace at the next turn boundary, steering a run
    /// already in progress. It lands before the next LLM call; if the agent is idle, it joins
    /// the first turn of the next [`run`](Self::run).
    pub fn steer(&self, text: impl Into<String>) {
        self.steering.push(Self::user_message(text));
    }

    /// Queue a message to deliver after the run would otherwise stop, continuing it instead of
    /// ending. A follow-up only fires on a natural stop — an aborted or failed run leaves it
    /// queued for next time.
    pub fn follow_up(&self, text: impl Into<String>) {
        self.follow_up.push(Self::user_message(text));
    }

    /// Discard any queued steering messages.
    pub fn clear_steering(&self) {
        self.steering.clear();
    }

    /// Discard any queued follow-up messages.
    pub fn clear_follow_up(&self) {
        self.follow_up.clear();
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
        self.file_access.reset();
        let cancel = self.reset_cancel();
        let _ = self.events.send(AgentEvent::RunStarted);
        let outcome = self.run_loop(&cancel).await;
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

    async fn run_loop(&self, cancel: &CancellationToken) -> Result<EndReason> {
        loop {
            if cancel.is_cancelled() {
                return Ok(EndReason::Aborted);
            }
            self.drain(&self.steering, QueueKind::Steering)?;
            match self.run_turn(cancel).await? {
                TurnOutcome::Tools(calls) => self.dispatch_tools(&calls, cancel).await?,
                // A natural stop ends the run only if nothing is queued to follow up with;
                // otherwise the queued input becomes the next turn. Abort/failure never
                // triggers a follow-up.
                TurnOutcome::Done(EndReason::Stopped) => {
                    if self.drain(&self.follow_up, QueueKind::FollowUp)? == 0 {
                        return Ok(EndReason::Stopped);
                    }
                }
                TurnOutcome::Done(reason) => return Ok(reason),
            }
        }
    }

    /// Drain a pending queue onto the workspace as user items, emitting a [`QueueDrained`]
    /// event per message. Returns how many were added.
    ///
    /// [`QueueDrained`]: AgentEvent::QueueDrained
    fn drain(&self, queue: &PendingQueue, kind: QueueKind) -> Result<usize> {
        let pending = queue.drain();
        let count = pending.len();
        for message in pending {
            let id = self.workspace.add(ItemPayload::Message(message))?;
            let _ = self.events.send(AgentEvent::QueueDrained { queue: kind, id });
        }
        Ok(count)
    }

    async fn run_turn(&self, cancel: &CancellationToken) -> Result<TurnOutcome> {
        let _ = self.events.send(AgentEvent::TurnStarted);
        let prompt = self.workspace.render(self.tools.descriptors());
        let provider = self.providers.get(&self.model.api)?;
        let mut stream =
            provider.stream(&self.model, prompt, self.options.clone(), cancel.clone()).await?;
        let outcome = self.consume(&mut stream).await?;
        let _ = self.events.send(AgentEvent::TurnEnded);
        Ok(match outcome {
            StreamOutcome::Completed { tool_calls } if tool_calls.is_empty() => {
                TurnOutcome::Done(EndReason::Stopped)
            }
            StreamOutcome::Completed { tool_calls } => TurnOutcome::Tools(tool_calls),
            StreamOutcome::Aborted => TurnOutcome::Done(EndReason::Aborted),
            StreamOutcome::Failed => TurnOutcome::Done(EndReason::Failed),
        })
    }

    async fn consume(&self, stream: &mut ProviderStream) -> Result<StreamOutcome> {
        while let Some(event) = stream.next().await {
            match event {
                ProviderEvent::Done { message, .. } => {
                    let tool_calls = Self::tool_calls_of(&message);
                    let id = self.workspace.add(ItemPayload::Message(*message))?;
                    let _ = self.events.send(AgentEvent::Settled { id });
                    return Ok(StreamOutcome::Completed { tool_calls });
                }
                ProviderEvent::Error { reason, message } => {
                    let _ = self.events.send(AgentEvent::Error { message });
                    return Ok(match reason {
                        ErrorReason::Cancelled => StreamOutcome::Aborted,
                        _ => StreamOutcome::Failed,
                    });
                }
                delta => {
                    let _ = self.events.send(AgentEvent::Streaming(delta));
                }
            }
        }

        // The stream ended without the Done/Error terminal the provider contract promises.
        // Surface it as a failure rather than silently reporting a clean, empty stop with no
        // assistant item recorded.
        let _ = self.events.send(AgentEvent::Error {
            message: "provider stream ended without a terminal event".into(),
        });
        Ok(StreamOutcome::Failed)
    }

    /// Run each requested tool in order, appending its result to the workspace. Stops before
    /// the next call once the run is aborted, so a multi-tool batch doesn't keep dispatching
    /// after cancellation; the run loop then settles as [`EndReason::Aborted`].
    async fn dispatch_tools(&self, calls: &[ToolCall], cancel: &CancellationToken) -> Result<()> {
        for call in calls {
            if cancel.is_cancelled() {
                break;
            }

            let _ = self.events.send(AgentEvent::ToolStarted {
                call_id: call.id.clone(),
                tool: call.name.clone(),
            });
            let result = self.run_tool(call, cancel).await;
            let is_error = matches!(&result.role, Role::ToolResult { is_error: true, .. });
            let id = self.workspace.add(ItemPayload::Message(result))?;
            let _ = self.events.send(AgentEvent::Settled { id });
            let _ = self.events.send(AgentEvent::ToolEnded { call_id: call.id.clone(), is_error });
        }
        Ok(())
    }

    async fn run_tool(&self, call: &ToolCall, cancel: &CancellationToken) -> Message {
        let outcome = match self.tools.get(&call.name) {
            Some(tool) => {
                let ctx =
                    ToolContext { root: self.root.clone(), file_access: self.file_access.clone() };
                tool.run(call.clone(), ctx, cancel.clone()).await
            }
            None => Err(RikoError::NotFound(format!("tool `{}` is not registered", call.name))),
        };
        Self::tool_result(call, outcome)
    }

    fn tool_calls_of(message: &Message) -> Vec<ToolCall> {
        message
            .content
            .iter()
            .filter_map(|block| match block {
                Content::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect()
    }

    fn tool_result(call: &ToolCall, outcome: Result<ToolOutput>) -> Message {
        let (content, is_error) = match outcome {
            Ok(output) => (output.content, false),
            Err(error) => (vec![Content::text(error.to_string())], true),
        };
        Message {
            role: Role::ToolResult { call_id: call.id.clone(), tool: call.name.clone(), is_error },
            content,
        }
    }

    /// Build a fresh `User`-role turn carrying one text block.
    fn user_message(text: impl Into<String>) -> Message {
        Message { role: Role::User, content: vec![Content::text(text)] }
    }
}

#[derive(Default)]
pub struct AgentBuilder {
    workspace: Option<Arc<Workspace>>,
    providers: Option<Arc<ProviderRegistry>>,
    model: Option<ModelSpec>,
    options: Option<StreamOptions>,
    tools: Option<Arc<ToolRegistry>>,
    root: Option<PathBuf>,
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

    pub fn tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Directory tools resolve relative paths against. Defaults to the current directory.
    pub fn root(mut self, root: impl Into<PathBuf>) -> Self {
        self.root = Some(root.into());
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
            tools: self.tools.unwrap_or_default(),
            root: self.root.unwrap_or_else(|| PathBuf::from(".")),
            events,
            running: AtomicBool::new(false),
            cancel: parking_lot::Mutex::new(CancellationToken::new()),
            file_access: FileAccess::new(),
            steering: PendingQueue::new(),
            follow_up: PendingQueue::new(),
        })
    }

    fn missing(field: &str) -> RikoError {
        RikoError::Config(format!("agent builder missing required field: {field}"))
    }
}

#[cfg(test)]
mod tests {
    use riko_tools::{Tool, ToolFuture};

    use super::*;
    use riko_core::{StopReason, ToolDescriptor};
    use riko_llm::{
        Api,
        mock::{MockProvider, MockScript, MockStep},
    };
    use serde_json::json;
    use std::time::Duration;

    struct EchoTool {
        descriptor: ToolDescriptor,
    }

    impl EchoTool {
        fn new() -> Self {
            Self {
                descriptor: ToolDescriptor {
                    name: "echo".into(),
                    description: "echo the call arguments".into(),
                    schema: Default::default(),
                },
            }
        }
    }

    impl Tool for EchoTool {
        fn descriptor(&self) -> &ToolDescriptor {
            &self.descriptor
        }

        fn run<'a>(
            &'a self,
            call: ToolCall,
            _ctx: ToolContext,
            _cancel: CancellationToken,
        ) -> ToolFuture<'a> {
            Box::pin(async move { Ok(ToolOutput::text(format!("echo: {}", call.arguments))) })
        }
    }

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

    fn text_of(payload: &ItemPayload) -> &str {
        match payload {
            ItemPayload::Message(Message { content, .. }) => match content.first() {
                Some(Content::Text(text)) => &text.text,
                _ => panic!("expected a leading text block"),
            },
            other => panic!("expected a message item, got {other:?}"),
        }
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

    #[tokio::test]
    async fn dispatches_a_tool_then_continues() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append(MockScript::tool_call("echo", json!({ "msg": "hi" })));
        mock.append_text("all done");
        let mut providers = ProviderRegistry::new();
        providers.register(mock);
        let mut tools = ToolRegistry::default();
        tools.register(EchoTool::new());
        let agent = Agent::builder()
            .workspace(workspace.clone())
            .providers(Arc::new(providers))
            .model(test_model())
            .tools(Arc::new(tools))
            .build()
            .unwrap();
        workspace.add(user("use echo")).unwrap();

        agent.run().await.unwrap();

        // user, assistant(tool call), tool result, assistant(text)
        let items = workspace.items();
        assert_eq!(items.len(), 4);
        assert!(matches!(
            &items[2].payload,
            ItemPayload::Message(Message { role: Role::ToolResult { is_error: false, .. }, .. })
        ));
    }

    #[tokio::test]
    async fn unknown_tool_becomes_an_error_result() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append(MockScript::tool_call("ghost", json!({})));
        mock.append_text("recovered");
        let agent = agent_with(mock, workspace.clone());
        workspace.add(user("go")).unwrap();

        agent.run().await.unwrap();

        let items = workspace.items();
        assert_eq!(items.len(), 4);
        assert!(matches!(
            &items[2].payload,
            ItemPayload::Message(Message { role: Role::ToolResult { is_error: true, .. }, .. })
        ));
    }

    #[tokio::test]
    async fn steering_is_drained_onto_the_workspace_before_the_turn() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append_text("ok");
        let agent = agent_with(mock, workspace.clone());
        workspace.add(user("start")).unwrap();
        agent.steer("actually, do this instead");
        let mut events = agent.subscribe();

        agent.run().await.unwrap();

        // user("start"), the steering user turn, then the assistant reply.
        let items = workspace.items();
        assert_eq!(items.len(), 3);
        assert!(matches!(
            &items[1].payload,
            ItemPayload::Message(Message { role: Role::User, .. })
        ));
        assert_eq!(text_of(&items[1].payload), "actually, do this instead");

        let mut drained = false;
        while let Ok(event) = events.try_recv() {
            if matches!(event, AgentEvent::QueueDrained { queue: QueueKind::Steering, .. }) {
                drained = true;
            }
        }
        assert!(drained, "expected a QueueDrained(Steering) event");
    }

    #[tokio::test]
    async fn follow_up_continues_the_run_after_a_natural_stop() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append_text("first answer");
        mock.append_text("second answer");
        let agent = agent_with(mock, workspace.clone());
        workspace.add(user("start")).unwrap();
        agent.follow_up("and now the follow-up");
        let mut events = agent.subscribe();

        agent.run().await.unwrap();

        // user, assistant(first), follow-up user, assistant(second) — the follow-up drove a
        // second turn instead of letting the run end.
        let items = workspace.items();
        assert_eq!(items.len(), 4);
        assert_eq!(text_of(&items[2].payload), "and now the follow-up");
        assert_eq!(text_of(&items[3].payload), "second answer");

        let mut drained = false;
        while let Ok(event) = events.try_recv() {
            if matches!(event, AgentEvent::QueueDrained { queue: QueueKind::FollowUp, .. }) {
                drained = true;
            }
        }
        assert!(drained, "expected a QueueDrained(FollowUp) event");
    }

    #[tokio::test]
    async fn cleared_follow_up_does_not_fire() {
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages);
        mock.append_text("only answer");
        let agent = agent_with(mock, workspace.clone());
        workspace.add(user("start")).unwrap();
        agent.follow_up("queued, then cancelled");
        agent.clear_follow_up();

        agent.run().await.unwrap();

        // No follow-up turn: just the user prompt and the single assistant answer.
        assert_eq!(workspace.items().len(), 2);
    }

    #[tokio::test]
    async fn steering_queued_mid_run_lands_before_the_next_turn() {
        // Turn 1 ends in a tool call so the loop continues; steering queued while turn 1
        // streams must be merged in before turn 2's LLM call.
        let workspace = Arc::new(Workspace::new());
        let mock = MockProvider::for_api(Api::AnthropicMessages)
            .with_step_delay(Duration::from_millis(30));
        mock.append(MockScript::tool_call("echo", json!({ "msg": "go" })));
        mock.append_text("done after steering");
        let mut providers = ProviderRegistry::new();
        providers.register(mock);
        let mut tools = ToolRegistry::default();
        tools.register(EchoTool::new());
        let agent = Arc::new(
            Agent::builder()
                .workspace(workspace.clone())
                .providers(Arc::new(providers))
                .model(test_model())
                .tools(Arc::new(tools))
                .build()
                .unwrap(),
        );
        workspace.add(user("start")).unwrap();

        let runner = {
            let agent = Arc::clone(&agent);
            tokio::spawn(async move { agent.run().await })
        };
        // Steer well before turn 1's ~30ms stream completes, so it is queued by the time the
        // loop reaches the steering drain ahead of turn 2.
        tokio::time::sleep(Duration::from_millis(10)).await;
        agent.steer("steered mid-run");
        runner.await.unwrap().unwrap();

        // user, assistant(tool call), tool result, steering user turn, assistant(text).
        let items = workspace.items();
        assert_eq!(items.len(), 5);
        assert_eq!(text_of(&items[3].payload), "steered mid-run");
        assert!(matches!(
            &items[4].payload,
            ItemPayload::Message(Message { role: Role::Assistant { .. }, .. })
        ));
    }
}
