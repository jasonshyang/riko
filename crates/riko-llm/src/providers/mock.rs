use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use parking_lot::Mutex;
use riko_core::{
    Content, Message, ModelRef, Prompt, Role, StopReason, ToolCall, ToolCallId, Usage,
};
use smol_str::SmolStr;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::{
    Api, ErrorReason, ModelSpec, Provider, ProviderEvent, ProviderFuture, ProviderStream,
    StreamOptions,
};

/// Scripted [`Provider`] for tests. Each [`Provider::stream`] call drains one queued
/// [`MockScript`] and replays its steps as events.
///
/// Cloning shares the script inbox, so a test can register one clone with an agent and keep
/// another to enqueue scripts.
#[derive(Clone)]
pub struct MockProvider {
    api: Api,
    inbox: Arc<Mutex<VecDeque<MockScript>>>,
    step_delay: Duration,
}

impl Provider for MockProvider {
    fn api(&self) -> Api {
        self.api.clone()
    }

    fn stream<'a>(
        &'a self,
        model: &'a ModelSpec,
        _prompt: Prompt,
        _options: StreamOptions,
        cancel: CancellationToken,
    ) -> ProviderFuture<'a> {
        let script = self.inbox.lock().pop_front();
        let model_id = model.id.clone();
        let delay = self.step_delay;
        Box::pin(async move {
            let stream: ProviderStream = Box::pin(Self::emit(script, model_id, delay, cancel));
            Ok(stream)
        })
    }
}

impl MockProvider {
    /// A mock claiming the given [`Api`], with no inter-step delay.
    pub fn for_api(api: Api) -> Self {
        Self { api, inbox: Arc::new(Mutex::new(VecDeque::new())), step_delay: Duration::ZERO }
    }

    /// Delay between steps — useful for exercising cancellation and backpressure.
    pub fn with_step_delay(mut self, delay: Duration) -> Self {
        self.step_delay = delay;
        self
    }

    /// Enqueue a script. Scripts drain FIFO across `stream` calls.
    pub fn append(&self, script: MockScript) {
        self.inbox.lock().push_back(script);
    }

    /// Enqueue the simplest script: one text chunk and a clean stop.
    pub fn append_text(&self, text: impl Into<String>) {
        self.append(MockScript::just_text(text));
    }

    /// Number of queued scripts.
    pub fn pending(&self) -> usize {
        self.inbox.lock().len()
    }

    fn emit(
        script: Option<MockScript>,
        model: ModelRef,
        delay: Duration,
        cancel: CancellationToken,
    ) -> impl futures::Stream<Item = ProviderEvent> {
        stream! {
            yield ProviderEvent::Start;
            let Some(script) = script else {
                yield ProviderEvent::Error {
                    reason: ErrorReason::Permanent,
                    message: "MockProvider: inbox empty".into(),
                };
                return;
            };

            let stop = script.stop;
            let mut text = String::new();
            let mut idx = 0u32;
            let mut text_open = false;
            let mut content: Vec<Content> = Vec::new();

            for step in script.steps {
                if cancel.is_cancelled() {
                    yield ProviderEvent::Error { reason: ErrorReason::Cancelled, message: "cancelled".into() };
                    return;
                }
                if !delay.is_zero() {
                    sleep(delay).await;
                }
                match step {
                    MockStep::Text(chunk) => {
                        if !text_open {
                            yield ProviderEvent::TextStart { idx };
                            text_open = true;
                        }
                        text.push_str(&chunk);
                        yield ProviderEvent::TextDelta { idx, text: chunk };
                    }
                    MockStep::EndText => {
                        if text_open {
                            yield ProviderEvent::TextEnd { idx };
                            content.push(Content::text(std::mem::take(&mut text)));
                            text_open = false;
                            idx += 1;
                        }
                    }
                    MockStep::ToolCall { id, name, arguments } => {
                        yield ProviderEvent::ToolCallStart { idx, call_id: id.clone(), name: name.clone() };
                        yield ProviderEvent::ToolCallEnd { idx, args: arguments.clone() };
                        content.push(Content::ToolCall(ToolCall { id, name, arguments }));
                        idx += 1;
                    }
                    MockStep::Error(message) => {
                        yield ProviderEvent::Error { reason: ErrorReason::Permanent, message };
                        return;
                    }
                }
            }
            if text_open {
                yield ProviderEvent::TextEnd { idx };
                content.push(Content::text(text));
            }

            let message = Message {
                role: Role::Assistant { model, usage: Usage::default(), stop },
                content,
            };
            yield ProviderEvent::Done { stop, message: Box::new(message) };
        }
    }
}

/// One scripted response, drained whole by a single [`Provider::stream`] call.
#[derive(Debug, Clone)]
pub struct MockScript {
    pub steps: Vec<MockStep>,
    pub stop: StopReason,
}

impl MockScript {
    /// An empty response that stops cleanly.
    pub fn empty() -> Self {
        Self { steps: Vec::new(), stop: StopReason::Stop }
    }

    /// One text chunk followed by a clean stop.
    pub fn just_text(text: impl Into<String>) -> Self {
        Self { steps: vec![MockStep::Text(text.into()), MockStep::EndText], stop: StopReason::Stop }
    }

    /// One tool call, stopping for tool use.
    pub fn tool_call(name: impl Into<SmolStr>, arguments: serde_json::Value) -> Self {
        Self {
            steps: vec![MockStep::ToolCall {
                id: ToolCallId::fresh(),
                name: name.into(),
                arguments,
            }],
            stop: StopReason::ToolUse,
        }
    }
}

/// One scripted streaming step.
#[derive(Debug, Clone)]
pub enum MockStep {
    /// Emit a text delta.
    Text(String),
    /// Close the open text block.
    EndText,
    /// Emit a complete tool call.
    ToolCall { id: ToolCallId, name: SmolStr, arguments: serde_json::Value },
    /// Abort the stream with a permanent error.
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn model() -> ModelSpec {
        ModelSpec::builder()
            .id("mock:test")
            .name("mock")
            .api(Api::AnthropicMessages)
            .context_window(8_192)
            .max_output(1_024)
            .build()
            .unwrap()
    }

    async fn collect(provider: &MockProvider) -> Vec<ProviderEvent> {
        let model = model();
        let mut stream = provider
            .stream(&model, Prompt::default(), StreamOptions::default(), CancellationToken::new())
            .await
            .unwrap();
        let mut out = Vec::new();
        while let Some(event) = stream.next().await {
            out.push(event);
        }
        out
    }

    #[tokio::test]
    async fn empty_inbox_errors() {
        let provider = MockProvider::for_api(Api::AnthropicMessages);
        let events = collect(&provider).await;
        assert!(matches!(events.first(), Some(ProviderEvent::Start)));
        assert!(matches!(events.last(), Some(ProviderEvent::Error { .. })));
    }

    #[tokio::test]
    async fn scripted_text_yields_start_delta_end_done() {
        let provider = MockProvider::for_api(Api::AnthropicMessages);
        provider.append_text("hello world");
        let events = collect(&provider).await;
        assert!(matches!(events[0], ProviderEvent::Start));
        assert!(matches!(events[1], ProviderEvent::TextStart { idx: 0 }));
        assert!(
            matches!(&events[2], ProviderEvent::TextDelta { idx: 0, text } if text == "hello world")
        );
        assert!(matches!(events[3], ProviderEvent::TextEnd { idx: 0 }));
        match events.last().unwrap() {
            ProviderEvent::Done { stop, message } => {
                assert_eq!(*stop, StopReason::Stop);
                assert_eq!(message.content, vec![Content::text("hello world")]);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancel_before_steps_yields_cancel_error() {
        let provider = MockProvider::for_api(Api::AnthropicMessages)
            .with_step_delay(Duration::from_millis(10));
        provider.append(MockScript {
            steps: vec![MockStep::Text("never seen".into())],
            stop: StopReason::Stop,
        });
        let cancel = CancellationToken::new();
        cancel.cancel();
        let model = model();
        let mut stream = provider
            .stream(&model, Prompt::default(), StreamOptions::default(), cancel)
            .await
            .unwrap();
        let mut last = None;
        while let Some(event) = stream.next().await {
            last = Some(event);
        }
        assert!(matches!(last, Some(ProviderEvent::Error { reason: ErrorReason::Cancelled, .. })));
    }

    #[tokio::test]
    async fn clones_share_inbox() {
        let provider = MockProvider::for_api(Api::AnthropicMessages);
        let handle = provider.clone();
        handle.append_text("from clone");
        assert_eq!(provider.pending(), 1);
        let events = collect(&provider).await;
        assert!(matches!(events.last(), Some(ProviderEvent::Done { .. })));
        assert_eq!(handle.pending(), 0);
    }

    #[tokio::test]
    async fn scripted_tool_call_appears_in_done_message() {
        let provider = MockProvider::for_api(Api::AnthropicMessages);
        provider.append(MockScript::tool_call("read", serde_json::json!({ "path": "/x" })));
        let events = collect(&provider).await;
        match events.last().unwrap() {
            ProviderEvent::Done { stop, message } => {
                assert_eq!(*stop, StopReason::ToolUse);
                assert!(matches!(message.content.first(), Some(Content::ToolCall(_))));
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }
}
