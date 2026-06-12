//! Anthropic Messages API provider.
//!
//! Translates a [`Prompt`] into the Anthropic request shape, opens a streaming HTTP
//! response, parses the SSE event stream, and emits normalized [`ProviderEvent`]s.

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use riko_core::{
    Content, Message, ModelRef, Prompt, Result, RikoError, Role, StopReason, Thinking, ToolCall,
    ToolCallId, Usage,
};
use riko_utils::b64;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use smol_str::SmolStr;
use tokio_util::sync::CancellationToken;

use crate::{
    Api, CacheRetention, ErrorReason, ModelSpec, Provider, ProviderEvent, ProviderFuture,
    ProviderStream, StreamOptions,
    sse::{SseEvent, sse_event_stream},
};

/// Default Anthropic API base URL.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Anthropic API version header value.
const API_VERSION: &str = "2023-06-01";
/// Beta feature header for 1h extended cache TTL.
const CACHE_LONG_BETA: &str = "extended-cache-ttl-2025-04-11";

/// Anthropic Messages provider.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    /// Construct a provider with a resolved API key. Uses the default Anthropic base URL.
    pub fn new(api_key: String) -> Self {
        Self { client: reqwest::Client::new(), api_key, base_url: DEFAULT_BASE_URL.into() }
    }

    /// Override the base URL — used for Anthropic-compatible endpoints behind a proxy.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Override the underlying HTTP client (e.g., to set a custom timeout or proxy).
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

impl Provider for AnthropicProvider {
    fn api(&self) -> Api {
        Api::AnthropicMessages
    }

    fn stream<'a>(
        &'a self,
        model: &'a ModelSpec,
        prompt: Prompt,
        opts: StreamOptions,
        cancel: CancellationToken,
    ) -> ProviderFuture<'a> {
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let model_id = model.id.clone();
        let model_wire = strip_provider_prefix(model_id.as_str()).to_string();
        let timeout = opts.timeout;
        let cache_long = opts.cache == CacheRetention::Persistent;

        Box::pin(async move {
            let payload = build_request(&model_wire, &prompt, &opts)?;
            let request = client
                .post(format!("{base_url}/v1/messages"))
                .header("x-api-key", &api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .header("accept", "text/event-stream");
            let request = if cache_long {
                request.header("anthropic-beta", CACHE_LONG_BETA)
            } else {
                request
            };

            let send_fut = request.timeout(timeout).json(&payload).send();
            let response = tokio::select! {
                res = send_fut => res.map_err(transport_err)?,
                _ = cancel.cancelled() => return Err(RikoError::Cancelled),
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(map_http_status(status, body));
            }

            let bytes = response.bytes_stream();
            let events = parse_anthropic_stream(bytes, model_id, cancel);
            Ok(Box::pin(events) as ProviderStream)
        })
    }
}

fn strip_provider_prefix(reference: &str) -> &str {
    reference.split_once(':').map(|(_, rest)| rest).unwrap_or(reference)
}

fn transport_err(err: reqwest::Error) -> RikoError {
    RikoError::Provider(format!("anthropic transport error: {err}"))
}

fn map_http_status(status: reqwest::StatusCode, body: String) -> RikoError {
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        RikoError::Provider(format!("anthropic auth failed ({status}): {body}"))
    } else if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        RikoError::Provider(format!("anthropic retryable error ({status}): {body}"))
    } else {
        RikoError::Provider(format!("anthropic error ({status}): {body}"))
    }
}

fn build_request(model: &str, prompt: &Prompt, opts: &StreamOptions) -> Result<Value> {
    let mut payload = Map::with_capacity(8);
    payload.insert("model".into(), Value::String(model.into()));
    payload.insert("max_tokens".into(), Value::from(opts.max_output_tokens));
    payload.insert("stream".into(), Value::Bool(true));

    if let Some(temp) = opts.temperature {
        payload.insert("temperature".into(), json!(temp));
    }

    if !prompt.system.is_empty() {
        payload.insert("system".into(), Value::String(prompt.system.clone()));
    }

    payload.insert("messages".into(), Value::Array(encode_messages(&prompt.messages)?));

    if !prompt.tools.is_empty() {
        let tools = prompt
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name.as_str(),
                    "description": t.description,
                    "input_schema": t.schema.as_value(),
                })
            })
            .collect();
        payload.insert("tools".into(), Value::Array(tools));
    }

    Ok(Value::Object(payload))
}

fn encode_messages(messages: &[Message]) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        match &msg.role {
            Role::User => {
                out.push(json!({
                    "role": "user",
                    "content": encode_content_blocks(&msg.content)?,
                }));
            }
            Role::Assistant { .. } => {
                out.push(json!({
                    "role": "assistant",
                    "content": encode_content_blocks(&msg.content)?,
                }));
            }
            Role::ToolResult { call_id, is_error, .. } => {
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": call_id.as_str(),
                        "is_error": is_error,
                        "content": encode_content_blocks(&msg.content)?,
                    }]
                }));
            }
        }
    }
    Ok(out)
}

fn encode_content_blocks(blocks: &[Content]) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        out.push(match block {
            Content::Text(text) => json!({ "type": "text", "text": text }),
            Content::Image(img) => {
                json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime.as_str(),
                        "data": b64::encode(&img.data),
                    }
                })
            }
            Content::Thinking(t) => {
                let mut obj = Map::with_capacity(3);
                obj.insert("type".into(), Value::String("thinking".into()));
                obj.insert("thinking".into(), Value::String(t.text.clone()));
                if let Some(sig) = &t.signature {
                    obj.insert("signature".into(), Value::String(sig.clone()));
                }
                Value::Object(obj)
            }
            Content::ToolCall(call) => json!({
                "type": "tool_use",
                "id": call.id.as_str(),
                "name": call.name.as_str(),
                "input": call.arguments,
            }),
        });
    }
    Ok(out)
}

fn parse_anthropic_stream<S>(
    bytes: S,
    model: ModelRef,
    cancel: CancellationToken,
) -> impl Stream<Item = ProviderEvent>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let sse = sse_event_stream(bytes, "anthropic");
    transform_to_provider_events(sse, model, cancel)
}

// ─── Anthropic event types ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessageHeader },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { index: u32, content_block: AnthropicBlock },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: AnthropicDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDelta,
        #[serde(default)]
        usage: Option<AnthropicUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: AnthropicError },
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageHeader {
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicBlock {
    #[serde(rename = "text")]
    Text {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
}

impl AnthropicUsage {
    fn into_usage(self) -> Usage {
        Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_read_tokens: self.cache_read_input_tokens,
            cache_write_tokens: self.cache_creation_input_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    message: String,
}

// ─── Provider-event projection ────────────────────────────────────────────────

#[derive(Default)]
struct StreamBuilder {
    blocks: Vec<BlockState>,
    usage: Usage,
}

enum BlockState {
    Text { text: String },
    Thinking { text: String, signature: Option<String> },
    ToolCall { call_id: ToolCallId, name: SmolStr, args_buffer: String },
}

impl StreamBuilder {
    fn block_at(&mut self, idx: u32) -> Option<&mut BlockState> {
        self.blocks.get_mut(idx as usize)
    }

    fn merge_usage(&mut self, partial: Option<AnthropicUsage>) {
        if let Some(u) = partial {
            self.usage.merge(u.into_usage());
        }
    }

    fn finalize_message(&self, model: ModelRef, stop: StopReason) -> Message {
        let mut content = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            match block {
                BlockState::Text { text } if !text.is_empty() => {
                    content.push(Content::Text(text.clone()));
                }
                BlockState::Thinking { text, signature } => {
                    content.push(Content::Thinking(Thinking {
                        text: text.clone(),
                        signature: signature.clone(),
                        redacted: false,
                    }));
                }
                BlockState::ToolCall { call_id, name, args_buffer } => {
                    let args: Value = serde_json::from_str(args_buffer)
                        .unwrap_or_else(|_| Value::String(args_buffer.clone()));
                    content.push(Content::ToolCall(ToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        arguments: args,
                    }));
                }
                _ => {}
            }
        }
        Message { role: Role::Assistant { model, usage: self.usage, stop }, content }
    }
}

fn transform_to_provider_events<S>(
    sse: S,
    model: ModelRef,
    cancel: CancellationToken,
) -> impl Stream<Item = ProviderEvent>
where
    S: Stream<Item = std::result::Result<SseEvent, RikoError>> + Send + 'static,
{
    stream! {
        yield ProviderEvent::Start;
        let mut sse = Box::pin(sse);
        let mut builder = StreamBuilder::default();
        let mut stop = StopReason::Stop;
        let mut errored = false;

        loop {
            let next = tokio::select! {
                evt = sse.next() => evt,
                _ = cancel.cancelled() => {
                    yield ProviderEvent::Error {
                        reason: ErrorReason::Cancelled,
                        message: "cancelled".into(),
                    };
                    return;
                }
            };
            let Some(evt_result) = next else { break };
            let sse_event = match evt_result {
                Ok(e) => e,
                Err(e) => {
                    errored = true;
                    yield ProviderEvent::Error { reason: ErrorReason::Transport, message: e.to_string() };
                    break;
                }
            };

            let parsed: std::result::Result<AnthropicEvent, _> = serde_json::from_str(&sse_event.data);
            let parsed = match parsed {
                Ok(p) => p,
                Err(e) => {
                    errored = true;
                    yield ProviderEvent::Error {
                        reason: ErrorReason::Permanent,
                        message: format!("anthropic parse error on `{}`: {e}", sse_event.event),
                    };
                    break;
                }
            };

            match parsed {
                AnthropicEvent::Ping => {}
                AnthropicEvent::MessageStart { message } => {
                    builder.merge_usage(message.usage);
                }
                AnthropicEvent::ContentBlockStart { index, content_block } => {
                    while builder.blocks.len() <= index as usize {
                        builder.blocks.push(BlockState::Text { text: String::new() });
                    }
                    match content_block {
                        AnthropicBlock::Text { text } => {
                            builder.blocks[index as usize] = BlockState::Text { text: String::new() };
                            yield ProviderEvent::TextStart { idx: index };
                            if !text.is_empty() {
                                if let Some(BlockState::Text { text: t }) = builder.block_at(index) {
                                    t.push_str(&text);
                                }
                                yield ProviderEvent::TextDelta { idx: index, text };
                            }
                        }
                        AnthropicBlock::Thinking { thinking, signature } => {
                            builder.blocks[index as usize] = BlockState::Thinking {
                                text: String::new(),
                                signature: signature.clone(),
                            };
                            yield ProviderEvent::ThinkingStart { idx: index };
                            if !thinking.is_empty() {
                                if let Some(BlockState::Thinking { text, .. }) = builder.block_at(index) {
                                    text.push_str(&thinking);
                                }
                                yield ProviderEvent::ThinkingDelta { idx: index, text: thinking, signature };
                            }
                        }
                        AnthropicBlock::ToolUse { id, name, input } => {
                            let call_id = ToolCallId::new(id);
                            let name = SmolStr::from(name);
                            let args_buffer = if input.is_null() { String::new() } else { input.to_string() };
                            builder.blocks[index as usize] = BlockState::ToolCall {
                                call_id: call_id.clone(),
                                name: name.clone(),
                                args_buffer,
                            };
                            yield ProviderEvent::ToolCallStart { idx: index, call_id, name };
                        }
                    }
                }
                AnthropicEvent::ContentBlockDelta { index, delta } => match delta {
                    AnthropicDelta::Text { text } => {
                        if let Some(BlockState::Text { text: t }) = builder.block_at(index) {
                            t.push_str(&text);
                        }
                        yield ProviderEvent::TextDelta { idx: index, text };
                    }
                    AnthropicDelta::Thinking { thinking } => {
                        if let Some(BlockState::Thinking { text, .. }) = builder.block_at(index) {
                            text.push_str(&thinking);
                        }
                        yield ProviderEvent::ThinkingDelta { idx: index, text: thinking, signature: None };
                    }
                    AnthropicDelta::Signature { signature } => {
                        if let Some(BlockState::Thinking { signature: sig, .. }) = builder.block_at(index) {
                            *sig = Some(signature);
                        }
                    }
                    AnthropicDelta::InputJson { partial_json } => {
                        if let Some(BlockState::ToolCall { args_buffer, .. }) = builder.block_at(index) {
                            args_buffer.push_str(&partial_json);
                            yield ProviderEvent::ToolCallArgsDelta {
                                idx: index,
                                fragment: partial_json,
                            };
                        }
                    }
                },
                AnthropicEvent::ContentBlockStop { index } => match builder.block_at(index) {
                    Some(BlockState::Text { .. }) => yield ProviderEvent::TextEnd { idx: index },
                    Some(BlockState::Thinking { .. }) => yield ProviderEvent::ThinkingEnd { idx: index },
                    Some(BlockState::ToolCall { args_buffer, .. }) => {
                        let args: Value = serde_json::from_str(args_buffer).unwrap_or(Value::Null);
                        yield ProviderEvent::ToolCallEnd { idx: index, args };
                    }
                    None => {}
                },
                AnthropicEvent::MessageDelta { delta, usage } => {
                    builder.merge_usage(usage);
                    if let Some(reason) = delta.stop_reason {
                        stop = map_stop_reason(&reason);
                    }
                }
                AnthropicEvent::MessageStop => {
                    yield ProviderEvent::UsageUpdate(builder.usage);
                    let message = builder.finalize_message(model.clone(), stop);
                    yield ProviderEvent::Done { stop, message: Box::new(message) };
                    return;
                }
                AnthropicEvent::Error { error } => {
                    errored = true;
                    yield ProviderEvent::Error {
                        reason: classify_anthropic_error(&error.kind),
                        message: error.message,
                    };
                    break;
                }
            }
        }

        if !errored {
            yield ProviderEvent::Error {
                reason: ErrorReason::Transport,
                message: "anthropic stream ended without message_stop".into(),
            };
        }
    }
}

fn map_stop_reason(raw: &str) -> StopReason {
    match raw {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "refusal" => StopReason::Stop,
        _ => StopReason::Stop,
    }
}

fn classify_anthropic_error(kind: &str) -> ErrorReason {
    match kind {
        "overloaded_error" | "api_error" => ErrorReason::Retryable,
        "rate_limit_error" => ErrorReason::Retryable,
        "authentication_error" | "permission_error" => ErrorReason::Auth,
        _ => ErrorReason::Permanent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelCost;
    use std::collections::HashMap;

    fn sample_model() -> ModelSpec {
        ModelSpec {
            id: ModelRef::new("anthropic:claude-test"),
            name: "claude-test".into(),
            api: Api::AnthropicMessages,
            base_url: None,
            context_window: 200_000,
            max_output: 4_096,
            cost: ModelCost::default(),
            thinking_levels: HashMap::new(),
            supports_images: true,
            supports_tools: true,
            api_key_env: None,
        }
    }

    #[test]
    fn build_request_includes_system_and_messages() {
        let prompt = Prompt {
            system: "be terse".into(),
            messages: vec![Message { role: Role::User, content: vec![Content::text("hi")] }],
            tools: Vec::new(),
        };
        let payload = build_request("claude-test", &prompt, &StreamOptions::default()).unwrap();
        assert_eq!(payload["model"], "claude-test");
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["system"], "be terse");
        assert_eq!(payload["messages"][0]["role"], "user");
        assert_eq!(payload["messages"][0]["content"][0]["type"], "text");
        assert_eq!(payload["messages"][0]["content"][0]["text"], "hi");
        assert!(payload.get("tools").is_none());
    }

    #[test]
    fn tool_result_translates_to_user_tool_result_block() {
        let prompt = Prompt {
            system: String::new(),
            messages: vec![Message {
                role: Role::ToolResult {
                    call_id: ToolCallId::new("tc_abc"),
                    tool: "read".into(),
                    is_error: false,
                },
                content: vec![Content::text("file body")],
            }],
            tools: Vec::new(),
        };
        let payload = build_request("claude-test", &prompt, &StreamOptions::default()).unwrap();
        let msg = &payload["messages"][0];
        assert_eq!(msg["role"], "user");
        assert_eq!(msg["content"][0]["type"], "tool_result");
        assert_eq!(msg["content"][0]["tool_use_id"], "tc_abc");
        assert_eq!(msg["content"][0]["is_error"], false);
    }

    #[test]
    fn strip_provider_prefix_removes_anthropic_prefix() {
        assert_eq!(strip_provider_prefix("anthropic:claude-opus-4-7"), "claude-opus-4-7");
        assert_eq!(strip_provider_prefix("claude-opus-4-7"), "claude-opus-4-7");
    }

    #[test]
    fn map_stop_reason_handles_known_codes() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::Length);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(map_stop_reason("nonsense"), StopReason::Stop);
    }

    #[test]
    fn classify_error_kinds() {
        assert_eq!(classify_anthropic_error("rate_limit_error"), ErrorReason::Retryable);
        assert_eq!(classify_anthropic_error("authentication_error"), ErrorReason::Auth);
        assert_eq!(classify_anthropic_error("invalid_request_error"), ErrorReason::Permanent);
    }

    fn build_text_stream_bytes() -> Bytes {
        let s = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":0,\"output_tokens\":2}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        Bytes::copy_from_slice(s.as_bytes())
    }

    #[tokio::test]
    async fn projects_text_stream_into_provider_events() {
        let bytes_stream =
            futures::stream::iter(vec![Ok::<_, reqwest::Error>(build_text_stream_bytes())]);
        let model = sample_model();
        let cancel = CancellationToken::new();
        let mut stream = Box::pin(parse_anthropic_stream(bytes_stream, model.id.clone(), cancel));
        let mut events = Vec::with_capacity(16);
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }
        assert!(matches!(events[0], ProviderEvent::Start));
        assert!(matches!(events[1], ProviderEvent::TextStart { idx: 0 }));
        assert!(matches!(&events[2], ProviderEvent::TextDelta { idx: 0, text } if text == "hello"));
        assert!(
            matches!(&events[3], ProviderEvent::TextDelta { idx: 0, text } if text == " world")
        );
        assert!(matches!(events[4], ProviderEvent::TextEnd { idx: 0 }));
        assert!(matches!(events[events.len() - 2], ProviderEvent::UsageUpdate(_)));
        match events.last() {
            Some(ProviderEvent::Done { stop, message }) => {
                assert_eq!(*stop, StopReason::Stop);
                assert_eq!(message.content.len(), 1);
                if let Content::Text(text) = &message.content[0] {
                    assert_eq!(text, "hello world");
                } else {
                    panic!("expected text content");
                }
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }
}
