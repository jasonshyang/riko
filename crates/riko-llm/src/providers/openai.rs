//! OpenAI Chat Completions API provider.
//!
//! One adapter covers every endpoint that speaks OpenAI's Chat Completions wire format —
//! OpenAI itself, plus Ollama, vLLM, LM Studio, llama.cpp server, xAI, Groq, Together,
//! DeepSeek, OpenRouter, Cerebras, etc. The model carries the `base_url` (so the right
//! endpoint is dialed) and an optional `api_key_env` (so the right auth header is sent —
//! or no header at all, for local runtimes).

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use riko_core::{
    Content, Message, ModelRef, Prompt, Result, RikoError, Role, StopReason, ToolCall, ToolCallId,
    Usage,
};
use riko_utils::b64;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use smol_str::SmolStr;
use tokio_util::sync::CancellationToken;

use crate::{
    Api, ErrorReason, ModelSpec, Provider, ProviderEvent, ProviderFuture, ProviderStream,
    StreamOptions,
    sse::{SseEvent, sse_event_stream},
};

/// Default base URL when a `ModelSpec` doesn't supply one. Matches OpenAI's public API.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// SSE sentinel that marks the end of an OpenAI completion stream.
const DONE_SENTINEL: &str = "[DONE]";

pub struct OpenAiCompletionsProvider {
    client: reqwest::Client,
}

impl OpenAiCompletionsProvider {
    pub fn new() -> Self {
        Self { client: reqwest::Client::new() }
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }
}

impl Default for OpenAiCompletionsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for OpenAiCompletionsProvider {
    fn api(&self) -> Api {
        Api::OpenAiCompletions
    }

    fn stream<'a>(
        &'a self,
        model: &'a ModelSpec,
        prompt: Prompt,
        opts: StreamOptions,
        cancel: CancellationToken,
    ) -> ProviderFuture<'a> {
        let base_url = model.base_url.clone().unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let model_wire = strip_provider_prefix(model.id.as_str()).to_string();
        let api_key = match resolve_api_key_for(model) {
            Ok(k) => k,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let client = self.client.clone();
        let model_id = model.id.clone();
        let timeout = opts.timeout;

        Box::pin(async move {
            let payload = build_request(&model_wire, &prompt, &opts)?;
            let mut request = client
                .post(format!("{}/chat/completions", base_url.trim_end_matches('/')))
                .header("content-type", "application/json")
                .header("accept", "text/event-stream");
            if let Some(key) = api_key {
                request = request.header("authorization", format!("Bearer {key}"));
            }

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
            let events = parse_openai_stream(bytes, model_id, cancel);
            Ok(Box::pin(events) as ProviderStream)
        })
    }
}

fn resolve_api_key_for(model: &ModelSpec) -> Result<Option<String>> {
    let Some(var) = model.api_key_env.as_ref() else {
        return Ok(None);
    };
    match std::env::var(var.as_str()) {
        Ok(v) if v.is_empty() => Err(RikoError::Config(format!(
            "model `{}` requires env `{var}` but it is empty",
            model.id
        ))),
        Ok(v) => Ok(Some(v)),
        Err(_) => Err(RikoError::Config(format!(
            "model `{}` requires env `{var}` but it is unset",
            model.id
        ))),
    }
}

fn strip_provider_prefix(reference: &str) -> &str {
    reference.split_once(':').map(|(_, rest)| rest).unwrap_or(reference)
}

fn transport_err(err: reqwest::Error) -> RikoError {
    RikoError::Provider(format!("openai-completions transport error: {err}"))
}

fn map_http_status(status: reqwest::StatusCode, body: String) -> RikoError {
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        RikoError::Provider(format!("openai-completions auth failed ({status}): {body}"))
    } else if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        RikoError::Provider(format!("openai-completions retryable error ({status}): {body}"))
    } else {
        RikoError::Provider(format!("openai-completions error ({status}): {body}"))
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

    let messages = encode_messages(&prompt.system, &prompt.messages)?;
    payload.insert("messages".into(), Value::Array(messages));

    if !prompt.tools.is_empty() {
        let tools = prompt
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name.as_str(),
                        "description": t.description,
                        "parameters": t.schema.as_value(),
                    }
                })
            })
            .collect();
        payload.insert("tools".into(), Value::Array(tools));
    }

    // Ask the server to include `usage` in the final chunk when supported.
    payload.insert("stream_options".into(), json!({ "include_usage": true }));

    Ok(Value::Object(payload))
}

fn encode_messages(system: &str, messages: &[Message]) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if !system.is_empty() {
        out.push(json!({ "role": "system", "content": system }));
    }
    for msg in messages {
        match &msg.role {
            Role::User => {
                out.push(json!({
                    "role": "user",
                    "content": encode_user_content(&msg.content),
                }));
            }
            Role::Assistant { .. } => {
                out.push(encode_assistant(&msg.content)?);
            }
            Role::ToolResult { call_id, .. } => {
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id.as_str(),
                    "content": text_of_blocks(&msg.content),
                }));
            }
        }
    }
    Ok(out)
}

fn encode_user_content(blocks: &[Content]) -> Value {
    let all_text = blocks.iter().all(|b| matches!(b, Content::Text { .. }));
    if all_text {
        return Value::String(text_of_blocks(blocks));
    }
    let parts: Vec<Value> = blocks
        .iter()
        .filter_map(|b| match b {
            Content::Text(text) => Some(json!({ "type": "text", "text": text })),
            Content::Image(img) => {
                let url = format!("data:{};base64,{}", img.mime.as_str(), b64::encode(&img.data));
                Some(json!({
                    "type": "image_url",
                    "image_url": { "url": url }
                }))
            }
            _ => None,
        })
        .collect();
    Value::Array(parts)
}

fn encode_assistant(blocks: &[Content]) -> Result<Value> {
    let mut text_parts: Vec<&str> = Vec::with_capacity(blocks.len());
    let mut tool_calls: Vec<Value> = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            Content::Text(text) => text_parts.push(text.as_str()),
            Content::ToolCall(tc) => tool_calls.push(json!({
                "id": tc.id.as_str(),
                "type": "function",
                "function": {
                    "name": tc.name.as_str(),
                    "arguments": serde_json::to_string(&tc.arguments)
                        .unwrap_or_else(|_| String::from("{}")),
                }
            })),
            // Thinking / Image are not part of the OpenAI assistant wire shape; drop them
            // silently so we don't fail the request.
            _ => {}
        }
    }
    let mut obj = Map::with_capacity(3);
    obj.insert("role".into(), Value::String("assistant".into()));
    if text_parts.is_empty() {
        obj.insert("content".into(), Value::Null);
    } else {
        obj.insert("content".into(), Value::String(text_parts.join("\n")));
    }
    if !tool_calls.is_empty() {
        obj.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    Ok(Value::Object(obj))
}

fn text_of_blocks(blocks: &[Content]) -> String {
    let mut out = String::with_capacity(64);
    for block in blocks {
        if let Content::Text(text) = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

// ─── Streaming response shape ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
struct ToolCallFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokenDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct PromptTokenDetails {
    #[serde(default)]
    cached_tokens: u32,
}

impl ChunkUsage {
    fn into_usage(self) -> Usage {
        Usage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            cache_read_tokens: self.prompt_tokens_details.map(|d| d.cached_tokens).unwrap_or(0),
            cache_write_tokens: 0,
        }
    }
}

// ─── Streaming projection ─────────────────────────────────────────────────────

struct ToolCallAccumulator {
    id: ToolCallId,
    name: SmolStr,
    args_buffer: String,
}

fn parse_openai_stream<S>(
    bytes: S,
    model: ModelRef,
    cancel: CancellationToken,
) -> impl Stream<Item = ProviderEvent>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let sse = sse_event_stream(bytes, "openai-completions");
    transform_to_provider_events(sse, model, cancel)
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
        let mut text_open = false;
        let mut text_buffer = String::new();
        let mut tool_calls: Vec<ToolCallAccumulator> = Vec::with_capacity(4);
        let mut text_idx: u32 = 0;
        let mut next_idx: u32 = 0;
        let mut tool_idx_lookup: std::collections::HashMap<u32, usize> = std::collections::HashMap::with_capacity(4);
        let mut usage = Usage::default();
        let mut finish_reason: Option<String> = None;
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
            let Some(item) = next else { break };
            let event = match item {
                Ok(e) => e,
                Err(e) => {
                    errored = true;
                    yield ProviderEvent::Error { reason: ErrorReason::Transport, message: e.to_string() };
                    break;
                }
            };
            if event.data == DONE_SENTINEL {
                break;
            }
            let chunk: ChatChunk = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(e) => {
                    errored = true;
                    yield ProviderEvent::Error {
                        reason: ErrorReason::Permanent,
                        message: format!("openai-completions parse error: {e}"),
                    };
                    break;
                }
            };

            if let Some(u) = chunk.usage {
                usage.merge(u.into_usage());
            }

            for choice in &chunk.choices {
                if let Some(content) = choice.delta.content.as_deref() {
                    if !text_open {
                        text_idx = next_idx;
                        next_idx += 1;
                        text_open = true;
                        yield ProviderEvent::TextStart { idx: text_idx };
                    }
                    text_buffer.push_str(content);
                    yield ProviderEvent::TextDelta { idx: text_idx, text: content.to_string() };
                }
                if let Some(deltas) = choice.delta.tool_calls.as_ref() {
                    for delta in deltas {
                        let position = match tool_idx_lookup.get(&delta.index).copied() {
                            Some(pos) => pos,
                            None => {
                                let pos = tool_calls.len();
                                let call_id = delta
                                    .id
                                    .clone()
                                    .map(ToolCallId::new)
                                    .unwrap_or_else(ToolCallId::fresh);
                                let name = delta
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.name.clone())
                                    .map(SmolStr::from)
                                    .unwrap_or_default();
                                tool_calls.push(ToolCallAccumulator {
                                    id: call_id.clone(),
                                    name: name.clone(),
                                    args_buffer: String::new(),
                                });
                                tool_idx_lookup.insert(delta.index, pos);
                                let assigned_idx = next_idx;
                                next_idx += 1;
                                yield ProviderEvent::ToolCallStart {
                                    idx: assigned_idx,
                                    call_id,
                                    name,
                                };
                                pos
                            }
                        };
                        if let Some(func) = &delta.function {
                            if let Some(name) = func.name.as_ref()
                                && tool_calls[position].name.is_empty()
                            {
                                tool_calls[position].name = SmolStr::from(name);
                            }
                            if let Some(args_chunk) = func.arguments.as_ref() {
                                tool_calls[position].args_buffer.push_str(args_chunk);
                                yield ProviderEvent::ToolCallArgsDelta {
                                    idx: (position as u32) + (text_idx + 1),
                                    fragment: args_chunk.clone(),
                                };
                            }
                        }
                    }
                }
                if let Some(reason) = &choice.finish_reason {
                    finish_reason = Some(reason.clone());
                }
            }
        }

        if errored {
            return;
        }

        if text_open {
            yield ProviderEvent::TextEnd { idx: text_idx };
        }
        for (position, acc) in tool_calls.iter().enumerate() {
            let args: Value = serde_json::from_str(&acc.args_buffer).unwrap_or(Value::Null);
            let idx = (position as u32) + if text_open { 1 } else { 0 };
            yield ProviderEvent::ToolCallEnd { idx, args };
        }

        yield ProviderEvent::UsageUpdate(usage);

        let stop = map_finish_reason(finish_reason.as_deref());
        let message = build_final_message(model, text_open, text_buffer, &tool_calls, usage, stop);
        yield ProviderEvent::Done { stop, message: Box::new(message) };
    }
}

fn map_finish_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length") => StopReason::Length,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("content_filter") => StopReason::Stop,
        _ => StopReason::Stop,
    }
}

fn build_final_message(
    model: ModelRef,
    has_text: bool,
    text: String,
    tool_calls: &[ToolCallAccumulator],
    usage: Usage,
    stop: StopReason,
) -> Message {
    let mut content = Vec::with_capacity(1 + tool_calls.len());
    if has_text && !text.is_empty() {
        content.push(Content::Text(text));
    }
    for acc in tool_calls {
        let arguments: Value = serde_json::from_str(&acc.args_buffer).unwrap_or(Value::Null);
        content.push(Content::ToolCall(ToolCall {
            id: acc.id.clone(),
            name: acc.name.clone(),
            arguments,
        }));
    }
    Message { role: Role::Assistant { model, usage, stop }, content }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelCost;
    use std::collections::HashMap;

    fn sample_model(api_key_env: Option<&str>) -> ModelSpec {
        ModelSpec {
            id: ModelRef::new("openai:gpt-test"),
            name: "gpt-test".into(),
            api: Api::OpenAiCompletions,
            base_url: Some("https://api.openai.com/v1".into()),
            context_window: 128_000,
            max_output: 4_096,
            cost: ModelCost::default(),
            thinking_levels: HashMap::new(),
            supports_images: true,
            supports_tools: true,
            api_key_env: api_key_env.map(SmolStr::from),
        }
    }

    #[test]
    fn build_request_prepends_system_message() {
        let prompt = Prompt {
            system: "be terse".into(),
            messages: vec![Message { role: Role::User, content: vec![Content::text("hi")] }],
            tools: Vec::new(),
        };
        let payload = build_request("gpt-test", &prompt, &StreamOptions::default()).unwrap();
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be terse");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hi");
    }

    #[test]
    fn tool_result_serializes_with_tool_role_and_call_id() {
        let prompt = Prompt {
            system: String::new(),
            messages: vec![Message {
                role: Role::ToolResult {
                    call_id: ToolCallId::new("call_42"),
                    tool: "read".into(),
                    is_error: false,
                },
                content: vec![Content::text("file body")],
            }],
            tools: Vec::new(),
        };
        let payload = build_request("gpt-test", &prompt, &StreamOptions::default()).unwrap();
        let msg = &payload["messages"][0];
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "call_42");
        assert_eq!(msg["content"], "file body");
    }

    #[test]
    fn assistant_with_tool_call_packs_tool_calls_array() {
        let prompt = Prompt {
            system: String::new(),
            messages: vec![Message {
                role: Role::Assistant {
                    model: ModelRef::new("openai:gpt-test"),
                    usage: Usage::default(),
                    stop: StopReason::ToolUse,
                },
                content: vec![
                    Content::text("let me check"),
                    Content::ToolCall(ToolCall {
                        id: ToolCallId::new("call_99"),
                        name: "read".into(),
                        arguments: json!({ "path": "src" }),
                    }),
                ],
            }],
            tools: Vec::new(),
        };
        let payload = build_request("gpt-test", &prompt, &StreamOptions::default()).unwrap();
        let msg = &payload["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "let me check");
        let calls = msg["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "read");
    }

    #[test]
    fn missing_env_var_for_required_key_fails_loudly() {
        let model = sample_model(Some("RIKO_NEVER_SET_TEST_VAR_XYZ"));
        let err = resolve_api_key_for(&model).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("RIKO_NEVER_SET_TEST_VAR_XYZ"));
    }

    #[test]
    fn no_api_key_env_resolves_to_none() {
        let model = sample_model(None);
        assert!(resolve_api_key_for(&model).unwrap().is_none());
    }

    #[test]
    fn strip_provider_prefix_removes_openai_prefix() {
        assert_eq!(strip_provider_prefix("openai:gpt-4o"), "gpt-4o");
        assert_eq!(strip_provider_prefix("ollama:llama-3.1-8b"), "llama-3.1-8b");
        assert_eq!(strip_provider_prefix("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn map_finish_reason_handles_known_codes() {
        assert_eq!(map_finish_reason(Some("stop")), StopReason::Stop);
        assert_eq!(map_finish_reason(Some("length")), StopReason::Length);
        assert_eq!(map_finish_reason(Some("tool_calls")), StopReason::ToolUse);
        assert_eq!(map_finish_reason(None), StopReason::Stop);
    }

    fn build_text_stream_bytes() -> Bytes {
        let s = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        Bytes::copy_from_slice(s.as_bytes())
    }

    #[tokio::test]
    async fn text_stream_produces_start_delta_end_done() {
        let bytes_stream =
            futures::stream::iter(vec![Ok::<_, reqwest::Error>(build_text_stream_bytes())]);
        let cancel = CancellationToken::new();
        let mut stream =
            Box::pin(parse_openai_stream(bytes_stream, ModelRef::new("openai:gpt-test"), cancel));
        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }
        assert!(matches!(events[0], ProviderEvent::Start));
        assert!(matches!(events[1], ProviderEvent::TextStart { idx: 0 }));
        assert!(matches!(&events[2], ProviderEvent::TextDelta { idx: 0, text } if text == "hello"));
        assert!(
            matches!(&events[3], ProviderEvent::TextDelta { idx: 0, text } if text == " world")
        );
        match events.last().unwrap() {
            ProviderEvent::Done { stop, message } => {
                assert_eq!(*stop, StopReason::Stop);
                assert_eq!(message.content.len(), 1);
                if let Content::Text(text) = &message.content[0] {
                    assert_eq!(text, "hello world");
                } else {
                    panic!("expected Text content");
                }
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    fn build_tool_call_stream_bytes() -> Bytes {
        let s = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\":\\\"x.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        Bytes::copy_from_slice(s.as_bytes())
    }

    #[tokio::test]
    async fn tool_call_stream_accumulates_arguments_and_emits_end() {
        let bytes_stream =
            futures::stream::iter(vec![Ok::<_, reqwest::Error>(build_tool_call_stream_bytes())]);
        let cancel = CancellationToken::new();
        let mut stream =
            Box::pin(parse_openai_stream(bytes_stream, ModelRef::new("openai:gpt-test"), cancel));
        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }
        let mut saw_start = false;
        let mut saw_end = false;
        for evt in &events {
            match evt {
                ProviderEvent::ToolCallStart { name, call_id, .. } => {
                    saw_start = true;
                    assert_eq!(name.as_str(), "read");
                    assert_eq!(call_id.as_str(), "call_a");
                }
                ProviderEvent::ToolCallEnd { args, .. } => {
                    saw_end = true;
                    assert_eq!(args["path"], "x.rs");
                }
                _ => {}
            }
        }
        assert!(saw_start, "expected ToolCallStart");
        assert!(saw_end, "expected ToolCallEnd");
        match events.last().unwrap() {
            ProviderEvent::Done { stop, message } => {
                assert_eq!(*stop, StopReason::ToolUse);
                let tool_call = message
                    .content
                    .iter()
                    .find_map(|c| match c {
                        Content::ToolCall(tc) => Some(tc),
                        _ => None,
                    })
                    .expect("tool call in final message");
                assert_eq!(tool_call.name.as_str(), "read");
                assert_eq!(tool_call.arguments["path"], "x.rs");
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }
}
