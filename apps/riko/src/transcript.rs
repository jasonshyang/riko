use std::sync::Arc;

use riko_context::{Item, ItemPayload};
use riko_core::{Content, Message, Role};

/// What produced a transcript entry; drives its label and styling in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// Seeded system context (instructions, env, skills).
    System,
    User,
    Assistant,
    ToolCall,
    ToolResult {
        is_error: bool,
    },
    Summary,
    /// In-progress assistant text being streamed — not yet a workspace item.
    Pending,
}

/// One renderable block of the transcript: a kind plus its flattened text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub kind: EntryKind,
    pub text: String,
}

/// Project `items` (and any `pending` streamed assistant text) into ordered entries. A single
/// assistant message may yield several entries — e.g. prose followed by a tool call.
pub fn entries(items: &[Arc<Item>], pending: Option<&str>) -> Vec<Entry> {
    let mut out = Vec::with_capacity(items.len() + 1);
    for item in items {
        match &item.payload {
            ItemPayload::Message(message) => push_message(&mut out, message),
            ItemPayload::System { text, .. } => {
                out.push(Entry { kind: EntryKind::System, text: text.clone() });
            }
            ItemPayload::Summary { text } => {
                out.push(Entry { kind: EntryKind::Summary, text: text.clone() });
            }
        }
    }
    if let Some(text) = pending.filter(|t| !t.is_empty()) {
        out.push(Entry { kind: EntryKind::Pending, text: text.to_owned() });
    }
    out
}

fn push_message(out: &mut Vec<Entry>, message: &Message) {
    match &message.role {
        Role::User => out.push(Entry { kind: EntryKind::User, text: join_text(&message.content) }),
        Role::Assistant { .. } => push_assistant_blocks(out, &message.content),
        Role::ToolResult { is_error, .. } => out.push(Entry {
            kind: EntryKind::ToolResult { is_error: *is_error },
            text: join_text(&message.content),
        }),
    }
}

/// An assistant turn interleaves prose and tool calls; emit each block as its own entry, in
/// order. Thinking and image blocks aren't shown in the baseline transcript.
fn push_assistant_blocks(out: &mut Vec<Entry>, content: &[Content]) {
    for block in content {
        match block {
            Content::Text(text) => {
                out.push(Entry { kind: EntryKind::Assistant, text: text.clone() });
            }
            Content::ToolCall(call) => out.push(Entry {
                kind: EntryKind::ToolCall,
                text: format!("{}({})", call.name, call.arguments),
            }),
            Content::Thinking(_) | Content::Image(_) => {}
        }
    }
}

/// Concatenate a message's text blocks, newline-separated, ignoring non-text content.
fn join_text(content: &[Content]) -> String {
    let mut out = String::new();
    for block in content {
        if let Content::Text(text) = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use riko_context::Workspace;
    use riko_core::{ModelRef, StopReason, ToolCall, ToolCallId, Usage};
    use serde_json::json;

    /// Build real `Arc<Item>`s by funneling payloads through an in-memory workspace, so we never
    /// hand-construct an `Item` (its timestamp/id are workspace concerns).
    fn items(payloads: Vec<ItemPayload>) -> Vec<Arc<Item>> {
        let ws = Workspace::new();
        for payload in payloads {
            ws.add(payload).unwrap();
        }
        ws.items()
    }

    fn user(text: &str) -> ItemPayload {
        ItemPayload::Message(Message { role: Role::User, content: vec![Content::text(text)] })
    }

    fn assistant(content: Vec<Content>) -> ItemPayload {
        ItemPayload::Message(Message {
            role: Role::Assistant {
                model: ModelRef::new("anthropic:test"),
                usage: Usage::default(),
                stop: StopReason::Stop,
            },
            content,
        })
    }

    #[test]
    fn user_and_assistant_text_become_two_entries() {
        let items = items(vec![user("hi"), assistant(vec![Content::text("hello")])]);
        let entries = entries(&items, None);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], Entry { kind: EntryKind::User, text: "hi".into() });
        assert_eq!(entries[1], Entry { kind: EntryKind::Assistant, text: "hello".into() });
    }

    #[test]
    fn assistant_text_and_tool_call_split_into_ordered_entries() {
        let content = vec![
            Content::text("let me look"),
            Content::ToolCall(ToolCall {
                id: ToolCallId::new("tc_1"),
                name: "ls".into(),
                arguments: json!({ "path": "src" }),
            }),
        ];
        let entries = entries(&items(vec![assistant(content)]), None);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, EntryKind::Assistant);
        assert_eq!(entries[1].kind, EntryKind::ToolCall);
        assert!(entries[1].text.contains("ls("));
        assert!(entries[1].text.contains("\"path\""));
    }

    #[test]
    fn tool_result_carries_its_error_flag() {
        let payload = ItemPayload::Message(Message {
            role: Role::ToolResult {
                call_id: ToolCallId::new("tc_1"),
                tool: "ls".into(),
                is_error: true,
            },
            content: vec![Content::text("boom")],
        });
        let entries = entries(&items(vec![payload]), None);
        assert_eq!(
            entries[0],
            Entry { kind: EntryKind::ToolResult { is_error: true }, text: "boom".into() }
        );
    }

    #[test]
    fn system_and_summary_keep_their_kinds() {
        let payloads = vec![
            ItemPayload::System { tag: None, text: "you are riko".into() },
            ItemPayload::Summary { text: "earlier turns".into() },
        ];
        let entries = entries(&items(payloads), None);
        assert_eq!(entries[0].kind, EntryKind::System);
        assert_eq!(entries[1].kind, EntryKind::Summary);
    }

    #[test]
    fn pending_text_is_appended_last_and_empty_is_skipped() {
        let base = items(vec![user("go")]);
        assert_eq!(entries(&base, Some("thinking")).last().unwrap().kind, EntryKind::Pending);
        assert_eq!(entries(&base, Some("")).len(), 1, "empty pending adds nothing");
        assert_eq!(entries(&base, None).len(), 1);
    }
}
