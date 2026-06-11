use std::sync::Arc;

use riko_core::{Prompt, ToolDescriptor};
use smol_str::SmolStr;

use crate::{Item, ItemPayload};

/// Project a branch's items into the wire payload: walk in order, skip hidden items, fold
/// System text into the system prompt, and collect Message turns.
///
/// Pure projection — it transforms nothing. Any compaction or redaction is an operation that
/// already ran and left its result in the items. `pinned` is a hint for those operations and
/// the UI, so it does not affect rendering; only `hidden` excludes an item.
pub(crate) fn render(items: &[Arc<Item>], tools: Vec<ToolDescriptor>) -> Prompt {
    let mut system = String::new();
    let mut messages = Vec::new();
    for item in items {
        if item.meta.hidden {
            continue;
        }
        match &item.payload {
            ItemPayload::System { tag, text } => append_section(&mut system, tag.as_ref(), text),
            ItemPayload::Summary { text } => {
                append_section(&mut system, Some(&SmolStr::new_static("summary")), text)
            }
            ItemPayload::Message(message) => messages.push(message.clone()),
        }
    }
    Prompt { system, messages, tools }
}

/// Append a system section, blank-line separated, wrapping it in `<tag>…</tag>` when named.
fn append_section(system: &mut String, tag: Option<&SmolStr>, text: &str) {
    if !system.is_empty() {
        system.push_str("\n\n");
    }
    match tag {
        Some(tag) => {
            system.push('<');
            system.push_str(tag);
            system.push_str(">\n");
            system.push_str(text);
            system.push_str("\n</");
            system.push_str(tag);
            system.push('>');
        }
        None => system.push_str(text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ItemId, ItemMeta};
    use riko_core::{Content, Message, Role};

    fn item(payload: ItemPayload, hidden: bool) -> Arc<Item> {
        Arc::new(Item {
            id: ItemId::fresh(),
            created: riko_utils::Timestamp::now(),
            payload,
            meta: ItemMeta { hidden, ..Default::default() },
        })
    }

    fn sys(tag: Option<&str>, text: &str) -> ItemPayload {
        ItemPayload::System { tag: tag.map(SmolStr::from), text: text.into() }
    }

    fn user(text: &str) -> ItemPayload {
        ItemPayload::Message(Message { role: Role::User, content: vec![Content::text(text)] })
    }

    #[test]
    fn empty_renders_empty() {
        let prompt = render(&[], Vec::new());
        assert!(prompt.system.is_empty());
        assert!(prompt.messages.is_empty());
    }

    #[test]
    fn system_sections_concatenate_and_wrap_tags() {
        let items =
            vec![item(sys(None, "base"), false), item(sys(Some("env"), "date: 2026"), false)];
        let prompt = render(&items, Vec::new());
        assert_eq!(prompt.system, "base\n\n<env>\ndate: 2026\n</env>");
    }

    #[test]
    fn messages_kept_in_order() {
        let items = vec![item(user("first"), false), item(user("second"), false)];
        let prompt = render(&items, Vec::new());
        assert_eq!(prompt.messages.len(), 2);
        assert_eq!(prompt.messages[0].content, vec![Content::text("first")]);
    }

    #[test]
    fn hidden_items_are_skipped() {
        let items = vec![
            item(user("shown"), false),
            item(user("gone"), true),
            item(sys(None, "shown"), false),
            item(sys(None, "gone"), true),
        ];
        let prompt = render(&items, Vec::new());
        assert_eq!(prompt.messages.len(), 1);
        assert_eq!(prompt.system, "shown");
    }

    #[test]
    fn tools_pass_through() {
        let tools = vec![ToolDescriptor {
            name: "read".into(),
            description: "read a file".into(),
            schema: Default::default(),
        }];
        assert_eq!(render(&[], tools).tools.len(), 1);
    }
}
