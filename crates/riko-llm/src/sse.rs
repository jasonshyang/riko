//! Generic Server-Sent Events parser used by every HTTP-based provider.
//!
//! Producers hand in a `Stream<Item = Result<Bytes, reqwest::Error>>` of raw response
//! bytes; the parser yields one [`SseEvent`] per `event: … / data: …` block delimited by
//! a blank line. Provider-specific code consumes the events and maps them to the
//! normalized [`crate::ProviderEvent`] taxonomy.

use async_stream::stream;
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use riko_core::RikoError;

/// One SSE block — optional `event:` name plus the joined `data:` lines.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
}

/// Adapter from raw response bytes to a stream of [`SseEvent`]s.
///
/// The producer's `label` is only used to compose error messages so the caller can tell
/// at a glance which provider's stream failed.
pub fn sse_event_stream<S>(
    bytes: S,
    label: &'static str,
) -> impl Stream<Item = std::result::Result<SseEvent, RikoError>>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    stream! {
        let mut buffer = BytesMut::with_capacity(4 * 1024);
        let mut bytes = Box::pin(bytes);
        while let Some(chunk) = bytes.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(RikoError::Provider(format!("{label} stream error: {e}")));
                    return;
                }
            };
            buffer.extend_from_slice(&chunk);
            while let Some(event) = take_event(&mut buffer) {
                yield Ok(event);
            }
        }
        if let Some(event) = flush_trailing(&mut buffer) {
            yield Ok(event);
        }
    }
}

fn take_event(buffer: &mut BytesMut) -> Option<SseEvent> {
    let pos = find_blank_line(buffer)?;
    let block = buffer.split_to(pos);
    let _ = buffer.split_to(buffer_eat_blank_line_prefix(buffer));
    parse_event_block(&block)
}

fn flush_trailing(buffer: &mut BytesMut) -> Option<SseEvent> {
    if buffer.is_empty() {
        return None;
    }
    let block = buffer.split_to(buffer.len());
    parse_event_block(&block)
}

fn find_blank_line(buffer: &BytesMut) -> Option<usize> {
    if let Some(pos) = find_subsequence(buffer, b"\n\n") {
        return Some(pos);
    }
    find_subsequence(buffer, b"\r\n\r\n")
}

fn buffer_eat_blank_line_prefix(buffer: &BytesMut) -> usize {
    if buffer.get(0..2) == Some(b"\n\n") {
        2
    } else if buffer.get(0..4) == Some(b"\r\n\r\n") {
        4
    } else {
        0
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Parse one event block (everything up to the blank line delimiter).
///
/// Public so individual providers can drive their own buffering when they need to (e.g.,
/// tests).
pub fn parse_event_block(block: &[u8]) -> Option<SseEvent> {
    let text = std::str::from_utf8(block).ok()?;
    let mut event_name = String::new();
    let mut data = String::with_capacity(64);
    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        }
    }
    if event_name.is_empty() && data.is_empty() {
        None
    } else {
        Some(SseEvent { event: event_name, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_event_block_extracts_name_and_data() {
        let block = b"event: chunk\ndata: {\"x\":1}";
        let parsed = parse_event_block(block).unwrap();
        assert_eq!(parsed.event, "chunk");
        assert_eq!(parsed.data, "{\"x\":1}");
    }

    #[test]
    fn parse_event_block_joins_multi_data_lines() {
        let block = b"data: {\"a\":1,\ndata: \"b\":2}";
        let parsed = parse_event_block(block).unwrap();
        assert_eq!(parsed.data, "{\"a\":1,\n\"b\":2}");
    }

    #[test]
    fn parse_event_block_handles_crlf() {
        let block = b"event: x\r\ndata: hello\r\n";
        let parsed = parse_event_block(block).unwrap();
        assert_eq!(parsed.event, "x");
        assert_eq!(parsed.data, "hello");
    }
}
