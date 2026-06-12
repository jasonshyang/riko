use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::app::App;
use crate::transcript::EntryKind;

pub fn draw(frame: &mut Frame, app: &App) {
    let areas = Layout::vertical([
        Constraint::Min(1),    // transcript
        Constraint::Length(3), // input box
        Constraint::Length(1), // status line
    ])
    .split(frame.area());

    draw_transcript(frame, app, areas[0]);
    draw_input(frame, app, areas[1]);
    draw_status(frame, app, areas[2]);
}

fn draw_transcript(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();
    for entry in app.entries() {
        // Seeded system context is hidden from the baseline chat view (a payload preview comes
        // with the editable-workspace surface in H2).
        if entry.kind == EntryKind::System {
            continue;
        }
        let (label, style) = label_style(entry.kind);
        lines.push(Line::from(Span::styled(label, style.add_modifier(Modifier::BOLD))));
        for text_line in entry.text.lines() {
            lines.push(Line::from(text_line.to_owned()));
        }
        lines.push(Line::from(""));
    }

    // Anchor to the bottom so the latest turn stays visible. Counts logical lines, not wrapped
    // rows — close enough for the baseline; wrap-aware scrolling is a later refinement.
    let scroll = (lines.len() as u16).saturating_sub(area.height);
    let transcript = Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((scroll, 0));
    frame.render_widget(transcript, area);
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let title = if app.is_running() {
        " input — running (Ctrl-C aborts) "
    } else {
        " input — Enter to send, Ctrl-C/Esc to quit "
    };
    let input = Paragraph::new(app.input())
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(input, area);
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let status = Paragraph::new(Line::from(Span::styled(
        format!(" {}", app.status()),
        Style::default().fg(Color::DarkGray),
    )));
    frame.render_widget(status, area);
}

/// A short role label and its accent color for one transcript entry.
fn label_style(kind: EntryKind) -> (&'static str, Style) {
    match kind {
        EntryKind::User => ("you", Style::default().fg(Color::Cyan)),
        EntryKind::Assistant | EntryKind::Pending => ("riko", Style::default().fg(Color::Green)),
        EntryKind::ToolCall => ("tool", Style::default().fg(Color::Yellow)),
        EntryKind::ToolResult { is_error: false } => ("tool", Style::default().fg(Color::Yellow)),
        EntryKind::ToolResult { is_error: true } => ("tool!", Style::default().fg(Color::Red)),
        EntryKind::Summary => ("summary", Style::default().fg(Color::Magenta)),
        EntryKind::System => ("system", Style::default().fg(Color::DarkGray)),
    }
}
