use std::io::{self, Stdout};
use std::sync::Arc;

use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::broadcast::error::RecvError;

use crate::app::{Action, App};
use crate::ui;

type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Enter the alternate screen, run the loop, and always restore the terminal afterward — even if
/// the loop errored — so a failure never leaves the user's terminal in raw mode.
pub async fn run(mut app: App) -> Result<()> {
    let mut terminal = setup().context("initializing the terminal")?;
    let result = event_loop(&mut app, &mut terminal).await;
    restore(&mut terminal).context("restoring the terminal")?;
    result
}

fn setup() -> Result<Tui> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

fn restore(terminal: &mut Tui) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

async fn event_loop(app: &mut App, terminal: &mut Tui) -> Result<()> {
    let mut keys = EventStream::new();
    let mut agent_events = app.engine().agent().subscribe();
    let mut workspace_events = app.engine().workspace().subscribe();

    terminal.draw(|frame| ui::draw(frame, app))?;
    let _ = app.take_dirty();

    loop {
        tokio::select! {
            key = keys.next() => match key {
                Some(Ok(Event::Key(key))) => match app.on_key(key) {
                    Action::Quit => break,
                    Action::Submit(text) => spawn_run(app, text),
                    Action::Abort => app.engine().agent().abort(),
                    Action::None => {}
                },
                Some(Ok(_)) => app.mark_dirty(), // resize / mouse / focus — just redraw
                Some(Err(_)) | None => break,    // input stream closed or errored
            },
            event = agent_events.recv() => match event {
                Ok(event) => app.on_agent_event(event),
                Err(RecvError::Lagged(_)) => app.mark_dirty(),
                Err(RecvError::Closed) => {}
            },
            event = workspace_events.recv() => match event {
                Ok(event) => app.on_workspace_event(event),
                Err(RecvError::Lagged(_)) => app.mark_dirty(),
                Err(RecvError::Closed) => {}
            },
        }

        if app.take_dirty() {
            terminal.draw(|frame| ui::draw(frame, app))?;
        }
    }
    Ok(())
}

/// Spawn the agent run concurrently so the loop keeps pumping events (streaming deltas, the user
/// hitting Ctrl-C to abort) while the turn is in flight. The run reports completion via
/// `RunEnded`, so the join handle is intentionally dropped.
fn spawn_run(app: &App, text: String) {
    let agent = Arc::clone(app.engine().agent());
    tokio::spawn(async move {
        let _ = agent.prompt(text).await;
    });
}
