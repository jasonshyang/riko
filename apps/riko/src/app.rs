use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use riko_agent::{AgentEvent, EndReason};
use riko_context::WorkspaceEvent;
use riko_engine::Engine;
use riko_llm::ProviderEvent;

use crate::transcript::{Entry, entries};

/// What a key press asks the event loop to do, beyond the local state change `on_key` already
/// applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    None,
    /// Tear down and exit.
    Quit,
    /// Start a run with this user input.
    Submit(String),
    /// Abort the in-progress run.
    Abort,
}

/// TUI model: the engine handle plus ephemeral UI state.
pub struct App {
    engine: Engine,
    input: String,
    /// Accumulated in-progress assistant text for the current turn; cleared once it settles.
    pending: String,
    running: bool,
    status: String,
    /// Set whenever state changed in a way that needs a redraw.
    dirty: bool,
}

impl App {
    pub fn new(engine: Engine) -> Self {
        Self {
            engine,
            input: String::new(),
            pending: String::new(),
            running: false,
            status: "ready".into(),
            dirty: true,
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    pub fn status(&self) -> &str {
        &self.status
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Take the pending-redraw flag, resetting it.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// The transcript to render: finalized workspace items plus any in-progress streamed text.
    pub fn entries(&self) -> Vec<Entry> {
        let items = self.engine.workspace().items();
        let pending = self.running.then_some(self.pending.as_str());
        entries(&items, pending)
    }

    /// Apply a key press and report what the loop should do next.
    pub fn on_key(&mut self, key: KeyEvent) -> Action {
        if key.kind != KeyEventKind::Press {
            return Action::None;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.running {
                    Action::Abort
                } else {
                    Action::Quit
                }
            }
            KeyCode::Esc => Action::Quit,
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => {
                self.input.pop();
                self.dirty = true;
                Action::None
            }
            KeyCode::Char(c) => {
                self.input.push(c);
                self.dirty = true;
                Action::None
            }
            _ => Action::None,
        }
    }

    /// Enter submits the input as a new run — unless one is already running or it is blank.
    fn submit(&mut self) -> Action {
        if self.running || self.input.trim().is_empty() {
            return Action::None;
        }
        let text = std::mem::take(&mut self.input);
        self.running = true;
        self.pending.clear();
        self.status = "running…".into();
        self.dirty = true;
        Action::Submit(text)
    }

    /// Fold an agent run-lifecycle / streaming event into UI state.
    pub fn on_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::RunStarted => {
                self.running = true;
                self.pending.clear();
                self.status = "running…".into();
            }
            AgentEvent::Streaming(ProviderEvent::TextDelta { text, .. }) => {
                self.pending.push_str(&text);
            }
            AgentEvent::Streaming(ProviderEvent::ToolCallStart { name, .. }) => {
                self.status = format!("calling {name}…");
            }
            AgentEvent::Streaming(_) => {}
            AgentEvent::ToolStarted { tool, .. } => self.status = format!("running {tool}…"),
            AgentEvent::ToolEnded { is_error, .. } => {
                self.status = if is_error { "tool failed".into() } else { "tool ok".into() };
            }
            // A finalized turn (assistant or tool result) is now in the workspace; the pending
            // buffer that mirrored its stream is no longer needed.
            AgentEvent::Settled { .. } => self.pending.clear(),
            AgentEvent::RunEnded { reason } => {
                self.running = false;
                self.pending.clear();
                self.status = Self::end_reason(reason).into();
            }
            AgentEvent::Error { message } => self.status = message,
            AgentEvent::TurnStarted | AgentEvent::TurnEnded | AgentEvent::QueueDrained { .. } => {}
        }
        self.dirty = true;
    }

    /// Durable workspace changes are read fresh at render time, so we just flag a redraw.
    pub fn on_workspace_event(&mut self, _event: WorkspaceEvent) {
        self.dirty = true;
    }

    fn end_reason(reason: EndReason) -> &'static str {
        match reason {
            EndReason::Stopped => "done",
            EndReason::Aborted => "aborted",
            EndReason::Failed => "failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::EntryKind;
    use riko_config::LoadOptions;
    use riko_llm::{Api, ProviderRegistry, mock::MockProvider};
    use std::sync::Arc;

    /// An app over an in-memory engine with a mock provider — hermetic (no disk, no network).
    fn app() -> App {
        let mut providers = ProviderRegistry::new();
        providers.register(MockProvider::for_api(Api::AnthropicMessages));
        let engine = Engine::builder()
            .workspace_root(".")
            .model("anthropic:test")
            .settings(LoadOptions::default())
            .skill_roots(Vec::new())
            .providers(Arc::new(providers))
            .build()
            .unwrap();
        App::new(engine)
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    fn delta(text: &str) -> AgentEvent {
        AgentEvent::Streaming(ProviderEvent::TextDelta { idx: 0, text: text.into() })
    }

    #[test]
    fn typing_accumulates_then_backspaces() {
        let mut app = app();
        app.on_key(press(KeyCode::Char('h')));
        app.on_key(press(KeyCode::Char('i')));
        assert_eq!(app.input(), "hi");
        app.on_key(press(KeyCode::Backspace));
        assert_eq!(app.input(), "h");
    }

    #[test]
    fn enter_submits_nonblank_input_and_clears_it() {
        let mut app = app();
        app.on_key(press(KeyCode::Char('y')));
        assert_eq!(app.on_key(press(KeyCode::Enter)), Action::Submit("y".into()));
        assert_eq!(app.input(), "");
        assert!(app.is_running());
    }

    #[test]
    fn enter_does_not_submit_when_blank_or_running() {
        let mut app = app();
        assert_eq!(app.on_key(press(KeyCode::Enter)), Action::None, "blank input");
        app.on_agent_event(AgentEvent::RunStarted);
        app.on_key(press(KeyCode::Char('x')));
        assert_eq!(app.on_key(press(KeyCode::Enter)), Action::None, "already running");
    }

    #[test]
    fn ctrl_c_quits_when_idle_and_aborts_when_running() {
        let mut app = app();
        assert_eq!(app.on_key(ctrl(KeyCode::Char('c'))), Action::Quit);
        app.on_agent_event(AgentEvent::RunStarted);
        assert_eq!(app.on_key(ctrl(KeyCode::Char('c'))), Action::Abort);
    }

    #[test]
    fn key_release_events_are_ignored() {
        let mut app = app();
        let mut release = press(KeyCode::Char('z'));
        release.kind = KeyEventKind::Release;
        assert_eq!(app.on_key(release), Action::None);
        assert_eq!(app.input(), "");
    }

    #[test]
    fn streaming_text_accumulates_into_pending_then_clears_at_run_end() {
        let mut app = app();
        app.on_agent_event(AgentEvent::RunStarted);
        app.on_agent_event(delta("hel"));
        app.on_agent_event(delta("lo"));
        let pending = app.entries().into_iter().find(|e| e.kind == EntryKind::Pending);
        assert_eq!(pending.unwrap().text, "hello");

        app.on_agent_event(AgentEvent::RunEnded { reason: EndReason::Stopped });
        assert!(!app.is_running());
        assert_eq!(app.status(), "done");
        assert!(app.entries().iter().all(|e| e.kind != EntryKind::Pending));
    }

    #[test]
    fn settled_clears_pending_mid_run() {
        let mut app = app();
        app.on_agent_event(AgentEvent::RunStarted);
        app.on_agent_event(delta("partial"));
        app.on_agent_event(AgentEvent::Settled {
            id: app.engine().workspace().add(system()).unwrap(),
        });
        assert!(app.entries().iter().all(|e| e.kind != EntryKind::Pending));
    }

    #[test]
    fn dirty_is_set_by_input_and_consumed_by_take() {
        let mut app = app();
        let _ = app.take_dirty(); // clear the initial dirty
        app.on_key(press(KeyCode::Char('a')));
        assert!(app.take_dirty());
        assert!(!app.take_dirty());
    }

    fn system() -> riko_context::ItemPayload {
        riko_context::ItemPayload::System { tag: None, text: "x".into() }
    }
}
