//! Agent panel backends: an AI agent designing in FigCraft, driven from the in-app Agent tab.
//!
//! A [`Session`] pairs a [`Backend`] with a channel of [`AgentEvent`]s. The UI sends a
//! [`Turn`] and polls events once per frame; backends run on their own threads, so the
//! window never blocks. Every backend drives the same running FigCraft app through the
//! `figcraft-app` MCP server (`figcraft-cli mcp --attach PORT`), exactly like the external
//! agents set up in Preferences ▸ AI agents.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(target_arch = "wasm32"))]
mod anthropic_api;
mod claude_cli;
mod codex_cli;
pub mod keys;
#[cfg(not(target_arch = "wasm32"))]
mod mcp_stdio;
#[cfg(not(target_arch = "wasm32"))]
mod openai_api;
mod opencode_cli;
mod pump;
pub mod vision;

#[cfg(test)]
mod anthropic_api_tests;
#[cfg(test)]
mod claude_cli_tests;
#[cfg(test)]
mod codex_cli_tests;
#[cfg(test)]
mod keys_tests;
#[cfg(test)]
mod live_tests;
#[cfg(test)]
mod mcp_stdio_tests;
#[cfg(test)]
mod openai_api_tests;
#[cfg(test)]
mod opencode_cli_tests;
#[cfg(test)]
mod vision_tests;

#[cfg(not(target_arch = "wasm32"))]
pub use anthropic_api::AnthropicApi;
pub use claude_cli::{ClaudeCodeCli, SYSTEM_PROMPT_NOTE};
pub use codex_cli::CodexCli;
pub use keys::{ApiKeys, FILE_NAME};
#[cfg(not(target_arch = "wasm32"))]
pub use openai_api::OpenAiApi;
pub use opencode_cli::OpenCodeCli;
pub use vision::{MAX_IMAGE_BYTES, image_media_type};

use std::sync::mpsc::{Receiver, Sender, channel};

/// A file the user attached to a turn (a text file, DESIGN.md, or an image).
#[derive(Clone, Debug)]
pub struct Attachment {
    /// The file name, shown on the panel and used as the agent's context label.
    pub name: String,
    /// MIME type: `text/plain`, `image/png`, …
    pub mime: String,
    /// The file's bytes.
    pub data: Vec<u8>,
}

/// One user turn: what the user typed, plus its attachments.
#[derive(Clone, Debug, Default)]
pub struct Turn {
    /// The user's message.
    pub text: String,
    /// Files the user attached; backends that cannot carry bytes name them in the prompt.
    pub attachments: Vec<Attachment>,
}

/// What a backend reports while it works; the panel renders these in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEvent {
    /// The session id: later turns continue it (`--resume`, `--session`).
    Session(String),
    /// A streamed assistant text delta.
    Text(String),
    /// The agent called a tool: its name and arguments as JSON text.
    ToolStart {
        /// The MCP tool name, e.g. `figcraft.build_layout`.
        name: String,
        /// The arguments as compact JSON.
        args: String,
    },
    /// The tool came back: `ok` or an error, with a one-line summary.
    ToolEnd {
        /// The MCP tool name.
        name: String,
        /// Whether the tool reported an error.
        ok: bool,
        /// The tool's first text, trimmed to one line.
        summary: String,
    },
    /// Tokens used, when the backend reports them.
    Tokens {
        /// Input (prompt) tokens.
        input: u64,
        /// Output tokens.
        output: u64,
    },
    /// The turn finished on its own (`end_turn`) or was cancelled.
    Done {
        /// The stop reason: `end_turn`, `cancelled`, the result subtype, …
        reason: String,
    },
    /// The backend could not start or died: shown as the panel's error row.
    Failed(String),
}

/// Why a backend refused a turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentError {
    /// The CLI binary is missing or could not be spawned.
    Spawn(String),
    /// The turn was refused before anything ran (already working, no config, …).
    Setup(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::Spawn(m) => write!(f, "Could not start the agent: {m}"),
            AgentError::Setup(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for AgentError {}

/// One backend's side of a session. A backend spawns work per turn and reports through the
/// channel wired by [`Session::new`].
pub trait Backend: Send {
    /// The name shown on the panel ("Claude Code", …).
    fn name(&self) -> &'static str;
    /// Called once by [`Session::new`] with the channel to report through.
    fn attach(&mut self, tx: Sender<AgentEvent>);
    /// Begin a turn. Events arrive on the session's channel; the first is usually
    /// [`AgentEvent::Session`].
    fn start(&mut self, turn: &Turn) -> Result<(), AgentError>;
    /// End the running turn early; the channel then reports [`AgentEvent::Done`] with
    /// reason `cancelled`.
    fn cancel(&mut self);
    /// Continue a conversation the panel restored from its chat history: `id` is the session
    /// the CLI reported earlier, so the next turn resumes it as if it had run here.
    fn resume_with(&mut self, _id: String) {}
}

/// A backend plus its event channel. Poll [`Session::poll`] from the UI thread.
pub struct Session {
    backend: Box<dyn Backend>,
    rx: Receiver<AgentEvent>,
}

impl Session {
    /// Wire a backend to a fresh event channel.
    pub fn new(mut backend: Box<dyn Backend>) -> Self {
        let (tx, rx) = channel();
        backend.attach(tx);
        Session { backend, rx }
    }

    /// The backend's display name.
    pub fn name(&self) -> &'static str {
        self.backend.name()
    }

    /// Begin a turn.
    pub fn send(&mut self, turn: &Turn) -> Result<(), AgentError> {
        self.backend.start(turn)
    }

    /// End the running turn early.
    pub fn cancel(&mut self) {
        self.backend.cancel();
    }

    /// Restore a conversation by id: the next turn continues it (`--resume`, `--session`,
    /// `exec resume`), as the backend's own session events would have.
    pub fn resume(&mut self, id: String) {
        self.backend.resume_with(id);
    }

    /// The events that arrived since the last poll; the UI calls this once per frame.
    pub fn poll(&self) -> Vec<AgentEvent> {
        self.rx.try_iter().collect()
    }
}
