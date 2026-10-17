//! OpenCode (`opencode run --standalone --format json`), driven headlessly.
//!
//! The CLI learns about the running app from an `opencode.json` written into a per-instance
//! temp dir used as the child's cwd (`mcp.figcraft-app` → `figcraft-cli mcp --attach PORT`);
//! project config merges with the user's own settings and auth, which live elsewhere and are
//! never touched. Output is JSONL of message parts, mapped to [`AgentEvent`]s. OpenCode says
//! nothing when a turn ends well — the pump turns a clean exit into `Done`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::Value;

use crate::claude_cli::SERVER_NAME;
use crate::pump::{Pump, line160};
use crate::{AgentError, AgentEvent, Backend, Turn};

/// OpenCode's `opencode` command, driven headlessly.
pub struct OpenCodeCli {
    bin: PathBuf,
    figcraft_cli: PathBuf,
    attach: u16,
    model: Option<String>,
    /// The session to `--session`; written by the pump when a `sessionID` arrives.
    session: Arc<Mutex<Option<String>>>,
    /// The temp dir holding this app's `opencode.json` (also the child's cwd).
    work_dir: PathBuf,
    tx: Option<Sender<AgentEvent>>,
    child: Arc<Mutex<Option<Child>>>,
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

/// Distinguishes one session's temp work dir from the next in the same process.
static NEXT_WORK_DIR: AtomicU64 = AtomicU64::new(0);

impl OpenCodeCli {
    /// `bin` is the `opencode` executable, `figcraft_cli` the one serving `mcp --attach`,
    /// `attach` the control port of the running app.
    pub fn new(bin: impl Into<PathBuf>, figcraft_cli: impl Into<PathBuf>, attach: u16) -> Self {
        let serial = NEXT_WORK_DIR.fetch_add(1, Ordering::Relaxed);
        OpenCodeCli {
            bin: bin.into(),
            figcraft_cli: figcraft_cli.into(),
            attach,
            model: None,
            session: Arc::new(Mutex::new(None)),
            work_dir: std::env::temp_dir().join(format!("figcraft-agent-oc-{}-{serial}", std::process::id())),
            tx: None,
            child: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Which model to run (`provider/model`); `None` keeps OpenCode's default.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The session id to continue, once one has arrived.
    pub fn session_id(&self) -> Option<String> {
        self.session.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// The cwd's `opencode.json`: `mcp.figcraft-app` → `figcraft-cli mcp --attach <port>`.
    fn write_mcp_config(&self) -> Result<(), AgentError> {
        let dir = &self.work_dir;
        std::fs::create_dir_all(dir).map_err(|e| AgentError::Setup(format!("{}: {e}", dir.display())))?;
        let config = serde_json::json!({
            "mcp": {
                SERVER_NAME: {
                    "type": "local",
                    "command": [
                        self.figcraft_cli.to_string_lossy(),
                        "mcp",
                        "--attach",
                        self.attach.to_string(),
                    ],
                    "enabled": true,
                }
            }
        });
        let text = serde_json::to_string_pretty(&config).map_err(|e| AgentError::Setup(e.to_string()))?;
        let path = dir.join("opencode.json");
        std::fs::write(&path, text).map_err(|e| AgentError::Setup(format!("{}: {e}", path.display())))?;
        Ok(())
    }

    fn command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.bin);
        cmd.arg("run").arg("--standalone").arg("--format").arg("json").arg("--auto").current_dir(&self.work_dir);
        if let Some(model) = &self.model {
            cmd.arg("--model").arg(model);
        }
        if let Some(session) = self.session_id() {
            cmd.arg("--session").arg(session);
        }
        cmd.arg(prompt);
        cmd
    }
}

impl Backend for OpenCodeCli {
    fn name(&self) -> &'static str {
        "OpenCode"
    }

    fn attach(&mut self, tx: Sender<AgentEvent>) {
        self.tx = Some(tx);
    }

    fn start(&mut self, turn: &Turn) -> Result<(), AgentError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(AgentError::Setup("OpenCode is still working on the last turn.".into()));
        }
        let mut prompt = turn.text.trim().to_string();
        if prompt.is_empty() && turn.attachments.is_empty() {
            return Err(AgentError::Setup("Type a message to start the agent.".into()));
        }
        if !turn.attachments.is_empty() {
            let names: Vec<&str> = turn.attachments.iter().map(|a| a.name.as_str()).collect();
            prompt.push_str(&format!("\n\n(Attached files: {}.)", names.join(", ")));
        }
        self.write_mcp_config()?;
        let mut cmd = self.command(&prompt);
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let child = cmd.spawn().map_err(|e| AgentError::Spawn(format!("{}: {e}", self.bin.display())))?;
        let tx = self.tx.clone().ok_or_else(|| AgentError::Setup("The session has no event channel.".into()))?;
        let pump = Pump {
            child,
            tx,
            session: Arc::clone(&self.session),
            slot: Arc::clone(&self.child),
            running: Arc::clone(&self.running),
            cancelled: Arc::clone(&self.cancelled),
            backend: self.name(),
        };
        // One parser for the whole turn: its dedupe state spans lines.
        let mut parser = Parser::default();
        pump.run(move |line| parser.line(line))
    }

    fn cancel(&mut self) {
        if !self.running.load(Ordering::SeqCst) {
            return;
        }
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(mut slot) = self.child.lock()
            && let Some(child) = slot.as_mut()
        {
            let _ = child.kill();
        }
    }
}

impl Drop for OpenCodeCli {
    fn drop(&mut self) {
        self.cancel();
        let _ = std::fs::remove_dir_all(&self.work_dir);
    }
}

/// Maps OpenCode's `--format json` parts to [`AgentEvent`]s; kinds it does not know are
/// ignored.
#[derive(Default)]
pub(crate) struct Parser {
    /// The `sessionID` already reported (each event carries it).
    last_session: Option<String>,
    /// Tool part ids already reported as started.
    started: BTreeSet<String>,
    /// Text part id → bytes already shown, so a re-emitted part shows only its growth.
    grown: BTreeMap<String, usize>,
}

impl Parser {
    pub fn line(&mut self, line: &str) -> Vec<AgentEvent> {
        let Ok(v) = serde_json::from_str::<Value>(line) else { return Vec::new() };
        let mut out = Vec::new();
        if let Some(id) = v["sessionID"].as_str()
            && self.last_session.as_deref() != Some(id)
        {
            self.last_session = Some(id.to_string());
            out.push(AgentEvent::Session(id.to_string()));
        }
        // Parts arrive as `{"type":…,"part":{…}}`; fall back to the object itself for a
        // flat shape a future version might use.
        let part = if v["part"].is_object() { &v["part"] } else { &v };
        match part["type"].as_str() {
            Some("text") => self.text(part, &mut out),
            Some("tool") => self.tool(part, &mut out),
            _ => {}
        }
        out
    }

    fn text(&mut self, part: &Value, out: &mut Vec<AgentEvent>) {
        let id = part["id"].as_str().unwrap_or_default().to_string();
        let text = part["text"].as_str().unwrap_or_default();
        let shown = self.grown.get(&id).copied().unwrap_or(0);
        match text.get(shown..) {
            // A part re-emitted with more text: show only what is new.
            Some(delta) if !delta.is_empty() => {
                self.grown.insert(id, text.len());
                out.push(AgentEvent::Text(delta.to_string()));
            }
            Some(_) => {}
            // The part changed in place (not an extension): start it over next time.
            None => {
                self.grown.remove(&id);
            }
        }
    }

    fn tool(&mut self, part: &Value, out: &mut Vec<AgentEvent>) {
        let id = part["id"].as_str().unwrap_or_default().to_string();
        let name = part["tool"].as_str().unwrap_or("tool").to_string();
        let state = &part["state"];
        let status = state["status"].as_str().unwrap_or_default();
        match status {
            "completed" | "error" | "failed" => {
                self.started.remove(&id);
                let ok = status == "completed";
                let raw = match (&state["output"], &state["error"]) {
                    (Value::String(s), _) => s.clone(),
                    (_, Value::String(s)) => s.clone(),
                    (other, _) if !other.is_null() => other.to_string(),
                    (_, other) => other.to_string(),
                };
                out.push(AgentEvent::ToolEnd { name, ok, summary: line160(&raw) });
            }
            _ => {
                // pending / running: one row per call, however often it is re-sent.
                if self.started.insert(id) {
                    let input = &state["input"];
                    let args = if input.is_null() { String::new() } else { input.to_string() };
                    out.push(AgentEvent::ToolStart { name, args });
                }
            }
        }
    }
}
