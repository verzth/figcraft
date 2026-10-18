//! Claude Code in print mode: one process per turn, continued with `--resume <session>`.
//!
//! The CLI talks to the running app through our own MCP server — a per-invocation
//! `--mcp-config` temp file pointing `figcraft-app` at `figcraft-cli mcp --attach PORT` —
//! so the user's own Claude Code config is never touched, and their existing login is
//! reused as-is. Output is `stream-json` with partial messages, mapped to [`AgentEvent`]s.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::Value;

use crate::pump::{line160, result_text};
use crate::{AgentError, AgentEvent, Backend, Turn};

/// The MCP server entry the agent CLI talks to; matches Preferences ▸ AI agents.
pub const SERVER_NAME: &str = "figcraft-app";

/// Stands next to the user's prompt so the agent knows where it is and what it may touch.
pub const SYSTEM_PROMPT_NOTE: &str = "You are designing inside the FigCraft design app. The \
    current document is open on the canvas. Use the figcraft-app MCP server's tools to build \
    and inspect the design; work in small steps and describe what you are doing in one or two \
    sentences per step.";

/// Claude Code (`claude -p --output-format stream-json`), driven headlessly.
pub struct ClaudeCodeCli {
    bin: PathBuf,
    figcraft_cli: PathBuf,
    attach: u16,
    model: Option<String>,
    /// The session to `--resume`; written by the reader thread when `system/init` arrives.
    session: Arc<Mutex<Option<String>>>,
    /// The temp dir holding this app's `mcp.json`.
    config_dir: PathBuf,
    /// What the installed CLI said it accepts (`--strict-mcp-config`, `--tools`), read from
    /// `--help` once per session. An unknown flag is a hard error, so a CLI that doesn't
    /// list a switch never receives it.
    supports: Arc<Mutex<Option<(bool, bool)>>>,
    tx: Option<Sender<AgentEvent>>,
    child: Arc<Mutex<Option<Child>>>,
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

/// Distinguishes one session's temp config dir from the next in the same process.
static NEXT_CONFIG_DIR: AtomicU64 = AtomicU64::new(0);

/// The two switches worth passing only to a CLI that lists them, read from its `--help`
/// (a sixtieth of a second, and only before a session's first turn). Help lands on either
/// stream depending on the CLI, so both count.
fn probe_support(bin: &Path) -> (bool, bool) {
    let Ok(out) = Command::new(bin).arg("--help").stdout(Stdio::piped()).stderr(Stdio::piped()).output() else {
        return (false, false);
    };
    let mut help = String::from_utf8_lossy(&out.stdout).into_owned();
    help.push_str(&String::from_utf8_lossy(&out.stderr));
    (help.contains("--strict-mcp-config"), help.contains("--tools"))
}

impl ClaudeCodeCli {
    /// `bin` is the `claude` executable, `figcraft_cli` the one serving `mcp --attach`,
    /// `attach` the control port of the running app.
    pub fn new(bin: impl Into<PathBuf>, figcraft_cli: impl Into<PathBuf>, attach: u16) -> Self {
        let serial = NEXT_CONFIG_DIR.fetch_add(1, Ordering::Relaxed);
        ClaudeCodeCli {
            bin: bin.into(),
            figcraft_cli: figcraft_cli.into(),
            attach,
            model: None,
            session: Arc::new(Mutex::new(None)),
            config_dir: std::env::temp_dir().join(format!("figcraft-agent-mcp-{}-{serial}", std::process::id())),
            supports: Arc::new(Mutex::new(None)),
            tx: None,
            child: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Which Claude model to run, e.g. `claude-sonnet-4-5`; `None` keeps the CLI's default.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The session id to continue, once one has arrived.
    pub fn session_id(&self) -> Option<String> {
        self.session.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// The per-invocation `mcp.json`: `figcraft-app` → `figcraft-cli mcp --attach <port>`.
    fn write_mcp_config(&self) -> Result<PathBuf, AgentError> {
        let dir = &self.config_dir;
        std::fs::create_dir_all(dir).map_err(|e| AgentError::Setup(format!("{}: {e}", dir.display())))?;
        let path = dir.join("mcp.json");
        let config = serde_json::json!({
            "mcpServers": {
                SERVER_NAME: {
                    "command": self.figcraft_cli.to_string_lossy(),
                    "args": ["mcp", "--attach", self.attach.to_string()],
                }
            }
        });
        let text = serde_json::to_string_pretty(&config).map_err(|e| AgentError::Setup(e.to_string()))?;
        std::fs::write(&path, text).map_err(|e| AgentError::Setup(format!("{}: {e}", path.display())))?;
        Ok(path)
    }

    /// The installed CLI's switches, probed once per session — before the first turn runs,
    /// on the worker thread that spawns the process (never the UI's).
    fn supports(&self) -> (bool, bool) {
        let mut slot = self.supports.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            // The fake CLIs in the tests answer `--help` with their transcript, so nothing
            // they say lists a switch: they run the plain turn, exactly like an old CLI.
            *slot = Some(probe_support(&self.bin));
        }
        slot.unwrap_or((false, false))
    }

    fn command(&self, prompt: &str, mcp_config: &Path) -> Command {
        // An unknown flag is a hard `error: unknown option` exit, so each switch is passed
        // only to a CLI whose `--help` listed it; a CLI too old gets the plain turn, which
        // works everywhere.
        let (strict, tools) = self.supports();
        let mut cmd = Command::new(&self.bin);
        cmd.arg("--print")
            .arg("--output-format")
            .arg("stream-json")
            .arg("--verbose")
            .arg("--include-partial-messages")
            .arg("--mcp-config")
            .arg(mcp_config);
        if strict {
            // Only this turn's server: the user's own MCP setup (which can hold hundreds
            // of tools) never enters the turn.
            cmd.arg("--strict-mcp-config");
        }
        if tools {
            // No built-in tools either — the agent designs here and touches nothing else
            // on this computer. Empty also spares it the tool-lookup round trip the CLI
            // makes when many tools are only deferred.
            cmd.arg("--tools").arg("");
        }
        cmd.arg("--allowedTools").arg(format!("mcp__{SERVER_NAME}__*")).arg("--append-system-prompt").arg(SYSTEM_PROMPT_NOTE);
        if let Some(model) = &self.model {
            cmd.arg("--model").arg(model);
        }
        if let Some(session) = self.session_id() {
            cmd.arg("--resume").arg(session);
        }
        cmd.arg(prompt);
        cmd
    }
}

impl Backend for ClaudeCodeCli {
    fn name(&self) -> &'static str {
        "Claude Code"
    }

    fn attach(&mut self, tx: Sender<AgentEvent>) {
        self.tx = Some(tx);
    }

    fn resume_with(&mut self, id: String) {
        if let Ok(mut slot) = self.session.lock() {
            *slot = Some(id);
        }
    }

    fn start(&mut self, turn: &Turn) -> Result<(), AgentError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(AgentError::Setup("Claude Code is still working on the last turn.".into()));
        }
        let mcp_config = self.write_mcp_config()?;
        let mut prompt = turn.text.trim().to_string();
        if prompt.is_empty() && turn.attachments.is_empty() {
            return Err(AgentError::Setup("Type a message to start the agent.".into()));
        }
        if !turn.attachments.is_empty() {
            let names: Vec<&str> = turn.attachments.iter().map(|a| a.name.as_str()).collect();
            prompt.push_str(&format!("\n\n(Attached files: {}.)", names.join(", ")));
        }
        let mut cmd = self.command(&prompt, &mcp_config);
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = cmd.spawn().map_err(|e| AgentError::Spawn(format!("{}: {e}", self.bin.display())))?;
        let stdout = child.stdout.take().ok_or_else(|| AgentError::Setup("Claude Code gave no output stream.".into()))?;
        let tx = self.tx.clone().ok_or_else(|| AgentError::Setup("The session has no event channel.".into()))?;
        *self.child.lock().unwrap_or_else(PoisonError::into_inner) = Some(child);
        self.cancelled.store(false, Ordering::SeqCst);
        self.running.store(true, Ordering::SeqCst);
        let session = Arc::clone(&self.session);
        let child_slot = Arc::clone(&self.child);
        let running = Arc::clone(&self.running);
        let cancelled = Arc::clone(&self.cancelled);
        std::thread::spawn(move || {
            let mut parser = Parser::default();
            let mut finished = false;
            for line in std::io::BufRead::lines(std::io::BufReader::new(stdout)).map_while(Result::ok) {
                for event in parser.line(&line) {
                    if let AgentEvent::Session(id) = &event {
                        *session.lock().unwrap_or_else(PoisonError::into_inner) = Some(id.clone());
                    }
                    if matches!(event, AgentEvent::Done { .. } | AgentEvent::Failed(_)) {
                        finished = true;
                    }
                    if tx.send(event).is_err() {
                        // The panel is gone: stop reading, but fall through to the reap
                        // below so the child never outlives the session as a zombie.
                        break;
                    }
                }
            }
            let mut child = child_slot.lock().unwrap_or_else(PoisonError::into_inner).take();
            let status = child.as_mut().and_then(|c| c.wait().ok());
            running.store(false, Ordering::SeqCst);
            if finished {
                return;
            }
            let event = if cancelled.load(Ordering::SeqCst) {
                AgentEvent::Done { reason: "cancelled".into() }
            } else {
                let how = status.map_or_else(|| "it stopped".into(), |s| format!("{s}"));
                AgentEvent::Failed(format!("Claude Code exited without finishing ({how})."))
            };
            let _ = tx.send(event);
        });
        Ok(())
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

impl Drop for ClaudeCodeCli {
    fn drop(&mut self) {
        self.cancel();
        let _ = std::fs::remove_dir_all(&self.config_dir);
    }
}

/// Maps Claude Code's `stream-json` lines to [`AgentEvent`]s; tolerant of every kind it
/// does not know (ignored), which is what keeps a CLI upgrade from breaking the panel.
#[derive(Default)]
pub(crate) struct Parser {
    /// Block index → (`tool_use_id`, tool name, streamed argument JSON), until the block
    /// stops. The stream numbers content blocks per message and repeats only the index in
    /// the delta and stop events — the `tool_use_id` they carry in some shapes is not
    /// dependable, so the index is what ties the three together.
    pending: BTreeMap<u64, (String, String, String)>,
    /// `tool_use_id` → tool name, for the tool results that follow.
    tools: BTreeMap<String, String>,
}

impl Parser {
    pub(crate) fn line(&mut self, line: &str) -> Vec<AgentEvent> {
        let Ok(v) = serde_json::from_str::<Value>(line) else { return Vec::new() };
        let Some(kind) = v["type"].as_str() else { return Vec::new() };
        let mut out = Vec::new();
        match kind {
            "system" if v["subtype"] == "init" => {
                if let Some(id) = v["session_id"].as_str() {
                    out.push(AgentEvent::Session(id.to_string()));
                }
            }
            "stream_event" => self.stream_event(&v["event"], &mut out),
            // With partial messages on, text and tool_use arrive as deltas; the complete
            // assistant messages repeat them, so they are skipped.
            "user" => self.tool_results(&v["message"]["content"], &mut out),
            "result" => {
                if let (Some(input), Some(output)) = (num(&v["usage"]["input_tokens"]), num(&v["usage"]["output_tokens"])) {
                    out.push(AgentEvent::Tokens { input, output });
                }
                let reason = v["subtype"].as_str().unwrap_or("end_turn").to_string();
                out.push(AgentEvent::Done { reason });
            }
            "error" => {
                let message = v["error"]["message"].as_str().unwrap_or("Claude Code reported an unknown error.");
                out.push(AgentEvent::Failed(message.to_string()));
            }
            _ => {}
        }
        out
    }

    fn stream_event(&mut self, event: &Value, out: &mut Vec<AgentEvent>) {
        let kind = event["type"].as_str().unwrap_or_default();
        match kind {
            "content_block_delta" => {
                let delta = &event["delta"];
                match delta["type"].as_str() {
                    // Text streams as its own block; the complete assistant message below
                    // repeats it and is skipped.
                    Some("text_delta") => {
                        if let Some(text) = delta["text"].as_str()
                            && !text.is_empty()
                        {
                            out.push(AgentEvent::Text(text.to_string()));
                        }
                    }
                    // A tool call's arguments stream into the block its index names.
                    Some("input_json_delta") => {
                        if let (Some(index), Some(part)) = (index(event), delta["partial_json"].as_str())
                            && let Some((_, _, args)) = self.pending.get_mut(&index)
                        {
                            args.push_str(part);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_start" => {
                let block = &event["content_block"];
                if block["type"] == "tool_use"
                    && let (Some(index), Some(id), Some(name)) = (index(event), block["id"].as_str(), block["name"].as_str())
                {
                    self.pending.insert(index, (id.to_string(), name.to_string(), String::new()));
                }
            }
            "content_block_stop" => {
                if let Some(index) = index(event)
                    && let Some((id, name, args)) = self.pending.remove(&index)
                {
                    self.tools.insert(id, name.clone());
                    // An empty `input` streams no argument deltas: the call is `{}`.
                    let args = if args.is_empty() { "{}".into() } else { args };
                    out.push(AgentEvent::ToolStart { name, args });
                }
            }
            _ => {}
        }
    }

    fn tool_results(&mut self, content: &Value, out: &mut Vec<AgentEvent>) {
        let Some(items) = content.as_array() else { return };
        for item in items {
            if item["type"] != "tool_result" {
                continue;
            }
            let Some(id) = item["tool_use_id"].as_str() else { continue };
            let name = self.tools.remove(id).unwrap_or_else(|| "tool".into());
            out.push(AgentEvent::ToolEnd { name, ok: item["is_error"] != true, summary: summary(&item["content"]) });
        }
    }
}

fn num(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))
}

/// The content-block index a stream event belongs to.
fn index(event: &Value) -> Option<u64> {
    num(&event["index"])
}

/// The tool result's first text, collapsed to one trimmed line for the panel's tool row.
fn summary(content: &Value) -> String {
    line160(&result_text(content))
}
