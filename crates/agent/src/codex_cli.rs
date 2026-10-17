//! Codex (`codex exec --json`), driven headlessly.
//!
//! The MCP server is injected per invocation with dotted-path `-c` overrides
//! (`mcp_servers.figcraft-app.command/args`) plus `approval_policy` so its tool calls run
//! without a prompt to nobody: the user's `~/.codex/config.toml` and login are untouched and
//! reused as-is. Output is JSONL of turn/item events, mapped to [`AgentEvent`]s.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::Value;

use crate::claude_cli::SERVER_NAME;
use crate::pump::{Pump, line160, num};
use crate::{AgentError, AgentEvent, Backend, Turn};

/// OpenAI's Codex CLI, driven headlessly.
pub struct CodexCli {
    bin: PathBuf,
    figcraft_cli: PathBuf,
    attach: u16,
    model: Option<String>,
    /// The session to `exec resume`; written by the pump when `thread.started` arrives.
    session: Arc<Mutex<Option<String>>>,
    tx: Option<Sender<AgentEvent>>,
    child: Arc<Mutex<Option<Child>>>,
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

impl CodexCli {
    /// `bin` is the `codex` executable, `figcraft_cli` the one serving `mcp --attach`,
    /// `attach` the control port of the running app.
    pub fn new(bin: impl Into<PathBuf>, figcraft_cli: impl Into<PathBuf>, attach: u16) -> Self {
        CodexCli {
            bin: bin.into(),
            figcraft_cli: figcraft_cli.into(),
            attach,
            model: None,
            session: Arc::new(Mutex::new(None)),
            tx: None,
            child: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Which model to run; `None` keeps Codex's default.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The session id to continue, once one has arrived.
    pub fn session_id(&self) -> Option<String> {
        self.session.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn command(&self, prompt: &str) -> Command {
        // JSON string escaping is TOML's too, for what a path can hold (`\n`, `\"`, `\\`).
        let mcp_exe = serde_json::to_string(&self.figcraft_cli.to_string_lossy().into_owned()).unwrap_or_else(|_| "\"figcraft-cli\"".into());
        let mcp_args = serde_json::json!(["mcp", "--attach", self.attach.to_string()]).to_string();
        let mut cmd = Command::new(&self.bin);
        cmd.arg("exec");
        if let Some(session) = self.session_id() {
            cmd.arg("resume").arg(session);
        }
        cmd.arg("--json")
            .arg("--skip-git-repo-check")
            // No one answers the approval prompt in a headless turn; the MCP tools are this
            // app's own, and the model's shell still runs inside its sandbox.
            .arg("-c")
            .arg("approval_policy=\"never\"")
            .arg("-c")
            .arg(format!("mcp_servers.{SERVER_NAME}.command={mcp_exe}"))
            .arg("-c")
            .arg(format!("mcp_servers.{SERVER_NAME}.args={mcp_args}"));
        if let Some(model) = &self.model {
            cmd.arg("-m").arg(model);
        }
        cmd.arg(prompt);
        cmd
    }
}

impl Backend for CodexCli {
    fn name(&self) -> &'static str {
        "Codex"
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
            return Err(AgentError::Setup("Codex is still working on the last turn.".into()));
        }
        let mut prompt = turn.text.trim().to_string();
        if prompt.is_empty() && turn.attachments.is_empty() {
            return Err(AgentError::Setup("Type a message to start the agent.".into()));
        }
        if !turn.attachments.is_empty() {
            let names: Vec<&str> = turn.attachments.iter().map(|a| a.name.as_str()).collect();
            prompt.push_str(&format!("\n\n(Attached files: {}.)", names.join(", ")));
        }
        let mut cmd = self.command(&prompt);
        // Null stdin: with a pipe, `codex exec` waits for input it will never get.
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

impl Drop for CodexCli {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Maps Codex's `--json` events to [`AgentEvent`]s; kinds it does not know are ignored.
#[derive(Default)]
pub(crate) struct Parser {
    /// Tool item ids already reported as started (`item.started` can repeat).
    started: BTreeSet<String>,
}

impl Parser {
    pub fn line(&mut self, line: &str) -> Vec<AgentEvent> {
        let Ok(v) = serde_json::from_str::<Value>(line) else { return Vec::new() };
        let mut out = Vec::new();
        match v["type"].as_str() {
            Some("thread.started") => {
                if let Some(id) = v["thread_id"].as_str() {
                    out.push(AgentEvent::Session(id.to_string()));
                }
            }
            Some("item.started") => self.item(&v, true, &mut out),
            Some("item.completed") => self.item(&v, false, &mut out),
            Some("turn.completed") => {
                let usage = &v["usage"];
                out.push(AgentEvent::Tokens { input: num(&usage["input_tokens"]).unwrap_or(0), output: num(&usage["output_tokens"]).unwrap_or(0) });
                out.push(AgentEvent::Done { reason: "success".into() });
            }
            Some("turn.failed") | Some("error") => {
                let message = v["error"]["message"].as_str().or_else(|| v["message"].as_str()).unwrap_or("Codex reported an error.");
                out.push(AgentEvent::Failed(message.to_string()));
            }
            _ => {}
        }
        out
    }

    fn item(&mut self, v: &Value, started: bool, out: &mut Vec<AgentEvent>) {
        let item = &v["item"];
        match item["type"].as_str() {
            // The model's prose: one event at the end of each message.
            Some("agent_message") if !started => {
                if let Some(text) = item["text"].as_str() {
                    out.push(AgentEvent::Text(text.to_string()));
                }
            }
            Some("mcp_tool_call") => {
                let id = item["id"].as_str().unwrap_or_default().to_string();
                let server = item["server"].as_str().unwrap_or_default();
                let tool = item["tool"].as_str().unwrap_or("tool");
                let name = format!("{server}.{tool}");
                if started {
                    if self.started.insert(id) {
                        let args = match &item["arguments"] {
                            Value::String(s) => s.clone(),
                            Value::Null => String::new(),
                            other => other.to_string(),
                        };
                        out.push(AgentEvent::ToolStart { name, args });
                    }
                } else {
                    self.started.remove(&id);
                    let ok = item["status"].as_str() != Some("failed");
                    let raw = item["output"].as_str().unwrap_or_default();
                    out.push(AgentEvent::ToolEnd { name, ok, summary: line160(raw) });
                }
            }
            Some("command_execution") => {
                let id = item["id"].as_str().unwrap_or_default().to_string();
                if started {
                    if self.started.insert(id) {
                        out.push(AgentEvent::ToolStart { name: "shell".into(), args: item["command"].as_str().unwrap_or_default().to_string() });
                    }
                } else {
                    self.started.remove(&id);
                    let code = item["exit_code"].as_i64();
                    let ok = item["status"].as_str() != Some("failed") && code.is_none_or(|c| c == 0);
                    let raw = item["aggregated_output"].as_str().unwrap_or_default();
                    let summary = if raw.trim().is_empty() { format!("exit {}", code.unwrap_or(0)) } else { line160(raw) };
                    out.push(AgentEvent::ToolEnd { name: "shell".into(), ok, summary });
                }
            }
            _ => {}
        }
    }
}
