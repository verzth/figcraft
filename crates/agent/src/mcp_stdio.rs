//! One `figcraft-cli mcp --attach PORT` child speaking newline-delimited JSON-RPC on stdio:
//! the in-process MCP client the HTTP backends drive their tools with — the same server the
//! external agents use, no agent CLI in the middle.
//!
//! A fresh client is spawned per turn (by the turn thread, lazily on the first tool call) and
//! dropped with it, so a wedged app can never leak a child across turns. Transport trouble
//! (spawn failure, EOF, a write that fails, a call that times out) is a hard
//! [`AgentError::Setup`]; the server answering `error` or `isError` is a normal tool failure —
//! `Ok((false, message))` — because that is the model's problem, not the backend's.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::AgentError;
use crate::pump::result_text;

/// How long the handshake may take: the child starts, attaches to the app's control port and
/// answers `initialize` — normally well under a second.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one tool call may take. Generous for a big export, short enough that a dead app
/// cannot park the turn thread forever (the panel's Stop cannot reach into a blocked read,
/// so the timeout is the backstop).
const CALL_TIMEOUT: Duration = Duration::from_secs(60);

/// A live `figcraft-cli mcp --attach` connection, spoken to over newline-delimited JSON-RPC.
pub(crate) struct McpStdio {
    child: Child,
    stdin: std::process::ChildStdin,
    /// Lines the reader thread has parsed off the child's stdout; a dropped sender (EOF or a
    /// decode error) surfaces as a disconnect here.
    lines: Receiver<String>,
    next_id: i64,
}

impl McpStdio {
    /// Spawn the child, run the MCP handshake, and return a ready client. `cli` is the
    /// `figcraft-cli` binary, `attach` the control port of the running app.
    pub(crate) fn open(cli: &Path, attach: u16) -> Result<Self, AgentError> {
        let mut child = Command::new(cli)
            .arg("mcp")
            .arg("--attach")
            .arg(attach.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| AgentError::Setup(format!("Could not start {} mcp --attach {attach}: {e}", cli.display())))?;
        let stdin = child.stdin.take().ok_or_else(|| AgentError::Setup("the MCP child gave no input stream.".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| AgentError::Setup("the MCP child gave no output stream.".into()))?;
        let (tx, lines) = channel::<String>();
        // The reader thread owns stdout for the client's life; the channel ends on EOF.
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut me = McpStdio { child, stdin, lines, next_id: 0 };
        me.request(
            "initialize",
            json!({
                "protocolVersion": figcraft_mcp::PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "figcraft-agent", "version": env!("CARGO_PKG_VERSION")},
            }),
            HANDSHAKE_TIMEOUT,
        )?;
        // The spec expects this after a successful initialize; the server treats it as
        // informational, so a failed write only matters if the transport is already gone.
        me.notify("notifications/initialized")?;
        Ok(me)
    }

    /// One `tools/call`: `Ok((true, text))` for a result, `Ok((false, message))` when the
    /// server answered `error` or `isError`, `Err` when the transport itself broke.
    pub(crate) fn call(&mut self, name: &str, args: &Value) -> Result<(bool, String), AgentError> {
        let reply = self.request("tools/call", json!({"name": name, "arguments": args}), CALL_TIMEOUT)?;
        let Some(result) = reply.get("result") else {
            let message =
                reply.get("error").and_then(|e| e["message"].as_str()).unwrap_or("the MCP server sent neither a result nor an error.").to_string();
            return Ok((false, message));
        };
        let ok = result["isError"] != true;
        let text = result.get("content").map(result_text).unwrap_or_default();
        Ok((ok, text))
    }

    /// One JSON-RPC request: write it, then wait for the reply with our id — notifications and
    /// any stragglers from earlier calls (a late reply after a timeout) are skipped.
    fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value, AgentError> {
        self.next_id += 1;
        let id = self.next_id;
        let line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| AgentError::Setup(format!("the MCP connection to the app broke: {e}")))?;
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(AgentError::Setup(format!("the MCP server did not answer {method} within {}s.", timeout.as_secs())));
            }
            match self.lines.recv_timeout(left) {
                // A notification, a foreign id or a line the server's own writer cannot
                // produce: skipped, the reply with our id is still coming.
                Ok(line) => {
                    let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                    if v.get("id").and_then(Value::as_i64) != Some(id) {
                        continue;
                    }
                    return Ok(v);
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(AgentError::Setup(format!("the MCP server did not answer {method} within {}s.", timeout.as_secs())));
                }
                // The child exited or the reader died: the app is most likely gone.
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(AgentError::Setup("the MCP connection closed (is the FigCraft app still running?)".into()));
                }
            }
        }
    }

    /// A JSON-RPC notification: no id, no reply.
    fn notify(&mut self, method: &str) -> Result<(), AgentError> {
        let line = json!({"jsonrpc": "2.0", "method": method}).to_string();
        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.write_all(b"\n"))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| AgentError::Setup(format!("the MCP connection to the app broke: {e}")))
    }
}

impl Drop for McpStdio {
    /// The child dies with the turn; reap it so nothing outlives the session as a zombie.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
