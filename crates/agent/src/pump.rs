//! The turn pump OpenCode and Codex share: keep the child reachable by `cancel`, stream its
//! stdout through the backend's parser, hold a stderr tail for when it dies badly, reap it,
//! and tell the panel how the turn ended when the CLI itself never says (OpenCode has no
//! terminal event; Codex's `turn.completed` is one, but a crash skips it).

use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, PoisonError};

use crate::{AgentError, AgentEvent};

/// One turn's plumbing, assembled by the backend before [`Pump::run`].
pub(crate) struct Pump {
    /// The spawned child (stdout/stderr still attached).
    pub child: Child,
    pub tx: Sender<AgentEvent>,
    /// The session id the parser recognized; the next turn continues it.
    pub session: Arc<Mutex<Option<String>>>,
    /// The slot `cancel` kills through (set before the reader thread starts).
    pub slot: Arc<Mutex<Option<Child>>>,
    pub running: Arc<AtomicBool>,
    pub cancelled: Arc<AtomicBool>,
    /// The backend's display name, for the exit message.
    pub backend: &'static str,
}

impl Pump {
    /// Put the child where `cancel` reaches it and run the reader thread; returns at once, so
    /// the window never waits on the CLI. `map` turns one stdout line into events — kinds it
    /// does not know come back empty, which is what keeps a CLI upgrade from breaking the
    /// panel.
    pub fn run(self, mut map: impl FnMut(&str) -> Vec<AgentEvent> + Send + 'static) -> Result<(), AgentError> {
        let Pump { mut child, tx, session, slot, running, cancelled, backend } = self;
        let stdout = child.stdout.take().ok_or_else(|| AgentError::Setup(format!("{backend} gave no output stream.")))?;
        let stderr = child.stderr.take();
        *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(child);
        cancelled.store(false, Ordering::SeqCst);
        running.store(true, Ordering::SeqCst);
        let tail = Arc::new(Mutex::new(String::new()));
        std::thread::spawn(move || {
            // The stderr tail rides along for the failure message ("permission denied", …).
            let drain = stderr.map(|mut pipe| {
                let tail = Arc::clone(&tail);
                std::thread::spawn(move || {
                    use std::io::BufRead;
                    for line in std::io::BufReader::new(&mut pipe).lines().map_while(Result::ok) {
                        let mut t = tail.lock().unwrap_or_else(PoisonError::into_inner);
                        t.push_str(&line);
                        t.push('\n');
                        let len = t.len();
                        if len > 800 {
                            t.replace_range(..len - 400, "");
                        }
                    }
                })
            });
            let mut finished = false;
            {
                use std::io::BufRead;
                'lines: for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
                    for event in map(&line) {
                        if let AgentEvent::Session(id) = &event {
                            *session.lock().unwrap_or_else(PoisonError::into_inner) = Some(id.clone());
                        }
                        if matches!(event, AgentEvent::Done { .. } | AgentEvent::Failed(_)) {
                            finished = true;
                        }
                        if tx.send(event).is_err() {
                            // The panel is gone: stop reading, but fall through to the reap
                            // below so the child never outlives the session as a zombie.
                            break 'lines;
                        }
                    }
                }
            }
            let mut guard = slot.lock().unwrap_or_else(PoisonError::into_inner);
            let mut me = guard.take();
            let status = me.as_mut().and_then(|c| c.wait().ok());
            drop(guard);
            running.store(false, Ordering::SeqCst);
            if let Some(handle) = drain {
                let _ = handle.join();
            }
            if finished {
                return;
            }
            let event = if cancelled.load(Ordering::SeqCst) {
                AgentEvent::Done { reason: "cancelled".into() }
            } else if status.is_some_and(|s| s.success()) {
                AgentEvent::Done { reason: "success".into() }
            } else {
                let how = status.map_or_else(|| "it stopped".into(), |s| format!("{s}"));
                let mut message = format!("{backend} exited without finishing ({how}).");
                let t = tail.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(last) = t.lines().rev().find(|l| !l.trim().is_empty()) {
                    message.push(' ');
                    message.push_str(&cut(last.trim()));
                }
                AgentEvent::Failed(message)
            };
            let _ = tx.send(event);
        });
        Ok(())
    }
}

/// A tool result's text for the panel's row: MCP answers as a content array of
/// `{type: "text", text}` items, some CLIs as a bare string.
pub(crate) fn result_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items.iter().find_map(|b| b["text"].as_str()).unwrap_or_default().to_string(),
        _ => String::new(),
    }
}

/// One line describing a tool's result for the panel's row. Our own server answers in
/// JSON, which opens with a lone `{` on its own line — that line says nothing, so a
/// bracket-only line gives way to the whole reply flattened into one.
pub(crate) fn line160(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or_default().trim();
    if line.chars().any(|c| c.is_alphanumeric()) { cut(line) } else { cut(&text.split_whitespace().collect::<Vec<_>>().join(" ")) }
}

/// One line to at most 160 characters (157 plus the ellipsis).
pub(crate) fn cut(trimmed: &str) -> String {
    if trimmed.chars().count() > 160 {
        let cut: String = trimmed.chars().take(157).collect();
        format!("{cut}…")
    } else {
        trimmed.to_string()
    }
}

/// A usage counter from a JSON number (the backends spell theirs differently).
pub(crate) fn num(v: &serde_json::Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))
}
