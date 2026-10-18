//! Anthropic's Messages API driven directly: no agent CLI to install and no session file —
//! the key comes from the local prefs file, the tools run over our own `figcraft-cli mcp
//! --attach` connection (spawned per turn, no MCP-server config anywhere), and the
//! conversation lives in this backend for as long as the chat. One turn streams
//! `/v1/messages` (`stream: true`) round by round: assistant text streams to the panel,
//! each `tool_use` block is executed when its message *ends* (never mid-stream — the panel
//! pairs a tool's end with its newest open row, so a `ToolStart` sent mid-stream could pair
//! against text that has not arrived yet), and the tool results go back as the next round's
//! user message until the model stops. An aborted turn rolls the history back to where it
//! began, so the conversation always ends on a finished assistant message — the API rejects
//! two user messages in a row.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};
use ureq::Agent;

use crate::mcp_stdio::McpStdio;
use crate::pump::line160;
use crate::vision::Vision;
use crate::{AgentError, AgentEvent, Backend, Turn};

/// The system prompt: plain design instructions. The tools arrive with every request, so
/// unlike the CLI backends there is no MCP server to point at.
pub const SYSTEM: &str = "You are designing inside the FigCraft design app. The current \
    document is open on the canvas. Use the tools to build and inspect the design; work in \
    small steps and describe what you are doing in one or two sentences per step.";

/// The model the panel offers when the user has not picked one.
const DEFAULT_MODEL: &str = "claude-sonnet-5-5";
/// The API's origin; `base_url` replaces it for proxies and regional endpoints.
const DEFAULT_BASE: &str = "https://api.anthropic.com";
/// The API version this request speaks.
const API_VERSION: &str = "2023-06-01";
/// One round's output cap. 8192 sits under every model's ceiling, and is far more than a
/// prose answer plus tool arguments ever needs — the model can ask for more rounds.
const MAX_TOKENS: u32 = 8192;
/// A turn's round cap: a runaway tool loop stops here instead of running all day.
const MAX_ROUNDS: u32 = 40;
/// A tool result's text cap: one huge reply (a whole document outline) must not blow up
/// the request. Characters, counted safely — the text is cut on a char boundary.
const MAX_TOOL_TEXT: usize = 100_000;

/// The direct Anthropic backend: a key, a model, and this chat's conversation.
pub struct AnthropicApi {
    key: String,
    model: String,
    base: String,
    figcraft_cli: PathBuf,
    attach: u16,
    /// The conversation as the API wants it (`messages`), from this chat's first turn on.
    /// Empty, or ending with an assistant message that finished — an aborted turn rolls
    /// back to where it started.
    history: Arc<Mutex<Vec<Value>>>,
    tx: Option<Sender<AgentEvent>>,
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

impl AnthropicApi {
    /// A backend ready to be wrapped in a session: `key` is the user's Anthropic API key
    /// (empty means "not configured" and every turn is refused), `figcraft_cli` the binary
    /// that attaches to the running app, `attach` its control port.
    pub fn new(key: impl Into<String>, figcraft_cli: impl Into<PathBuf>, attach: u16) -> Self {
        AnthropicApi {
            key: key.into(),
            model: DEFAULT_MODEL.to_string(),
            base: DEFAULT_BASE.to_string(),
            figcraft_cli: figcraft_cli.into(),
            attach,
            history: Arc::new(Mutex::new(Vec::new())),
            tx: None,
            running: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Pick a model (default `claude-sonnet-5-5`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Point at another endpoint — a proxy or a regional host. A trailing slash is not part
    /// of the path; an empty URL keeps the default rather than failing every turn later.
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        let trimmed = base.into();
        let trimmed = trimmed.trim_end_matches('/');
        if !trimmed.is_empty() {
            self.base = trimmed.to_string();
        }
        self
    }
}

impl Backend for AnthropicApi {
    fn name(&self) -> &'static str {
        "Anthropic API"
    }

    fn attach(&mut self, tx: Sender<AgentEvent>) {
        self.tx = Some(tx);
    }

    fn start(&mut self, turn: &Turn) -> Result<(), AgentError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(AgentError::Setup("The Anthropic API is still working on the last turn.".into()));
        }
        if self.key.trim().is_empty() {
            return Err(AgentError::Setup("No Anthropic API key is set for this chat.".into()));
        }
        let vision = Vision::of(turn);
        if vision.prompt.is_empty() {
            return Err(AgentError::Setup("Type a message to start the agent.".into()));
        }
        // The images the bytes proved to be travel as image blocks beside the text; the
        // prompt already names every attachment, carried or not, the way the CLIs do.
        let content = user_content(&vision);
        let tx = self.tx.clone().ok_or_else(|| AgentError::Setup("The session has no event channel.".into()))?;
        self.cancelled.store(false, Ordering::SeqCst);
        self.running.store(true, Ordering::SeqCst);
        let ctx = TurnCtx {
            key: self.key.clone(),
            model: self.model.clone(),
            base: self.base.clone(),
            figcraft_cli: self.figcraft_cli.clone(),
            attach: self.attach,
            history: Arc::clone(&self.history),
            tx,
            running: Arc::clone(&self.running),
            cancelled: Arc::clone(&self.cancelled),
        };
        std::thread::spawn(move || {
            let terminal = ctx.turn(content);
            // The flag drops before the terminal event: a panel that reacts to `Done` at
            // once and sends the next turn must not be refused as "still working".
            ctx.running.store(false, Ordering::SeqCst);
            let _ = ctx.tx.send(terminal);
        });
        Ok(())
    }

    fn cancel(&mut self) {
        if !self.running.load(Ordering::SeqCst) {
            return;
        }
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

impl Drop for AnthropicApi {
    /// The chat closed mid-turn: stop the thread's work at its next safe point. The thread
    /// holds its own `Arc`s, so nothing dangles — a send that fails once the panel is gone
    /// ends the turn the same way.
    fn drop(&mut self) {
        self.cancel();
    }
}

/// What one turn's thread carries: the configuration, the shared conversation, and the
/// channel back to the panel.
struct TurnCtx {
    key: String,
    model: String,
    base: String,
    figcraft_cli: PathBuf,
    attach: u16,
    history: Arc<Mutex<Vec<Value>>>,
    tx: Sender<AgentEvent>,
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

impl TurnCtx {
    /// The whole turn, round by round; returns the event that ends it. The caller sets
    /// `running` false and sends it — never the other way around. `content` is the user
    /// message already built: plain text, or text with image blocks beside it.
    fn turn(&self, content: Value) -> AgentEvent {
        let mark = {
            let mut history = self.history();
            let mark = history.len();
            history.push(json!({"role": "user", "content": content}));
            mark
        };
        let tools = api_tools();
        let agent = http_agent();
        let mut mcp: Option<McpStdio> = None;
        let (mut input, mut output) = (0u64, 0u64);
        let (mut sent_input, mut sent_output) = (0u64, 0u64);

        for _round in 0..MAX_ROUNDS {
            if self.cancelled.load(Ordering::SeqCst) {
                return self.abort(mark, AgentEvent::Done { reason: "cancelled".into() });
            }
            let body = json!({
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "system": SYSTEM,
                "messages": self.history().clone(),
                "tools": tools,
                "stream": true,
            });
            let reader = match post_stream(&agent, &self.base, &self.key, &body) {
                Ok(reader) => reader,
                Err(message) => return self.abort(mark, AgentEvent::Failed(message)),
            };
            let (parser, cancelled, gone) = self.stream(reader);
            let round = parser.into_round();
            input += round.input;
            output += round.output;
            // Cumulative usage reaches the panel before any terminal event that follows.
            if (input, output) != (sent_input, sent_output) && (input != 0 || output != 0) {
                sent_input = input;
                sent_output = output;
                let _ = self.tx.send(AgentEvent::Tokens { input, output });
            }
            if cancelled || gone || self.cancelled.load(Ordering::SeqCst) {
                return self.abort(mark, AgentEvent::Done { reason: "cancelled".into() });
            }
            if let Some(message) = round.error {
                return self.abort(mark, AgentEvent::Failed(format!("The API reported an error: {message}")));
            }
            if !round.stopped {
                return self.abort(mark, AgentEvent::Failed("The API's stream ended before the reply was finished.".into()));
            }
            // Tools before the stop reason: an assistant message carrying a `tool_use`
            // block must always get its `tool_result`, or the next request is invalid.
            if !round.tools.is_empty() {
                self.push(json!({"role": "assistant", "content": round.content}));
                match self.execute_tools(&round.tools, &mut mcp) {
                    ToolRound::Done(results) => self.push(json!({"role": "user", "content": results})),
                    ToolRound::Fatal(message) => return self.abort(mark, AgentEvent::Failed(message)),
                    ToolRound::Dead => return self.abort(mark, AgentEvent::Done { reason: "cancelled".into() }),
                }
                continue;
            }
            if round.content.is_empty() {
                return self.abort(mark, AgentEvent::Failed("The API returned an empty reply.".into()));
            }
            self.push(json!({"role": "assistant", "content": round.content}));
            // A stream that stopped without a reason ended normally: the panel shows only
            // the reasons it knows, so anything else must arrive here spelled out.
            let reason = round.stop.unwrap_or_else(|| "end_turn".to_string());
            return AgentEvent::Done { reason };
        }
        self.abort(mark, AgentEvent::Failed(format!("The API kept calling tools; the turn stopped after {MAX_ROUNDS} rounds.")))
    }

    /// One round's SSE stream: text events reach the panel as they arrive, and Stop (or a
    /// panel that is gone) ends the read between lines — the `ping` events the API sends
    /// keep that latency to seconds, and the request's global timeout is the backstop when
    /// a line never comes. `cancelled` and `gone` say why the stream stopped early.
    fn stream(&self, reader: impl Read) -> (Parser, bool, bool) {
        let mut parser = Parser::default();
        for line in BufReader::new(reader).lines() {
            if self.cancelled.load(Ordering::SeqCst) {
                return (parser, true, false);
            }
            let Ok(line) = line else { break };
            for event in parser.feed(&line) {
                if self.tx.send(event).is_err() {
                    return (parser, false, true);
                }
            }
        }
        (parser, false, false)
    }

    /// Run one round's tool calls. Each `ToolStart` is sent when the call begins (after
    /// the message stopped — see the module doc), each `ToolEnd` when it answers.
    /// `Dead` stops the turn without a `tool_result`: the panel is gone or the user
    /// pressed Stop. `Fatal` is the MCP connection itself breaking.
    fn execute_tools(&self, tools: &[ToolUse], mcp: &mut Option<McpStdio>) -> ToolRound {
        let mut results = Vec::with_capacity(tools.len());
        for tool in tools {
            if self.cancelled.load(Ordering::SeqCst) {
                return ToolRound::Dead;
            }
            let args = if tool.raw.is_empty() { "{}".to_string() } else { tool.raw.clone() };
            if !self.emit(AgentEvent::ToolStart { name: tool.name.clone(), args }) {
                return ToolRound::Dead;
            }
            let outcome: Result<(bool, String), AgentError> = match &tool.input {
                // Arguments that never parsed: the call still shows, and the model is
                // told to send them again. The app is never asked to run garbage.
                None => Ok((false, "The API's tool arguments were not valid JSON.".into())),
                Some(input) => match ensure_client(mcp, &self.figcraft_cli, self.attach) {
                    Ok(client) => client.call(&tool.name, input),
                    Err(e) => Err(e),
                },
            };
            match outcome {
                Ok((ok, text)) => {
                    let summary = line160(&text);
                    if !self.emit(AgentEvent::ToolEnd { name: tool.name.clone(), ok, summary }) {
                        return ToolRound::Dead;
                    }
                    results.push(tool_result(tool, ok, &text));
                }
                Err(e) => {
                    let _ = self.emit(AgentEvent::ToolEnd { name: tool.name.clone(), ok: false, summary: "the app connection broke".into() });
                    return ToolRound::Fatal(e.to_string());
                }
            }
        }
        ToolRound::Done(results)
    }

    /// One event to the panel. `false` means the channel is gone — the turn stops like a
    /// cancel, because there is nobody left to tell.
    fn emit(&self, event: AgentEvent) -> bool {
        let ok = self.tx.send(event).is_ok();
        if !ok {
            self.cancelled.store(true, Ordering::SeqCst);
        }
        ok
    }

    /// Roll the conversation back to its state before this turn and return the terminal
    /// event: an aborted turn leaves no half-finished messages behind.
    fn abort(&self, mark: usize, event: AgentEvent) -> AgentEvent {
        self.history().truncate(mark);
        event
    }

    fn push(&self, message: Value) {
        self.history().push(message);
    }

    fn history(&self) -> MutexGuard<'_, Vec<Value>> {
        self.history.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// How one round of tool calls ended: results to send back, a broken connection, or a turn
/// that must stop (the panel is gone or the user pressed Stop).
enum ToolRound {
    Done(Vec<Value>),
    Fatal(String),
    Dead,
}

/// The turn's MCP client, spawned lazily on the first tool call — a chat that never calls
/// a tool never spawns a child, and a spawn failure fails the turn with no orphan row.
fn ensure_client<'a>(slot: &'a mut Option<McpStdio>, cli: &std::path::Path, attach: u16) -> Result<&'a mut McpStdio, AgentError> {
    if slot.is_none() {
        *slot = Some(McpStdio::open(cli, attach)?);
    }
    slot.as_mut().ok_or_else(|| AgentError::Setup("The MCP client for this turn is gone.".into()))
}

/// One streamed round: the response's body reader, or the API's own answer for a refusal
/// (`error.message`, cut to length). A 4xx/5xx is a response, not a transport error — the
/// message in the body is the real explanation ("invalid x-api-key", an over quota, …).
fn post_stream(agent: &Agent, base: &str, key: &str, body: &Value) -> Result<impl Read, String> {
    let mut response = agent
        .post(format!("{base}/v1/messages"))
        .header("x-api-key", key.trim())
        .header("anthropic-version", API_VERSION)
        .send_json(body)
        .map_err(|e| format!("The API could not be reached: {e}"))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let text = response.body_mut().read_to_string().unwrap_or_default();
        return Err(refusal_message(status, &text));
    }
    Ok(response.into_body().into_reader())
}

/// The body of a refused round as the panel's error row: the API's `error.message` when it
/// sent one, else the body's first characters. Capped — a response body must not become an
/// unbounded error row.
fn refusal_message(status: u16, text: &str) -> String {
    let parsed = serde_json::from_str::<Value>(text).ok();
    let detail = parsed.as_ref().and_then(|v| v["error"]["message"].as_str()).unwrap_or_else(|| text.trim());
    let detail: String = detail.chars().take(400).collect();
    if detail.is_empty() {
        format!("The API refused the request (HTTP {status}).")
    } else {
        format!("The API refused the request (HTTP {status}): {detail}")
    }
}

/// The HTTP agent for a turn: 4xx/5xx arrive as responses (the body explains), and the
/// timeouts are sized for a streamed round — the app reachable within seconds, response
/// headers within two minutes, the whole round within ten. There is deliberately no
/// *body* budget: that is a total timer and would kill any long stream; a stalled read is
/// caught by the global timeout, and the API's `ping` lines keep Stop responsive in
/// between.
fn http_agent() -> Agent {
    Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(120)))
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .into()
}

/// The MCP tools in the API's shape: `inputSchema` becomes `input_schema`, and the MCP
/// extras (`title`, `annotations`) ride along nowhere — the model sees name, description
/// and schema.
fn api_tools() -> Vec<Value> {
    figcraft_mcp::tool_definitions()
        .iter()
        .map(|t| {
            json!({
                "name": t["name"].as_str().unwrap_or_default(),
                "description": t["description"].as_str().unwrap_or_default(),
                "input_schema": t["inputSchema"],
            })
        })
        .collect()
}

/// The user message: the prompt alone when nothing was carried, or a text block with the
/// images beside it — each one's bytes base64-encoded under the media type they proved
/// to be. (The prompt names every attachment either way, so the model can refer to it.)
fn user_content(vision: &Vision) -> Value {
    if vision.images.is_empty() {
        return Value::String(vision.prompt.clone());
    }
    let mut blocks = vec![json!({"type": "text", "text": vision.prompt})];
    for (attachment, media_type) in vision.images.iter().copied() {
        blocks.push(json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": crate::vision::encode(&attachment.data)},
        }));
    }
    Value::Array(blocks)
}

/// One `tool_result` block answering a call: the text capped, and `is_error` whenever the
/// tool (or its arguments) failed, so the model knows to correct itself.
fn tool_result(tool: &ToolUse, ok: bool, text: &str) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": tool.id,
        "content": cap_tool_text(text),
        "is_error": !ok,
    })
}

/// A tool result's text for the API: long enough for real output, capped so one huge
/// reply cannot blow up the request. Cut on a char boundary — never mid-codepoint.
fn cap_tool_text(text: &str) -> String {
    if text.chars().count() <= MAX_TOOL_TEXT {
        return text.to_string();
    }
    let cut: String = text.chars().take(MAX_TOOL_TEXT).collect();
    format!("{cut}\n… (truncated at {MAX_TOOL_TEXT} characters)")
}

/// One streamed round's state: the text the panel already saw as events, plus everything
/// only the turn loop needs. Built by [`Parser::into_round`] when the stream ends.
#[derive(Default)]
pub(crate) struct Round {
    /// The assistant message's blocks, in stream order — text and `tool_use` interleaved.
    pub(crate) content: Vec<Value>,
    /// The `tool_use` blocks to execute, in stream order.
    pub(crate) tools: Vec<ToolUse>,
    /// `message_delta.stop_reason` — `None` when the stream never carried one.
    pub(crate) stop: Option<String>,
    /// The API's own `error` event message, if any.
    pub(crate) error: Option<String>,
    /// `message_stop` arrived: the round really finished.
    pub(crate) stopped: bool,
    /// The round's usage, added to the turn's cumulative totals.
    pub(crate) input: u64,
    pub(crate) output: u64,
}

/// One `tool_use` block: the id the result must answer, the name, the streamed argument
/// JSON, and `input` parsed — `None` when the JSON is not a usable object (the call still
/// shows; its result tells the model the arguments were cut off).
pub(crate) struct ToolUse {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) raw: String,
    pub(crate) input: Option<Value>,
}

/// Maps one streamed reply (SSE lines) to panel text events and to the [`Round`] it left
/// behind. Text deltas stream out as they arrive; tool arguments accumulate per block
/// index until the message stops — the panel pairs a tool's end with its newest open row,
/// so a `ToolStart` mid-stream could pair against text that has not arrived yet.
#[derive(Default)]
pub(crate) struct Parser {
    /// Block index → the block. A `BTreeMap` because the index *is* the stream order.
    blocks: BTreeMap<u64, Block>,
    stop: Option<String>,
    error: Option<String>,
    input: u64,
    output: u64,
    stopped: bool,
}

enum Block {
    Text(String),
    Tool { id: String, name: String, raw: String },
}

/// An absent block starts empty: text to append to.
impl Default for Block {
    fn default() -> Self {
        Block::Text(String::new())
    }
}

impl Parser {
    /// One SSE line: only `data:` lines carry payloads; `event:`, `:` comments and blanks
    /// are framing. Unknown payload kinds ride along ignored — an API addition must not
    /// break the panel.
    pub(crate) fn feed(&mut self, line: &str) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else { return out };
        let Ok(v) = serde_json::from_str::<Value>(data) else { return out };
        let Some(kind) = v["type"].as_str() else { return out };
        match kind {
            "message_start" => {
                self.input = crate::pump::num(&v["message"]["usage"]["input_tokens"]).unwrap_or(self.input);
            }
            "content_block_start" => {
                let Some(index) = crate::pump::num(&v["index"]) else { return out };
                let block = &v["content_block"];
                match block["type"].as_str() {
                    Some("text") => {
                        // Usually empty here; a non-empty start must still reach the panel.
                        let text = block["text"].as_str().unwrap_or_default();
                        self.blocks.insert(index, Block::Text(text.to_string()));
                        if !text.is_empty() {
                            out.push(AgentEvent::Text(text.to_string()));
                        }
                    }
                    Some("tool_use") => {
                        // Without an id there is no result to answer with, and without a
                        // name no call to make: the block is dropped, not half-kept.
                        let (Some(id), Some(name)) = (block["id"].as_str(), block["name"].as_str()) else { return out };
                        self.blocks.insert(index, Block::Tool { id: id.to_string(), name: name.to_string(), raw: String::new() });
                    }
                    // Thinking blocks and whatever comes next: recorded as nothing, their
                    // deltas are skipped along with them.
                    _ => {}
                }
            }
            "content_block_delta" => {
                let Some(index) = crate::pump::num(&v["index"]) else { return out };
                match v["delta"]["type"].as_str() {
                    Some("text_delta") => {
                        let Some(text) = v["delta"]["text"].as_str() else { return out };
                        if text.is_empty() {
                            return out;
                        }
                        let Block::Text(buf) = self.blocks.entry(index).or_insert(Block::Text(String::new())) else { return out };
                        buf.push_str(text);
                        out.push(AgentEvent::Text(text.to_string()));
                    }
                    Some("input_json_delta") => {
                        let Some(part) = v["delta"]["partial_json"].as_str() else { return out };
                        if part.is_empty() {
                            return out;
                        }
                        let Block::Tool { raw, .. } =
                            self.blocks.entry(index).or_insert(Block::Tool { id: String::new(), name: String::new(), raw: String::new() })
                        else {
                            return out;
                        };
                        raw.push_str(part);
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(stop) = v["delta"]["stop_reason"].as_str() {
                    self.stop = Some(stop.to_string());
                }
                if let Some(tokens) = crate::pump::num(&v["usage"]["output_tokens"]) {
                    self.output = tokens;
                }
            }
            "message_stop" => self.stopped = true,
            "error" => {
                self.error = Some(v["error"]["message"].as_str().unwrap_or("an error with no message").to_string());
            }
            // "ping", "content_block_stop" and anything new: ignored.
            _ => {}
        }
        out
    }

    /// The round as the turn loop needs it: the assistant message in stream order, the
    /// tool calls with their arguments parsed, the stop reason and the usage.
    pub(crate) fn into_round(self) -> Round {
        let mut content = Vec::new();
        let mut tools = Vec::new();
        for block in self.blocks.into_values() {
            match block {
                Block::Text(text) => {
                    // An empty text block says nothing and the API may reject it.
                    if !text.is_empty() {
                        content.push(json!({"type": "text", "text": text}));
                    }
                }
                Block::Tool { id, name, raw } => {
                    let input = parse_tool_input(&raw);
                    content.push(json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        // The API rejects anything but an object here, so unparseable
                        // arguments go back as `{}` — their result says what happened.
                        "input": input.clone().unwrap_or_else(|| json!({})),
                    }));
                    tools.push(ToolUse { id, name, raw, input });
                }
            }
        }
        Round { content, tools, stop: self.stop, error: self.error, stopped: self.stopped, input: self.input, output: self.output }
    }
}

/// A tool call's arguments: the API streams them as JSON text, normally an object. Empty
/// means "no arguments"; anything that does not parse as an object means the arguments
/// were cut off (or the model misbehaved) — the call still shows, and its result says so.
pub(crate) fn parse_tool_input(raw: &str) -> Option<Value> {
    if raw.trim().is_empty() {
        return Some(json!({}));
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(v) if v.is_object() => Some(v),
        _ => None,
    }
}
