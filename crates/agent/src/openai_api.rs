//! OpenAI's Chat Completions API driven directly — and with it every server that speaks the
//! same protocol: the base URL is the user's to point at api.openai.com, OpenRouter, Ollama
//! or a regional proxy. The same shape as the Anthropic backend: the key comes from the
//! local prefs file, the tools run over our own `figcraft-cli mcp --attach` connection
//! (spawned per turn, no MCP-server config anywhere), and one turn streams
//! `stream: true` round by round — assistant text reaching the panel as it arrives, each
//! `tool_use` executed when its message *ends* (the panel pairs a tool's end with its
//! newest open row), the results going back as `role: "tool"` messages until the model
//! stops. An aborted turn rolls the history back to where it began, so the conversation
//! always ends on a finished assistant message — the API rejects two user messages in a
//! row.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};
use ureq::Agent;

use crate::anthropic_api::SYSTEM;
use crate::mcp_stdio::McpStdio;
use crate::pump::line160;
use crate::{AgentError, AgentEvent, Backend, Turn};

/// The model the panel offers when the user has not picked one.
const DEFAULT_MODEL: &str = "gpt-6.1-sol";
/// The API's origin; `base_url` replaces it for compatible servers and proxies.
const DEFAULT_BASE: &str = "https://api.openai.com";
/// A turn's round cap: a runaway tool loop stops here instead of running all day. The
/// per-round output is deliberately uncapped — the parameter's name differs between
/// OpenAI (`max_completion_tokens`) and the compatible servers (`max_tokens`), and a
/// wrong one makes whole classes of server refuse the request.
const MAX_ROUNDS: u32 = 40;
/// A tool result's text cap: one huge reply (a whole document outline) must not blow up
/// the request. Characters, counted safely — the text is cut on a char boundary.
const MAX_TOOL_TEXT: usize = 100_000;

/// The direct OpenAI-compatible backend: a key, a base URL, a model, and this chat's
/// conversation.
pub struct OpenAiApi {
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

impl OpenAiApi {
    /// A backend ready to be wrapped in a session: `key` is the user's OpenAI (or
    /// compatible) API key — empty means "not configured" and every turn is refused —
    /// `figcraft_cli` the binary that attaches to the running app, `attach` its control
    /// port.
    pub fn new(key: impl Into<String>, figcraft_cli: impl Into<PathBuf>, attach: u16) -> Self {
        OpenAiApi {
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

    /// Pick a model (default `gpt-6.1-sol`). Any name the server carries works — that is
    /// the point of a compatible endpoint.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Point at another endpoint — OpenRouter, Ollama, a proxy. A trailing slash is not
    /// part of the path; an empty URL keeps the default rather than failing every turn
    /// later.
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        let trimmed = base.into();
        let trimmed = trimmed.trim_end_matches('/');
        if !trimmed.is_empty() {
            self.base = trimmed.to_string();
        }
        self
    }
}

impl Backend for OpenAiApi {
    fn name(&self) -> &'static str {
        "OpenAI"
    }

    fn attach(&mut self, tx: Sender<AgentEvent>) {
        self.tx = Some(tx);
    }

    fn start(&mut self, turn: &Turn) -> Result<(), AgentError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(AgentError::Setup("The OpenAI API is still working on the last turn.".into()));
        }
        if self.key.trim().is_empty() {
            return Err(AgentError::Setup("No OpenAI API key is set for this chat.".into()));
        }
        let mut prompt = turn.text.trim().to_string();
        if prompt.is_empty() && turn.attachments.is_empty() {
            return Err(AgentError::Setup("Type a message to start the agent.".into()));
        }
        if !turn.attachments.is_empty() {
            // The API takes text, not files: the names keep the model aware of what the
            // user attached, exactly as the CLI backends phrase it.
            let names: Vec<&str> = turn.attachments.iter().map(|a| a.name.as_str()).collect();
            prompt.push_str(&format!("\n\n(Attached files: {}.)", names.join(", ")));
        }
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
            let terminal = ctx.turn(&prompt);
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

impl Drop for OpenAiApi {
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
    /// `running` false and sends it — never the other way around.
    fn turn(&self, prompt: &str) -> AgentEvent {
        let mark = {
            let mut history = self.history();
            let mark = history.len();
            history.push(json!({"role": "user", "content": prompt}));
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
                "messages": self.request_messages(),
                "tools": tools,
                "stream": true,
                // Usage arrives in its own chunk after the finish reason; without this the
                // totals stay zero.
                "stream_options": {"include_usage": true},
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
            // Tool calls before the finish reason is settled: an assistant message carrying
            // `tool_calls` must always get its `role: "tool"` answers, or the next request
            // is invalid.
            if !round.tools.is_empty() {
                self.push(assistant_message(&round));
                match self.execute_tools(&round.tools, &mut mcp) {
                    ToolRound::Done(messages) => {
                        for message in messages {
                            self.push(message);
                        }
                    }
                    ToolRound::Fatal(message) => return self.abort(mark, AgentEvent::Failed(message)),
                    ToolRound::Dead => return self.abort(mark, AgentEvent::Done { reason: "cancelled".into() }),
                }
                continue;
            }
            if round.text.is_empty() {
                return self.abort(mark, AgentEvent::Failed("The API returned an empty reply.".into()));
            }
            self.push(json!({"role": "assistant", "content": round.text}));
            return AgentEvent::Done { reason: finish_reason(round.stop.as_deref()) };
        }
        self.abort(mark, AgentEvent::Failed(format!("The API kept calling tools; the turn stopped after {MAX_ROUNDS} rounds.")))
    }

    /// The request's messages: the system prompt first (this API takes it as a message),
    /// then the conversation.
    fn request_messages(&self) -> Vec<Value> {
        let mut messages = Vec::with_capacity(self.history().len() + 1);
        messages.push(json!({"role": "system", "content": SYSTEM}));
        messages.extend(self.history().clone());
        messages
    }

    /// One round's SSE stream: text events reach the panel as they arrive, and Stop (or a
    /// panel that is gone) ends the read between lines — the `ping`-equivalent chatter
    /// keeps that latency to seconds, and the request's global timeout is the backstop when
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
    /// `Dead` stops the turn without answers: the panel is gone or the user pressed Stop.
    /// `Fatal` is the MCP connection itself breaking.
    fn execute_tools(&self, tools: &[ToolUse], mcp: &mut Option<McpStdio>) -> ToolRound {
        let mut answers = Vec::with_capacity(tools.len());
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
                    // This API marks no answer as failed itself: the tool's own words say
                    // what happened, exactly what the model reads.
                    answers.push(json!({"role": "tool", "tool_call_id": tool.id, "content": cap_tool_text(&text)}));
                }
                Err(e) => {
                    let _ = self.emit(AgentEvent::ToolEnd { name: tool.name.clone(), ok: false, summary: "the app connection broke".into() });
                    return ToolRound::Fatal(e.to_string());
                }
            }
        }
        ToolRound::Done(answers)
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

/// How one round of tool calls ended: the answers to send back, a broken connection, or a
/// turn that must stop (the panel is gone or the user pressed Stop).
enum ToolRound {
    Done(Vec<Value>),
    Fatal(String),
    Dead,
}

/// The assistant's message for the history: its prose (or `null` — the API takes an empty
/// content alongside `tool_calls`) and the calls it asked for, arguments exactly as sent.
fn assistant_message(round: &Round) -> Value {
    let content = if round.text.is_empty() { Value::Null } else { json!(round.text) };
    let tool_calls: Vec<Value> =
        round.tools.iter().map(|t| json!({"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.raw}})).collect();
    json!({"role": "assistant", "content": content, "tool_calls": tool_calls})
}

/// The finish reason the panel shows: `stop` is this API's word for a clean end, which the
/// panel knows as `end_turn`; anything else rides along spelled out, so a truncated reply
/// (`length`) or a refusal (`content_filter`) reads as itself.
pub(crate) fn finish_reason(stop: Option<&str>) -> String {
    match stop.unwrap_or("stop") {
        "stop" | "end_turn" => "end_turn".to_string(),
        other => other.to_string(),
    }
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
/// message in the body is the real explanation ("Incorrect API key provided", an over
/// quota, …).
fn post_stream(agent: &Agent, base: &str, key: &str, body: &Value) -> Result<impl Read, String> {
    let auth = format!("Bearer {}", key.trim());
    let mut response = agent
        .post(completions_url(base))
        .header("Authorization", auth.as_str())
        .send_json(body)
        .map_err(|e| format!("The API could not be reached: {e}"))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let text = response.body_mut().read_to_string().unwrap_or_default();
        return Err(refusal_message(status, &text));
    }
    Ok(response.into_body().into_reader())
}

/// The endpoint: `{base}/chat/completions` when the base already carries `/v1` (OpenRouter,
/// Ollama and vLLM write their bases that way), else `{base}/v1/chat/completions`. A
/// trailing slash is not part of the path.
pub(crate) fn completions_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.split('/').next_back() == Some("v1") { format!("{base}/chat/completions") } else { format!("{base}/v1/chat/completions") }
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
/// caught by the global timeout.
fn http_agent() -> Agent {
    Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(120)))
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .into()
}

/// The MCP tools in this API's shape: a `function` tool, the schema under `parameters`.
fn api_tools() -> Vec<Value> {
    figcraft_mcp::tool_definitions()
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t["name"].as_str().unwrap_or_default(),
                    "description": t["description"].as_str().unwrap_or_default(),
                    "parameters": t["inputSchema"],
                }
            })
        })
        .collect()
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

/// One streamed reply's state: the text the panel already saw as events, plus everything
/// only the turn loop needs. Built by [`Parser::into_round`] when the stream ends.
#[derive(Default)]
pub(crate) struct Round {
    /// The assistant's prose, as it streamed.
    pub(crate) text: String,
    /// The tool calls to execute, in stream order.
    pub(crate) tools: Vec<ToolUse>,
    /// `finish_reason` — `None` when the stream never carried one.
    pub(crate) stop: Option<String>,
    /// An error object some servers stream mid-round instead of refusing up front.
    pub(crate) error: Option<String>,
    /// The reply finished: a `finish_reason` arrived, or the `[DONE]` marker (a few
    /// compatible servers send neither and just close — those are not finished).
    pub(crate) stopped: bool,
    /// The round's usage, added to the turn's cumulative totals.
    pub(crate) input: u64,
    pub(crate) output: u64,
}

/// One `tool_calls` entry: the id the answers must match, the name, the streamed argument
/// JSON, and `input` parsed — `None` when the JSON is not a usable object (the call still
/// shows; its answer tells the model the arguments were cut off).
pub(crate) struct ToolUse {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) raw: String,
    pub(crate) input: Option<Value>,
}

/// Maps one streamed reply (SSE lines) to panel text events and to the [`Round`] it left
/// behind. Text deltas stream out as they arrive; tool calls accumulate per chunk index
/// until the stream ends — the panel pairs a tool's end with its newest open row, so a
/// `ToolStart` mid-stream could pair against text that has not arrived yet.
#[derive(Default)]
pub(crate) struct Parser {
    /// The prose so far (already streamed out as events).
    text: String,
    /// Chunk tool-call index → the call being assembled: the API splits one call's
    /// arguments across chunks, keyed by this index.
    calls: BTreeMap<u64, ToolCall>,
    stop: Option<String>,
    error: Option<String>,
    input: u64,
    output: u64,
    done_marker: bool,
}

#[derive(Default)]
struct ToolCall {
    id: String,
    name: String,
    raw: String,
}

impl Parser {
    /// One SSE line: only `data:` lines carry payloads; `event:`, `:` comments and blanks
    /// are framing, and `[DONE]` is the stream's end marker. Unknown payload shapes ride
    /// along ignored — a server addition must not break the panel.
    pub(crate) fn feed(&mut self, line: &str) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        let Some(data) = line.trim().strip_prefix("data:").map(str::trim) else { return out };
        if data == "[DONE]" {
            self.done_marker = true;
            return out;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { return out };
        // Some servers stream their errors mid-round instead of refusing up front.
        if v.get("error").is_some_and(|e| !e.is_null()) {
            self.error = Some(v["error"]["message"].as_str().unwrap_or("an error with no message").to_string());
            return out;
        }
        if let Some(tokens) = crate::pump::num(&v["usage"]["prompt_tokens"]) {
            self.input = tokens;
        }
        if let Some(tokens) = crate::pump::num(&v["usage"]["completion_tokens"]) {
            self.output = tokens;
        }
        // Choices carry the reply itself: the finish reason ends it, the delta adds to it.
        let Some(choice) = v["choices"].get(0) else { return out };
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.stop = Some(reason.to_string());
        }
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str()
            && !text.is_empty()
        {
            self.text.push_str(text);
            out.push(AgentEvent::Text(text.to_string()));
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for chunk in calls {
                let index = crate::pump::num(&chunk["index"]).unwrap_or(0);
                let call = self.calls.entry(index).or_default();
                if let Some(id) = chunk["id"].as_str() {
                    call.id = id.to_string();
                }
                if let Some(name) = chunk["function"]["name"].as_str() {
                    call.name = name.to_string();
                }
                if let Some(part) = chunk["function"]["arguments"].as_str() {
                    call.raw.push_str(part);
                }
            }
        }
        out
    }

    /// The round as the turn loop needs it: the prose, the tool calls with their arguments
    /// parsed, the finish reason and the usage.
    pub(crate) fn into_round(self) -> Round {
        // A finish reason or the [DONE] marker ends the reply; a stream with neither (a
        // compatible server closing without ceremony) is unfinished.
        let stopped = self.done_marker || self.stop.is_some();
        let mut tools = Vec::with_capacity(self.calls.len());
        for (i, (_, call)) in self.calls.into_iter().enumerate() {
            // A server that never sent an id still needs one its answers can match.
            let id = if call.id.is_empty() { format!("call_{i}") } else { call.id };
            let input = parse_tool_input(&call.raw);
            tools.push(ToolUse { id, name: call.name, raw: call.raw, input });
        }
        Round { text: self.text, tools, stop: self.stop, error: self.error, stopped, input: self.input, output: self.output }
    }
}

/// A tool call's arguments: the API streams them as JSON text, normally an object. Empty
/// means "no arguments"; anything that does not parse as an object means the arguments
/// were cut off (or the model misbehaved) — the call still shows, and its answer says so.
pub(crate) fn parse_tool_input(raw: &str) -> Option<Value> {
    if raw.trim().is_empty() {
        return Some(json!({}));
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(v) if v.is_object() => Some(v),
        _ => None,
    }
}
