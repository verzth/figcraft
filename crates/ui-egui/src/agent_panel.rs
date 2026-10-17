//! The Agent tab: an AI agent designing in the open document, chat-style.
//!
//! One worker thread per conversation owns a [`figcraft_agent::Session`] (Claude Code in
//! print mode for now) and forwards its events over a channel the panel drains once per
//! frame, so the window never blocks. The agent reaches this app through the same
//! `figcraft-app` MCP server the external clients use in Preferences ▸ AI agents, and its
//! tool calls land here as ordinary control-channel commands.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::Duration;

use egui::{Color32, CornerRadius, Key, Layout, RichText, Sense, Ui, vec2};
use figcraft_agent::{AgentEvent, ClaudeCodeCli, Session, Turn};

use crate::FigApp;
use crate::dialogs::PrefsTab;
use crate::i18n::tr;
use crate::theme::{Tokens, regular, semibold};
use crate::widgets;

/// How often a conversation's worker wakes to check for commands.
const POLL_MS: u64 = 30;
/// The input block's height below the transcript (backend line + field), so the input sits at
/// the panel's bottom edge; slightly generous to keep the outer scroll area from appearing.
const INPUT_H: f32 = 112.0;
/// The backend's name, on the panel and in the tab-strip pill while it works.
pub const BACKEND_NAME: &str = "Claude Code";

/// One transcript row.
#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    /// What the user typed.
    User(String),
    /// The agent's text, in the deltas it streamed.
    Assistant(Vec<String>),
    /// A tool call: its name, its arguments on one line, and its result once it returns.
    Tool { name: String, args: String, result: Option<(bool, String)> },
    /// Something went wrong, or the turn stopped for a reason worth naming.
    Error(String),
}

/// One conversation: its transcript and the worker thread behind it.
#[derive(Default)]
pub struct Chat {
    pub rows: Vec<Row>,
    /// The session id the backend reported (shown in the header; the next turn resumes it).
    pub id: Option<String>,
    /// A turn is running.
    pub running: bool,
    /// Tokens used, when the backend reports them.
    pub tokens: Option<(u64, u64)>,
    link: Option<Link>,
}

/// The worker thread's ends. Dropping it tells the thread to stop.
struct Link {
    cmd: Sender<Cmd>,
    events: Receiver<AgentEvent>,
}

impl Drop for Link {
    fn drop(&mut self) {
        let _ = self.cmd.send(Cmd::Shutdown);
    }
}

enum Cmd {
    Turn(String),
    Cancel,
    Shutdown,
}

/// The Agent tab's state (kept in [`crate::UiState`]).
#[derive(Default)]
pub struct AgentUi {
    pub chats: Vec<Chat>,
    pub active: usize,
    /// The text being typed (shared across conversations, as one draft box).
    pub draft: String,
    /// Why the last send couldn't start (no server, no Claude Code…), shown above the input.
    pub setup_error: Option<String>,
}

/// Draw the tab: header, transcript, then the input box. (Events are drained every frame by
/// [`poll`], whether this tab shows or not.)
pub fn panel(app: &mut FigApp, ui: &mut Ui) {
    if app.ui.agent.chats.is_empty() {
        app.ui.agent.chats.push(Chat::default());
    }
    let ctx = ui.ctx().clone();
    header(app, ui);
    transcript(app, ui);
    input(app, ui, &ctx);
}

/// The top row: a new chat on the left, the turn's status (and tokens) on the right.
fn header(app: &mut FigApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.add_space(widgets::PAD);
        if widgets::button(ui, tr("New chat"), false, true).clicked() {
            app.ui.agent.chats.push(Chat::default());
            app.ui.agent.active = app.ui.agent.chats.len() - 1;
            app.ui.agent.setup_error = None;
        }
        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(widgets::PAD);
            let chat = app.ui.agent.chats.get(app.ui.agent.active);
            let mut bits: Vec<String> = Vec::new();
            if let Some((input, output)) = chat.and_then(|c| c.tokens) {
                bits.push(format!("{} ↑ {} ↓", fmt_tokens(input), fmt_tokens(output)));
            }
            let status = match chat {
                Some(c) if c.running => tr("Working…").to_string(),
                _ => tr("Idle").to_string(),
            };
            bits.push(status);
            let color = if chat.is_some_and(|c| c.running) { t.accent } else { t.text_secondary };
            ui.label(RichText::new(bits.join(" · ")).font(semibold(10.5)).color(color));
        });
    });
}

/// The transcript: assistant prose, the user's bubbles, and one row per tool call.
fn transcript(app: &mut FigApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width();
    let active = app.ui.agent.active;
    let Some(chat) = app.ui.agent.chats.get_mut(active) else { return };
    if chat.rows.is_empty() {
        empty_state(app, ui);
        return;
    }
    let running = chat.running;
    let last_is_assistant = matches!(chat.rows.last(), Some(Row::Assistant(_)));
    // The transcript fills what the header and the input block leave, so the input sits at the
    // panel's bottom edge (chat panels read top-down, write at the bottom).
    let h = (ui.available_height() - INPUT_H).max(180.0);
    egui::ScrollArea::vertical().id_salt(("agent_rows", active)).stick_to_bottom(true).auto_shrink([false, false]).max_height(h).show(ui, |ui| {
        ui.add_space(6.0);
        for row in &chat.rows {
            row_ui(ui, row, w, &t);
        }
        // The caret where the next words land.
        if running && last_is_assistant {
            ui.add_space(4.0);
            ui.label(RichText::new("▍").font(regular(11.5)).color(t.accent));
        }
        ui.add_space(4.0);
    });
}

/// What the tab says before anything was asked.
fn empty_state(app: &mut FigApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(30.0);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new(tr("Ask the agent to design anything")).font(semibold(12.0)).color(t.text));
        ui.add_space(6.0);
        ui.label(
            RichText::new(tr("It designs on the canvas with the figcraft-app tools, while you watch.")).font(regular(10.5)).color(t.text_secondary),
        );
    });
    ui.add_space(16.0);
    let chips: [&str; 3] = [tr("Design a login card"), tr("Make a pricing table"), tr("Lay out these frames")];
    for chip in chips {
        ui.vertical_centered(|ui| {
            let r = widgets::button(ui, chip, false, !app.ui.agent.draft.contains(chip));
            if r.clicked() {
                app.ui.agent.draft = chip.to_string();
            }
        });
        ui.add_space(6.0);
    }
}

fn row_ui(ui: &mut Ui, row: &Row, w: f32, t: &Tokens) {
    match row {
        Row::User(text) => bubble(ui, text, w, t),
        Row::Assistant(segs) => {
            for seg in segs {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add_space(widgets::PAD);
                    ui.set_max_width(w - widgets::PAD * 2.0);
                    ui.add(egui::Label::new(RichText::new(seg).font(regular(11.5)).color(t.text)).wrap());
                });
            }
        }
        Row::Tool { name, args, result } => tool_row(ui, name, args, result.as_ref(), w, t),
        Row::Error(message) => {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space(widgets::PAD);
                ui.set_max_width(w - widgets::PAD * 2.0);
                ui.add(egui::Label::new(RichText::new(message).font(regular(10.5)).color(t.danger)).wrap());
            });
        }
    }
}

/// The user's message, right-aligned on the accent colour.
fn bubble(ui: &mut Ui, text: &str, w: f32, t: &Tokens) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.with_layout(Layout::right_to_left(egui::Align::Min), |ui| {
            ui.add_space(widgets::PAD);
            egui::Frame::NONE.fill(t.accent).corner_radius(CornerRadius::same(8)).inner_margin(egui::Margin::symmetric(10, 6)).show(ui, |ui| {
                ui.set_max_width((w * 0.8).max(120.0));
                ui.add(egui::Label::new(RichText::new(text).font(regular(11.5)).color(Color32::WHITE)).wrap());
            });
        });
    });
}

/// One tool call: a status dot, `name(args…)` on one line, and the result line once it lands.
fn tool_row(ui: &mut Ui, name: &str, args: &str, result: Option<&(bool, String)>, w: f32, t: &Tokens) {
    let short = name.rsplit('.').next().unwrap_or(name);
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add_space(widgets::PAD);
        let (dot, line_color) = match result {
            None => (t.accent, t.text_secondary),
            Some((true, _)) => (t.success, t.text_secondary),
            Some((false, _)) => (t.danger, t.danger),
        };
        let (dr, _) = ui.allocate_exact_size(vec2(7.0, 7.0), Sense::hover());
        ui.painter().circle_filled(dr.center(), 3.5, dot);
        ui.add_space(6.0);
        let text = if args.is_empty() || args == "{}" {
            short.to_string()
        } else {
            format!("{short} {}", widgets::elide_font(ui, args, w - 60.0, regular(10.5)))
        };
        ui.label(RichText::new(text).font(regular(10.5)).color(line_color));
        if result.is_none() {
            ui.label(RichText::new("…").font(regular(10.5)).color(t.text_secondary));
        }
    });
    if let Some((_, summary)) = result
        && !summary.is_empty()
    {
        ui.horizontal(|ui| {
            ui.add_space(widgets::PAD + 13.0);
            ui.set_max_width(w - widgets::PAD * 2.0);
            ui.add(egui::Label::new(RichText::new(summary).font(regular(10.0)).color(t.text_tertiary)).wrap());
        });
    }
}

/// The input box, with the backend line above it: New chat's twin, Stop while it works.
fn input(app: &mut FigApp, ui: &mut Ui, ctx: &egui::Context) {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width();
    let running = app.ui.agent.chats.get(app.ui.agent.active).is_some_and(|c| c.running);
    // Why the last send couldn't start, with a way to fix it.
    if let Some(why) = app.ui.agent.setup_error.clone() {
        ui.horizontal(|ui| {
            ui.add_space(widgets::PAD);
            ui.set_max_width(w - 90.0);
            ui.add(egui::Label::new(RichText::new(why).font(regular(10.0)).color(t.warning)).wrap());
            if widgets::button(ui, tr("Settings"), false, true).clicked() {
                app.ui.preferences = true;
                app.ui.prefs_tab = PrefsTab::Agents;
                app.ui.agent.setup_error = None;
            }
        });
        ui.add_space(4.0);
    }
    // Backend line: Stop while a turn runs, Send when there's something to say.
    ui.horizontal(|ui| {
        ui.add_space(widgets::PAD);
        ui.label(RichText::new(BACKEND_NAME).font(semibold(10.5)).color(t.text_secondary));
        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(widgets::PAD);
            let sendable = !running && !app.ui.agent.draft.trim().is_empty();
            if running {
                if widgets::button(ui, tr("Stop"), false, true).clicked() {
                    cancel(app);
                }
            } else if widgets::button(ui, tr("Send"), true, sendable).clicked() {
                send_turn(app, ctx);
            }
        });
    });
    // The draft: Enter sends, Shift+Enter breaks the line.
    let mut send = false;
    ui.horizontal(|ui| {
        ui.add_space(widgets::PAD);
        let field_w = w - widgets::PAD * 2.0;
        let draft = &mut app.ui.agent.draft;
        let shown = egui::Frame::NONE.fill(t.field).corner_radius(CornerRadius::same(6)).inner_margin(egui::Margin::same(8)).show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(draft)
                    .id(ui.id().with("agent_draft"))
                    .frame(egui::Frame::NONE)
                    .font(regular(11.5))
                    .hint_text(tr("Design anything…"))
                    .desired_rows(3)
                    .desired_width(field_w - 16.0),
            )
        });
        if shown.inner.lost_focus()
            && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift)
            && !app.ui.agent.draft.trim().is_empty()
            && !running
        {
            send = true;
        }
    });
    if send {
        send_turn(app, ctx);
    }
}

/// Ask the active conversation's worker for a turn (starting its thread the first time).
fn send_turn(app: &mut FigApp, ctx: &egui::Context) {
    let text = app.ui.agent.draft.trim().to_string();
    if text.is_empty() {
        return;
    }
    if app.ui.agent.chats.is_empty() {
        app.ui.agent.chats.push(Chat::default());
    }
    let active = app.ui.agent.active;
    app.ui.agent.draft.clear();
    app.ui.agent.setup_error = None;
    match backend_info(app) {
        Err(why) => {
            app.ui.agent.setup_error = Some(why.clone());
            if let Some(chat) = app.ui.agent.chats.get_mut(active) {
                chat.rows.push(Row::Error(why));
            }
        }
        Ok((claude, exe, port)) => {
            if let Some(chat) = app.ui.agent.chats.get_mut(active) {
                if chat.running {
                    return;
                }
                if chat.link.is_none() {
                    let (tx_cmd, rx_cmd) = mpsc::channel();
                    let (tx_ev, rx_ev) = mpsc::channel();
                    // The worker outlives this call: it needs its own clone of the context.
                    let ctx = ctx.clone();
                    std::thread::spawn(move || worker(claude, exe, port, rx_cmd, tx_ev, ctx));
                    chat.link = Some(Link { cmd: tx_cmd, events: rx_ev });
                }
                chat.rows.push(Row::User(text.clone()));
                chat.running = true;
                if let Some(link) = &chat.link {
                    let _ = link.cmd.send(Cmd::Turn(text));
                }
            }
        }
    }
}

/// End the active turn early (its `Done` arrives as `cancelled`).
fn cancel(app: &mut FigApp) {
    let active = app.ui.agent.active;
    if let Some(link) = app.ui.agent.chats.get(active).and_then(|c| c.link.as_ref()) {
        let _ = link.cmd.send(Cmd::Cancel);
    }
}

/// What the panel needs to spawn an agent: the `claude` binary, this app's executable (the
/// MCP server the agent drives), and the port the control server listens on.
fn backend_info(app: &mut FigApp) -> Result<(PathBuf, PathBuf, u16), String> {
    let host = app.services.agents().ok_or_else(|| tr("The AI agents server is not running. Turn it on in Preferences ▸ AI agents.").to_string())?;
    let port =
        host.status().listening.ok_or_else(|| tr("The AI agents server is not running. Turn it on in Preferences ▸ AI agents.").to_string())?;
    let (claude, exe) = host.panel_bins().ok_or_else(|| tr("Claude Code was not found on this computer.").to_string())?;
    Ok((claude, exe, port))
}

/// One conversation's worker: it owns the session, takes commands, forwards events.
fn worker(claude: PathBuf, exe: PathBuf, port: u16, cmd: Receiver<Cmd>, events: Sender<AgentEvent>, ctx: egui::Context) {
    let mut session = Session::new(Box::new(ClaudeCodeCli::new(claude, exe, port)));
    // While a turn runs, repaint steadily even when the model is silent: the panel's header,
    // the caret and the tab-strip pill all track "working" through the per-frame poll.
    let mut busy = false;
    loop {
        match cmd.try_recv() {
            Ok(Cmd::Turn(text)) => {
                let turn = Turn { text, attachments: Vec::new() };
                match session.send(&turn) {
                    Ok(()) => busy = true,
                    Err(e) => {
                        let _ = events.send(AgentEvent::Failed(e.to_string()));
                    }
                }
            }
            Ok(Cmd::Cancel) => session.cancel(),
            Ok(Cmd::Shutdown) | Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }
        let mut woke = false;
        for event in session.poll() {
            woke = true;
            if matches!(event, AgentEvent::Done { .. } | AgentEvent::Failed(_)) {
                busy = false;
            }
            if events.send(event).is_err() {
                return;
            }
        }
        if woke || busy {
            ctx.request_repaint();
        }
        std::thread::sleep(Duration::from_millis(POLL_MS));
    }
}

/// Drain the active conversation's events into its transcript — every frame, whether the
/// Agent tab shows or not, so the pill and the cursor keep up with a hidden panel.
pub(crate) fn poll(app: &mut FigApp, ctx: &egui::Context) {
    let active = app.ui.agent.active;
    let Some(chat) = app.ui.agent.chats.get_mut(active) else { return };
    // Drain the channel first: the events must leave `link` before they land on `chat`.
    let events: Vec<AgentEvent> = match &chat.link {
        Some(link) => link.events.try_iter().collect(),
        None => Vec::new(),
    };
    let mut changed = false;
    for event in events {
        apply(chat, event);
        changed = true;
    }
    if changed || chat.running {
        app.ui.agents.note_panel_activity(ctx.input(|i| i.time));
    }
}

/// One event's effect on the transcript (pure; this is what the tests cover).
pub(crate) fn apply(chat: &mut Chat, event: AgentEvent) {
    match event {
        AgentEvent::Session(id) => chat.id = Some(id),
        AgentEvent::Text(delta) => {
            if !matches!(chat.rows.last(), Some(Row::Assistant(_))) {
                chat.rows.push(Row::Assistant(Vec::new()));
            }
            if let Some(Row::Assistant(segs)) = chat.rows.last_mut() {
                segs.push(delta);
            }
        }
        AgentEvent::ToolStart { name, args } => {
            chat.rows.push(Row::Tool { name, args: one_line(&args), result: None });
        }
        AgentEvent::ToolEnd { ok, summary, .. } => {
            // Attach to the call still waiting for its result (the newest first).
            let pending = chat.rows.iter_mut().rev().find(|r| matches!(r, Row::Tool { result: None, .. }));
            match pending {
                Some(Row::Tool { result, .. }) => *result = Some((ok, summary)),
                // A result with no visible call still gets a row (never lose the outcome).
                _ => chat.rows.push(Row::Tool { name: "tool".into(), args: String::new(), result: Some((ok, summary)) }),
            }
        }
        AgentEvent::Tokens { input, output } => chat.tokens = Some((input, output)),
        AgentEvent::Done { reason } => {
            chat.running = false;
            if !matches!(reason.as_str(), "success" | "end_turn" | "cancelled") {
                chat.rows.push(Row::Error(reason));
            }
        }
        AgentEvent::Failed(why) => {
            chat.running = false;
            chat.rows.push(Row::Error(why));
        }
    }
}

/// Tool arguments on one line, for the row.
pub(crate) fn one_line(args: &str) -> String {
    let flat: String = args.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat == "{}" { String::new() } else { flat }
}

/// 1234 → "1.2k" (token counts in the header).
pub(crate) fn fmt_tokens(n: u64) -> String {
    if n >= 10_000 {
        format!("{}k", n / 1000)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}
