//! The Agent tab (in the left rail, after File): an AI agent designing in the open document,
//! chat-style, laid out like pen.dev's panel — the session dropdown and New chat on top,
//! the transcript in the middle, the field with its controls row pinned at the bottom.
//!
//! One worker thread per conversation owns a [`figcraft_agent::Session`] and forwards its
//! events over a channel the panel drains once per frame, so the window never blocks. The
//! backend is whichever CLI this computer has — Claude Code, OpenCode or Codex — chosen by the
//! model the picker routes it to; the agent reaches this app through the same
//! `figcraft-app` MCP server the external clients use in Preferences ▸ AI agents, and its
//! tool calls land here as ordinary control-channel commands. Every conversation, its
//! transcript and the session id that continues it are kept in a local history file and come
//! back next launch.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::Duration;

use egui::{Align2, Color32, CornerRadius, Key, Layout, Rect, RichText, Sense, Stroke, Ui, pos2, vec2};
#[cfg(not(target_arch = "wasm32"))]
use figcraft_agent::AnthropicApi;
#[cfg(not(target_arch = "wasm32"))]
use figcraft_agent::OpenAiApi;
use figcraft_agent::{
    AgentEvent, ApiKeys, Attachment, Backend, ClaudeCodeCli, CodexCli, MAX_IMAGE_BYTES, OpenCodeCli, Session, Turn, image_media_type,
};
use figcraft_engine::{AgentParallel, AgentRefine};

use crate::FigApp;
use crate::agent_models::{ModelChoice, Models, Route};
use crate::agents::{BackendKind, PanelBackend};
use crate::dialogs::PrefsTab;
use crate::i18n::{tr, trf};
use crate::theme::{Tokens, regular, semibold};
use crate::widgets;

/// How often a conversation's worker wakes to check for commands.
const POLL_MS: u64 = 30;
/// The input block's height below the transcript (field + controls row), so the input sits
/// [`EDGE_PAD`] clear of the panel's bottom edge; slightly generous to keep the outer scroll
/// area from appearing.
const INPUT_H: f32 = 100.0;
/// The content's distance from the panel's top and bottom edges: the frame carries no margin
/// of its own (`lib.rs` draws it `Frame::NONE`), so the tab insets its own rows — the same
/// padding the sides already use.
const EDGE_PAD: f32 = widgets::PAD;
/// Extra input-block height for the setup note above the field, and for the drop chips.
const ERROR_H: f32 = 40.0;
const DROPS_H: f32 = 36.0;
/// Attachments: at most this many (text files and images together), each text file read
/// to at most [`MAX_DROP_CHARS`] characters and each image to [`MAX_IMAGE_BYTES`] bytes.
/// The attach and drop cap: at most this many files wait for the next send.
pub(crate) const MAX_DROPS: usize = 4;
const MAX_DROP_CHARS: usize = 32_768;
/// The composer's counts: at most this many agents run in parallel on one send, and this
/// many refine rounds each of them runs.
const MAX_RUNS: u32 = 4;
/// What the attach button offers: text files whose text joins the message, and images that
/// travel with the turn (the same two kinds the panel accepts as drops).
const ATTACH_EXTENSIONS: &[&str] = &["md", "markdown", "txt", "json", "csv", "yaml", "yml", "toml", "svg", "png", "jpg", "jpeg", "webp", "gif"];

/// One transcript row.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

/// One conversation: its transcript, the worker thread behind it, and what the history file
/// needs to bring it back.
#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Chat {
    pub rows: Vec<Row>,
    /// The session id the backend reported (the next turn resumes it; the history keeps it).
    pub id: Option<String>,
    /// A turn is running.
    pub running: bool,
    /// Tokens used, when the backend reports them.
    pub tokens: Option<(u64, u64)>,
    /// Whom this conversation runs (set on the first send; the pill names it).
    pub kind: Option<BackendKind>,
    /// When the chat was opened (epoch seconds; the history groups days from it).
    pub created: f64,
    /// Which parallel variant of one send this chat is: 1…n when a send runs several agents
    /// at once, 0 for a chat of its own. The history names it "… · n".
    pub variant: usize,
    /// Refine rounds this agent still runs by itself after a turn completes.
    pub rounds_left: u32,
    /// The live worker (never saved; crate-visible so a test can play the worker).
    #[serde(skip)]
    pub(crate) link: Option<Link>,
}

impl Chat {
    /// A fresh conversation, stamped now (the history groups it under Today).
    fn new() -> Self {
        Self { created: now_secs(), ..Default::default() }
    }

    /// The session dropdown's name: the first thing asked, else "New Agent" (pen.dev's).
    pub fn title(&self) -> String {
        for row in &self.rows {
            let Row::User(text) = row else { continue };
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let mut out = String::new();
            for (i, ch) in text.chars().enumerate() {
                if i >= 48 {
                    out.push('…');
                    break;
                }
                out.push(ch);
            }
            // A parallel variant says so after the words, so chats from one send tell apart.
            if self.variant > 0 {
                out.push_str(&format!(" · {}", self.variant));
            }
            return out;
        }
        tr("New Agent").to_string()
    }
}

/// The worker thread's ends. Dropping it tells the thread to stop. (Crate-visible so the
/// tests can play the worker: hold a channel end and speak into it.)
pub(crate) struct Link {
    pub(crate) cmd: Sender<Cmd>,
    pub(crate) events: Receiver<AgentEvent>,
}

impl Drop for Link {
    fn drop(&mut self) {
        let _ = self.cmd.send(Cmd::Shutdown);
    }
}

pub(crate) enum Cmd {
    Turn(Turn),
    Cancel,
    Shutdown,
}

/// One thing dropped on the panel, waiting for the next send: a text file (its text goes
/// into the message) or an image (its bytes travel with the turn — the API backends see
/// it, the CLIs name it in the prompt).
#[derive(Clone)]
pub enum Pending {
    /// A text file, already read to text.
    Text {
        /// The file name.
        name: String,
        /// The file's text, capped to [`MAX_DROP_CHARS`] characters.
        content: String,
    },
    /// An image whose bytes proved to be one (sniffed by [`image_media_type`]).
    Image {
        /// The file name.
        name: String,
        /// The media type its magic bytes claimed: `image/png`, …
        mime: &'static str,
        /// The image's bytes, at most [`MAX_IMAGE_BYTES`] of them.
        data: Vec<u8>,
    },
}

impl Pending {
    /// The file's name, as the chip draws it.
    fn name(&self) -> &str {
        match self {
            Pending::Text { name, .. } | Pending::Image { name, .. } => name,
        }
    }
}

/// The Agent tab's state (kept in [`crate::UiState`]).
#[derive(Default)]
pub struct AgentUi {
    pub chats: Vec<Chat>,
    pub active: usize,
    /// The text being typed (shared across conversations, as one draft box).
    pub draft: String,
    /// Text files and images dropped on the panel, waiting for the next send.
    pub attachments: Vec<Pending>,
    /// Why the last send couldn't start (no server, no backend…), shown above the input.
    pub setup_error: Option<String>,
    /// The backends found on this computer (scanned once, when the tab first draws).
    pub backends: Vec<PanelBackend>,
    /// Whether `backends` has been looked up yet.
    pub resolved: bool,
    /// Which backend the next turn prefers (an index into `backends`).
    pub backend: usize,
    /// The model picker: the whole list, and what's picked.
    pub models: Models,
    /// The chat history was read from the disk (or there was none to read).
    pub loaded: bool,
    /// "Clear Agent History" waits for a second click.
    pub clear_armed: bool,
}

impl AgentUi {
    /// The backend the next turn prefers, as shown; "Agent" when this computer has none.
    pub fn selected_name(&self) -> String {
        self.backends.get(self.backend).map(|b| b.kind.name().to_string()).unwrap_or_else(|| tr("Agent").to_string())
    }

    /// Whom the tab-strip pill names: the active conversation's backend once it has spoken,
    /// else the backend the next turn would use.
    pub fn pill_name(&self) -> String {
        match self.chats.get(self.active).and_then(|c| c.kind) {
            Some(kind) => kind.name().to_string(),
            None => self.selected_name(),
        }
    }
}

/// What the history file holds: which chat is open, and every conversation in it.
#[derive(serde::Serialize, serde::Deserialize)]
struct History {
    active: usize,
    chats: Vec<Chat>,
}

/// The same, borrowed, for writing it out: a save shouldn't copy every transcript.
#[derive(serde::Serialize)]
struct HistoryRef<'a> {
    active: usize,
    chats: &'a [Chat],
}

/// Draw the tab: header, transcript, then the input box. (Events are drained every frame by
/// [`poll`], whether this tab shows or not.)
pub fn panel(app: &mut FigApp, ui: &mut Ui) {
    ensure_backends(app);
    // The picker's choices, resolved once: the curated lists immediately, OpenCode's
    // own list arriving on a thread a moment later.
    app.ui.agent.models.ensure(&app.ui.agent.backends);
    restore(app);
    if app.ui.agent.chats.is_empty() {
        app.ui.agent.chats.push(Chat::new());
    }
    let ctx = ui.ctx().clone();
    dropped(app, &ctx);
    // The frame carries no margin, so the header keeps its own distance from the top edge.
    ui.add_space(EDGE_PAD);
    header(app, ui);
    transcript(app, ui);
    input(app, ui, &ctx);
}

/// Look the backends up once, the first time the tab draws (a PATH scan, not a per-frame one).
fn ensure_backends(app: &mut FigApp) {
    if app.ui.agent.resolved {
        return;
    }
    app.ui.agent.backends = app.services.agents().map(|h| h.panel_backends()).unwrap_or_default();
    app.ui.agent.resolved = true;
    if app.ui.agent.backend >= app.ui.agent.backends.len() {
        app.ui.agent.backend = 0;
    }
}

/// Read the chat history once, the first time the tab draws: every conversation with its
/// transcript, still-warm sessions included (no turn is running after a launch).
fn restore(app: &mut FigApp) {
    if app.ui.agent.loaded {
        return;
    }
    app.ui.agent.loaded = true;
    let Some(json) = app.services.load_agent_sessions() else { return };
    let Ok(history) = serde_json::from_str::<History>(&json) else { return };
    let mut chats = history.chats;
    if chats.is_empty() {
        return;
    }
    for chat in &mut chats {
        chat.running = false;
        chat.link = None;
    }
    app.ui.agent.active = history.active.min(chats.len() - 1);
    app.ui.agent.chats = chats;
}

/// Keep the chat history beside the preferences: every conversation, its transcript, and the
/// session id that continues each one.
fn persist(app: &mut FigApp) {
    let json = {
        let history = HistoryRef { active: app.ui.agent.active, chats: &app.ui.agent.chats };
        match serde_json::to_string(&history) {
            Ok(json) => json,
            Err(_) => return,
        }
    };
    app.services.save_agent_sessions(&json);
}

/// Now, in seconds since the epoch.
#[cfg(not(target_arch = "wasm32"))]
fn now_secs() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Now, in seconds since the epoch (the browser has no history file to group by days).
#[cfg(target_arch = "wasm32")]
fn now_secs() -> f64 {
    0.0
}

/// The top row, as pen.dev's: the session dropdown (its name, click for the history), the
/// turn's status and tokens, then New chat and the collapse button.
fn header(app: &mut FigApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.add_space(widgets::PAD);
        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(widgets::PAD);
            if widgets::icon_button(ui, crate::icons::Icon::Panels, tr("Collapse panel")) {
                app.ui.left_collapsed = true;
            }
            if widgets::button(ui, tr("New chat"), false, true).clicked() {
                app.ui.agent.chats.push(Chat::new());
                app.ui.agent.active = app.ui.agent.chats.len() - 1;
                app.ui.agent.setup_error = None;
                app.ui.agent.clear_armed = false;
                persist(app);
            }
            // Which CLI runs the turns, and how the turn's going. This sits after the session
            // dropdown in the right-to-left flow, so it elides into whatever the dropdown and
            // the two buttons leave; the session name itself stays readable.
            if !app.ui.agent.backends.is_empty() {
                let chat = app.ui.agent.chats.get(app.ui.agent.active);
                let status = match chat {
                    Some(c) if c.running => tr("Working…").to_string(),
                    _ => tr("Idle").to_string(),
                };
                let mut bits: Vec<String> = Vec::new();
                if let Some((input, output)) = chat.and_then(|c| c.tokens) {
                    bits.push(format!("{} ↑ {} ↓", fmt_tokens(input), fmt_tokens(output)));
                }
                bits.push(status);
                let color = if chat.is_some_and(|c| c.running) { t.accent } else { t.text_secondary };
                let bits = bits.join(" · ");
                // Placed before the session dropdown in this right-to-left row, so the chat's
                // name gets its room first: this label only takes what's above ~96px for the
                // dropdown, and steps out entirely when the row is too tight to hold both.
                let room = (ui.available_width() - 96.0).clamp(0.0, 72.0);
                if room >= 24.0 {
                    let shown = widgets::elide_font(ui, &bits, room, semibold(10.5));
                    ui.label(RichText::new(shown).font(semibold(10.5)).color(color));
                }
            }
            session_trigger(app, ui, &t);
        });
    });
}

/// The session dropdown in the header: the current chat's name, and the history below it.
fn session_trigger(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    let title = app.ui.agent.chats.get(app.ui.agent.active).map(|c| c.title()).unwrap_or_else(|| tr("New Agent").to_string());
    // Take what's left of the header row; a floor here would over-allocate and blow the
    // panel's rect leftward (the row already stacks the collapse button, New chat and status).
    // The row's outer padding already clears the panel's edge, so no slack on top of it.
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 30.0), Sense::click());
    ui.painter().rect_filled(rect, CornerRadius::same(8), t.field);
    if resp.hovered() {
        ui.painter().rect_filled(rect, CornerRadius::same(8), t.hover);
    }
    ui.painter().rect_stroke(rect, CornerRadius::same(8), Stroke::new(1.0, t.border), egui::StrokeKind::Inside);
    let shown = widgets::elide_font(ui, &title, w - 34.0, semibold(11.5));
    ui.painter().text(pos2(rect.min.x + 10.0, rect.center().y), Align2::LEFT_CENTER, shown, semibold(11.5), t.text);
    crate::icons::paint(
        ui.painter(),
        Rect::from_center_size(pos2(rect.max.x - 15.0, rect.center().y), vec2(10.0, 10.0)),
        crate::icons::Icon::Chevron,
        t.icon_secondary,
    );
    crate::a11y::button(&resp, &title);
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tr("Chat history"));
    let id = egui::Id::new("agent_sessions");
    egui::Popup::from_toggle_button_response(&resp)
        .id(id)
        .align(egui::RectAlign::TOP_START)
        .width(280.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| history_menu(app, ui, t));
}

/// One history group: its label, and the test that decides which chats belong to it.
type HistoryGroup = (&'static str, Box<dyn Fn(f64) -> bool>);

/// A short stamp beside a history entry so chats inside one group tell apart: the clock time for
/// today, "Yesterday", then the day count. Several new chats all fall back to "New Agent", so
/// only their time separates them in the list.
pub(crate) fn hist_stamp(created: f64, now: f64) -> String {
    let day = 86_400.0;
    let days = ((now / day).floor() - (created / day).floor()).max(0.0) as i64;
    if days <= 0 {
        let secs = created.rem_euclid(day) as i32;
        format!("{:02}:{:02}", secs / 3600, (secs % 3600) / 60)
    } else if days == 1 {
        tr("Yesterday").to_string()
    } else {
        format!("{days}d")
    }
}

/// The history popup: Today, Previous 7 days and Older, each chat named by its first message
/// (pen.dev's list), with Clear Agent History below — a second click confirms it.
fn history_menu(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    // Pin the popup's width: `Popup::width` only seeds a default, and the area then shrinks to
    // its content's min size — a narrow list would clip "Clear Agent History" and elide even the
    // short "New Agent" title. Fixing the width here keeps every row a predictable size.
    ui.set_width(280.0);
    let now = now_secs();
    let day = 86_400.0;
    // The days count from the epoch: the popup says "Today" for the same UTC day — close
    // enough without a date library, and every other day sits in a group that fits.
    let today = (now / day).floor();
    // The three groups must not overlap: each chat lands in exactly one, or the same conversation
    // lists under every heading at once (a chat from today also reads as "within the last 7 days"
    // and — with a catch-all — as "older" too). "Today" claims the same UTC day and any unset time;
    // the rest split what's left on how many days old it is.
    let groups: [HistoryGroup; 3] = [
        (tr("Today"), Box::new(move |at| at <= 0.0 || (at / day).floor() == today)),
        (tr("Previous 7 days"), Box::new(move |at| at > 0.0 && (at / day).floor() != today && now - at < 7.0 * day)),
        (tr("Older"), Box::new(move |at| at > 0.0 && (at / day).floor() != today && now - at >= 7.0 * day)),
    ];
    let mut switch: Option<usize> = None;
    ui.add_space(4.0);
    for (label, in_group) in groups {
        let members: Vec<usize> =
            (0..app.ui.agent.chats.len()).filter(|i| app.ui.agent.chats.get(*i).is_some_and(|c| in_group(c.created))).rev().collect();
        if members.is_empty() {
            continue;
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_space(10.0);
            ui.label(RichText::new(label).font(semibold(9.5)).color(t.text_secondary));
        });
        for i in members {
            let Some(chat) = app.ui.agent.chats.get(i) else { continue };
            let title = chat.title();
            let active = i == app.ui.agent.active;
            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width() - 8.0, 26.0), Sense::click());
            if active || resp.hovered() {
                ui.painter().rect_filled(rect, CornerRadius::same(5), t.hover);
            }
            // A short stamp on the right tells chats apart inside a group — several empty chats
            // all fall back to "New Agent", and only their time separates them.
            let stamp = ui.painter().layout_no_wrap(hist_stamp(chat.created, now), regular(9.5), t.text_secondary);
            let (stamp_w, stamp_h) = (stamp.size().x, stamp.size().y);
            let room = (rect.width() - 16.0 - stamp_w - 8.0).max(20.0);
            let shown = widgets::elide_font(ui, &title, room, regular(11.0));
            let color = if active { t.accent } else { t.text };
            ui.painter().text(pos2(rect.min.x + 8.0, rect.center().y), Align2::LEFT_CENTER, shown, regular(11.0), color);
            ui.painter().galley(pos2(rect.max.x - 8.0 - stamp_w, rect.center().y - stamp_h * 0.5), stamp, t.text_secondary);
            if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                switch = Some(i);
            }
        }
    }
    if let Some(i) = switch {
        app.ui.agent.active = i;
        app.ui.agent.setup_error = None;
        app.ui.agent.clear_armed = false;
        persist(app);
        ui.close();
    }
    ui.add_space(6.0);
    ui.painter().line_segment([ui.cursor().left_bottom(), ui.cursor().right_bottom()], Stroke::new(1.0, t.border));
    ui.add_space(6.0);
    // Clear Agent History, red — a second click says yes (it forgets every chat).
    let label = if app.ui.agent.clear_armed { tr("Click again to confirm.") } else { tr("Clear Agent History") };
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width() - 8.0, 26.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(rect, CornerRadius::same(5), t.hover);
    }
    ui.painter().text(pos2(rect.min.x + 8.0, rect.center().y), Align2::LEFT_CENTER, label, regular(11.0), t.danger);
    if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
        if app.ui.agent.clear_armed {
            app.ui.agent.chats = vec![Chat::new()];
            app.ui.agent.active = 0;
            app.ui.agent.clear_armed = false;
            persist(app);
            ui.close();
        } else {
            app.ui.agent.clear_armed = true;
        }
    }
    ui.add_space(6.0);
}

/// The transcript: assistant prose, the user's bubbles, and one row per tool call. The middle
/// fills what the header and the input leave, so the input pins to the bottom edge (chat
/// panels read top-down, write at the bottom — pen.dev's placement even before the first ask).
fn transcript(app: &mut FigApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width();
    let active = app.ui.agent.active;
    // What the input block needs below the transcript: its own height, the gap that keeps it
    // clear of the panel's bottom edge, and any note/drop rows above the field.
    let input_h = INPUT_H
        + EDGE_PAD
        + if app.ui.agent.setup_error.is_some() { ERROR_H } else { 0.0 }
        + if app.ui.agent.attachments.is_empty() { 0.0 } else { DROPS_H };
    let Some(chat) = app.ui.agent.chats.get_mut(active) else { return };
    if chat.rows.is_empty() {
        empty_state(app, ui);
        let want = ui.max_rect().max.y - input_h;
        ui.add_space((want - ui.cursor().min.y).max(0.0));
        return;
    }
    let running = chat.running;
    let last_is_assistant = matches!(chat.rows.last(), Some(Row::Assistant(_)));
    let h = (ui.available_height() - input_h).max(180.0);
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

/// What the tab says before anything was asked: pen.dev's shape — a title, the suggestion
/// chips as full-width rows, and the context-files note, all left-aligned.
fn empty_state(app: &mut FigApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(26.0);
    ui.horizontal(|ui| {
        ui.add_space(widgets::PAD);
        ui.label(RichText::new(tr("Ask me to design anything")).font(semibold(13.0)).color(t.text));
    });
    ui.add_space(14.0);
    let chips: [&str; 4] = [
        tr("Design a landing page for a coffee roastery"),
        tr("Design a dashboard for a home energy monitor"),
        tr("Design a booking flow for a yoga studio"),
        tr("Design an editorial layout for a photo magazine"),
    ];
    for chip in chips {
        ui.horizontal(|ui| {
            ui.add_space(widgets::PAD);
            let taken = app.ui.agent.draft == chip;
            if chip_row(ui, chip, &t, taken) {
                app.ui.agent.draft = chip.to_string();
            }
        });
        ui.add_space(8.0);
    }
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        ui.add_space(widgets::PAD);
        ui.label(RichText::new(tr("Attach context files.")).font(semibold(11.5)).color(t.text));
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.set_max_width((ui.available_width() - widgets::PAD).max(80.0));
        ui.add_space(widgets::PAD);
        ui.add(
            egui::Label::new(
                RichText::new(tr("Drop a DESIGN.md, a text file or an image into the chat; it goes with your message."))
                    .font(regular(10.5))
                    .color(t.text_secondary),
            )
            .wrap(),
        );
    });
}

/// One suggestion chip: a full-width bordered row with left-aligned, elided text (pen.dev's).
fn chip_row(ui: &mut Ui, label: &str, t: &Tokens, taken: bool) -> bool {
    // The row starts after the left PAD inside a horizontal; keep the same PAD on the right
    // so the chip's border doesn't hug the panel edge.
    let w = (ui.available_width() - widgets::PAD).max(60.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 34.0), Sense::click());
    if taken || resp.hovered() {
        ui.painter().rect_filled(rect, CornerRadius::same(8), if taken { t.selected_row } else { t.hover });
    }
    ui.painter().rect_stroke(rect, CornerRadius::same(8), Stroke::new(1.0, t.border), egui::StrokeKind::Inside);
    let shown = widgets::elide_font(ui, label, w - 24.0, regular(11.5));
    ui.painter().text(pos2(rect.min.x + 12.0, rect.center().y), Align2::LEFT_CENTER, shown, regular(11.5), t.text);
    resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked()
}

fn row_ui(ui: &mut Ui, row: &Row, w: f32, t: &Tokens) {
    match row {
        Row::User(text) => bubble(ui, text, w, t),
        Row::Assistant(segs) => {
            for seg in segs {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.set_max_width(w - widgets::PAD * 2.0);
                    ui.add_space(widgets::PAD);
                    ui.add(egui::Label::new(RichText::new(seg).font(regular(11.5)).color(t.text)).wrap());
                });
            }
        }
        Row::Tool { name, args, result } => tool_row(ui, name, args, result.as_ref(), w, t),
        Row::Error(message) => {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.set_max_width(w - widgets::PAD * 2.0);
                ui.add_space(widgets::PAD);
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
    let short = short_tool_name(name);
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
            ui.set_max_width((w - widgets::PAD * 2.0 - 13.0).max(80.0));
            ui.add_space(widgets::PAD + 13.0);
            ui.add(egui::Label::new(RichText::new(summary).font(regular(10.0)).color(t.text_tertiary)).wrap());
        });
    }
}

/// The input box, as pen.dev's: one rounded container holding the field, and below it —
/// inside the same border — the controls row: attach, the parallel-agents count and the
/// refine-rounds count at the left; the model picker and the round send button at the right.
fn input(app: &mut FigApp, ui: &mut Ui, ctx: &egui::Context) {
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width();
    let running = app.ui.agent.chats.get(app.ui.agent.active).is_some_and(|c| c.running);
    // Why the last send couldn't start, with a way to fix it.
    if let Some(why) = app.ui.agent.setup_error.clone() {
        ui.horizontal(|ui| {
            ui.set_max_width(w - 90.0);
            ui.add_space(widgets::PAD);
            ui.add(egui::Label::new(RichText::new(why).font(regular(10.0)).color(t.warning)).wrap());
            if widgets::button(ui, tr("Settings"), false, true).clicked() {
                app.ui.preferences = true;
                app.ui.prefs_tab = PrefsTab::Agents;
                app.ui.agent.setup_error = None;
            }
        });
        ui.add_space(4.0);
    }
    // Dropped files, waiting for the next send (click one to take it back).
    if !app.ui.agent.attachments.is_empty() {
        let mut remove: Option<usize> = None;
        ui.horizontal_wrapped(|ui| {
            ui.add_space(widgets::PAD);
            for (i, pending) in app.ui.agent.attachments.iter().enumerate() {
                let shown = format!("× {}", widgets::elide_font(ui, pending.name(), 140.0, regular(10.5)));
                let picked =
                    egui::Frame::NONE.fill(t.field).corner_radius(CornerRadius::same(6)).inner_margin(egui::Margin::symmetric(8, 3)).show(ui, |ui| {
                        let r = ui
                            .add(egui::Label::new(RichText::new(shown).font(regular(10.5)).color(t.text_secondary)).sense(Sense::click()))
                            .on_hover_cursor(egui::CursorIcon::PointingHand);
                        r.clicked()
                    });
                if picked.inner {
                    remove = Some(i);
                }
            }
        });
        if let Some(i) = remove {
            app.ui.agent.attachments.remove(i);
        }
        ui.add_space(4.0);
    }
    let has_text = !app.ui.agent.draft.trim().is_empty();
    let has_context = !app.ui.agent.attachments.is_empty();
    let sendable = !running && (has_text || has_context);
    let mut send = false;
    ui.horizontal(|ui| {
        ui.set_max_width((ui.available_width() - widgets::PAD).max(120.0));
        ui.add_space(widgets::PAD);
        egui::Frame::NONE
            .fill(t.field)
            .corner_radius(CornerRadius::same(10))
            .inner_margin(egui::Margin::same(8))
            .stroke(Stroke::new(1.0, t.border))
            .show(ui, |ui| {
                // The field stacks: the draft on top, the controls row under it. The frame's
                // ui inherits the row's horizontal layout, so it has to be told — left as it
                // was, the draft took the field's whole width and shoved the controls past
                // the panel's right edge.
                ui.vertical(|ui| {
                    let field_w = ui.available_width();
                    // The draft: Enter sends, Shift+Enter breaks the line.
                    let draft = &mut app.ui.agent.draft;
                    let shown = ui.add(
                        egui::TextEdit::multiline(draft)
                            .id(ui.id().with("agent_draft"))
                            .frame(egui::Frame::NONE)
                            .font(regular(11.5))
                            .hint_text(tr("Design anything…"))
                            .desired_rows(3)
                            .desired_width(field_w - 4.0),
                    );
                    // Enter sends (NumpadEnter maps to it too): a multiline field keeps its focus
                    // when Enter is pressed, so watch for the key while the field holds it.
                    if shown.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift) && (has_text || has_context) && !running {
                        send = true;
                    }
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        // Left: attach a file, then the shape of the send — how many agents
                        // work in parallel (each in its own chat, the model picker chooses
                        // what they run) and how many refine rounds each one makes.
                        attach_button(app, ui, &t);
                        parallel_pill(app, ui, &t);
                        iterate_pill(app, ui, &t);
                        // Right: the model picker, then the round send button (Stop while it works).
                        ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                            if send_stop(ui, running, sendable, &t) {
                                if running {
                                    cancel(app);
                                } else {
                                    send = true;
                                }
                            }
                            model_trigger(app, ui, &t);
                        });
                    });
                });
            });
    });
    if send {
        send_turn(app, ctx);
    }
}

/// The round send button at the field's bottom-right — an up arrow on the accent colour,
/// a square to stop while a turn runs.
fn send_stop(ui: &mut Ui, running: bool, sendable: bool, t: &Tokens) -> bool {
    let (rect, resp) = ui.allocate_exact_size(vec2(26.0, 26.0), Sense::click());
    let p = ui.painter();
    if running {
        p.circle_stroke(rect.center(), 13.0, Stroke::new(1.0, t.border));
        p.rect_filled(Rect::from_center_size(rect.center(), vec2(9.0, 9.0)), CornerRadius::same(2), t.text_secondary);
        crate::a11y::button(&resp, tr("Stop"));
        return resp.clicked();
    }
    p.circle_filled(rect.center(), 13.0, if sendable { t.accent } else { t.border });
    let color = if sendable { Color32::WHITE } else { t.icon_secondary };
    let c = rect.center();
    let stroke = Stroke::new(1.6, color);
    p.line_segment([pos2(c.x, c.y + 5.0), pos2(c.x, c.y - 4.5)], stroke);
    p.line_segment([pos2(c.x - 4.0, c.y + 0.5), pos2(c.x, c.y - 4.5)], stroke);
    p.line_segment([pos2(c.x + 4.0, c.y + 0.5), pos2(c.x, c.y - 4.5)], stroke);
    crate::a11y::button(&resp, tr("Send"));
    resp.clicked() && sendable
}

/// The attach button at the field's left: a paperclip opening the system picker — the picked
/// file joins the dropped ones, with the same caps and the same two kinds.
fn attach_button(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    let (rect, resp) = ui.allocate_exact_size(vec2(20.0, 20.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(rect, CornerRadius::same(5), t.hover);
    }
    crate::icons::paint(ui.painter(), Rect::from_center_size(rect.center(), vec2(14.0, 14.0)), crate::icons::Icon::Plus, t.icon_secondary);
    let what = tr("Attach a file");
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(what);
    crate::a11y::button(&resp, what);
    if resp.clicked() {
        app.pick_open(tr("Attach context file"), ATTACH_EXTENSIONS, |app, path| take_file(app, &path));
    }
}

/// A pill in the controls row: its icon and "×N", opening a popover (returned as the response
/// so the popup hangs off it). The hover says what the pill controls.
fn pill(ui: &mut Ui, t: &Tokens, icon: crate::icons::Icon, count: &str, hover: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(46.0, 20.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(rect, CornerRadius::same(5), t.hover);
    }
    crate::icons::paint(ui.painter(), Rect::from_center_size(pos2(rect.min.x + 12.0, rect.center().y), vec2(13.0, 13.0)), icon, t.text_secondary);
    ui.painter().text(pos2(rect.min.x + 21.0, rect.center().y), Align2::LEFT_CENTER, count, semibold(10.5), t.text);
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(hover);
    crate::a11y::button(&resp, hover);
    resp
}

/// The "parallel agents" pill: how many agents run at once and how they share the work.
fn parallel_pill(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    let n = app.session.prefs.agent_runs.0.clamp(1, MAX_RUNS);
    let hover = trf("{n} agents work in parallel, each in its own chat", &[("n", &n.to_string())]);
    let resp = pill(ui, t, crate::icons::Icon::Agents, &format!("×{n}"), &hover);
    egui::Popup::from_toggle_button_response(&resp)
        .id(egui::Id::new("agent_parallel"))
        .align(egui::RectAlign::TOP_START)
        .width(232.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| parallel_menu(app, ui, t));
}

/// The "iteration" pill: how many refine rounds each agent runs and what they aim at.
fn iterate_pill(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    let m = app.session.prefs.agent_runs.1.clamp(1, MAX_RUNS);
    let hover = trf("{n} refine rounds per agent", &[("n", &m.to_string())]);
    let resp = pill(ui, t, crate::icons::Icon::Iterate, &format!("×{m}"), &hover);
    egui::Popup::from_toggle_button_response(&resp)
        .id(egui::Id::new("agent_iterate"))
        .align(egui::RectAlign::TOP_START)
        .width(232.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| iterate_menu(app, ui, t));
}

/// The parallel-agents popover: how the agents share the work (two cards), then how many run.
fn parallel_menu(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    ui.set_width(216.0);
    section(ui, t, tr("Parallel agents"));
    ui.label(RichText::new(tr("Let it cook")).font(regular(10.0)).color(t.text_secondary));
    ui.add_space(8.0);
    let now = app.session.prefs.agent_parallel;
    let cards = [
        (AgentParallel::SplitWork, crate::icons::Icon::SplitWork, tr("Split Work"), tr("Each agent takes a different slice.")),
        (AgentParallel::SideBySide, crate::icons::Icon::SideBySide, tr("Side by side"), tr("Each agent designs its own version.")),
    ];
    for (i, (mode, icon, title, desc)) in cards.into_iter().enumerate() {
        if i > 0 {
            ui.add_space(6.0);
        }
        if mode_card(ui, t, icon, title, desc, now == mode) {
            app.session.prefs.agent_parallel = mode;
            app.services.save_prefs(&app.session.prefs);
        }
    }
    ui.add_space(10.0);
    section(ui, t, tr("Agents"));
    let n = app.session.prefs.agent_runs.0.clamp(1, MAX_RUNS) as usize;
    let labels: Vec<String> = (1..=MAX_RUNS).map(|i| i.to_string()).collect();
    if let Some(pick) = choice_row(ui, t, &labels, n - 1) {
        app.session.prefs.agent_runs.0 = (pick + 1) as u32;
        app.services.save_prefs(&app.session.prefs);
    }
}

/// The iteration popover: what the refine rounds fix (layout or style), then how many run.
fn iterate_menu(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    ui.set_width(216.0);
    section(ui, t, tr("Iteration"));
    let now = app.session.prefs.agent_refine;
    let cards = [
        (AgentRefine::Layout, crate::icons::Icon::Layout, tr("Layout"), tr("Spacing, alignment and hierarchy.")),
        (AgentRefine::Style, crate::icons::Icon::Style, tr("Style"), tr("Colour, typography and visual weight.")),
    ];
    for (i, (mode, icon, title, desc)) in cards.into_iter().enumerate() {
        if i > 0 {
            ui.add_space(6.0);
        }
        if mode_card(ui, t, icon, title, desc, now == mode) {
            app.session.prefs.agent_refine = mode;
            app.services.save_prefs(&app.session.prefs);
        }
    }
    ui.add_space(10.0);
    section(ui, t, tr("Variants"));
    let m = app.session.prefs.agent_runs.1.clamp(1, MAX_RUNS) as usize;
    if let Some(pick) = choice_row(ui, t, &variant_labels(), m - 1) {
        app.session.prefs.agent_runs.1 = (pick + 1) as u32;
        app.services.save_prefs(&app.session.prefs);
    }
}

/// A small section label at the top of a popover group.
fn section(ui: &mut Ui, t: &Tokens, text: &str) {
    ui.label(RichText::new(text).font(semibold(9.5)).color(t.text_secondary));
    ui.add_space(4.0);
}

/// A selectable card in a popover: an icon, a title and a line describing it; the chosen one
/// is filled and outlined in the accent colour.
fn mode_card(ui: &mut Ui, t: &Tokens, icon: crate::icons::Icon, title: &str, desc: &str, selected: bool) -> bool {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 46.0), Sense::click());
    let bg = if selected {
        t.selected_row
    } else if resp.hovered() {
        t.hover
    } else {
        t.field
    };
    ui.painter().rect_filled(rect, CornerRadius::same(8), bg);
    let stroke = if selected { Stroke::new(1.5, t.accent) } else { Stroke::new(1.0, t.border) };
    ui.painter().rect_stroke(rect, CornerRadius::same(8), stroke, egui::StrokeKind::Inside);
    let icon_col = if selected { t.accent_text } else { t.text_secondary };
    crate::icons::paint(ui.painter(), Rect::from_center_size(pos2(rect.min.x + 21.0, rect.center().y), vec2(16.0, 16.0)), icon, icon_col);
    let text_x = rect.min.x + 40.0;
    ui.painter().text(pos2(text_x, rect.min.y + 15.0), Align2::LEFT_CENTER, title, semibold(11.0), t.text);
    let room = (rect.width() - 48.0).max(20.0);
    let desc = widgets::elide_font(ui, desc, room, regular(9.5));
    ui.painter().text(pos2(text_x, rect.min.y + 32.0), Align2::LEFT_CENTER, desc, regular(9.5), t.text_secondary);
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    crate::a11y::button(&resp, title);
    resp.clicked()
}

/// A row of small selectable chips; the chosen one is filled. Returns the picked index.
fn choice_row(ui: &mut Ui, t: &Tokens, labels: &[String], selected: usize) -> Option<usize> {
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (i, label) in labels.iter().enumerate() {
            let chosen = i == selected;
            let (bg, fg) = if chosen { (t.accent, Color32::WHITE) } else { (t.field, t.text_secondary) };
            let resp = ui.add(
                egui::Button::new(RichText::new(label).font(regular(10.5)).color(fg))
                    .fill(bg)
                    .corner_radius(CornerRadius::same(6))
                    .min_size(vec2(0.0, 24.0)),
            );
            if resp.clicked() {
                picked = Some(i);
            }
        }
    });
    picked
}

/// The refine-round choices: "Don't iterate" for a single round, then the count.
fn variant_labels() -> Vec<String> {
    let mut out = vec![tr("Don't iterate").to_string()];
    for i in 2..=MAX_RUNS {
        out.push(i.to_string());
    }
    out
}

/// Take files dropped onto the panel: an image's bytes travel with the next send, and a
/// text file's text becomes part of the message (the empty state's note says so).
fn dropped(app: &mut FigApp, ctx: &egui::Context) {
    let files: Vec<egui::DroppedFileHandle> = ctx.input(|i| i.raw.dropped_files.clone());
    if files.is_empty() {
        return;
    }
    for file in files {
        if app.ui.agent.attachments.len() >= MAX_DROPS {
            break;
        }
        let path = file.path().to_path_buf();
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "context.txt".to_string());
        // A file on disk is read by path; a drop with no file behind it carries its bytes.
        let bytes = if path.exists() { std::fs::read(&path).ok() } else { dropped_now(&file) };
        if let Some(bytes) = bytes {
            add_pending(app, name, bytes);
        }
    }
}

/// A drop's bytes, read right here: the browser hands its own over asynchronously and this panel
/// still takes them in the frame it sees the drop in (as it did before the shared helper grew a
/// future; `pending_open` queues those for the rest of the app).
#[cfg(not(target_arch = "wasm32"))]
fn dropped_now(f: &egui::DroppedFileHandle) -> Option<Vec<u8>> {
    f.bytes().ok()
}

#[cfg(target_arch = "wasm32")]
fn dropped_now(_f: &egui::DroppedFileHandle) -> Option<Vec<u8>> {
    None
}

/// One file chosen with the attach button: read it by path, and let it in as a drop would.
pub(crate) fn take_file(app: &mut FigApp, path: &std::path::Path) {
    if app.ui.agent.attachments.len() >= MAX_DROPS {
        return;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "context.txt".to_string());
    let Ok(bytes) = std::fs::read(path) else {
        app.toast(tr("Couldn't read that file."));
        return;
    };
    add_pending(app, name, bytes);
}

/// What a file's bytes become waiting for the next send: a real image's bytes ride along
/// with the turn (up to the API's own cap), a text file's text joins the message, and a
/// binary — mostly replacement characters — is not this send's context.
pub(crate) fn add_pending(app: &mut FigApp, name: String, bytes: Vec<u8>) {
    if app.ui.agent.attachments.len() >= MAX_DROPS {
        return;
    }
    // A real image the APIs can see (sniffed, not named): its bytes ride along —
    // up to the API's own cap; over it, the file is skipped as any binary is.
    if let Some(mime) = image_media_type(&bytes) {
        if bytes.len() <= MAX_IMAGE_BYTES {
            app.ui.agent.attachments.push(Pending::Image { name, mime, data: bytes });
        }
        return;
    }
    let content: String = String::from_utf8_lossy(&bytes).chars().take(MAX_DROP_CHARS).collect();
    // Mostly replacement characters: a binary — not this send's context.
    if content.matches('\u{fffd}').count() * 40 > content.len() {
        return;
    }
    if content.trim().is_empty() {
        return;
    }
    app.ui.agent.attachments.push(Pending::Text { name, content });
}

/// Which model the next turn runs (pen.dev's picker, one list for every provider): its name
/// in the controls row, and a search-and-pick popup above it.
fn model_trigger(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    let label = app.ui.agent.models.label();
    let id = egui::Id::new("agent_model_picker");
    let was_open = egui::Popup::is_id_open(ui.ctx(), id);
    // Room for the chevron, painted into the gap between the name and the send button. This sits
    // in a right-to-left group whose available_width() has already shrunk past the send button,
    // so the name elides into a fixed share of the row rather than what happens to be left —
    // the backend name on the far left keeps its own share and neither clips the other.
    ui.add_space(16.0);
    let room = ui.available_width().min(150.0);
    let shown = widgets::elide_font(ui, &label, room, semibold(10.5));
    let resp = ui
        .add(egui::Label::new(RichText::new(shown).font(semibold(10.5)).color(t.text)).sense(Sense::click()))
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    crate::a11y::button(&resp, &trf("Model: {name}", &[("name", &label)]));
    let chevron = Rect::from_center_size(pos2(resp.rect.right() + 8.0, resp.rect.center().y), vec2(9.0, 9.0));
    crate::icons::paint(ui.painter(), chevron, crate::icons::Icon::Chevron, t.icon_secondary);
    egui::Popup::from_toggle_button_response(&resp)
        .id(id)
        .align(egui::RectAlign::TOP_START)
        .width(240.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| model_menu(app, ui, t));
    if !was_open && egui::Popup::is_id_open(ui.ctx(), id) {
        app.ui.agent.models.focus_search();
    }
}

/// The popup: a search box, then every model of every provider, grouped.
fn model_menu(app: &mut FigApp, ui: &mut Ui, t: &Tokens) {
    ui.add_space(6.0);
    let search = egui::TextEdit::singleline(&mut app.ui.agent.models.search)
        .hint_text(tr("Search models..."))
        .font(regular(11.0))
        .desired_width(ui.available_width() - 20.0);
    let search = ui.add(search);
    if app.ui.agent.models.take_focus() {
        search.request_focus();
    }
    ui.add_space(4.0);
    let query = app.ui.agent.models.search.to_lowercase();
    let picked = app.ui.agent.models.picked();
    let mut clicked = None;
    egui::ScrollArea::vertical().auto_shrink([false, true]).max_height(320.0).show(ui, |ui| {
        let choices = app.ui.agent.models.choices();
        if choices.is_empty() {
            ui.horizontal(|ui| {
                ui.add_space(10.0);
                ui.label(RichText::new(tr("No models found.")).font(regular(10.5)).color(t.text_secondary));
            });
            return;
        }
        let mut provider = String::new();
        for (i, choice) in choices.iter().enumerate() {
            if !query.is_empty() && !choice.label.to_lowercase().contains(&query) && !choice.id.contains(&query) {
                continue;
            }
            if choice.provider != provider {
                provider = choice.provider.clone();
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.label(RichText::new(provider.to_uppercase()).font(semibold(9.5)).color(t.text_secondary));
                });
                ui.add_space(2.0);
            }
            if model_row(ui, t, choice, picked == Some(i)) {
                clicked = Some(i);
            }
        }
    });
    if let Some(i) = clicked {
        app.ui.agent.models.pick(i);
        ui.close();
    }
}

/// One model row: full width, highlighted when picked or hovered.
fn model_row(ui: &mut Ui, t: &Tokens, choice: &ModelChoice, picked: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click());
    if picked || resp.hovered() {
        ui.painter().rect_filled(rect, CornerRadius::same(5), t.hover);
    }
    let galley = ui.painter().layout_no_wrap(choice.label.clone(), regular(11.5), t.text);
    let y = rect.min.y + (rect.height() - galley.size().y) * 0.5;
    ui.painter().galley(pos2(rect.min.x + 10.0, y), galley, t.text);
    resp.clicked()
}

/// Ask the active conversation's worker for a turn (starting its thread the first time),
/// routing the picked model to a backend that runs it. The composer's agents count decides
/// how many variants run: the active chat takes the first, and every other one is a fresh
/// chat beside it, all working the same prompt in parallel under the same model.
fn send_turn(app: &mut FigApp, ctx: &egui::Context) {
    let original = app.ui.agent.draft.trim().to_string();
    let has_context = !app.ui.agent.attachments.is_empty();
    if original.is_empty() && !has_context {
        return;
    }
    ensure_backends(app);
    if app.ui.agent.chats.is_empty() {
        app.ui.agent.chats.push(Chat::new());
    }
    let active = app.ui.agent.active;
    if app.ui.agent.chats.get(active).is_some_and(|c| c.running) {
        return;
    }
    app.ui.agent.draft.clear();
    app.ui.agent.setup_error = None;
    // The MCP server the agent drives.
    let (exe, port) = match server(app) {
        Ok(parts) => parts,
        Err(why) => return refuse(app, active, why, &original),
    };
    // Where this turn runs: the picked model on a backend that takes it, else the selected
    // backend with the CLI's own default model.
    let route = app.ui.agent.models.route(&app.ui.agent.backends, app.ui.agent.backend);
    let (kind, model) = match route {
        Route::NoBackend => return refuse(app, active, tr("No agent backend was found on this computer.").to_string(), &original),
        Route::Unsupported(label) => {
            let why = trf("No agent on this computer can run {model}.", &[("model", &label)]);
            return refuse(app, active, why, &original);
        }
        Route::Selected(kind) => (kind, None),
        Route::Model { kind, id } => (kind, Some(id)),
    };
    // The CLIs carry their executable; the API backends take their key from the prefs file.
    let bin = app.ui.agent.backends.iter().find(|b| b.kind == kind).and_then(|b| b.bin.clone());
    let keys = app.services.agents().and_then(|h| h.agent_keys()).unwrap_or_default();
    // What the agent sees: the typed message, then every attached text file's content.
    // An image's bytes travel with the turn instead — the API backends see them, the
    // CLIs name them — and the transcript records every attachment by name either way.
    let mut text = original.clone();
    let mut shown = original.clone();
    let mut images = Vec::new();
    for pending in &app.ui.agent.attachments {
        match pending {
            Pending::Text { name, content } => {
                let line = format!("\n\n(Attached file: {name})\n{content}");
                text.push_str(&line);
                shown.push_str(&line);
            }
            Pending::Image { name, mime, data } => {
                shown.push_str(&format!("\n\n(Attached file: {name})"));
                images.push(Attachment { name: name.clone(), mime: mime.to_string(), data: data.clone() });
            }
        }
    }
    // One agent, or several in parallel: each with its own chat, its own worker and its own
    // refine rounds, all on the same prompt and the same model.
    let count = app.session.prefs.agent_runs.0.clamp(1, MAX_RUNS) as usize;
    let rounds = app.session.prefs.agent_runs.1.clamp(1, MAX_RUNS);
    // Several agents: say once, on every one's prompt, how they share the work.
    if count > 1 {
        let mode = match app.session.prefs.agent_parallel {
            AgentParallel::SideBySide => {
                trf("{n} agents work side by side on the same request — each builds a complete, independent design.", &[("n", &count.to_string())])
            }
            AgentParallel::SplitWork => {
                trf("{n} agents split one design between them — each takes a different slice of it.", &[("n", &count.to_string())])
            }
        };
        text.push_str(&format!("\n\n{mode}"));
        shown.push_str(&format!("\n\n{mode}"));
    }
    let mut started = 0;
    for i in 0..count {
        let idx = if i == 0 {
            active
        } else {
            app.ui.agent.chats.push(Chat::new());
            app.ui.agent.chats.len() - 1
        };
        let v =
            Variant { idx, variant: if count > 1 { i + 1 } else { 0 }, rounds, kind, model: model.clone(), bin: bin.clone(), exe: exe.clone(), port };
        if !start_variant(app, ctx, &v, &keys, &text, &shown, &images, &original) {
            break;
        }
        started += 1;
    }
    // The drops belong to this turn now — and everything that could still refuse has run,
    // so a send that never started keeps them for the retry.
    if started > 0 {
        app.ui.agent.attachments.clear();
    }
    persist(app);
}

/// One variant of a send: which chat it lands in, and what it runs on.
struct Variant {
    idx: usize,
    variant: usize,
    rounds: u32,
    kind: BackendKind,
    model: Option<String>,
    bin: Option<PathBuf>,
    exe: PathBuf,
    port: u16,
}

/// One chat's half of a send: its worker (built and started the first time), the prompt on
/// its transcript, and the refine rounds it will run by itself once a turn completes. False
/// when this variant couldn't start — the reason is on its transcript and above the field.
fn start_variant(
    app: &mut FigApp,
    ctx: &egui::Context,
    v: &Variant,
    keys: &ApiKeys,
    text: &str,
    shown: &str,
    images: &[Attachment],
    original: &str,
) -> bool {
    // A chat restored from the history continues its CLI session — under the backend that
    // owns that session id, or a fresh one after a backend switch.
    let resume = match app.ui.agent.chats.get(v.idx) {
        Some(chat) if chat.link.is_none() => chat.id.clone().filter(|_| chat.kind == Some(v.kind)),
        _ => None,
    };
    let mut built = None;
    if app.ui.agent.chats.get(v.idx).is_some_and(|c| c.link.is_none()) {
        match build_backend(v.kind, v.bin.clone(), v.exe.clone(), v.port, v.model.clone(), keys) {
            Ok(b) => built = Some(b),
            Err(why) => {
                refuse(app, v.idx, why, original);
                return false;
            }
        }
    }
    let Some(chat) = app.ui.agent.chats.get_mut(v.idx) else { return false };
    if let Some(built) = built {
        let (tx_cmd, rx_cmd) = mpsc::channel();
        let (tx_ev, rx_ev) = mpsc::channel();
        // The worker outlives this call: it needs its own clone of the context.
        let ctx = ctx.clone();
        std::thread::spawn(move || worker(built, resume, rx_cmd, tx_ev, ctx));
        chat.link = Some(Link { cmd: tx_cmd, events: rx_ev });
    }
    chat.kind = Some(v.kind);
    chat.variant = v.variant;
    chat.rounds_left = v.rounds.saturating_sub(1);
    if chat.created <= 0.0 {
        chat.created = now_secs();
    }
    chat.rows.push(Row::User(shown.to_string()));
    chat.running = true;
    if let Some(link) = &chat.link {
        let _ = link.cmd.send(Cmd::Turn(Turn { text: text.to_string(), attachments: images.to_vec() }));
    }
    true
}

/// Why the last send couldn't start, said above the input and in the transcript — with the
/// typed message kept in the draft to try again.
fn refuse(app: &mut FigApp, active: usize, why: String, restore: &str) {
    app.ui.agent.setup_error = Some(why.clone());
    app.ui.agent.draft = restore.to_string();
    if let Some(chat) = app.ui.agent.chats.get_mut(active) {
        chat.rows.push(Row::Error(why));
    }
    persist(app);
}

/// End the active turn early (its `Done` arrives as `cancelled`).
fn cancel(app: &mut FigApp) {
    let active = app.ui.agent.active;
    if let Some(link) = app.ui.agent.chats.get(active).and_then(|c| c.link.as_ref()) {
        let _ = link.cmd.send(Cmd::Cancel);
    }
}

/// What the panel needs to spawn an agent: this app's executable (the MCP server the agent
/// drives) and the port the control server listens on.
fn server(app: &mut FigApp) -> Result<(PathBuf, u16), String> {
    let no_server = || tr("The AI agents server is not running. Turn it on in Preferences ▸ AI agents.").to_string();
    let host = app.services.agents().ok_or_else(no_server)?;
    let port = host.status().listening.ok_or_else(no_server)?;
    let exe = host.mcp_exe().ok_or_else(no_server)?;
    Ok((exe, port))
}

/// The backend this conversation's worker runs, from what the host found on this computer.
/// `model` pins the run to one model; `None` leaves the backend's own default. The CLIs
/// take the binary the scan found; the API kinds take the key from the prefs file.
fn build_backend(
    kind: BackendKind,
    bin: Option<PathBuf>,
    exe: PathBuf,
    port: u16,
    model: Option<String>,
    keys: &ApiKeys,
) -> Result<Box<dyn Backend>, String> {
    let no_bin = || tr("No agent backend was found on this computer.").to_string();
    match kind {
        BackendKind::ClaudeCode => {
            let bin = bin.ok_or_else(no_bin)?;
            let backend = ClaudeCodeCli::new(bin, exe, port);
            Ok(Box::new(match model {
                Some(model) => backend.model(model),
                None => backend,
            }))
        }
        BackendKind::OpenCode => {
            let bin = bin.ok_or_else(no_bin)?;
            let backend = OpenCodeCli::new(bin, exe, port);
            Ok(Box::new(match model {
                Some(model) => backend.model(model),
                None => backend,
            }))
        }
        BackendKind::Codex => {
            let bin = bin.ok_or_else(no_bin)?;
            let backend = CodexCli::new(bin, exe, port);
            Ok(Box::new(match model {
                Some(model) => backend.model(model),
                None => backend,
            }))
        }
        BackendKind::AnthropicApi | BackendKind::OpenAiApi => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                api_backend(kind, exe, port, model, keys)
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = (exe, port, model, keys);
                Err(tr("API keys run in the desktop app: a browser tab can't keep them.").to_string())
            }
        }
    }
}

/// The direct-API backends: the key (and optional base URL) from the prefs file, the app
/// itself reached over the same `mcp --attach` connection the CLIs use.
#[cfg(not(target_arch = "wasm32"))]
fn api_backend(kind: BackendKind, exe: PathBuf, port: u16, model: Option<String>, keys: &ApiKeys) -> Result<Box<dyn Backend>, String> {
    let no_key = |provider: &str| trf("No {provider} API key is set for this chat.", &[("provider", &provider)]);
    match kind {
        BackendKind::AnthropicApi => {
            let key = keys.anthropic.clone().filter(|k| !k.trim().is_empty()).ok_or_else(|| no_key("Anthropic"))?;
            let mut backend = AnthropicApi::new(key, exe, port);
            if let Some(base) = keys.anthropic_base.as_deref().filter(|b| !b.trim().is_empty()) {
                backend = backend.base_url(base);
            }
            if let Some(model) = model {
                backend = backend.model(model);
            }
            Ok(Box::new(backend))
        }
        BackendKind::OpenAiApi => {
            let key = keys.openai.clone().filter(|k| !k.trim().is_empty()).ok_or_else(|| no_key("OpenAI"))?;
            let mut backend = OpenAiApi::new(key, exe, port);
            if let Some(base) = keys.openai_base.as_deref().filter(|b| !b.trim().is_empty()) {
                backend = backend.base_url(base);
            }
            if let Some(model) = model {
                backend = backend.model(model);
            }
            Ok(Box::new(backend))
        }
        _ => Err(tr("No agent backend was found on this computer.").to_string()),
    }
}

/// One conversation's worker: it owns the session (continuing `resume` first, for a chat
/// restored from history), takes commands, forwards events.
fn worker(backend: Box<dyn Backend>, resume: Option<String>, cmd: Receiver<Cmd>, events: Sender<AgentEvent>, ctx: egui::Context) {
    let mut session = Session::new(backend);
    if let Some(id) = resume {
        session.resume(id);
    }
    // While a turn runs, repaint steadily even when the model is silent: the panel's header,
    // the caret and the tab-strip pill all track "working" through the per-frame poll.
    let mut busy = false;
    loop {
        match cmd.try_recv() {
            Ok(Cmd::Turn(turn)) => match session.send(&turn) {
                Ok(()) => busy = true,
                Err(e) => {
                    let _ = events.send(AgentEvent::Failed(e.to_string()));
                }
            },
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

/// Drain every conversation's events into its transcript — every frame, whether the Agent tab
/// shows or not (a chat keeps working while another is open), and keep the history file
/// current whenever a transcript changes.
pub(crate) fn poll(app: &mut FigApp, ctx: &egui::Context) {
    let mut changed = false;
    let mut refine: Vec<usize> = Vec::new();
    for (i, chat) in app.ui.agent.chats.iter_mut().enumerate() {
        // Drain the channels first: the events must leave `link` before they land on `chat`.
        let events: Vec<AgentEvent> = match &chat.link {
            Some(link) => link.events.try_iter().collect(),
            None => Vec::new(),
        };
        // A turn that ended well is the moment for this chat's next refine round (if any);
        // a cancelled or failed one ends the rounds here.
        let finished = events.iter().any(|e| matches!(e, AgentEvent::Done { reason } if matches!(reason.as_str(), "success" | "end_turn")));
        for event in events {
            apply(chat, event);
            changed = true;
        }
        if finished && !chat.running && chat.rounds_left > 0 {
            refine.push(i);
        }
    }
    for i in refine {
        changed |= refine_once(app, ctx, i);
    }
    let busy = app.ui.agent.chats.iter().any(|c| c.running);
    if changed || busy {
        app.ui.agents.note_panel_activity(ctx.input(|i| i.time));
    }
    if changed {
        persist(app);
    }
}

/// One chat's next refine round: the prompt the composer's rounds count promised, asked of
/// the same worker. Only a chat with a live worker and rounds left goes again — a chat come
/// back from the history has no worker to ask, and its rounds end here.
pub(crate) fn refine_once(app: &mut FigApp, ctx: &egui::Context, idx: usize) -> bool {
    // What a refine round asks for, set by the composer's iteration popover: the layout
    // rounds fix spacing and hierarchy, the style rounds fix colour and typography.
    let text = match app.session.prefs.agent_refine {
        AgentRefine::Layout => {
            tr("Review the canvas and your last changes, then refine the layout: fix spacing, alignment, hierarchy and the rhythm between elements.")
        }
        AgentRefine::Style => {
            tr("Review the canvas and your last changes, then refine the style: fix colour, typography, visual weight and consistency.")
        }
    }
    .to_string();
    let Some(chat) = app.ui.agent.chats.get_mut(idx) else { return false };
    if chat.running || chat.rounds_left == 0 {
        return false;
    }
    let Some(link) = &chat.link else {
        chat.rounds_left = 0;
        return false;
    };
    chat.rounds_left -= 1;
    chat.rows.push(Row::User(text.clone()));
    chat.running = true;
    let sent = link.cmd.send(Cmd::Turn(Turn { text, attachments: Vec::new() })).is_ok();
    if sent {
        ctx.request_repaint();
    } else {
        chat.running = false;
    }
    sent
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

/// What a tool row calls the tool: the last dotted part of a namespaced name, and the
/// bare tool of an MCP name (`mcp__figcraft-app__create_node` → `create_node`). The panel
/// is narrow; the server prefix is what the agent CLI needed, not what the row does.
pub(crate) fn short_tool_name(name: &str) -> String {
    let tail = name.rsplit('.').next().unwrap_or(name);
    match tail.strip_prefix("mcp__") {
        Some(rest) => rest.rsplit("__").next().unwrap_or(rest).to_string(),
        None => tail.to_string(),
    }
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
