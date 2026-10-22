//! What each agent event does to the transcript — the pure heart of the Agent tab — and the tab
//! itself drawn headless through egui: the pen.dev layout, its history dropdown, and Enter-to-send.

use figcraft_agent::AgentEvent;
use figcraft_engine::{AgentParallel, AgentRefine};

use crate::agent_panel::{Chat, Cmd, Link, Pending, Row, apply, fmt_tokens, hist_stamp, one_line, refine_once, short_tool_name, take_file};
use crate::agents::BackendKind;
use crate::i18n::tr;

fn text_chat() -> (Chat, Vec<AgentEvent>) {
    let events = vec![AgentEvent::Session("sess-1".into()), AgentEvent::Text("Saya ".into()), AgentEvent::Text("membuat frame.".into())];
    (Chat::default(), events)
}

#[test]
fn text_deltas_group_into_one_assistant_row() {
    let (mut chat, events) = text_chat();
    for event in events {
        apply(&mut chat, event);
    }
    assert_eq!(chat.id.as_deref(), Some("sess-1"));
    assert_eq!(chat.rows, vec![Row::Assistant(vec!["Saya ".into(), "membuat frame.".into()])]);
}

#[test]
fn a_tool_row_takes_its_result_and_the_next_call_gets_its_own_row() {
    let (mut chat, events) = text_chat();
    for event in events {
        apply(&mut chat, event);
    }
    chat.rows.push(Row::User("satu".into()));
    apply(&mut chat, AgentEvent::Text("oke".into()));
    apply(&mut chat, AgentEvent::ToolStart { name: "figcraft.build_layout".into(), args: "{\"kind\":\"stack\"}".into() });
    apply(&mut chat, AgentEvent::Text("selesai.".into()));
    apply(&mut chat, AgentEvent::ToolEnd { name: "figcraft.build_layout".into(), ok: true, summary: "3 nodes".into() });
    apply(&mut chat, AgentEvent::ToolStart { name: "figcraft.check_layout".into(), args: "{}".into() });
    assert_eq!(
        chat.rows,
        vec![
            Row::Assistant(vec!["Saya ".into(), "membuat frame.".into()]),
            Row::User("satu".into()),
            Row::Assistant(vec!["oke".into()]),
            Row::Tool {
                name: "figcraft.build_layout".into(),
                // Arguments land on one line.
                args: "{\"kind\":\"stack\"}".into(),
                result: Some((true, "3 nodes".into())),
            },
            Row::Assistant(vec!["selesai.".into()]),
            // The result attached to the call that waited for it, newest first; this one waits.
            Row::Tool { name: "figcraft.check_layout".into(), args: String::new(), result: None },
        ]
    );
}

#[test]
fn a_result_with_no_visible_call_still_gets_a_row() {
    let mut chat = Chat::default();
    apply(&mut chat, AgentEvent::ToolEnd { name: "x".into(), ok: false, summary: "no frame".into() });
    assert_eq!(chat.rows, vec![Row::Tool { name: "tool".into(), args: String::new(), result: Some((false, "no frame".into())) }]);
}

#[test]
fn done_clears_running_and_only_a_surprising_reason_says_so() {
    let mut chat = Chat { running: true, ..Chat::default() };
    apply(&mut chat, AgentEvent::Tokens { input: 10, output: 20 });
    apply(&mut chat, AgentEvent::Done { reason: "success".into() });
    assert!(!chat.running, "the turn ended");
    assert_eq!(chat.tokens, Some((10, 20)));
    assert!(chat.rows.is_empty(), "success is not worth a row");
    chat.running = true;
    apply(&mut chat, AgentEvent::Done { reason: "cancelled".into() });
    assert!(chat.rows.is_empty(), "the user asked for the stop");
    chat.running = true;
    apply(&mut chat, AgentEvent::Done { reason: "error_max_turns".into() });
    assert!(!chat.running);
    assert_eq!(chat.rows, vec![Row::Error("error_max_turns".into())]);
}

#[test]
fn a_failed_backend_ends_the_turn_and_says_why() {
    let mut chat = Chat { running: true, ..Chat::default() };
    apply(&mut chat, AgentEvent::Failed("Claude Code was not found on this computer.".into()));
    assert!(!chat.running);
    assert_eq!(chat.rows, vec![Row::Error("Claude Code was not found on this computer.".into())]);
}

#[test]
fn arguments_and_counts_stay_compact() {
    assert_eq!(one_line("{}"), "");
    // Arguments stream in as JSON fragments, possibly spread over lines: one line, one space.
    assert_eq!(one_line("{\n  \"kind\": \"stack\",\n  \"gap\": 8\n}"), "{ \"kind\": \"stack\", \"gap\": 8 }");
    assert_eq!(fmt_tokens(0), "0");
    assert_eq!(fmt_tokens(999), "999");
    assert_eq!(fmt_tokens(1234), "1.2k");
    assert_eq!(fmt_tokens(12_345), "12k");
}

#[test]
fn a_tool_row_calls_the_tool_by_its_bare_name() {
    // The agent CLIs name a tool four ways (Codex and OpenCode prefix the server); the row
    // shows the one the user acts on.
    assert_eq!(short_tool_name("mcp__figcraft-app__create_node"), "create_node");
    assert_eq!(short_tool_name("figcraft-app.create_node"), "create_node");
    assert_eq!(short_tool_name("figcraft.build_layout"), "build_layout");
    assert_eq!(short_tool_name("create_node"), "create_node");
}

/// The history file's contract: a whole conversation survives the round trip — session id,
/// model and all — and the live worker thread never rides along.
#[test]
fn a_chat_survives_the_history_json_whole_and_without_its_live_link() {
    let (mut chat, events) = text_chat();
    for event in events {
        apply(&mut chat, event);
    }
    chat.tokens = Some((1200, 34));
    chat.kind = Some(BackendKind::ClaudeCode);
    chat.created = 1_760_000_000.0;
    chat.variant = 3;
    chat.rounds_left = 2;
    let json = serde_json::to_string(&chat).expect("a chat serializes");
    assert!(!json.contains("link"), "the live worker never rides along: {json}");
    let back: Chat = serde_json::from_str(&json).expect("the same chat reads back");
    assert_eq!(back.rows, chat.rows);
    assert_eq!(back.id, chat.id);
    assert_eq!(back.tokens, chat.tokens);
    assert_eq!(back.kind, chat.kind);
    assert_eq!(back.created, chat.created);
    assert_eq!(back.variant, chat.variant);
    assert_eq!(back.rounds_left, chat.rounds_left);
}

/// A chat is named after its first real words, elided when the thought runs long.
#[test]
fn a_chat_is_named_after_its_first_words_and_elides_long_thoughts() {
    let mut chat = Chat::default();
    assert_eq!(chat.title(), tr("New Agent"));
    chat.rows.push(Row::User("   ".into()));
    chat.rows.push(Row::User("Real ask".into()));
    assert_eq!(chat.title(), "Real ask");
    chat.rows.insert(0, Row::User("m".repeat(60)));
    assert_eq!(chat.title(), format!("{}…", "m".repeat(48)));
    // A parallel variant says so in its name, so chats from one send can be told apart.
    chat.rows.truncate(1);
    chat.variant = 2;
    assert_eq!(chat.title(), format!("{}… · 2", "m".repeat(48)));
}

/// A history entry's stamp: the clock time for today, "Yesterday", then a day count — so several
/// chats that all fall back to "New Agent" still tell apart in the list.
#[test]
fn a_history_stamp_reads_as_a_clock_time_today_and_a_day_count_after() {
    let day = 86_400.0;
    let now = 12.5 * day;
    // Today: the same UTC day, shown as the clock time it was opened.
    assert_eq!(hist_stamp(12.0 * day + 3661.0, now), "01:01");
    // One day back reads "Yesterday".
    assert_eq!(hist_stamp(now - day - 500.0, now), "Yesterday");
    // Older still: the day count.
    assert_eq!(hist_stamp(now - 2.0 * day, now), "2d");
    assert_eq!(hist_stamp(now - 10.0 * day, now), "10d");
}

// --- The tab drawn headless: the pen.dev layout, its dropdown, and the send rules. ---

use egui::{Event, Key, Modifiers, PointerButton, Pos2, Rect, Shape};
use figcraft_engine::Session;

use crate::layers_drag_tests::Ui;
use crate::live_tests::{flat, has_text, text_center};

/// The Agent tab on a test screen, and one frame of what it draws.
fn tab() -> Ui {
    let mut u = Ui::new(Session::new());
    u.app.ui.left_tab = crate::LeftTab::Agent;
    u.frame(vec![]);
    u
}

/// What the Agent panel painted, and the span it paints inside: the icon rail is 56 wide
/// (lib.rs) and the panel is asked for `left_width`. The shapes are taken by clip rect rather
/// than by folding over every rect in the frame — the rail, the canvas and the right panel all
/// paint their own shapes past these edges, so a check over the whole frame's shapes holds no
/// matter what the panel does. The returned rect is the widest of those clip rects, i.e. the
/// panel's own area (top to bottom included), against which its rows keep their padding.
fn panel_shapes(u: &mut Ui) -> (Vec<Shape>, Rect) {
    u.frame(vec![]);
    let (left, right) = (56.0, 56.0 + u.app.ui.left_width);
    let mut painted = Vec::new();
    let mut panel = Rect::NOTHING;
    for clipped in &u.frame_output(vec![]).shapes {
        let clip = clipped.clip_rect;
        if clip.min.x >= left - 0.5 && clip.max.x <= right + 0.5 {
            painted.extend(flat(std::slice::from_ref(clipped)));
            // The rows inside the scroll area (and any popup) clip tighter than the panel.
            if clip.area() > panel.area() {
                panel = clip;
            }
        }
    }
    (painted, panel)
}

/// Nothing in the panel may start left of the content edge: in the header's right-to-left row a
/// widget whose floor exceeded the space left allocated past the panel's left edge, and egui then
/// painted the whole panel 34px narrower than it was asked for (the canvas showed a gap beside it).
#[test]
fn no_text_starts_left_of_the_panel() {
    let mut u = tab();
    let (painted, span) = panel_shapes(&mut u);
    assert!(has_text(&painted, "New chat"), "the panel's own shapes are found by their clip rect");
    for (text, at, _) in crate::live_tests::texts(&painted) {
        assert!(at.x >= span.min.x - 0.5, "{text:?} starts at x={} inside the panel's left edge x={}", at.x, span.min.x);
    }
}

/// Nothing may run past the panel's right edge either: the names at the right end of the header
/// and of the field's controls row elide to the room they have rather than shoving their group
/// off the panel — the draft and the controls were once laid out side by side inside the field,
/// which pushed the controls clean off it.
#[test]
fn no_text_runs_past_the_panel() {
    let mut u = tab();
    let (painted, span) = panel_shapes(&mut u);
    assert!(has_text(&painted, "New chat"), "the panel's own shapes are found by their clip rect");
    for (text, at, size) in crate::live_tests::texts(&painted) {
        assert!(at.x + size.x <= span.max.x + 0.5, "{text:?} ends at x={} past the panel's right edge x={}", at.x + size.x, span.max.x);
    }
}

/// The panel at its narrow default width with a CLI actually installed — the state the shipped
/// app is in. The session name stays whole, the status beside it steps aside rather than
/// squeezing the dropdown, and the field's two ends (attach and the send's counts at the left,
/// the model at the right) sit on their own row under the draft instead of being shoved past
/// the panel's right edge.
#[test]
fn with_a_backend_the_names_stay_inside_the_panel() {
    let mut u = tab();
    u.app.ui.agent.backends = vec![crate::agents::PanelBackend { kind: crate::agents::BackendKind::ClaudeCode, bin: Some("claude".into()) }];
    // Plant the backend rather than scanning this machine: `resolved` makes the scan a no-op.
    u.app.ui.agent.resolved = true;
    let (painted, span) = panel_shapes(&mut u);
    assert!(has_text(&painted, "New Agent"), "the session name is whole beside the status");
    // The counts draw fresh as "×1" twice — parallel agents first, refine rounds beside it.
    // The CLI's name no longer takes the row: the model picker routes it.
    assert_eq!(
        crate::live_tests::texts(&painted).iter().filter(|(t, _, _)| t == "×1").count(),
        2,
        "the parallel-agents and refine-rounds counts draw in the field's controls row"
    );
    for (text, at, size) in crate::live_tests::texts(&painted) {
        assert!(at.x >= span.min.x - 0.5, "{text:?} starts at x={} inside the panel's left edge x={}", at.x, span.min.x);
        assert!(at.x + size.x <= span.max.x + 0.5, "{text:?} ends at x={} past the panel's right edge x={}", at.x + size.x, span.max.x);
    }
}

/// The model picker's top section is whom the turn runs on — the fix for "the models are read
/// but there's no way to send them to Codex or OpenCode": every installed CLI is listed there,
/// and picking one takes a model it runs along with it.
#[test]
fn the_model_pickers_top_section_chooses_the_backend_and_the_model_moves_with_it() {
    let mut u = tab();
    u.app.ui.agent.backends = vec![
        crate::agents::PanelBackend { kind: BackendKind::ClaudeCode, bin: Some("/bin/claude".into()) },
        crate::agents::PanelBackend { kind: BackendKind::Codex, bin: Some("/bin/codex".into()) },
        crate::agents::PanelBackend { kind: BackendKind::OpenCode, bin: Some("/bin/opencode".into()) },
    ];
    // Plant the backends rather than scanning this machine: `resolved` makes the scan a no-op.
    u.app.ui.agent.resolved = true;
    u.frame(vec![]);
    egui::Popup::open_id(&u.ctx, egui::Id::new("agent_model_picker"));
    // The popup's first frame is an invisible sizing pass; it appears on the next.
    u.frame_output(vec![]);
    let s = drawn(&mut u);
    assert!(has_text(&s, "RUN ON"), "the picker's top section names whom a turn runs on");
    for name in ["Claude Code", "Codex", "OpenCode"] {
        assert!(has_text(&s, name), "the installed {name} is listed");
    }
    assert_eq!(u.app.ui.agent.effective_name(), "Claude Code", "the default turn runs on Claude Code");
    // Codex: a GPT model comes with it, and the name follows the route.
    click(&mut u, text_center(&s, "Codex"));
    u.frame(vec![]);
    assert_eq!(u.app.ui.agent.backend, 1, "Codex is the backend now");
    assert_eq!(u.app.ui.agent.models.label(), "GPT 6.1 Sol", "a Codex backend takes a GPT model");
    assert_eq!(u.app.ui.agent.effective_name(), "Codex", "the row beside the picker would name the same");
}

/// The name stays on the CLI that really runs the turn: picking a model from another provider
/// moves the backend with it instead of promising one that can't run the model.
#[test]
fn picking_a_model_moves_the_backend_that_runs_it() {
    let mut u = tab();
    u.app.ui.agent.backends = vec![
        crate::agents::PanelBackend { kind: BackendKind::ClaudeCode, bin: Some("/bin/claude".into()) },
        crate::agents::PanelBackend { kind: BackendKind::Codex, bin: Some("/bin/codex".into()) },
    ];
    u.app.ui.agent.resolved = true;
    u.frame(vec![]);
    egui::Popup::open_id(&u.ctx, egui::Id::new("agent_model_picker"));
    u.app.ui.agent.models.focus_search(); // what opening via the trigger does
    u.frame_output(vec![]);
    // The GPT rows sit under twelve Claude rows, past the list's viewport: search first, as a
    // user does.
    u.frame(vec![egui::Event::Text("gpt".into())]);
    let s = drawn(&mut u);
    click(&mut u, text_center(&s, "GPT 6.1 Sol"));
    u.frame(vec![]);
    assert_eq!(u.app.ui.agent.backend, 1, "the model picker moved the backend to the CLI that runs it");
    assert_eq!(u.app.ui.agent.effective_name(), "Codex", "and the name tells the truth");
}

/// One more frame, and the shapes it drew.
fn drawn(u: &mut Ui) -> Vec<Shape> {
    u.frame(vec![]);
    flat(&u.frame_output(vec![]).shapes)
}

/// The middle of the first drawn text that starts with `prefix` (long rows elide in the panel).
fn elided_center(s: &[Shape], prefix: &str) -> Pos2 {
    let (_, p, size) =
        crate::live_tests::texts(s).into_iter().find(|(t, _, _)| t.starts_with(prefix)).unwrap_or_else(|| panic!("no text starting {prefix:?}"));
    p + size / 2.0
}

fn has_prefix(s: &[Shape], prefix: &str) -> bool {
    crate::live_tests::texts(s).iter().any(|(t, _, _)| t.starts_with(prefix))
}

/// The history dropdown lays out at a pinned width: `Popup::width` only seeds a default and the
/// area then shrinks to its content's min size, which could clip the long "Clear Agent History"
/// row. `history_menu` pins the content width instead, so the frame stays wide enough for every
/// row (a regression: the popup once collapsed to its content and clipped the clear row).
#[test]
fn the_history_popup_stays_wide_enough_to_fit_its_clear_row() {
    let mut u = tab();
    egui::Popup::open_id(&u.ctx, egui::Id::new("agent_sessions"));
    // The popup's first frame is an invisible sizing pass; it appears on the next.
    u.frame_output(vec![]);
    let out = u.frame_output(vec![]);

    // The popup's own frame background: the widest rect that starts just right of the rail and
    // drops below the header row (chat rows are inset a further 8 and are narrower than it).
    let mut frame_rect: Option<Rect> = None;
    let mut clear: Option<(f32, f32)> = None;
    let mut title_seen = false;
    for clipped in &out.shapes {
        let shapes = crate::live_tests::flat(std::slice::from_ref(clipped));
        for s in &shapes {
            if let Shape::Rect(rs) = s {
                let r = rs.rect;
                if r.min.x > 60.0 && r.min.x < 120.0 && r.min.y > 80.0 && r.width() > 200.0 {
                    frame_rect = Some(frame_rect.map_or(r, |f: Rect| f.union(r)));
                }
            }
        }
        for (t, p, size) in crate::live_tests::texts(&shapes) {
            if t == tr("Clear Agent History") {
                clear = Some((p.x, p.x + size.x));
            }
            if t == tr("New Agent") {
                title_seen = true;
            }
        }
    }
    assert!(title_seen, "the dropdown lists a chat");
    let frame = frame_rect.expect("the popup drew its frame background");
    assert!(frame.width() >= 250.0, "the popup kept a usable width, got {}", frame.width());
    let (clear_left, clear_right) = clear.expect("the popup drew its clear-history row");
    assert!(clear_right <= frame.max.x, "Clear Agent History fits inside the popup ({clear_right} vs {})", frame.max.x);
    assert!(clear_left >= frame.min.x, "the clear row starts inside the popup");
}

/// The three history groups don't overlap: a chat from today lists under "Today" alone. The
/// predicates once fell through ("Previous 7 days" also matched today, "Older" matched
/// everything), so one conversation showed up beneath all three headings at once.
#[test]
fn a_todays_chat_lists_under_today_alone_not_under_every_group() {
    let mut u = tab();
    egui::Popup::open_id(&u.ctx, egui::Id::new("agent_sessions"));
    // The popup's first frame is an invisible sizing pass; it appears on the next.
    u.frame_output(vec![]);
    let out = u.frame_output(vec![]);
    let mut painted: Vec<String> = Vec::new();
    for clipped in &out.shapes {
        let shapes = crate::live_tests::flat(std::slice::from_ref(clipped));
        painted.extend(crate::live_tests::texts(&shapes).into_iter().map(|(t, _, _)| t));
    }
    assert!(painted.iter().any(|t| t == tr("Today")), "the fresh chat is listed under Today");
    assert!(!painted.iter().any(|t| t == tr("Previous 7 days")), "a today chat isn't also under the 7-day group");
    assert!(!painted.iter().any(|t| t == tr("Older")), "a today chat isn't also under Older");
}

/// Press and release at `p` (a frame each).
fn click(u: &mut Ui, p: Pos2) {
    for pressed in [true, false] {
        u.frame(vec![Event::PointerMoved(p), Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE }]);
    }
}

/// The pen.dev shape: the session dropdown and New chat in the header, the left-aligned empty
/// state with its chips, and the context-files note above the input.
#[test]
fn the_tab_draws_its_empty_state_and_a_chip_fills_the_draft() {
    let mut u = tab();
    let s = drawn(&mut u);
    assert!(has_text(&s, "New Agent"), "the session dropdown names the fresh chat");
    assert!(has_text(&s, "New chat"));
    assert!(has_text(&s, "Ask me to design anything"));
    for chip in ["Design a landing", "Design a dashboard", "Design a booking", "Design an editorial"] {
        assert!(has_prefix(&s, chip), "the chip row {chip:?} draws (elided to the panel)");
    }
    assert!(has_text(&s, "Attach context files."));
    // A chip is a shortcut: it fills the draft with its whole words.
    click(&mut u, elided_center(&s, "Design a landing"));
    assert_eq!(u.app.ui.agent.draft, "Design a landing page for a coffee roastery");
}

/// The session dropdown is the chat history: Today's chats, and Clear Agent History below —
/// which asks once more before it wipes anything.
#[test]
fn the_dropdown_lists_the_history_and_clearing_it_asks_twice() {
    let mut u = tab();
    let s = drawn(&mut u);
    click(&mut u, text_center(&s, "New Agent"));
    let s = drawn(&mut u);
    assert!(has_text(&s, "Today"), "the chat is grouped under Today");
    assert!(has_text(&s, "Clear Agent History"));
    click(&mut u, text_center(&s, "Clear Agent History"));
    let s = drawn(&mut u);
    assert!(has_text(&s, "Click again to confirm."), "clearing the history asks twice");
    assert_eq!(u.app.ui.agent.chats.len(), 1, "one click clears nothing yet");
    click(&mut u, text_center(&s, "Click again to confirm."));
    u.frame(vec![]);
    assert!(u.app.ui.agent.chats.iter().all(|c| c.rows.is_empty()), "the second click wipes the transcripts");
}

/// Enter in the draft sends the turn; Shift+Enter only breaks the line. With no server to run
/// against the send is refused, says why, and gives the words back to the draft.
#[test]
fn enter_sends_shift_enter_breaks_a_line_and_a_refusal_keeps_the_words() {
    let mut u = tab();
    let s = drawn(&mut u);
    click(&mut u, text_center(&s, "Design anything…"));
    u.frame(vec![Event::Text("A landing page".into())]);
    assert_eq!(u.app.ui.agent.draft, "A landing page");
    for pressed in [true, false] {
        u.frame_with(vec![Event::Key { key: Key::Enter, physical_key: None, pressed, repeat: false, modifiers: Modifiers::SHIFT }], Modifiers::SHIFT);
    }
    assert!(u.app.ui.agent.chats.iter().all(|c| c.rows.is_empty()), "Shift+Enter doesn't send");
    for pressed in [true, false] {
        u.frame_with(vec![Event::Key { key: Key::Enter, physical_key: None, pressed, repeat: false, modifiers: Modifiers::NONE }], Modifiers::NONE);
    }
    let s = drawn(&mut u);
    assert!(has_text(&s, "The AI agents server is not running. Turn it on in Preferences ▸ AI agents."));
    assert_eq!(u.app.ui.agent.draft, "A landing page", "a refused turn gives the words back");
}

/// A dropped image waits as a chip like a text file, and a refused send keeps it (and the
/// words) for the retry: nothing about the drop is spent until a backend really runs it.
#[test]
fn an_image_drop_draws_a_chip_and_survives_a_refusal() {
    let mut u = tab();
    let s = drawn(&mut u);
    click(&mut u, text_center(&s, "Design anything…"));
    u.frame(vec![Event::Text("Lihat skema".into())]);
    u.app.ui.agent.attachments.push(Pending::Image { name: "skema.png".into(), mime: "image/png", data: b"\x89PNG\r\n\x1a\nfigcraft".to_vec() });
    let s = drawn(&mut u);
    assert!(has_prefix(&s, "× skema.png"), "the image waits as a chip above the field");
    // Enter sends — the server is not running, so the send is refused and says so.
    for pressed in [true, false] {
        u.frame_with(vec![Event::Key { key: Key::Enter, physical_key: None, pressed, repeat: false, modifiers: Modifiers::NONE }], Modifiers::NONE);
    }
    let s = drawn(&mut u);
    assert!(has_text(&s, "The AI agents server is not running. Turn it on in Preferences ▸ AI agents."));
    assert_eq!(u.app.ui.agent.draft, "Lihat skema", "a refused turn gives the words back");
    assert_eq!(u.app.ui.agent.attachments.len(), 1, "a refused turn keeps the drop");
}

// --- The send's shape: attach, parallel agents, refine rounds. ---

/// The `set_max_width` trap: it resets the row's cursor, so padding added after it was
/// silently thrown away — the note and the field sat flush against the panel's edge while
/// the rows around them kept theirs.
#[test]
fn the_empty_states_content_sits_inside_the_padding() {
    let mut u = tab();
    let (painted, span) = panel_shapes(&mut u);
    for wanted in ["Ask me to design anything", "Attach context files.", "Drop a DESIGN.md", "Design anything…"] {
        let (_, at, _) = crate::live_tests::texts(&painted)
            .into_iter()
            .find(|(t, _, _)| t.starts_with(wanted))
            .unwrap_or_else(|| panic!("no text starting {wanted:?}"));
        assert!(
            at.x >= span.min.x + crate::widgets::PAD - 0.5,
            "{wanted:?} starts at x={} outside the panel's {}px content padding (edge x={})",
            at.x,
            crate::widgets::PAD,
            span.min.x
        );
    }
}

/// The panel frame carries no margin of its own (`lib.rs` draws it `Frame::NONE`), so the tab's
/// own rows keep the vertical padding: the header must not touch the tab bar above it and the
/// composer must not touch the window's bottom edge. Painted content therefore stays `PAD` clear
/// of both panel edges — checked over every shape the panel paints, not just the texts, so the
/// field's own frame and the send button are held to the same rule.
#[test]
fn no_painted_content_touches_the_panels_top_or_bottom_edge() {
    let mut u = tab();
    let (painted, span) = panel_shapes(&mut u);
    assert!(has_text(&painted, "New chat"), "the panel's own shapes are found by their clip rect");
    // The panel's own background fill spans its whole area; skip it and measure what's on it.
    let content: Vec<Rect> = painted
        .iter()
        .filter_map(|s| match s {
            Shape::Rect(r) if r.rect.height() < span.height() - 1.0 => Some(r.rect),
            _ => None,
        })
        .collect();
    let pad = crate::widgets::PAD;
    let top = content.iter().map(|r| r.min.y).fold(f32::INFINITY, f32::min);
    let bottom = content.iter().map(|r| r.max.y).fold(f32::NEG_INFINITY, f32::max);
    assert!(
        top >= span.min.y + pad - 0.5,
        "painted content starts {top}px from the window top, inside the panel's {pad}px vertical padding (edge y={})",
        span.min.y
    );
    assert!(
        bottom <= span.max.y - pad + 0.5,
        "painted content ends at {bottom}px, inside the panel's {pad}px vertical padding (bottom edge y={})",
        span.max.y
    );
}

/// The middle of the `n`th count drawn in the controls row (0: the agents pill, left of the
/// rounds pill). Counts are "×2"-shaped; a chip's "× name" never counts.
fn count_center(s: &[Shape], n: usize) -> Pos2 {
    let (_, p, size) = crate::live_tests::texts(s)
        .into_iter()
        .filter(|(t, _, _)| t.strip_prefix('×').is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit())))
        .nth(n)
        .unwrap_or_else(|| panic!("no count pill {n} drawn"));
    p + size / 2.0
}

/// Every shape a frame painted, across all its clip rects.
fn shapes_of(out: &egui::FullOutput) -> Vec<Shape> {
    out.shapes.iter().flat_map(|c| flat(std::slice::from_ref(c))).collect()
}

/// The parallel-agents pill opens a popover instead of stepping its count: the popover names
/// what it sets up, offers how the agents share the work (split one design, or each build their
/// own), and the number that runs at once — each choice landing in the saved prefs.
#[test]
fn the_parallel_popover_picks_the_mode_and_the_count() {
    let mut u = tab();
    assert_eq!(u.app.session.prefs.agent_runs, (1, 1), "one agent, one round by default");
    assert_eq!(u.app.session.prefs.agent_parallel, AgentParallel::SideBySide, "side by side by default");
    // The pill opens its popover.
    let s = drawn(&mut u);
    click(&mut u, count_center(&s, 0));
    assert!(egui::Popup::is_id_open(&u.ctx, egui::Id::new("agent_parallel")), "the parallel pill opens its popover");
    // The popover draws its header, its two mode cards and its count row.
    u.frame_output(vec![]);
    let painted = shapes_of(&u.frame_output(vec![]));
    for want in [tr("Parallel agents"), tr("Split Work"), tr("Side by side"), tr("Agents")] {
        assert!(has_text(&painted, want), "the popover draws {want:?}");
    }
    // Pick "Split Work": the agents now split one design between them.
    click(&mut u, text_center(&painted, tr("Split Work")));
    assert_eq!(u.app.session.prefs.agent_parallel, AgentParallel::SplitWork, "the split-work card is picked");
    // Pick three agents; the refine rounds beside them are untouched.
    click(&mut u, text_center(&painted, "3"));
    assert_eq!(u.app.session.prefs.agent_runs.0, 3, "the count chip sets how many agents run");
    assert_eq!(u.app.session.prefs.agent_runs.1, 1, "the refine rounds are untouched");
}

/// The iteration popover does the same for the refine rounds: what they fix (the layout, or the
/// style) and how many run, "Don't iterate" standing for the single round that isn't one.
#[test]
fn the_iteration_popover_picks_what_the_rounds_fix_and_how_many() {
    let mut u = tab();
    assert_eq!(u.app.session.prefs.agent_refine, AgentRefine::Layout, "the layout by default");
    let s = drawn(&mut u);
    click(&mut u, count_center(&s, 1));
    assert!(egui::Popup::is_id_open(&u.ctx, egui::Id::new("agent_iterate")), "the iteration pill opens its popover");
    u.frame_output(vec![]);
    let painted = shapes_of(&u.frame_output(vec![]));
    for want in [tr("Iteration"), tr("Layout"), tr("Style"), tr("Variants"), tr("Don't iterate")] {
        assert!(has_text(&painted, want), "the popover draws {want:?}");
    }
    // Focus the rounds on the style.
    click(&mut u, text_center(&painted, tr("Style")));
    assert_eq!(u.app.session.prefs.agent_refine, AgentRefine::Style, "the style card is picked");
    // Three rounds.
    click(&mut u, text_center(&painted, "3"));
    assert_eq!(u.app.session.prefs.agent_runs.1, 3, "the count chip sets how many rounds run");
    assert_eq!(u.app.session.prefs.agent_runs.0, 1, "the agents count is untouched");
}

/// Refine rounds: when a turn ends well the chat asks its own worker for the next round — one
/// round at a time — and a chat without a live worker (one come back from the history) ends
/// its rounds instead of asking nobody.
#[test]
fn a_chat_with_rounds_left_asks_itself_for_the_next_round() {
    let mut u = tab();
    u.app.ui.agent.chats.push(Chat::default());
    let idx = u.app.ui.agent.chats.len() - 1;
    let (tx_cmd, rx_cmd) = std::sync::mpsc::channel();
    let (tx_ev, rx_ev) = std::sync::mpsc::channel();
    {
        let chat = &mut u.app.ui.agent.chats[idx];
        chat.link = Some(Link { cmd: tx_cmd, events: rx_ev });
        chat.rounds_left = 1;
        chat.rows.push(Row::User("Design a hero".into()));
    }
    let ctx = egui::Context::default();
    assert!(refine_once(&mut u.app, &ctx, idx), "the promised round was asked for");
    let chat = &u.app.ui.agent.chats[idx];
    assert_eq!(chat.rounds_left, 0, "one round at a time");
    assert!(chat.running, "the worker is on the round");
    assert!(matches!(chat.rows.last(), Some(Row::User(_))), "the round shows on the transcript");
    let cmd = rx_cmd.try_recv().expect("the refine turn reached the worker");
    let Cmd::Turn(turn) = cmd else { panic!("a turn, not a stop") };
    assert!(turn.text.contains("refine"), "the prompt asks for a refinement: {}", turn.text);
    drop(tx_ev);
    // No rounds left: nothing more is asked, of a busy worker least of all.
    assert!(!refine_once(&mut u.app, &ctx, idx));
    // A chat with no live worker ends its rounds instead of asking nobody.
    u.app.ui.agent.chats[idx].link = None;
    u.app.ui.agent.chats[idx].rounds_left = 2;
    u.app.ui.agent.chats[idx].running = false;
    assert!(!refine_once(&mut u.app, &ctx, idx));
    assert_eq!(u.app.ui.agent.chats[idx].rounds_left, 0, "the rounds end with the worker");
}

/// The attach button's pick lands exactly as a drop does — the text joins the next message —
/// and never queues past the drops' cap.
#[test]
fn a_picked_file_becomes_pending_like_a_drop() {
    let path = std::env::temp_dir().join("figcraft-attach-pick.md");
    std::fs::write(&path, "# Notes\nthe body").expect("a scratch file");
    let mut u = tab();
    take_file(&mut u.app, &path);
    let [Pending::Text { name, content }] = u.app.ui.agent.attachments.as_slice() else {
        panic!("one text file waits for the next send");
    };
    assert_eq!(name, "figcraft-attach-pick.md");
    assert!(content.contains("the body"), "the text is what a drop would carry");
    while u.app.ui.agent.attachments.len() < crate::agent_panel::MAX_DROPS {
        u.app.ui.agent.attachments.push(Pending::Text { name: "extra.txt".into(), content: String::new() });
    }
    take_file(&mut u.app, &path);
    assert_eq!(u.app.ui.agent.attachments.len(), crate::agent_panel::MAX_DROPS, "the drop cap holds");
    std::fs::remove_file(&path).ok();
}
