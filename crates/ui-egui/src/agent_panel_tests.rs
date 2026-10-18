//! What each agent event does to the transcript — the pure heart of the Agent tab — and the tab
//! itself drawn headless through egui: the pen.dev layout, its history dropdown, and Enter-to-send.

use figcraft_agent::AgentEvent;

use crate::agent_panel::{Chat, Row, apply, fmt_tokens, one_line, short_tool_name};
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
    let mut chat = Chat::default();
    chat.running = true;
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
    let mut chat = Chat::default();
    chat.running = true;
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
    let json = serde_json::to_string(&chat).expect("a chat serializes");
    assert!(!json.contains("link"), "the live worker never rides along: {json}");
    let back: Chat = serde_json::from_str(&json).expect("the same chat reads back");
    assert_eq!(back.rows, chat.rows);
    assert_eq!(back.id, chat.id);
    assert_eq!(back.tokens, chat.tokens);
    assert_eq!(back.kind, chat.kind);
    assert_eq!(back.created, chat.created);
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
}

// --- The tab drawn headless: the pen.dev layout, its dropdown, and the send rules. ---

use egui::{Event, Key, Modifiers, PointerButton, Pos2, Rect, Shape, pos2};
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
/// matter what the panel does.
fn panel_shapes(u: &mut Ui) -> (Vec<Shape>, Rect) {
    u.frame(vec![]);
    let (left, right) = (56.0, 56.0 + u.app.ui.left_width);
    let mut painted = Vec::new();
    for clipped in &u.frame_output(vec![]).shapes {
        let clip = clipped.clip_rect;
        if clip.min.x >= left - 0.5 && clip.max.x <= right + 0.5 {
            painted.extend(flat(std::slice::from_ref(clipped)));
        }
    }
    (painted, Rect::from_min_max(pos2(left, 0.0), pos2(right, 0.0)))
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
/// squeezing the dropdown, and the field's two ends (which CLI, which model) sit on their own
/// row under the draft instead of being shoved past the panel's right edge.
#[test]
fn with_a_backend_the_names_stay_inside_the_panel() {
    let mut u = tab();
    u.app.ui.agent.backends = vec![crate::agents::PanelBackend { kind: crate::agents::BackendKind::ClaudeCode, bin: Some("claude".into()) }];
    // Plant the backend rather than scanning this machine: `resolved` makes the scan a no-op.
    u.app.ui.agent.resolved = true;
    let (painted, span) = panel_shapes(&mut u);
    assert!(has_text(&painted, "New Agent"), "the session name is whole beside the status");
    assert!(has_text(&painted, "Claude Code"), "the CLI's name draws in the field's controls row");
    for (text, at, size) in crate::live_tests::texts(&painted) {
        assert!(at.x >= span.min.x - 0.5, "{text:?} starts at x={} inside the panel's left edge x={}", at.x, span.min.x);
        assert!(at.x + size.x <= span.max.x + 0.5, "{text:?} ends at x={} past the panel's right edge x={}", at.x + size.x, span.max.x);
    }
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
