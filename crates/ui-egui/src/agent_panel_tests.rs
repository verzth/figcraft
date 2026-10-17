//! What each agent event does to the transcript: the pure heart of the Agent tab (the panel
//! itself is drawn and screenshot by hand; these are the rules it renders).

use figcraft_agent::AgentEvent;

use crate::agent_panel::{Chat, Row, apply, fmt_tokens, one_line};

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
