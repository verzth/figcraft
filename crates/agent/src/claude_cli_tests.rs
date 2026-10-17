//! The stream-json parser's mapping and the spawn wiring, against a fake `claude` that
//! speaks the same JSON on stdout (the real CLI is exercised by hand from the panel).

use crate::claude_cli::{ClaudeCodeCli, Parser};
use crate::{AgentEvent, Session, Turn};

fn events(parser: &mut Parser, lines: &[&str]) -> Vec<AgentEvent> {
    lines.iter().flat_map(|l| parser.line(l)).collect()
}

#[test]
fn init_text_tool_result_and_result_map_in_order() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            r#"{"type":"system","subtype":"init","session_id":"sess-1","model":"claude-sonnet-4-5"}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Saya "}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"membuat frame."}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"figcraft.build_layout","input":{}}}}"#,
            r#"{"type":"stream_event","event":{"type":"input_json_delta","tool_use_id":"toolu_1","partial_json":"{\"frame\":"}}"#,
            r#"{"type":"stream_event","event":{"type":"input_json_delta","tool_use_id":"toolu_1","partial_json":"100}"}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop","index":1,"tool_use_id":"toolu_1"}}"#,
            r#"{"type":"assistant","message":{"id":"msg_1","content":[{"type":"text","text":"Saya membuat frame."},{"type":"tool_use","id":"toolu_1","name":"figcraft.build_layout","input":{"frame":100}}]}}"#,
            r#"{"type":"user","message":{"id":"msg_2","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"built frame 100\n2 children","is_error":false}]}}"#,
            r#"{"type":"result","subtype":"success","is_error":false,"duration_ms":1234,"usage":{"input_tokens":10,"output_tokens":20},"session_id":"sess-1"}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            AgentEvent::Session("sess-1".into()),
            AgentEvent::Text("Saya ".into()),
            AgentEvent::Text("membuat frame.".into()),
            // The streamed tool_use block flushes its assembled arguments at the stop…
            AgentEvent::ToolStart { name: "figcraft.build_layout".into(), args: "{\"frame\":100}".into() },
            // …and only the streamed events produce rows: the complete assistant message
            // below repeats the same text and tool_use and must not double them.
            AgentEvent::ToolEnd { name: "figcraft.build_layout".into(), ok: true, summary: "built frame 100".into() },
            AgentEvent::Tokens { input: 10, output: 20 },
            AgentEvent::Done { reason: "success".into() },
        ]
    );
    // The flushed tool_use block reports the assembled arguments, not the empty start.
    let flush: Vec<AgentEvent> = events(
        &mut Parser::default(),
        &[
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t","name":"n","input":{}}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop","index":1,"tool_use_id":"t"}}"#,
        ],
    );
    assert_eq!(flush, vec![AgentEvent::ToolStart { name: "n".into(), args: "{}".into() }]);
}

#[test]
fn a_tool_that_failed_reports_its_error_and_keeps_later_results_working() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"figcraft.check_layout","input":{}}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop","index":0,"tool_use_id":"a"}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"a","content":[{"type":"text","text":"no frame selected"}],"is_error":true}]}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"b","name":"figcraft.inspect_document","input":{}}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop","index":2,"tool_use_id":"b"}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"b","content":"ok"}]}}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            AgentEvent::ToolStart { name: "figcraft.check_layout".into(), args: "{}".into() },
            AgentEvent::ToolEnd { name: "figcraft.check_layout".into(), ok: false, summary: "no frame selected".into() },
            AgentEvent::ToolStart { name: "figcraft.inspect_document".into(), args: "{}".into() },
            // A result without `is_error` is a success; its row still names the tool.
            AgentEvent::ToolEnd { name: "figcraft.inspect_document".into(), ok: true, summary: "ok".into() },
        ]
    );
}

#[test]
fn errors_and_junk_lines_are_tolerated() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            "not json at all",
            r#"{"type":"stream_event","event":{"type":"message_start","message":{}}}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"error","error":{"type":"api_error","message":"overloaded"}}"#,
            r#"{"type":"result","subtype":"error_max_turns","is_error":true}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            AgentEvent::Failed("overloaded".into()),
            // No usage in this result: only the stop reason, which names the subtype.
            AgentEvent::Done { reason: "error_max_turns".into() },
        ]
    );
}

#[test]
fn long_summaries_are_cut_to_one_line() {
    let mut p = Parser::default();
    let long = "x".repeat(400);
    let line =
        format!(r#"{{"type":"user","message":{{"content":[{{"type":"tool_result","tool_use_id":"gone","content":"{long}\nsecond line"}}]}}}}"#);
    let got = p.line(&line);
    assert_eq!(got.len(), 1);
    let AgentEvent::ToolEnd { summary, .. } = &got[0] else { panic!("expected ToolEnd") };
    assert_eq!(summary.chars().count(), 158, "157 cut characters plus the ellipsis");
    assert!(!summary.contains('\n'));
}

#[cfg(unix)]
mod spawn {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    /// A fake `claude`: it reports its `--mcp-config` content and whether `--resume` was
    /// passed, so one script covers the config file, the wiring and session continuation.
    fn fake_claude(dir: &std::path::Path, sleep: Option<Duration>) -> PathBuf {
        let path = dir.join("fake-claude.sh");
        let sleep = sleep.map_or(String::new(), |d| format!("sleep {};\n", d.as_secs_f64()));
        let script = format!(
            "#!/bin/sh\n\
             {sleep}cfg=\"\"\n\
             prev=\"\"\n\
             resumed=no\n\
             for a in \"$@\"; do\n\
             \x20 if [ \"$prev\" = \"--mcp-config\" ]; then cfg=\"$a\"; fi\n\
             \x20 if [ \"$prev\" = \"--resume\" ]; then resumed=yes; fi\n\
             \x20 prev=\"$a\"\n\
             done\n\
             esc=$(tr -d '\\n' < \"$cfg\" | sed 's/\"/\\\\\"/g')\n\
             printf '%s\\n' '{{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"fake-1\"}}'\n\
             printf '%s\\n' \"{{\\\"type\\\":\\\"stream_event\\\",\\\"event\\\":{{\\\"type\\\":\\\"content_block_delta\\\",\\\"index\\\":0,\\\"delta\\\":{{\\\"type\\\":\\\"text_delta\\\",\\\"text\\\":\\\"cfg:$esc resumed:$resumed\\\"}}}}}}\"\n\
             printf '%s\\n' '{{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"usage\":{{\"input_tokens\":1,\"output_tokens\":2}}}}'\n"
        );
        std::fs::write(&path, script).unwrap();
        let mut perm = std::fs::metadata(&path).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&path, perm).unwrap();
        path
    }

    fn turn(text: &str) -> Turn {
        Turn { text: text.into(), attachments: Vec::new() }
    }

    fn collect_until_done(session: &Session) -> Vec<AgentEvent> {
        // A real `claude` turn can take a while; the fakes return at once, so the cap only
        // bounds a hang.
        let deadline = Instant::now() + Duration::from_secs(120);
        let mut got = Vec::new();
        while Instant::now() < deadline {
            got.extend(session.poll());
            if got.iter().any(|e| matches!(e, AgentEvent::Done { .. } | AgentEvent::Failed(_))) {
                return got;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        panic!("the agent never finished: {got:?}");
    }

    #[test]
    fn a_turn_reports_session_tokens_and_done_and_writes_the_mcp_config() {
        let dir = std::env::temp_dir().join(format!("figcraft-agent-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = fake_claude(&dir, None);
        let mut session = Session::new(Box::new(ClaudeCodeCli::new(&bin, "/usr/local/bin/figcraft-cli", 7979)));
        session.send(&turn("halo")).unwrap();
        let got = collect_until_done(&session);
        assert!(got.contains(&AgentEvent::Session("fake-1".into())), "{got:?}");
        assert!(got.contains(&AgentEvent::Tokens { input: 1, output: 2 }), "{got:?}");
        assert!(got.contains(&AgentEvent::Done { reason: "success".into() }), "{got:?}");
        let text: String = got
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(text.contains("figcraft-app"), "{text}");
        assert!(text.contains("--attach") && text.contains("7979"), "{text}");
        assert!(text.contains("resumed:no"), "{text}");
        // The next turn continues the session the init event named.
        session.send(&turn(" lagi")).unwrap();
        let got = collect_until_done(&session);
        let text: String = got
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(text.contains("resumed:yes"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_restored_session_resumes_from_the_very_first_turn() {
        let dir = std::env::temp_dir().join(format!("figcraft-agent-resume-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = fake_claude(&dir, None);
        let mut session = Session::new(Box::new(ClaudeCodeCli::new(&bin, "/usr/local/bin/figcraft-cli", 7979)));
        // The panel opened a chat from its history: the CLI's old session id continues it.
        session.resume("old-session".into());
        session.send(&turn("halo lagi")).unwrap();
        let got = collect_until_done(&session);
        let text: String = got
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(text.contains("resumed:yes"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_turn_refused_while_working_says_so() {
        let dir = std::env::temp_dir().join(format!("figcraft-agent-busy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = fake_claude(&dir, Some(Duration::from_secs(2)));
        let mut session = Session::new(Box::new(ClaudeCodeCli::new(bin, "/usr/local/bin/figcraft-cli", 1)));
        session.send(&turn("satu")).unwrap();
        let err = session.send(&turn("dua")).unwrap_err();
        assert!(err.to_string().contains("still working"), "{err}");
        session.cancel();
        let got = collect_until_done(&session);
        assert!(got.contains(&AgentEvent::Done { reason: "cancelled".into() }), "{got:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The real `claude` against the real flags — run by hand, never in CI:
    /// `cargo test -p figcraft-agent -- --ignored live_claude --nocapture`
    #[ignore = "needs the real claude CLI and an account"]
    #[test]
    fn live_claude_says_hello_without_touching_the_canvas() {
        let mut session = Session::new(Box::new(ClaudeCodeCli::new("claude", "figcraft-cli", 9)));
        session.send(&Turn { text: "Reply with exactly: siap. Do not call any tools.".into(), attachments: Vec::new() }).unwrap();
        let got = collect_until_done(&session);
        println!("{got:#?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Done { .. })), "{got:?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Session(_))), "{got:?}");
    }
}
