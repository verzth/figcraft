//! The Codex JSONL parser's mapping and the spawn wiring, against a fake `codex` that speaks
//! the same JSON on stdout (the real CLI is exercised by hand from the panel).

use crate::codex_cli::{CodexCli, Parser};
use crate::{AgentEvent, Session, Turn};

fn events(parser: &mut Parser, lines: &[&str]) -> Vec<AgentEvent> {
    lines.iter().flat_map(|l| parser.line(l)).collect()
}

/// The lines the real `codex exec --json` printed for a one-word reply, plus the tool and
/// shell items a designing turn goes through.
#[test]
fn thread_message_tools_and_usage_map_in_order() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            r#"{"type":"thread.started","thread_id":"01a14a6c-05e3-7551-abb5-6ba0fccc9cca"}"#,
            r#"{"type":"turn.started"}"#,
            r#"{"type":"item.started","item":{"id":"item_1","type":"mcp_tool_call","server":"figcraft-app","tool":"figcraft.build_layout","arguments":"{\"x\":10}"}}"#,
            // The same item started again (retries repeat): still one row.
            r#"{"type":"item.started","item":{"id":"item_1","type":"mcp_tool_call","server":"figcraft-app","tool":"figcraft.build_layout","arguments":"{\"x\":10}"}}"#,
            r#"{"type":"item.completed","item":{"id":"item_1","type":"mcp_tool_call","server":"figcraft-app","tool":"figcraft.build_layout","status":"completed","arguments":"{\"x\":10}","output":"rectangle at 10,20\n2 children"}}"#,
            r#"{"type":"item.started","item":{"id":"item_2","type":"command_execution","command":"ls"}}"#,
            r#"{"type":"item.completed","item":{"id":"item_2","type":"command_execution","command":"ls","status":"completed","exit_code":0,"aggregated_output":"a.png"}}"#,
            r#"{"type":"item.completed","item":{"id":"item_3","type":"agent_message","text":"siap"}}"#,
            r#"{"type":"turn.completed","usage":{"input_tokens":28386,"cached_input_tokens":13312,"cache_write_input_tokens":0,"output_tokens":6,"reasoning_output_tokens":0}}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            AgentEvent::Session("01a14a6c-05e3-7551-abb5-6ba0fccc9cca".into()),
            AgentEvent::ToolStart { name: "figcraft-app.figcraft.build_layout".into(), args: "{\"x\":10}".into() },
            AgentEvent::ToolEnd { name: "figcraft-app.figcraft.build_layout".into(), ok: true, summary: "rectangle at 10,20".into() },
            AgentEvent::ToolStart { name: "shell".into(), args: "ls".into() },
            AgentEvent::ToolEnd { name: "shell".into(), ok: true, summary: "a.png".into() },
            AgentEvent::Text("siap".into()),
            AgentEvent::Tokens { input: 28386, output: 6 },
            AgentEvent::Done { reason: "success".into() },
        ]
    );
}

#[test]
fn a_failed_turn_says_why_and_a_failed_tool_marks_its_row() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            r#"{"type":"item.completed","item":{"id":"t","type":"mcp_tool_call","server":"figcraft-app","tool":"figcraft.apply_props","status":"failed","output":"no selection"}}"#,
            r#"{"type":"turn.failed","error":{"type":"some_error","message":"quota exceeded"}}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            AgentEvent::ToolEnd { name: "figcraft-app.figcraft.apply_props".into(), ok: false, summary: "no selection".into() },
            AgentEvent::Failed("quota exceeded".into()),
        ]
    );
    // Not JSON, events without meaning for the panel, unknown item types: all ignored.
    assert!(
        events(
            &mut Parser::default(),
            &["not json", r#"{"type":"turn.started"}"#, r#"{"type":"item.completed","item":{"id":"r","type":"reasoning"}}"#, r#"{"type":"wat"}"#]
        )
        .is_empty()
    );
}

/// The real `codex exec --json`, captured against a live app: a call's answer is MCP
/// content under `result` (`output` is null), arguments are a JSON object rather than a
/// string, and an item carries `error`/`status` fields the row must respect. Regression
/// for reading `output` alone, which live Codex never fills for an MCP call — the panel
/// then showed every result row empty.
#[test]
fn a_real_codex_call_reports_its_mcp_result_text() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            r#"{"type":"item.started","item":{"id":"item_1","type":"mcp_tool_call","server":"figcraft-app","tool":"create_node","arguments":{"type":"rectangle","name":"probe","x":40,"y":40,"width":120,"height":80},"result":null,"error":null,"status":"in_progress"}}"#,
            r#"{"type":"item.completed","item":{"id":"item_1","type":"mcp_tool_call","server":"figcraft-app","tool":"create_node","arguments":{"type":"rectangle","name":"probe"},"result":{"content":[{"type":"text","text":"{\n  \"id\": 4,\n  \"node\": {\"name\": \"probe\"}\n}"}],"structured_content":null},"error":null,"status":"completed"}}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            AgentEvent::ToolStart {
                name: "figcraft-app.create_node".into(),
                // Object arguments round-trip through JSON, so the keys come back sorted.
                args: r#"{"height":80,"name":"probe","type":"rectangle","width":120,"x":40,"y":40}"#.into()
            },
            AgentEvent::ToolEnd { name: "figcraft-app.create_node".into(), ok: true, summary: r#"{ "id": 4, "node": {"name": "probe"} }"#.into() },
        ]
    );
}

#[cfg(unix)]
mod spawn {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    /// A fake `codex`: it reports its own arguments, then a complete turn.
    fn fake_codex(dir: &std::path::Path, sleep: Option<Duration>) -> PathBuf {
        let path = dir.join("fake-codex.sh");
        let sleep = sleep.map_or(String::new(), |d| format!("sleep {};\n", d.as_secs_f64()));
        let script = format!(
            "#!/bin/sh\n\
             {sleep}args=$(printf '%s' \"$*\" | sed 's/\"/\\\\\"/g')\n\
             printf '%s\\n' '{{\"type\":\"thread.started\",\"thread_id\":\"fake-1\"}}'\n\
             printf '%s\\n' \"{{\\\"type\\\":\\\"item.completed\\\",\\\"item\\\":{{\\\"id\\\":\\\"m1\\\",\\\"type\\\":\\\"agent_message\\\",\\\"text\\\":\\\"args:$args\\\"}}}}\"\n\
             printf '%s\\n' '{{\"type\":\"turn.completed\",\"usage\":{{\"input_tokens\":3,\"output_tokens\":4}}}}'\n"
        );
        std::fs::write(&path, script).unwrap();
        let mut perm = std::fs::metadata(&path).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&path, perm).unwrap();
        path
    }

    /// A `codex` that dies before saying anything, complaining on stderr.
    fn broken_codex(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("broken-codex.sh");
        let script = "#!/bin/sh\nprintf '%s\\n' 'permission denied' >&2\nexit 1\n";
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

    fn text_of(got: &[AgentEvent]) -> String {
        got.iter()
            .filter_map(|e| match e {
                AgentEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_turn_carries_the_mcp_overrides_the_usage_and_resumes_the_thread() {
        let dir = std::env::temp_dir().join(format!("figcraft-agent-cx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = fake_codex(&dir, None);
        let mut session = Session::new(Box::new(CodexCli::new(&bin, "/usr/local/bin/figcraft-cli", 7979)));
        session.send(&turn("halo")).unwrap();
        let got = collect_until_done(&session);
        assert!(got.contains(&AgentEvent::Session("fake-1".into())), "{got:?}");
        assert!(got.contains(&AgentEvent::Tokens { input: 3, output: 4 }), "{got:?}");
        assert!(got.contains(&AgentEvent::Done { reason: "success".into() }), "{got:?}");
        let text = text_of(&got);
        assert!(text.contains("--json") && text.contains("--skip-git-repo-check"), "{text}");
        assert!(text.contains("mcp_servers.figcraft-app.command="), "{text}");
        assert!(text.contains("\"/usr/local/bin/figcraft-cli\""), "{text}");
        assert!(text.contains("mcp_servers.figcraft-app.args=") && text.contains("7979"), "{text}");
        assert!(text.contains("approval_policy=\"never\""), "{text}");
        assert!(!text.contains("resume"), "the first turn starts fresh: {text}");
        // The next turn resumes the thread `thread.started` named.
        session.send(&turn(" lagi")).unwrap();
        let got = collect_until_done(&session);
        let text = text_of(&got);
        assert!(text.contains("resume") && text.contains("fake-1"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_backend_that_dies_without_a_turn_says_why() {
        let dir = std::env::temp_dir().join(format!("figcraft-agent-cx-dead-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = broken_codex(&dir);
        let mut session = Session::new(Box::new(CodexCli::new(bin, "/usr/local/bin/figcraft-cli", 1)));
        session.send(&turn("halo")).unwrap();
        let got = collect_until_done(&session);
        let failed: Vec<&str> = got
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Failed(m) => Some(m.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(failed.len(), 1, "{got:?}");
        assert!(failed[0].contains("Codex") && failed[0].contains("permission denied"), "{}", failed[0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The real `codex` against the real flags — run by hand, never in CI:
    /// `cargo test -p figcraft-agent -- --ignored live_codex --nocapture`
    #[ignore = "needs the real codex CLI and an account"]
    #[test]
    fn live_codex_says_hello_without_touching_the_canvas() {
        let mut session = Session::new(Box::new(CodexCli::new("codex", "figcraft-cli", 9)));
        session.send(&Turn { text: "Reply with exactly: siap. Do not call any tools.".into(), attachments: Vec::new() }).unwrap();
        let got = collect_until_done(&session);
        println!("{got:#?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Done { .. })), "{got:?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Session(_))), "{got:?}");
    }

    /// The whole loop against a live app: the real `codex` designs on the canvas of the
    /// FigCraft it is attached to, through `figcraft-cli mcp --attach`. Run by hand with
    /// the app up, never in CI:
    /// `FIGCRAFT_TEST_PORT=<control port> cargo test -p figcraft-agent -- --ignored live_codex_designs --nocapture`
    #[ignore = "needs a running app, the real codex CLI and an account"]
    #[test]
    fn live_codex_designs_on_the_attached_canvas() {
        let Some(port) = crate::live_tests::live_port() else { return };
        let mut session = Session::new(Box::new(CodexCli::new("codex", "figcraft-cli", port)));
        session.send(&Turn { text: crate::live_tests::live_design_prompt("codex"), attachments: Vec::new() }).unwrap();
        let got = collect_until_done(&session);
        println!("{got:#?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Done { .. })), "{got:?}");
        let tools: Vec<&str> = got
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert!(!tools.is_empty(), "no MCP tool was called: {got:?}");
    }
}
