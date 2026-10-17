//! The OpenCode JSONL parser's mapping and the spawn wiring, against a fake `opencode` that
//! speaks the same JSON on stdout (the real CLI is exercised by hand from the panel).

use crate::opencode_cli::{OpenCodeCli, Parser};
use crate::{AgentEvent, Session, Turn};

fn events(parser: &mut Parser, lines: &[&str]) -> Vec<AgentEvent> {
    lines.iter().flat_map(|l| parser.line(l)).collect()
}

/// The two lines the real `opencode run --format json` printed for a one-word reply.
const STEP_START: &str = r#"{"type":"step_start","timestamp":1792249936256,"sessionID":"ses_abc","part":{"id":"prt_1","sessionID":"ses_abc","messageID":"msg_1","type":"step-start"}}"#;
const TEXT: &str = r#"{"type":"text","timestamp":1792249936533,"sessionID":"ses_abc","part":{"id":"prt_1_text-0","sessionID":"ses_abc","messageID":"msg_1","type":"text","text":"siap","time":{"start":1,"end":2}}}"#;

#[test]
fn session_text_and_tool_parts_map_in_order() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            STEP_START,
            TEXT,
            r#"{"type":"tool","sessionID":"ses_abc","part":{"id":"prt_tool","type":"tool","tool":"figcraft.build_layout","state":{"status":"running","input":{"x":10,"y":20}}}}"#,
            // The same running part again (state snapshots repeat): still one row.
            r#"{"type":"tool","sessionID":"ses_abc","part":{"id":"prt_tool","type":"tool","tool":"figcraft.build_layout","state":{"status":"running","input":{"x":10,"y":20}}}}"#,
            r#"{"type":"tool","sessionID":"ses_abc","part":{"id":"prt_tool","type":"tool","tool":"figcraft.build_layout","state":{"status":"completed","output":"rectangle at 10,20\n2 children"}}}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            // The session arrives with the first event that carries it, once.
            AgentEvent::Session("ses_abc".into()),
            AgentEvent::Text("siap".into()),
            AgentEvent::ToolStart { name: "figcraft.build_layout".into(), args: "{\"x\":10,\"y\":20}".into() },
            AgentEvent::ToolEnd { name: "figcraft.build_layout".into(), ok: true, summary: "rectangle at 10,20".into() },
        ]
    );
}

#[test]
fn a_failed_tool_reports_its_error_and_a_reemitted_text_shows_only_its_growth() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            r#"{"type":"tool","sessionID":"ses_abc","part":{"id":"p","type":"tool","tool":"figcraft.apply_props","state":{"status":"error","error":"no selection"}}}"#,
            r#"{"type":"text","sessionID":"ses_abc","part":{"id":"t","type":"text","text":"Halo du"}}"#,
            r#"{"type":"text","sessionID":"ses_abc","part":{"id":"t","type":"text","text":"Halo dunia"}}"#,
        ],
    );
    assert_eq!(
        got,
        vec![
            AgentEvent::Session("ses_abc".into()),
            AgentEvent::ToolEnd { name: "figcraft.apply_props".into(), ok: false, summary: "no selection".into() },
            AgentEvent::Text("Halo du".into()),
            // The part arrived again with more text: only what is new, never a repeat.
            AgentEvent::Text("nia".into()),
        ]
    );
}

#[test]
fn long_tool_output_is_cut_to_one_line_and_junk_is_tolerated() {
    let mut p = Parser::default();
    let long = "x".repeat(400);
    let line = format!(
        r#"{{"type":"tool","sessionID":"ses_abc","part":{{"id":"g","type":"tool","tool":"t","state":{{"status":"completed","output":"{long}\nsecond"}}}}}}"#
    );
    let got = p.line(&line);
    // The line also carries the session, which arrives first.
    let Some(summary) = got.iter().find_map(|e| match e {
        AgentEvent::ToolEnd { summary, .. } => Some(summary),
        _ => None,
    }) else {
        panic!("expected ToolEnd in {got:?}")
    };
    assert_eq!(summary.chars().count(), 158, "157 cut characters plus the ellipsis");
    assert!(!summary.contains('\n'));

    // Not JSON, an unknown part, an unknown session line: all ignored, nothing panics.
    assert!(events(&mut p, &["not json", r#"{"type":"step_start","part":{"type":"step-start"}}"#, r#"{"type":"wat"}"#]).is_empty());
}

#[cfg(unix)]
mod spawn {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    /// A fake `opencode`: it reports its own arguments and the cwd's `opencode.json`, so one
    /// script covers the config file, the flags and session continuation. Unlike Claude Code
    /// it says nothing at the end of a turn — the pump turns its clean exit into `Done`.
    fn fake_opencode(dir: &std::path::Path, sleep: Option<Duration>) -> PathBuf {
        let path = dir.join("fake-opencode.sh");
        let sleep = sleep.map_or(String::new(), |d| format!("sleep {};\n", d.as_secs_f64()));
        let script = format!(
            "#!/bin/sh\n\
             {sleep}esc=$(tr -d '\\n' < opencode.json | sed 's/\"/\\\\\"/g')\n\
             printf '%s\\n' '{{\"type\":\"step_start\",\"sessionID\":\"ses_fake_1\",\"part\":{{\"type\":\"step-start\"}}}}'\n\
             printf '%s\\n' \"{{\\\"type\\\":\\\"text\\\",\\\"sessionID\\\":\\\"ses_fake_1\\\",\\\"part\\\":{{\\\"id\\\":\\\"t1\\\",\\\"type\\\":\\\"text\\\",\\\"text\\\":\\\"cfg:$esc args:$*\\\"}}}}\"\n"
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
        // The fakes return at once; the cap only bounds a hang.
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
    fn a_turn_writes_the_mcp_config_reports_its_session_and_a_clean_exit_is_success() {
        let dir = std::env::temp_dir().join(format!("figcraft-agent-oc-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = fake_opencode(&dir, None);
        let mut session = Session::new(Box::new(OpenCodeCli::new(&bin, "/usr/local/bin/figcraft-cli", 7979)));
        session.send(&turn("halo")).unwrap();
        let got = collect_until_done(&session);
        assert!(got.contains(&AgentEvent::Session("ses_fake_1".into())), "{got:?}");
        // No terminal event was printed: the clean exit became Done.
        assert!(got.contains(&AgentEvent::Done { reason: "success".into() }), "{got:?}");
        let text = text_of(&got);
        assert!(text.contains("figcraft-app"), "{text}");
        assert!(text.contains("--attach") && text.contains("7979"), "{text}");
        assert!(text.contains("\"type\": \"local\""), "{text}");
        assert!(text.contains("--standalone") && text.contains("--format") && text.contains("--auto"), "{text}");
        assert!(!text.contains("--session"), "the first turn starts fresh: {text}");
        // The next turn continues the session the events named.
        session.send(&turn(" lagi")).unwrap();
        let got = collect_until_done(&session);
        let text = text_of(&got);
        assert!(text.contains("--session") && text.contains("ses_fake_1"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_turn_refused_while_working_says_so_and_a_cancel_stops_the_child() {
        let dir = std::env::temp_dir().join(format!("figcraft-agent-oc-busy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = fake_opencode(&dir, Some(Duration::from_secs(2)));
        let mut session = Session::new(Box::new(OpenCodeCli::new(bin, "/usr/local/bin/figcraft-cli", 1)));
        session.send(&turn("satu")).unwrap();
        let err = session.send(&turn("dua")).unwrap_err();
        assert!(err.to_string().contains("still working"), "{err}");
        session.cancel();
        let got = collect_until_done(&session);
        assert!(got.contains(&AgentEvent::Done { reason: "cancelled".into() }), "{got:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The real `opencode` against the real flags — run by hand, never in CI:
    /// `cargo test -p figcraft-agent -- --ignored live_opencode --nocapture`
    #[ignore = "needs the real opencode CLI and an account"]
    #[test]
    fn live_opencode_says_hello_without_touching_the_canvas() {
        let mut session = Session::new(Box::new(OpenCodeCli::new("opencode", "figcraft-cli", 9)));
        session.send(&Turn { text: "Reply with exactly: siap. Do not call any tools.".into(), attachments: Vec::new() }).unwrap();
        let got = collect_until_done(&session);
        println!("{got:#?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Done { .. })), "{got:?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Session(_))), "{got:?}");
    }
}
