//! The SSE parser's mapping, and a full turn against a fake API on a local socket: what
//! goes out (headers, model, tools, the tool_result round), what comes back to the panel
//! in order, and that an aborted turn leaves the conversation exactly as it found it.

use serde_json::Value;

use crate::anthropic_api::{Parser, SYSTEM, parse_tool_input};
use crate::{AgentEvent, Session, Turn};

/// Feed whole SSE lines and collect the panel events they produced.
fn events(parser: &mut Parser, lines: &[&str]) -> Vec<AgentEvent> {
    lines.iter().flat_map(|l| parser.feed(l)).collect()
}

#[test]
fn text_streams_out_and_tool_arguments_assemble_by_block_index() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            "event: message_start",
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":1}}}"#,
            "", // SSE framing: a blank line closes each event.
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Saya membuat "}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"kotak."}}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"create_node","input":{}}}"#,
            // The arguments arrive as fragments; only their concatenation is real JSON.
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"name\":\"box\""}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":",\"x\":10}"}}"#,
            r#"data: {"type":"content_block_stop","index":1}"#,
            ": a comment line the API may send at any time",
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
            r#"data: {"type":"message_stop"}"#,
        ],
    );
    assert_eq!(got, vec![AgentEvent::Text("Saya membuat ".into()), AgentEvent::Text("kotak.".into())]);
    let round = p.into_round();
    assert!(round.stopped);
    assert_eq!(round.stop.as_deref(), Some("tool_use"));
    assert_eq!((round.input, round.output), (11, 7));
    assert_eq!(round.error, None);
    // The assistant message keeps text and tool_use in stream order; the tool carries
    // the assembled arguments, not the empty start.
    assert_eq!(round.content.len(), 2);
    assert_eq!(round.content[0]["type"], "text");
    assert_eq!(round.content[1]["type"], "tool_use");
    assert_eq!(round.content[1]["input"], serde_json::json!({"name":"box","x":10}));
    let tools: &crate::anthropic_api::ToolUse = &round.tools[0];
    assert_eq!((tools.id.as_str(), tools.name.as_str()), ("toolu_1", "create_node"));
    assert_eq!(tools.raw, "{\"name\":\"box\",\"x\":10}");
    assert_eq!(tools.input, Some(serde_json::json!({"name":"box","x":10})));
}

#[test]
fn an_error_event_a_ping_and_a_stream_without_message_stop_report_honestly() {
    let mut p = Parser::default();
    // `ping` and unknown payloads are ignored, an error keeps its message.
    let got = events(
        &mut p,
        &[
            r#"data: {"type":"ping"}"#,
            r#"data: {"type":"new_kind_from_the_future","whatever":1}"#,
            r#"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ],
    );
    assert!(got.is_empty(), "{got:?}");
    let round = p.into_round();
    assert_eq!(round.error.as_deref(), Some("Overloaded"));
    assert!(!round.stopped, "an error is not a finished message");

    // A stream that just ends (no message_stop) is unfinished, even with text in it.
    let mut p = Parser::default();
    events(&mut p, &[r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"terpot"}}"#]);
    let round = p.into_round();
    assert!(!round.stopped);
    assert_eq!(round.stop, None, "a stream without message_delta carried no stop reason");
    assert_eq!(round.content[0]["text"], "terpot");
}

#[test]
fn a_max_tokens_stop_and_unparsable_arguments_stay_usable() {
    let mut p = Parser::default();
    events(
        &mut p,
        &[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_2","name":"export","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"format\":\"pdf\""}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":9}}"#,
            r#"data: {"type":"message_stop"}"#,
        ],
    );
    let round = p.into_round();
    assert!(round.stopped);
    assert_eq!(round.stop.as_deref(), Some("max_tokens"));
    // The arguments were cut off: the call survives with an empty input, and the
    // turn loop will tell the model its arguments never parsed.
    assert_eq!(round.tools[0].input, None);
    assert_eq!(round.content[0]["input"], serde_json::json!({}));

    // A tool with no arguments at all: empty stream means `{}`, not garbage.
    assert_eq!(parse_tool_input(""), Some(serde_json::json!({})));
    assert_eq!(parse_tool_input("  \n"), Some(serde_json::json!({})));
    assert_eq!(parse_tool_input("{\"a\":1}"), Some(serde_json::json!({"a":1})));
    assert_eq!(parse_tool_input("[1,2]"), None, "a non-object is not arguments");
    assert_eq!(parse_tool_input("{broken"), None);
}

#[test]
fn a_tool_use_block_without_an_id_is_dropped_not_half_kept() {
    let mut p = Parser::default();
    events(
        &mut p,
        &[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"create_node","input":{}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ],
    );
    let round = p.into_round();
    assert!(round.tools.is_empty(), "a tool with no id has no result to answer with");
    assert!(round.content.is_empty());
    assert!(SYSTEM.contains("FigCraft"), "the system prompt stands alone");
}

#[cfg(unix)]
mod spawn {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use crate::AnthropicApi;

    /// A canned API on a local socket: it records every request's head and body, then
    /// answers with the next canned response — one per request, the last repeating, each
    /// with its own delay before the bytes go out (used to hold a round open while the
    /// test presses Stop).
    struct FakeApi {
        base: String,
        seen: Arc<Mutex<Vec<(String, String)>>>,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl FakeApi {
        fn start(responses: Vec<(Duration, String)>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            listener.set_nonblocking(true).unwrap();
            let seen = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let queue = Arc::new(Mutex::new(responses));
            let (seen_t, stop_t) = (seen.clone(), stop.clone());
            let handle = std::thread::spawn(move || {
                while !stop_t.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let Some((head, body)) = read_request(&mut stream) else { continue };
                            seen_t.lock().unwrap().push((head, body));
                            let (delay, response) = {
                                let mut queue = queue.lock().unwrap();
                                if queue.is_empty() {
                                    (Duration::ZERO, "HTTP/1.1 500 Server Error\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_string())
                                } else if queue.len() == 1 {
                                    queue[0].clone()
                                } else {
                                    queue.remove(0)
                                }
                            };
                            if !delay.is_zero() {
                                std::thread::sleep(delay);
                            }
                            let _ = stream.write_all(response.as_bytes());
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(10)),
                        Err(_) => break,
                    }
                }
            });
            FakeApi { base, seen, stop, handle: Some(handle) }
        }

        /// Every request so far: (head lowercased, body), in arrival order.
        fn seen(&self) -> Vec<(String, String)> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl Drop for FakeApi {
        /// The accept loop is nonblocking with a 10ms nap: joining ends at once.
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    /// Read one HTTP/1.1 request: head until the blank line, then exactly `content-length`
    /// body bytes. Lowercased head, so the assertions do not care about the casing ureq sends.
    fn read_request(stream: &mut TcpStream) -> Option<(String, String)> {
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        let head_end = loop {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
            if buf.len() > 1 << 20 {
                return None;
            }
            let n = stream.read(&mut chunk).ok()?;
            if n == 0 {
                return None;
            }
            buf.extend_from_slice(&chunk[..n]);
        };
        let mut head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
        // Lowercased once, so both the length parse and the assertions see one casing.
        head.make_ascii_lowercase();
        let length: usize = head.lines().find_map(|line| line.strip_prefix("content-length:")).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        let mut body = buf[head_end..].to_vec();
        while body.len() < length {
            let n = stream.read(&mut chunk).ok()?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..n]);
        }
        Some((head.to_ascii_lowercase(), String::from_utf8_lossy(&body).into_owned()))
    }

    /// A 200 SSE response with a content length and a closed connection: the shape the
    /// client reads line by line, ended cleanly.
    fn sse(body: &str) -> (Duration, String) {
        (
            Duration::ZERO,
            format!("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()),
        )
    }

    /// A non-2xx JSON response, as the API sends for a refusal.
    fn refused(status: u16, body: &str) -> (Duration, String) {
        (
            Duration::ZERO,
            format!(
                "HTTP/1.1 {status} Refused\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            ),
        )
    }

    /// Round one asks to run a tool: text streams first, then a `tool_use` whose arguments
    /// arrive as two fragments.
    const ROUND_TOOL: &str = r#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Saya membuat kotak."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"create_node","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"name\":\"box\""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":",\"x\":10}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}
"#;

    /// A normal finishing round: one text delta and `end_turn`.
    const ROUND_END: &str = r#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Kotak jadi."}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
"#;

    /// The second turn's own finishing round.
    const ROUND_TURN2: &str = r#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":40,"output_tokens":1}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Siap."}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}
"#;

    /// A round that stopped without saying anything at all.
    const ROUND_EMPTY: &str = r#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}

event: message_stop
data: {"type":"message_stop"}
"#;

    /// A fake `figcraft-cli mcp --attach` answering every `tools/call` with success.
    fn fake_cli(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("fake-cli.sh");
        let script = "#!/bin/sh\nwhile IFS= read -r line; do\n\
            \x20 id=$(printf '%s\\n' \"$line\" | sed -n 's/^{\"id\":\\([0-9-]*\\),.*/\\1/p')\n\
            \x20 case \"$line\" in\n\
            \x20   *'\"method\":\"initialize\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"test\",\"capabilities\":{\"tools\":{}}}}\\n' \"$id\" ;;\n\
            \x20   *'\"method\":\"tools/call\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"created node 42\"}],\"isError\":false}}\\n' \"$id\" ;;\n\
            \x20 esac\n\
            done\n";
        std::fs::write(&path, script).unwrap();
        let mut perm = std::fs::metadata(&path).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&path, perm).unwrap();
        path
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("figcraft-anthropic-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn turn(text: &str) -> Turn {
        Turn { text: text.into(), attachments: Vec::new() }
    }

    fn collect_until_done(session: &Session) -> Vec<AgentEvent> {
        // A real API turn can take a while; the fakes return at once, so the cap only
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

    fn failed_message(got: &[AgentEvent]) -> String {
        got.iter()
            .find_map(|e| match e {
                AgentEvent::Failed(m) => Some(m.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no error row: {got:?}"))
    }

    #[test]
    fn a_turn_streams_text_runs_the_tool_and_commits_the_conversation() {
        let api = FakeApi::start(vec![sse(ROUND_TOOL), sse(ROUND_END), sse(ROUND_TURN2)]);
        let dir = scratch("turn");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(AnthropicApi::new("sk-ant-test", &cli, 7979).base_url(api.base.clone())));
        session.send(&turn("buat kotak")).unwrap();
        let got = collect_until_done(&session);
        // Text streams as it arrives; usage lands before the tool; the tool's row pairs
        // because its start is sent only after the message stopped.
        assert_eq!(
            got,
            vec![
                AgentEvent::Text("Saya membuat kotak.".into()),
                AgentEvent::Tokens { input: 11, output: 7 },
                AgentEvent::ToolStart { name: "create_node".into(), args: "{\"name\":\"box\",\"x\":10}".into() },
                AgentEvent::ToolEnd { name: "create_node".into(), ok: true, summary: "created node 42".into() },
                AgentEvent::Text("Kotak jadi.".into()),
                AgentEvent::Tokens { input: 36, output: 12 },
                AgentEvent::Done { reason: "end_turn".into() },
            ]
        );
        // One request per round; the first carries our headers, model, system prompt and
        // the tools in the API's spelling.
        let seen = api.seen();
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(seen[0].0.contains("x-api-key: sk-ant-test"), "{}", seen[0].0);
        assert!(seen[0].0.contains("anthropic-version: 2023-06-01"), "{}", seen[0].0);
        let first: Value = serde_json::from_str(&seen[0].1).unwrap();
        assert_eq!(first["model"], "claude-sonnet-5-5");
        assert_eq!(first["system"], SYSTEM);
        assert_eq!(first["stream"], true);
        assert_eq!(first["max_tokens"], 8192);
        assert_eq!(first["messages"].as_array().unwrap().len(), 1, "the turn starts with its one prompt");
        assert_eq!(first["messages"][0]["content"], "buat kotak");
        let tools = first["tools"].as_array().unwrap();
        let create = tools.iter().find(|t| t["name"] == "create_node").unwrap_or_else(|| panic!("no create_node in {tools:?}"));
        assert!(create.get("input_schema").is_some(), "{create}");
        assert!(create.get("inputSchema").is_none(), "the MCP spelling leaked into the API request: {create}");
        // Round two carries the assistant's tool_use and the tool_result answering it.
        let second: Value = serde_json::from_str(&seen[1].1).unwrap();
        let messages = second["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["content"][1]["type"], "tool_use");
        assert_eq!(messages[1]["content"][1]["input"], serde_json::json!({"name":"box","x":10}));
        let results = messages[2]["content"].as_array().unwrap();
        assert_eq!(results[0]["type"], "tool_result");
        assert_eq!(results[0]["tool_use_id"], "toolu_1");
        assert_eq!(results[0]["is_error"], false);
        // The second turn continues the conversation: all four committed messages, then
        // its own prompt.
        session.send(&turn("sekarang lingkaran")).unwrap();
        let got2 = collect_until_done(&session);
        assert!(got2.contains(&AgentEvent::Done { reason: "end_turn".into() }), "{got2:?}");
        let seen = api.seen();
        let third: Value = serde_json::from_str(&seen[2].1).unwrap();
        let messages = third["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 5, "the first turn's four messages must still be there");
        assert_eq!(messages[4]["content"], "sekarang lingkaran");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A dropped image travels: the user message becomes blocks — the text (with the file
    /// named, the words the CLI backends add) and the image's bytes base64-encoded under
    /// the media type they proved to be — and the next turn goes back to a plain string.
    #[test]
    fn an_image_turn_sends_its_bytes_as_an_image_block_and_still_names_the_file() {
        let api = FakeApi::start(vec![sse(ROUND_END), sse(ROUND_TURN2)]);
        let dir = scratch("image");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(AnthropicApi::new("sk-ant-test", &cli, 7979).base_url(api.base.clone())));
        let png = b"\x89PNG\r\n\x1a\nfigcraft";
        let mut t = Turn {
            text: "lihat skema ini".into(),
            attachments: vec![crate::Attachment { name: "skema.png".into(), mime: "image/png".into(), data: png.to_vec() }],
        };
        session.send(&t).unwrap();
        let got = collect_until_done(&session);
        assert!(got.contains(&AgentEvent::Done { reason: "end_turn".into() }), "{got:?}");
        let seen = api.seen();
        let first: Value = serde_json::from_str(&seen[0].1).unwrap();
        let content = &first["messages"][0]["content"];
        let blocks = content.as_array().unwrap_or_else(|| panic!("a turn with an image sends content blocks, not a string: {content}"));
        assert_eq!(blocks[0]["type"], "text");
        let text = blocks[0]["text"].as_str().unwrap();
        assert!(text.contains("lihat skema ini") && text.contains("(Attached files: skema.png.)"), "{text}");
        assert_eq!(blocks.len(), 2, "one text block, one image block: {blocks:?}");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
        assert_eq!(blocks[1]["source"]["data"], crate::vision::encode(png), "the bytes arrive whole");
        // Without an image the message is a string again — the shapes never mix.
        t.attachments.clear();
        t.text = "sekarang lingkaran".into();
        session.send(&t).unwrap();
        let got2 = collect_until_done(&session);
        assert!(got2.contains(&AgentEvent::Done { reason: "end_turn".into() }), "{got2:?}");
        // One request per turn: the second carries the committed first turn, then its
        // own prompt as a plain string again.
        let second: Value = serde_json::from_str(&api.seen()[1].1).unwrap();
        let messages = second["messages"].as_array().unwrap();
        assert_eq!(messages[2]["content"], "sekarang lingkaran");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_refusal_names_the_status_and_the_api_s_own_message() {
        let api = FakeApi::start(vec![refused(401, r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#)]);
        let dir = scratch("refusal");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(AnthropicApi::new("sk-ant-wrong", &cli, 7979).base_url(api.base.clone())));
        session.send(&turn("halo")).unwrap();
        let got = collect_until_done(&session);
        let message = failed_message(&got);
        assert!(message.contains("401"), "{message}");
        assert!(message.contains("invalid x-api-key"), "{message}");
        // A refused turn leaves the conversation empty: the next turn is one message.
        session.send(&turn("lagi")).unwrap();
        let got2 = collect_until_done(&session);
        assert!(!failed_message(&got2).is_empty(), "the second turn should also be refused");
        let seen = api.seen();
        let second: Value = serde_json::from_str(&seen[1].1).unwrap();
        assert_eq!(second["messages"].as_array().unwrap().len(), 1, "the refused turn left messages behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_reply_with_nothing_in_it_fails_the_turn() {
        let api = FakeApi::start(vec![sse(ROUND_EMPTY)]);
        let dir = scratch("empty");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(AnthropicApi::new("sk-ant-test", &cli, 7979).base_url(api.base.clone())));
        session.send(&turn("halo")).unwrap();
        let got = collect_until_done(&session);
        assert!(failed_message(&got).contains("empty reply"), "{got:?}");
        // Its usage still reached the panel before the failure.
        assert!(got.contains(&AgentEvent::Tokens { input: 5, output: 0 }), "{got:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_cancelled_turn_stops_and_rolls_the_conversation_back() {
        // Round two's response is delayed: the turn sits waiting for it, the user presses
        // Stop, and the round is abandoned before its first line is fed to the parser.
        let api = FakeApi::start(vec![sse(ROUND_TOOL), (Duration::from_secs(3), sse(ROUND_END).1), sse(ROUND_TURN2)]);
        let dir = scratch("cancel");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(AnthropicApi::new("sk-ant-test", &cli, 7979).base_url(api.base.clone())));
        session.send(&turn("buat kotak")).unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut got = Vec::new();
        while !got.iter().any(|e| matches!(e, AgentEvent::ToolEnd { .. })) {
            assert!(Instant::now() < deadline, "the first round never ran its tool: {got:?}");
            got.extend(session.poll());
            std::thread::sleep(Duration::from_millis(25));
        }
        session.cancel();
        let mut tail = Vec::new();
        while !tail.iter().any(|e| matches!(e, AgentEvent::Done { .. })) {
            assert!(Instant::now() < deadline, "cancel never ended the turn: {tail:?}");
            tail.extend(session.poll());
            std::thread::sleep(Duration::from_millis(25));
        }
        assert!(tail.contains(&AgentEvent::Done { reason: "cancelled".into() }), "{tail:?}");
        // The next turn sees an empty conversation: an aborted turn leaves nothing to
        // resume — not even its prompt.
        session.send(&turn("halo lagi")).unwrap();
        let got2 = collect_until_done(&session);
        assert!(got2.contains(&AgentEvent::Done { reason: "end_turn".into() }), "{got2:?}");
        let seen = api.seen();
        let last: Value = serde_json::from_str(&seen.last().unwrap().1).unwrap();
        assert_eq!(last["messages"].as_array().unwrap().len(), 1, "the cancelled turn left messages behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "needs a real ANTHROPIC_API_KEY and a running app (FIGCRAFT_TEST_PORT)"]
    fn live_anthropic_designs_on_the_attached_canvas() {
        let Ok(key) = std::env::var("ANTHROPIC_API_KEY") else {
            println!("skipped: set ANTHROPIC_API_KEY to run this against the real API");
            return;
        };
        let Some(port) = crate::live_tests::live_port() else { return };
        let prompt = crate::live_tests::live_design_prompt("anthropic");
        let mut session = Session::new(Box::new(AnthropicApi::new(key, "figcraft-cli", port)));
        session.send(&turn(&prompt)).unwrap();
        let got = collect_until_done(&session);
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Done { .. })), "{got:?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::ToolStart { .. })), "the agent ran no tools: {got:?}");
    }
}
