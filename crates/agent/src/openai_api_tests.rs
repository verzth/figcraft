//! The Chat Completions parser's mapping, and a full turn against a fake API on a local
//! socket: what goes out (the Bearer header, the system message, the `function` tools, the
//! `role: "tool"` answers), what comes back to the panel in order, and that an aborted turn
//! leaves the conversation exactly as it found it.

use serde_json::Value;

use crate::anthropic_api::SYSTEM;
use crate::openai_api::{Parser, completions_url, finish_reason, parse_tool_input};
use crate::{AgentEvent, Session, Turn};

/// Feed whole SSE lines and collect the panel events they produced.
fn events(parser: &mut Parser, lines: &[&str]) -> Vec<AgentEvent> {
    lines.iter().flat_map(|l| parser.feed(l)).collect()
}

#[test]
fn text_streams_out_and_tool_calls_assemble_by_chunk_index() {
    let mut p = Parser::default();
    let got = events(
        &mut p,
        &[
            "event: message_start",
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            "", // SSE framing: a blank line closes each event.
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"content":"Saya membuat "},"finish_reason":null}]}"#,
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"content":"kotak."},"finish_reason":null}]}"#,
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_7","type":"function","function":{"name":"create_node","arguments":""}}]},"finish_reason":null}]}"#,
            // The arguments arrive as fragments; only their concatenation is real JSON.
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"name\":\"box\""}}]},"finish_reason":null}]}"#,
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\"x\":10}"}}]},"finish_reason":null}]}"#,
            ": a comment line the server may send at any time",
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#,
            r#"data: [DONE]"#,
        ],
    );
    assert_eq!(got, vec![AgentEvent::Text("Saya membuat ".into()), AgentEvent::Text("kotak.".into())]);
    let round = p.into_round();
    assert!(round.stopped, "finish_reason plus [DONE] is a finished reply");
    assert_eq!(round.stop.as_deref(), Some("tool_calls"));
    assert_eq!((round.input, round.output), (11, 7));
    assert_eq!(round.error, None);
    assert_eq!(round.text, "Saya membuat kotak.");
    let tools: &crate::openai_api::ToolUse = &round.tools[0];
    assert_eq!((tools.id.as_str(), tools.name.as_str()), ("call_7", "create_node"));
    assert_eq!(tools.raw, "{\"name\":\"box\",\"x\":10}");
    assert_eq!(tools.input, Some(serde_json::json!({"name":"box","x":10})));
}

#[test]
fn an_error_object_a_finish_without_done_and_a_dead_stream_report_honestly() {
    // Some servers stream their errors mid-round instead of refusing up front.
    let mut p = Parser::default();
    let got = events(&mut p, &[r#"data: {"error":{"message":"model overloaded","type":"server_error"}}"#]);
    assert!(got.is_empty(), "{got:?}");
    let round = p.into_round();
    assert_eq!(round.error.as_deref(), Some("model overloaded"));
    assert!(!round.stopped, "an error is not a finished message");

    // A finish reason with no [DONE] (a compatible server that just closes) counts.
    let mut p = Parser::default();
    events(&mut p, &[r#"data: {"choices":[{"index":0,"delta":{"content":"halo"},"finish_reason":"stop"}]}"#]);
    let round = p.into_round();
    assert!(round.stopped);
    assert_eq!(round.stop.as_deref(), Some("stop"));

    // A stream that ends with neither (a connection cut mid-reply) is unfinished.
    let mut p = Parser::default();
    events(&mut p, &[r#"data: {"choices":[{"index":0,"delta":{"content":"terpo"}}]}"#]);
    let round = p.into_round();
    assert!(!round.stopped, "no finish and no [DONE]: the reply never ended");
    assert_eq!(round.stop, None);
    assert_eq!(round.text, "terpo");
}

#[test]
fn unparsable_arguments_a_missing_id_and_the_endpoint_all_stay_usable() {
    // Arguments cut off mid-JSON: the call survives with no input, and the turn loop will
    // tell the model its arguments never parsed.
    let mut p = Parser::default();
    events(
        &mut p,
        &[
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"export","arguments":"{\"format\":\"pdf\""}}]},"finish_reason":null}]}"#,
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
            r#"data: [DONE]"#,
        ],
    );
    let round = p.into_round();
    assert_eq!(round.tools[0].input, None);
    // A server that never sent an id gets one its answers can still match.
    assert_eq!(round.tools[0].id, "call_0");
    // "length" rides through spelled out: a truncated reply reads as itself.
    assert_eq!(finish_reason(round.stop.as_deref()), "length");
    assert_eq!(finish_reason(Some("stop")), "end_turn");
    assert_eq!(finish_reason(None), "end_turn");

    assert_eq!(parse_tool_input(""), Some(serde_json::json!({})));
    assert_eq!(parse_tool_input("{broken"), None);
    // A base that already carries /v1 keeps it; one that doesn't gains it.
    assert_eq!(completions_url("https://openrouter.ai/api/v1"), "https://openrouter.ai/api/v1/chat/completions");
    assert_eq!(completions_url("https://api.openai.com"), "https://api.openai.com/v1/chat/completions");
    assert_eq!(completions_url("http://127.0.0.1:11434/v1/"), "http://127.0.0.1:11434/v1/chat/completions");
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

    use crate::OpenAiApi;

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
        Some((head, String::from_utf8_lossy(&body).into_owned()))
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

    /// Round one asks to run a tool: text streams first, then a `tool_calls` entry whose
    /// arguments arrive as two fragments.
    const ROUND_TOOL: &str = r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"content":"Saya membuat kotak."},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_7","type":"function","function":{"name":"create_node","arguments":""}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"name\":\"box\""}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\"x\":10}"}}]},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}

data: [DONE]
"#;

    /// A normal finishing round: one content chunk and `stop`.
    const ROUND_END: &str = r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"content":"Kotak jadi."},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":5,"total_tokens":30}}

data: [DONE]
"#;

    /// The second turn's own finishing round.
    const ROUND_TURN2: &str = r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"content":"Siap."},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[],"usage":{"prompt_tokens":40,"completion_tokens":3,"total_tokens":43}}

data: [DONE]
"#;

    /// A round that finished without saying anything at all.
    const ROUND_EMPTY: &str = r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-6.1-sol","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5}}

data: [DONE]
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
        let dir = std::env::temp_dir().join(format!("figcraft-openai-test-{}-{name}", std::process::id()));
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
        let mut session = Session::new(Box::new(OpenAiApi::new("sk-test-123", &cli, 7979).base_url(api.base.clone())));
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
        // One request per round; the first carries the Bearer header, the system message
        // and the tools in this API's shape.
        let seen = api.seen();
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(seen[0].0.contains("authorization: bearer sk-test-123"), "{}", seen[0].0);
        let first: Value = serde_json::from_str(&seen[0].1).unwrap();
        assert_eq!(first["model"], "gpt-6.1-sol");
        assert_eq!(first["stream"], true);
        assert_eq!(first["stream_options"]["include_usage"], true);
        let messages = first["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "system prompt + the one prompt");
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], SYSTEM);
        assert_eq!(messages[1]["content"], "buat kotak");
        let tools = first["tools"].as_array().unwrap();
        let create = tools.iter().find(|t| t["function"]["name"] == "create_node").unwrap_or_else(|| panic!("no create_node in {tools:?}"));
        assert_eq!(create["type"], "function");
        assert!(create["function"]["parameters"].is_object(), "{create}");
        assert!(create.get("parameters_schema").is_none() && create.get("inputSchema").is_none(), "wrong spelling: {create}");
        // Round two carries the assistant's tool_calls and the tool answer matching them.
        let second: Value = serde_json::from_str(&seen[1].1).unwrap();
        let messages = second["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 4, "system + prompt + assistant + answer");
        assert_eq!(messages[2]["role"], "assistant");
        // This round streamed prose AND asked for a tool: the message carries both.
        assert_eq!(messages[2]["content"], "Saya membuat kotak.");
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call_7");
        assert_eq!(messages[2]["tool_calls"][0]["function"]["arguments"], "{\"name\":\"box\",\"x\":10}");
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_7");
        assert_eq!(messages[3]["content"], "created node 42");
        // The second turn continues the conversation: all four committed messages, then
        // its own prompt.
        session.send(&turn("sekarang lingkaran")).unwrap();
        let got2 = collect_until_done(&session);
        assert!(got2.contains(&AgentEvent::Done { reason: "end_turn".into() }), "{got2:?}");
        let seen = api.seen();
        let third: Value = serde_json::from_str(&seen[2].1).unwrap();
        let messages = third["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 6, "the first turn's five messages must still be there");
        assert_eq!(messages[5]["content"], "sekarang lingkaran");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A dropped image travels: the user message becomes parts — the text (with the file
    /// named, the words the CLI backends add) and the image's bytes as a `data:` URL under
    /// the media type they proved to be — and the next turn goes back to a plain string.
    #[test]
    fn an_image_turn_sends_its_bytes_as_a_data_url_and_still_names_the_file() {
        let api = FakeApi::start(vec![sse(ROUND_END), sse(ROUND_TURN2)]);
        let dir = scratch("image");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(OpenAiApi::new("sk-test-123", &cli, 7979).base_url(api.base.clone())));
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
        let messages = first["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2, "system prompt + the one prompt");
        let content = &messages[1]["content"];
        let parts = content.as_array().unwrap_or_else(|| panic!("a turn with an image sends content parts, not a string: {content}"));
        assert_eq!(parts[0]["type"], "text");
        let text = parts[0]["text"].as_str().unwrap();
        assert!(text.contains("lihat skema ini") && text.contains("(Attached files: skema.png.)"), "{text}");
        assert_eq!(parts.len(), 2, "one text part, one image part: {parts:?}");
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert_eq!(url, format!("data:image/png;base64,{}", crate::vision::encode(png)));
        // Without an image the message is a string again — the shapes never mix.
        t.attachments.clear();
        t.text = "sekarang lingkaran".into();
        session.send(&t).unwrap();
        let got2 = collect_until_done(&session);
        assert!(got2.contains(&AgentEvent::Done { reason: "end_turn".into() }), "{got2:?}");
        // One request per turn: the second carries system + the committed first turn,
        // then its own prompt as a plain string again.
        let second: Value = serde_json::from_str(&api.seen()[1].1).unwrap();
        let messages = second["messages"].as_array().unwrap();
        assert_eq!(messages[3]["content"], "sekarang lingkaran");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_refusal_names_the_status_and_the_api_s_own_message() {
        let api = FakeApi::start(vec![refused(401, r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#)]);
        let dir = scratch("refusal");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(OpenAiApi::new("sk-wrong", &cli, 7979).base_url(api.base.clone())));
        session.send(&turn("halo")).unwrap();
        let got = collect_until_done(&session);
        let message = failed_message(&got);
        assert!(message.contains("401"), "{message}");
        assert!(message.contains("Incorrect API key provided"), "{message}");
        // A refused turn leaves the conversation empty: the next turn is one message.
        session.send(&turn("lagi")).unwrap();
        let got2 = collect_until_done(&session);
        assert!(!failed_message(&got2).is_empty(), "the second turn should also be refused");
        let seen = api.seen();
        let second: Value = serde_json::from_str(&seen[1].1).unwrap();
        // The system message plus the new prompt, and nothing of the refused one.
        assert_eq!(second["messages"].as_array().unwrap().len(), 2, "the refused turn left messages behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_reply_with_nothing_in_it_fails_the_turn() {
        let api = FakeApi::start(vec![sse(ROUND_EMPTY)]);
        let dir = scratch("empty");
        let cli = fake_cli(&dir);
        let mut session = Session::new(Box::new(OpenAiApi::new("sk-test-123", &cli, 7979).base_url(api.base.clone())));
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
        let mut session = Session::new(Box::new(OpenAiApi::new("sk-test-123", &cli, 7979).base_url(api.base.clone())));
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
        // The system message plus the new prompt, and nothing of the cancelled turn.
        assert_eq!(last["messages"].as_array().unwrap().len(), 2, "the cancelled turn left messages behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "needs a real OPENAI_API_KEY and a running app (FIGCRAFT_TEST_PORT)"]
    fn live_openai_designs_on_the_attached_canvas() {
        let Ok(key) = std::env::var("OPENAI_API_KEY") else {
            println!("skipped: set OPENAI_API_KEY to run this against the real API");
            return;
        };
        let Some(port) = crate::live_tests::live_port() else { return };
        let prompt = crate::live_tests::live_design_prompt("openai");
        let mut session = Session::new(Box::new(OpenAiApi::new(key, "figcraft-cli", port)));
        session.send(&turn(&prompt)).unwrap();
        let got = collect_until_done(&session);
        assert!(got.iter().any(|e| matches!(e, AgentEvent::Done { .. })), "{got:?}");
        assert!(got.iter().any(|e| matches!(e, AgentEvent::ToolStart { .. })), "the agent ran no tools: {got:?}");
    }
}
