//! The stdio MCP client against a fake `figcraft-cli` that speaks the same JSON-RPC: the
//! handshake, the id matching (a notification, a stale id and a garbage line must all be
//! skipped), and the split between a tool that failed and a connection that broke.

use crate::mcp_stdio::McpStdio;

#[cfg(unix)]
mod spawn {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    /// A fake `figcraft-cli`: it answers `initialize` (after a notification, a stale-id
    /// line and a garbage line that the client must skip) and canned `tools/call` replies
    /// keyed on the tool name — `tool_iserror` is a failed tool, `tool_error` a JSON-RPC
    /// error, `tool_exit` kills the child mid-call.
    fn fake_cli(dir: &std::path::Path, body: &str) -> PathBuf {
        let path = dir.join("fake-cli.sh");
        let script = format!("#!/bin/sh\n{body}");
        std::fs::write(&path, script).unwrap();
        let mut perm = std::fs::metadata(&path).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(&path, perm).unwrap();
        path
    }

    const LOOP: &str = "while IFS= read -r line; do\n\
        \x20 id=$(printf '%s\\n' \"$line\" | sed -n 's/^{\"id\":\\([0-9-]*\\),.*/\\1/p')\n\
        \x20 case \"$line\" in\n\
        \x20   *'\"method\":\"initialize\"'*)\n\
        \x20     printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}'\n\
        \x20     printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{\"stale\":true}}'\n\
        \x20     printf 'garbage that is not json\\n'\n\
        \x20     printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"test\",\"capabilities\":{\"tools\":{}}}}\\n' \"$id\"\n\
        \x20     ;;\n\
        \x20   *'\"method\":\"tools/call\"'*)\n\
        \x20     case \"$line\" in\n\
        \x20       *'\"name\":\"tool_iserror\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"no such node\"}],\"isError\":true}}\\n' \"$id\" ;;\n\
        \x20       *'\"name\":\"tool_error\"'*) printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"error\":{\"code\":-32601,\"message\":\"unknown tool\"}}\\n' \"$id\" ;;\n\
        \x20       *'\"name\":\"tool_exit\"'*) exit 1 ;;\n\
        \x20       *) printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"created node 42\"}],\"isError\":false}}\\n' \"$id\" ;;\n\
        \x20     esac ;;\n\
        \x20 esac\n\
        done\n";

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("figcraft-mcp-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_handshake_survives_notifications_stale_ids_and_garbage() {
        let dir = scratch("handshake");
        let cli = fake_cli(&dir, LOOP);
        let mut client = McpStdio::open(&cli, 7979).unwrap();
        // Every reply's id matched ours: the notification, the stale id 99 and the
        // garbage line were all skipped, not taken as answers.
        assert_eq!(client.call("tool_ok", &serde_json::json!({"id": 1})).unwrap(), (true, "created node 42".to_string()));
        // A second call gets its own id: the transport is still healthy.
        assert_eq!(client.call("tool_ok", &serde_json::json!({})).unwrap(), (true, "created node 42".to_string()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_tool_and_a_json_rpc_error_are_both_normal_failures() {
        let dir = scratch("errors");
        let cli = fake_cli(&dir, LOOP);
        let mut client = McpStdio::open(&cli, 7979).unwrap();
        assert_eq!(client.call("tool_iserror", &serde_json::json!({})).unwrap(), (false, "no such node".to_string()));
        assert_eq!(client.call("tool_error", &serde_json::json!({})).unwrap(), (false, "unknown tool".to_string()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_child_that_dies_is_a_hard_transport_error() {
        let dir = scratch("death");
        let cli = fake_cli(&dir, "exit 0\n");
        assert!(matches!(McpStdio::open(&cli, 7979), Err(crate::AgentError::Setup(_))), "a dead child must not look ready");
        // …and a child that dies mid-call ends the call the same way.
        let cli = fake_cli(&dir, LOOP);
        let mut client = McpStdio::open(&cli, 7979).unwrap();
        let err = client.call("tool_exit", &serde_json::json!({})).unwrap_err();
        assert!(matches!(err, crate::AgentError::Setup(_)), "{err:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
