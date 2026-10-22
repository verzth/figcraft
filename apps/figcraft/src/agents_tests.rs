#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use std::net::{TcpListener, TcpStream};

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("figcraft-agents-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn wait_for(mut ok: impl FnMut() -> bool) {
    let start = Instant::now();
    while !ok() {
        assert!(start.elapsed() < Duration::from_secs(5), "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn the_entry_goes_in_and_out_leaving_every_other_setting() {
    let theirs = json!({"theme": "dark", "mcpServers": {"other": {"command": "x"}}});
    let entry = json!({"command": "/Applications/FigCraft.app/Contents/MacOS/FigCraft", "args": ["mcp", "--attach", "7979"]});
    let with = with_entry(Some(theirs.clone()), entry.clone());
    assert_eq!(with["theme"], "dark");
    assert_eq!(with["mcpServers"]["other"], json!({"command": "x"}));
    assert_eq!(with["mcpServers"][SERVER_NAME], entry);
    assert_eq!(entry_port(&with), Some(7979));
    assert_eq!(without_entry(with), theirs);
    // A file that isn't an object of servers gets one.
    assert_eq!(with_entry(Some(json!({"mcpServers": 3})), entry.clone())["mcpServers"][SERVER_NAME], entry);
    assert_eq!(with_entry(None, entry.clone())["mcpServers"][SERVER_NAME], entry);
    // host:port and numbers count too; anything else has no port.
    assert_eq!(entry_port(&json!({"mcpServers": {SERVER_NAME: {"args": ["mcp", "--attach", "127.0.0.1:8123"]}}})), Some(8123));
    assert_eq!(entry_port(&json!({"mcpServers": {SERVER_NAME: {"args": ["mcp", "--attach", 8124]}}})), Some(8124));
    assert_eq!(entry_port(&json!({"mcpServers": {SERVER_NAME: {"args": ["mcp"]}}})), None);
}

#[test]
fn installing_in_cursor_writes_its_settings_and_removing_takes_only_figcraft_out() {
    let home = scratch("cursor");
    let mut a = DesktopAgents::with_home(&home, Path::new("/opt/FigCraft/figcraft"));
    let cursor = |a: &mut DesktopAgents, port| a.clients(port).into_iter().find(|c| c.id == "cursor").unwrap();
    assert_eq!(cursor(&mut a, 7979).state, ClientState::Missing, "no ~/.cursor yet");
    let file = home.join(".cursor/mcp.json");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, r#"{"mcpServers": {"github": {"command": "gh-mcp"}}, "keep": true}"#).unwrap();
    assert_eq!(cursor(&mut a, 7979).state, ClientState::Available);

    a.install("cursor", 7979).unwrap();
    let v = read_json(&file).unwrap();
    assert_eq!(v["keep"], true);
    assert_eq!(v["mcpServers"]["github"]["command"], "gh-mcp");
    assert_eq!(v["mcpServers"][SERVER_NAME]["command"], "/opt/FigCraft/figcraft");
    assert_eq!(cursor(&mut a, 7979).state, ClientState::Installed { same_port: true });
    assert_eq!(cursor(&mut a, 8000).state, ClientState::Installed { same_port: false }, "another port asks for an update");
    assert!(cursor(&mut a, 7979).detail.starts_with("~/"), "{}", cursor(&mut a, 7979).detail);

    a.uninstall("cursor").unwrap();
    let v = read_json(&file).unwrap();
    assert!(v["mcpServers"].get(SERVER_NAME).is_none());
    assert_eq!(v["mcpServers"]["github"]["command"], "gh-mcp");
    assert_eq!(cursor(&mut a, 7979).state, ClientState::Available);
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn installing_in_opencode_writes_its_settings_and_removing_takes_only_figcraft_out() {
    let home = scratch("opencode");
    let mut a = DesktopAgents::with_home(&home, Path::new("/opt/FigCraft/figcraft"));
    let open = |a: &mut DesktopAgents, port| a.clients(port).into_iter().find(|c| c.id == "opencode").unwrap();
    assert!(!matches!(open(&mut a, 7979).state, ClientState::Installed { .. }), "no ~/.config/opencode yet");
    let file = home.join(".config/opencode/opencode.json");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, r#"{"mcp": {"github": {"type": "remote", "enabled": true, "url": "https://x"}}, "keep": true}"#).unwrap();

    a.install("opencode", 7979).unwrap();
    let v = read_json(&file).unwrap();
    assert_eq!(v["keep"], true);
    assert_eq!(v["mcp"]["github"]["type"], "remote");
    assert_eq!(v["mcp"][SERVER_NAME]["type"], "local");
    assert_eq!(v["mcp"][SERVER_NAME]["enabled"], true);
    assert_eq!(v["mcp"][SERVER_NAME]["command"], json!(["/opt/FigCraft/figcraft", "mcp", "--attach", "7979"]));
    assert_eq!(open(&mut a, 7979).state, ClientState::Installed { same_port: true });
    assert_eq!(open(&mut a, 8000).state, ClientState::Installed { same_port: false }, "another port asks for an update");
    assert!(!open(&mut a, 7979).detail.is_empty(), "the row names where it was found");

    a.uninstall("opencode").unwrap();
    let v = read_json(&file).unwrap();
    assert!(v["mcp"].get(SERVER_NAME).is_none());
    assert_eq!(v["mcp"]["github"]["type"], "remote");
    assert!(!matches!(open(&mut a, 7979).state, ClientState::Installed { .. }));
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn opencode_keeps_the_spelling_the_file_already_uses() {
    let entry = json!({"type": "local", "enabled": true, "command": ["/opt/FigCraft/figcraft", "mcp", "--attach", "8123"]});
    // A file with servers nested under `mcp` keeps that spelling, and so does the reading.
    let theirs = json!({"mcp": {"servers": {"github": {"command": "x"}}, "theme": "dark"}});
    let with = with_opencode_entry(Some(theirs.clone()), entry.clone());
    assert_eq!(with["mcp"]["servers"]["github"]["command"], "x");
    assert_eq!(with["mcp"]["servers"][SERVER_NAME], entry);
    assert_eq!(opencode_entry_port(&with), Some(8123));
    assert_eq!(without_opencode_entry(with), theirs);
    // A new file starts flat, as OpenCode writes it.
    let with = with_opencode_entry(None, entry.clone());
    assert_eq!(with["mcp"][SERVER_NAME], entry);
    assert_eq!(opencode_entry_port(&with), Some(8123));
    assert_eq!(without_opencode_entry(with), json!({"mcp": {}}));
    // A settings file that isn't an object of servers gets one.
    assert_eq!(with_opencode_entry(Some(json!({"mcp": 3})), entry.clone())["mcp"][SERVER_NAME], entry);
    // `host:port` counts too; anything else has no port.
    assert_eq!(opencode_entry_port(&json!({"mcp": {SERVER_NAME: {"command": ["x", "--attach", "127.0.0.1:8124"]}}})), Some(8124));
    assert_eq!(opencode_entry_port(&json!({"mcp": {"servers": {SERVER_NAME: {"command": ["x", "mcp"]}}}})), None);
}

#[test]
fn a_codex_entry_is_read_from_its_config_toml() {
    let home = scratch("codex");
    let mut a = DesktopAgents::with_home(&home, Path::new("/opt/FigCraft/figcraft"));
    let codex = |a: &mut DesktopAgents, port| a.clients(port).into_iter().find(|c| c.id == "codex").unwrap();
    assert!(!matches!(codex(&mut a, 7979).state, ClientState::Installed { .. }), "no config.toml yet");
    std::fs::create_dir_all(&a.codex_home).unwrap();
    std::fs::write(
        a.codex_home.join("config.toml"),
        "[mcp_servers.other]\ncommand = \"npx\"\nargs = [\"-y\", \"other-mcp\"]\n\n[mcp_servers.figcraft]\ncommand = \"/opt/FigCraft/figcraft\"\nargs = [\"mcp\", \"--attach\", \"7979\"]\n",
    )
    .unwrap();
    assert_eq!(codex(&mut a, 7979).state, ClientState::Installed { same_port: true });
    assert_eq!(codex(&mut a, 8000).state, ClientState::Installed { same_port: false }, "another port asks for an update");
    assert!(!codex(&mut a, 7979).detail.is_empty(), "the row names where it was found");
    // A file Codex left half-written is just "no entry here", never an error.
    std::fs::write(a.codex_home.join("config.toml"), "this is not toml [[[").unwrap();
    assert!(!matches!(codex(&mut a, 7979).state, ClientState::Installed { .. }));
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_settings_file_that_isnt_json_is_never_overwritten() {
    let home = scratch("broken");
    let mut a = DesktopAgents::with_home(&home, Path::new("/opt/FigCraft/figcraft"));
    let file = home.join(".cursor/mcp.json");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, "{ this is not json").unwrap();
    let e = a.install("cursor", 7979).unwrap_err();
    assert!(e.contains("isn't valid JSON"), "{e}");
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "{ this is not json");
    assert!(a.uninstall("cursor").is_err());
    // The same for OpenCode's file, and its `opencode.jsonc` is left to the person who wrote it.
    let open = home.join(".config/opencode/opencode.json");
    std::fs::create_dir_all(open.parent().unwrap()).unwrap();
    std::fs::write(&open, "{ this is not json").unwrap();
    let e = a.install("opencode", 7979).unwrap_err();
    assert!(e.contains("isn't valid JSON"), "{e}");
    assert_eq!(std::fs::read_to_string(&open).unwrap(), "{ this is not json");
    assert!(a.uninstall("opencode").is_err());
    std::fs::remove_file(&open).unwrap();
    let jsonc = home.join(".config/opencode/opencode.jsonc");
    std::fs::write(&jsonc, "// comments and all").unwrap();
    let e = a.install("opencode", 7979).unwrap_err();
    assert!(e.contains("opencode.jsonc"), "{e}");
    assert_eq!(std::fs::read_to_string(&jsonc).unwrap(), "// comments and all");
    assert!(a.install("no-such-client", 7979).is_err());
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn the_config_to_copy_runs_this_app_attached_to_the_port() {
    let home = scratch("config");
    let a = DesktopAgents::with_home(&home, Path::new("/opt/FigCraft/figcraft"));
    let v: Value = serde_json::from_str(&a.config_json(8123)).unwrap();
    assert_eq!(v["mcpServers"][SERVER_NAME], json!({"command": "/opt/FigCraft/figcraft", "args": ["mcp", "--attach", "8123"]}));
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn the_server_starts_counts_connections_disconnects_and_frees_its_port() {
    let home = scratch("server");
    let mut a = DesktopAgents::with_home(&home, Path::new("figcraft"));
    let port = a.free_port(20_000).unwrap();
    a.start(port).unwrap();
    assert_eq!(a.status().listening, Some(port));
    let c1 = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let c2 = TcpStream::connect(("127.0.0.1", port)).unwrap();
    wait_for(|| a.status().connections == 2);
    drop(c1);
    wait_for(|| a.status().connections == 1);
    a.disconnect();
    wait_for(|| a.status().connections == 0);
    assert_eq!(a.status().listening, Some(port), "disconnecting keeps listening");
    // A client closes its end once the server closed the connection (as `figcraft mcp` does); until
    // then the server's end waits for it, and macOS keeps the port.
    drop(c2);
    a.stop();
    assert_eq!(a.status(), ServerStatus::default());
    // Free once the kernel has finished closing the connections (milliseconds on loopback).
    wait_for(|| TcpListener::bind(("127.0.0.1", port)).is_ok());
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn a_busy_port_is_reported_and_a_free_one_offered() {
    let home = scratch("busy");
    let mut a = DesktopAgents::with_home(&home, Path::new("figcraft"));
    // Below the ephemeral range, so the ports next to it aren't handed to other tests' connections.
    let port = a.free_port(21_000).unwrap();
    let _taken = TcpListener::bind(("127.0.0.1", port)).unwrap();
    assert!(a.start(port).is_err());
    let s = a.status();
    assert_eq!((s.listening, s.error_port), (None, Some(port)));
    assert!(s.error.is_some());
    let free = a.free_port(port).unwrap();
    assert_ne!(free, port);
    a.start(free).unwrap();
    assert_eq!(a.status().error, None, "starting clears the error");
    // Moving to a taken port keeps listening on the old one, with the error to show.
    assert!(a.start(port).is_err());
    assert_eq!((a.status().listening, a.status().error_port), (Some(free), Some(port)));
    let still = TcpStream::connect(("127.0.0.1", free));
    assert!(still.is_ok(), "the old port still answers");
    a.stop();
    let _ = std::fs::remove_dir_all(&home);
}
