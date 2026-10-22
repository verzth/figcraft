//! Preferences ▸ AI agents, the desktop side: the control server MCP clients reach the app through,
//! and FigCraft's entry in the clients' settings (Claude Code, Codex, OpenCode, Claude Desktop,
//! Cursor).
//!
//! A client runs `<this app> mcp --attach PORT` (main.rs), the MCP server attached to this window.
//! Claude Code and Codex are set up through their own CLIs (`claude mcp add`, `codex mcp add`),
//! never by editing their settings files, which a running client rewrites; Claude Desktop, Cursor
//! and OpenCode keep their servers in a JSON file that only gains or loses FigCraft's entry,
//! everything else as it was.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use figcraft_ui_egui::ControlRequest;
use figcraft_ui_egui::agents::{AgentHost, BackendKind, Client, ClientState, PanelBackend, ServerStatus};
use serde_json::{Map, Value, json};

use crate::control_server::ControlServer;

type Queue = Arc<Mutex<Vec<ControlRequest>>>;
type CtxSlot = Arc<Mutex<Option<egui::Context>>>;

/// The server's name in the clients' settings.
pub const SERVER_NAME: &str = "figcraft";
/// How long `claude mcp …` may take before the install gives up.
const CLI_TIMEOUT: Duration = Duration::from_secs(20);
/// Ports [`DesktopAgents::free_port`] tries.
const PORT_SEARCH: u16 = 50;

pub struct DesktopAgents {
    queue: Queue,
    ctx: CtxSlot,
    server: Option<ControlServer>,
    error: Option<(u16, String)>,
    /// Where the clients' settings and tools live (the home folder; tests use a scratch one).
    home: Option<PathBuf>,
    /// This app's executable, which the clients run as `… mcp --attach PORT`.
    exe: PathBuf,
    /// Codex's settings folder: `CODEX_HOME` when it is set, else `<home>/.codex` — the folder
    /// `codex` itself reads and writes, so the row here and what the CLI does never disagree.
    codex_home: PathBuf,
    /// The direct-API keys from the config folder (AgentKeys.json), loaded at startup.
    keys: figcraft_agent::ApiKeys,
}

impl DesktopAgents {
    pub fn new(queue: Queue, ctx: CtxSlot) -> Self {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("figcraft"));
        let codex_home = std::env::var_os("CODEX_HOME").map(PathBuf::from).unwrap_or_else(|| home.clone().unwrap_or_default().join(".codex"));
        let keys = crate::services::config_dir().map(|d| figcraft_agent::ApiKeys::load(&d.join(figcraft_agent::FILE_NAME))).unwrap_or_default();
        Self { queue, ctx, server: None, error: None, home, exe, codex_home, keys }
    }

    #[cfg(test)]
    fn with_home(home: &Path, exe: &Path) -> Self {
        // The scratch home keeps its own keys empty: a test never reads the real keyring.
        Self {
            queue: Arc::default(),
            ctx: Arc::default(),
            server: None,
            error: None,
            home: Some(home.to_path_buf()),
            exe: exe.to_path_buf(),
            codex_home: home.join(".codex"),
            keys: figcraft_agent::ApiKeys::default(),
        }
    }

    /// The command a client runs to reach this app on `port`.
    fn entry(&self, port: u16) -> Value {
        json!({"command": self.exe.to_string_lossy(), "args": ["mcp", "--attach", port.to_string()]})
    }

    /// The same command in OpenCode's shape: one `command` array, a type and an enabled flag.
    fn opencode_entry(&self, port: u16) -> Value {
        json!({"type": "local", "enabled": true, "command": [self.exe.to_string_lossy(), "mcp", "--attach", port.to_string()]})
    }

    fn json_config(&self, id: &str) -> Option<PathBuf> {
        let home = self.home.as_deref()?;
        match id {
            "claude-desktop" => Some(claude_desktop_dir(home)?.join("claude_desktop_config.json")),
            "cursor" => Some(home.join(".cursor").join("mcp.json")),
            _ => None,
        }
    }

    fn claude_code(&self, port: u16) -> Client {
        let bin = self.home.as_deref().and_then(find_claude);
        let installed = self.home.as_deref().and_then(|h| read_json(&h.join(".claude.json")).ok()).and_then(|v| entry_port(&v));
        let state = match (installed, &bin) {
            (Some(p), _) => ClientState::Installed { same_port: p == port },
            (None, Some(_)) => ClientState::Available,
            (None, None) => ClientState::Missing,
        };
        let detail = bin.map(|b| tilde(self.home.as_deref(), &b)).unwrap_or_default();
        Client { id: "claude-code", name: "Claude Code", state, detail }
    }

    fn json_client(&self, id: &'static str, name: &'static str, port: u16) -> Client {
        let Some(path) = self.json_config(id) else {
            return Client { id, name, state: ClientState::Missing, detail: String::new() };
        };
        // The client's folder exists once it's installed and has run.
        let present = path.parent().is_some_and(Path::is_dir);
        let state = match read_json(&path).ok().and_then(|v| entry_port(&v)) {
            Some(p) => ClientState::Installed { same_port: p == port },
            None if present => ClientState::Available,
            None => ClientState::Missing,
        };
        Client { id, name, state, detail: tilde(self.home.as_deref(), &path) }
    }

    fn claude_cli(&self, args: &[&str]) -> Result<String, String> {
        let bin = self.home.as_deref().and_then(find_claude).ok_or("Claude Code's `claude` command wasn't found")?;
        run(Command::new(bin).args(args))
    }

    /// Codex: its MCP servers live in `config.toml` (`CODEX_HOME`, else `~/.codex`). The file is
    /// only read here — `codex mcp add` / `remove` write it, since a running Codex rewrites it.
    fn codex(&self, port: u16) -> Client {
        let home = self.home.as_deref();
        let bin = home.and_then(|h| find_command(h, "codex", &[]));
        let file = self.codex_home.join("config.toml");
        let installed = read_toml(&file).and_then(|v| toml_entry_port(&v));
        let state = match (installed, &bin) {
            (Some(p), _) => ClientState::Installed { same_port: p == port },
            (None, Some(_)) => ClientState::Available,
            (None, None) => ClientState::Missing,
        };
        let detail = match (bin, file) {
            (Some(b), _) => tilde(home, &b),
            (None, f) => tilde(home, &f),
        };
        Client { id: "codex", name: "Codex", state, detail }
    }

    /// OpenCode: its MCP servers live in the global `opencode.json`, edited here — its CLI adds
    /// servers but has none for taking one out again.
    fn opencode(&self, port: u16) -> Client {
        let home = self.home.as_deref();
        let bin = home.and_then(|h| find_command(h, "opencode", &[]));
        let file = opencode_config(home);
        let installed = file.as_deref().and_then(|p| read_json(p).ok()).and_then(|v| opencode_entry_port(&v));
        let state = match (installed, &bin) {
            (Some(p), _) => ClientState::Installed { same_port: p == port },
            (None, Some(_)) => ClientState::Available,
            (None, None) => ClientState::Missing,
        };
        let detail = match (bin, file) {
            (Some(b), _) => tilde(home, &b),
            (None, Some(f)) => tilde(home, &f),
            (None, None) => String::new(),
        };
        Client { id: "opencode", name: "OpenCode", state, detail }
    }
}

impl AgentHost for DesktopAgents {
    fn status(&self) -> ServerStatus {
        ServerStatus {
            listening: self.server.as_ref().map(ControlServer::port),
            connections: self.server.as_ref().map_or(0, ControlServer::connections),
            error: self.error.as_ref().map(|(_, e)| e.clone()),
            error_port: self.error.as_ref().map(|(p, _)| *p),
        }
    }

    fn start(&mut self, port: u16) -> Result<(), String> {
        if self.server.as_ref().is_some_and(|s| s.port() == port) {
            return Ok(());
        }
        // The new port first: when it's taken, the agents connected on the old one stay connected.
        match ControlServer::start(port, self.queue.clone(), self.ctx.clone()) {
            Ok(s) => {
                if let Some(old) = self.server.replace(s) {
                    old.stop();
                }
                self.error = None;
                Ok(())
            }
            Err(e) => {
                log::error!("control server: can't listen on 127.0.0.1:{port}: {e}");
                self.error = Some((port, e.to_string()));
                Err(e.to_string())
            }
        }
    }

    fn stop(&mut self) {
        if let Some(s) = self.server.take() {
            s.stop();
        }
        self.error = None;
    }

    fn disconnect(&mut self) {
        if let Some(s) = &self.server {
            s.disconnect();
        }
    }

    fn free_port(&self, from: u16) -> Option<u16> {
        (from..from.saturating_add(PORT_SEARCH)).find(|p| std::net::TcpListener::bind(("127.0.0.1", *p)).is_ok())
    }

    fn clients(&mut self, port: u16) -> Vec<Client> {
        vec![
            self.claude_code(port),
            self.codex(port),
            self.opencode(port),
            self.json_client("claude-desktop", "Claude Desktop", port),
            self.json_client("cursor", "Cursor", port),
        ]
    }

    fn install(&mut self, id: &str, port: u16) -> Result<String, String> {
        match id {
            "claude-code" => {
                // A FigCraft entry with another port goes first (`add` refuses an existing name).
                let _ = self.claude_cli(&["mcp", "remove", SERVER_NAME, "-s", "user"]);
                let exe = self.exe.to_string_lossy().into_owned();
                let port = port.to_string();
                self.claude_cli(&["mcp", "add", SERVER_NAME, "-s", "user", "--", &exe, "mcp", "--attach", &port])?;
                Ok("FigCraft is installed in Claude Code: start a new Claude Code session to use it".into())
            }
            "claude-desktop" | "cursor" => {
                let path = self.json_config(id).ok_or("the settings folder wasn't found")?;
                let current = match read_json(&path) {
                    Ok(v) => Some(v),
                    Err(ReadError::Missing) => None,
                    Err(ReadError::Invalid(e)) => return Err(format!("{} isn't valid JSON ({e}): fix or remove it first", path.display())),
                };
                write_json(&path, &with_entry(current, self.entry(port)))?;
                Ok(if id == "cursor" {
                    "FigCraft is installed in Cursor: it shows under Settings ▸ MCP".into()
                } else {
                    "FigCraft is installed in Claude Desktop: quit and reopen Claude Desktop to load it".into()
                })
            }
            "codex" => {
                let bin = self.home.as_deref().and_then(|h| find_command(h, "codex", &[])).ok_or("Codex's `codex` command wasn't found")?;
                // A FigCraft entry with another port goes first (`add` keeps the name it is given).
                let _ = run(Command::new(&bin).args(["mcp", "remove", SERVER_NAME]));
                let exe = self.exe.to_string_lossy().into_owned();
                let port = port.to_string();
                run(Command::new(&bin).args(["mcp", "add", SERVER_NAME, "--", &exe, "mcp", "--attach", &port]))?;
                Ok("FigCraft is installed in Codex: start a new Codex session to use it".into())
            }
            "opencode" => {
                let path = opencode_config(self.home.as_deref()).ok_or("the settings folder wasn't found")?;
                let jsonc = path.with_extension("jsonc");
                if !path.exists() && jsonc.exists() {
                    return Err(format!(
                        "{} holds OpenCode's settings: add the entry there by hand, or copy the settings",
                        tilde(self.home.as_deref(), &jsonc)
                    ));
                }
                let current = match read_json(&path) {
                    Ok(v) => Some(v),
                    Err(ReadError::Missing) => None,
                    Err(ReadError::Invalid(e)) => return Err(format!("{} isn't valid JSON ({e}): fix or remove it first", path.display())),
                };
                write_json(&path, &with_opencode_entry(current, self.opencode_entry(port)))?;
                Ok("FigCraft is installed in OpenCode: start a new OpenCode session to use it".into())
            }
            _ => Err(format!("unknown client `{id}`")),
        }
    }

    fn uninstall(&mut self, id: &str) -> Result<(), String> {
        match id {
            "claude-code" => self.claude_cli(&["mcp", "remove", SERVER_NAME, "-s", "user"]).map(|_| ()),
            "claude-desktop" | "cursor" => {
                let path = self.json_config(id).ok_or("the settings folder wasn't found")?;
                match read_json(&path) {
                    Ok(v) => write_json(&path, &without_entry(v)),
                    Err(ReadError::Missing) => Ok(()),
                    Err(ReadError::Invalid(e)) => Err(format!("{} isn't valid JSON ({e})", path.display())),
                }
            }
            "codex" => {
                let bin = self.home.as_deref().and_then(|h| find_command(h, "codex", &[])).ok_or("Codex's `codex` command wasn't found")?;
                run(Command::new(&bin).args(["mcp", "remove", SERVER_NAME])).map(|_| ())
            }
            "opencode" => {
                let path = opencode_config(self.home.as_deref()).ok_or("the settings folder wasn't found")?;
                match read_json(&path) {
                    Ok(v) => write_json(&path, &without_opencode_entry(v)),
                    Err(ReadError::Missing) => Ok(()),
                    Err(ReadError::Invalid(e)) => Err(format!("{} isn't valid JSON ({e})", path.display())),
                }
            }
            _ => Err(format!("unknown client `{id}`")),
        }
    }

    fn config_json(&self, port: u16) -> String {
        serde_json::to_string_pretty(&json!({"mcpServers": {SERVER_NAME: self.entry(port)}})).unwrap_or_default()
    }

    /// The Agent tab's backends, in the order the tab prefers them (empty when this
    /// computer has none). The direct-API backends join the list only while the prefs file
    /// holds their key — a key typed in Preferences ▸ AI agents shows up on the next scan.
    fn panel_backends(&mut self) -> Vec<PanelBackend> {
        let mut found: Vec<PanelBackend> = Vec::new();
        if let Some(home) = self.home.as_deref() {
            let claude_extra = [home.join(".claude/local")];
            found.extend(
                [
                    (BackendKind::ClaudeCode, "claude", &claude_extra[..]),
                    (BackendKind::OpenCode, "opencode", &[][..]),
                    (BackendKind::Codex, "codex", &[][..]),
                ]
                .iter()
                .filter_map(|(kind, cmd, extra)| find_command(home, cmd, extra).map(|bin| PanelBackend { kind: *kind, bin: Some(bin) })),
            );
        }
        if self.keys.anthropic.as_deref().is_some_and(|k| !k.trim().is_empty()) {
            found.push(PanelBackend { kind: BackendKind::AnthropicApi, bin: None });
        }
        if self.keys.openai.as_deref().is_some_and(|k| !k.trim().is_empty()) {
            found.push(PanelBackend { kind: BackendKind::OpenAiApi, bin: None });
        }
        found
    }

    /// This app's executable, which the backends run as `… mcp --attach PORT`.
    fn mcp_exe(&self) -> Option<PathBuf> {
        Some(self.exe.clone())
    }

    /// The direct-API keys from AgentKeys.json (loaded at startup).
    fn agent_keys(&mut self) -> Option<figcraft_agent::ApiKeys> {
        Some(self.keys.clone())
    }

    /// Write the keys to AgentKeys.json (atomically, owner-only) and keep them for the
    /// backends this session.
    fn set_agent_keys(&mut self, keys: figcraft_agent::ApiKeys) -> Result<(), String> {
        let dir = crate::services::config_dir().ok_or_else(|| "the configuration folder wasn't found".to_string())?;
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        keys.save(&dir.join(figcraft_agent::FILE_NAME)).map_err(|e| e.to_string())?;
        self.keys = keys;
        Ok(())
    }
}

/// Claude Desktop's settings folder on this platform.
fn claude_desktop_dir(home: &Path) -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        Some(home.join("Library/Application Support/Claude"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("Claude"))
    } else {
        Some(std::env::var_os("XDG_CONFIG_HOME").map_or_else(|| home.join(".config"), PathBuf::from).join("Claude"))
    }
}

/// Claude Code's `claude` command wherever it is.
fn find_claude(home: &Path) -> Option<PathBuf> {
    find_command(home, "claude", &[home.join(".claude/local")])
}

/// A command: on the PATH, or where its installers put it (an app started from the Finder or
/// the Dock has a short PATH) — including nvm's Node, where the npm globals (`opencode`,
/// `codex`, `claude`) live.
fn find_command(home: &Path, name: &str, extra: &[PathBuf]) -> Option<PathBuf> {
    let names: Vec<String> = if cfg!(windows) { vec![format!("{name}.exe"), format!("{name}.cmd")] } else { vec![name.to_string()] };
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default();
    dirs.extend([home.join(".local/bin"), home.join(".npm-global/bin"), PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin")]);
    dirs.extend(extra.iter().cloned());
    dirs.extend(nvm_bins(home));
    dirs.iter().flat_map(|d| names.iter().map(move |n| d.join(n))).find(|p| p.is_file())
}

/// The `bin` of every Node version nvm installed; the newest sorts first on a version that
/// sorts last lexicographically ("v9" > "v10"), which only matters when a command is in both.
fn nvm_bins(home: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(home.join(".nvm/versions/node")) else { return Vec::new() };
    let mut bins: Vec<PathBuf> = entries.flatten().map(|e| e.path().join("bin")).filter(|p| p.is_dir()).collect();
    bins.sort();
    bins.reverse();
    bins
}

/// `~/…` for paths in the home folder, as people read them.
fn tilde(home: Option<&Path>, p: &Path) -> String {
    match home.and_then(|h| p.strip_prefix(h).ok()) {
        Some(rest) => format!("~/{}", rest.display()),
        None => p.display().to_string(),
    }
}

#[derive(Debug)]
pub enum ReadError {
    Missing,
    Invalid(String),
}

/// Settings files bigger than this aren't settings.
const MAX_SETTINGS: u64 = 16 * 1024 * 1024;

fn read_json(path: &Path) -> Result<Value, ReadError> {
    let meta = std::fs::metadata(path).map_err(|_| ReadError::Missing)?;
    if meta.len() > MAX_SETTINGS {
        return Err(ReadError::Invalid("too large".into()));
    }
    let text = std::fs::read_to_string(path).map_err(|e| ReadError::Invalid(e.to_string()))?;
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(&text).map_err(|e| ReadError::Invalid(e.to_string()))
}

/// Write `v` next to `path` and move it into place, so a crash never leaves half a file.
fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension("json.figcraft-tmp");
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, text + "\n").map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

/// `settings` with FigCraft's server entry set, every other key kept.
pub fn with_entry(settings: Option<Value>, entry: Value) -> Value {
    let mut root = match settings {
        Some(Value::Object(o)) => o,
        _ => Map::new(),
    };
    let servers = root.entry("mcpServers").or_insert_with(|| json!({}));
    if !servers.is_object() {
        *servers = json!({});
    }
    if let Some(s) = servers.as_object_mut() {
        s.insert(SERVER_NAME.into(), entry);
    }
    Value::Object(root)
}

/// `settings` without FigCraft's server entry, every other key kept.
pub fn without_entry(mut settings: Value) -> Value {
    if let Some(s) = settings.get_mut("mcpServers").and_then(Value::as_object_mut) {
        s.remove(SERVER_NAME);
    }
    settings
}

/// The port FigCraft's entry in `settings` attaches to, if it has one.
pub fn entry_port(settings: &Value) -> Option<u16> {
    let args = settings.get("mcpServers")?.get(SERVER_NAME)?.get("args")?.as_array()?;
    let i = args.iter().position(|a| a == "--attach")?;
    let port = args.get(i + 1)?;
    port.as_str().and_then(|p| p.rsplit(':').next()?.parse().ok()).or_else(|| port.as_u64().and_then(|p| u16::try_from(p).ok()))
}

/// OpenCode's global settings file: the one `opencode mcp add --global` writes.
fn opencode_config(home: Option<&Path>) -> Option<PathBuf> {
    Some(home?.join(".config/opencode/opencode.json"))
}

/// FigCraft's entry in OpenCode's `mcp` section: OpenCode reads servers both as `mcp.<name>` and
/// as `mcp.servers.<name>`, so both are looked for here.
fn opencode_server(settings: &Value) -> Option<&Value> {
    let mcp = settings.get("mcp")?;
    mcp.get(SERVER_NAME).or_else(|| mcp.get("servers")?.get(SERVER_NAME))
}

/// The port FigCraft's OpenCode entry attaches to, if it has one.
pub fn opencode_entry_port(settings: &Value) -> Option<u16> {
    let words = opencode_server(settings)?.get("command")?.as_array()?;
    let words: Vec<&str> = words.iter().filter_map(Value::as_str).collect();
    let i = words.iter().position(|w| *w == "--attach")?;
    words.get(i + 1)?.rsplit(':').next()?.parse().ok()
}

/// `settings` with FigCraft's server entry set in OpenCode's spelling, every other key kept.
pub fn with_opencode_entry(settings: Option<Value>, entry: Value) -> Value {
    let mut root = match settings {
        Some(Value::Object(o)) => o,
        _ => Map::new(),
    };
    let mcp = root.entry("mcp").or_insert_with(|| json!({}));
    if !mcp.is_object() {
        *mcp = json!({});
    }
    // The spelling the file already uses; a new one starts flat, as OpenCode writes it.
    let nested = mcp.get("servers").is_some_and(Value::is_object);
    if nested {
        if let Some(servers) = mcp.get_mut("servers").and_then(Value::as_object_mut) {
            servers.insert(SERVER_NAME.into(), entry);
        }
    } else if let Some(mcp) = mcp.as_object_mut() {
        mcp.insert(SERVER_NAME.into(), entry);
    }
    Value::Object(root)
}

/// `settings` without FigCraft's server entry, every other key kept.
pub fn without_opencode_entry(mut settings: Value) -> Value {
    if let Some(mcp) = settings.get_mut("mcp") {
        if let Some(servers) = mcp.get_mut("servers").and_then(Value::as_object_mut) {
            servers.remove(SERVER_NAME);
        }
        if let Some(mcp) = mcp.as_object_mut() {
            mcp.remove(SERVER_NAME);
        }
    }
    settings
}

/// Codex's `config.toml`, read with the same TOML dialect its CLI writes. A file that doesn't
/// parse is simply "no entry here": `codex mcp add` writes it, this never does.
fn read_toml(path: &Path) -> Option<toml::Value> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > MAX_SETTINGS {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

/// The port FigCraft's entry in a Codex `config.toml` attaches to, if it has one.
fn toml_entry_port(v: &toml::Value) -> Option<u16> {
    let args = v.get("mcp_servers")?.get(SERVER_NAME)?.get("args")?.as_array()?;
    let words: Vec<String> = args
        .iter()
        .filter_map(|a| match a {
            toml::Value::String(s) => Some(s.clone()),
            toml::Value::Integer(i) => Some(i.to_string()),
            _ => None,
        })
        .collect();
    let i = words.iter().position(|w| w == "--attach")?;
    words.get(i + 1)?.rsplit(':').next()?.parse().ok()
}

/// Run a command, its output as the error when it fails, giving up after [`CLI_TIMEOUT`].
fn run(cmd: &mut Command) -> Result<String, String> {
    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() > CLI_TIMEOUT => {
                let _ = child.kill();
                return Err(format!("no answer after {} s", CLI_TIMEOUT.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(e) => return Err(e.to_string()),
        }
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let text = |b: &[u8]| String::from_utf8_lossy(b).trim().to_string();
    if out.status.success() {
        Ok(text(&out.stdout))
    } else {
        Err(Some(text(&out.stderr)).filter(|s| !s.is_empty()).unwrap_or_else(|| text(&out.stdout)))
    }
}

#[cfg(test)]
#[path = "agents_tests.rs"]
mod tests;
