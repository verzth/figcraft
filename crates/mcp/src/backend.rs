//! Where MCP tool calls end up: a control-channel method call, answered either by an in-process
//! engine session ([`crate::Headless`]) or by a running FigCraft app ([`Attached`]).

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use base64::Engine as _;
use serde_json::{Value, json};

/// Something that answers the FigCraft control-channel methods (`engine.execute`,
/// `engine.commands`, `document.inspect`, `node.get`, `ui.pointer`, `ui.key`, `ui.type`,
/// `ui.render`, `ui.screenshot`, `ui.inspect`). See `docs/control-protocol.md`.
pub trait Backend {
    /// Call one method. `Ok` carries the `result`, `Err` the error message.
    fn call(&mut self, method: &str, params: Value) -> Result<Value, String>;

    /// True when a real UI window is attached (`ui.screenshot` captures the window).
    fn has_ui(&self) -> bool;

    /// Short human description ("headless", "attached to 127.0.0.1:7979").
    fn describe(&self) -> String;

    /// The MCP client that connected (`clientInfo` from `initialize`). An attached app shows it;
    /// best effort: an app that doesn't know the method just says so.
    fn hello(&mut self, _client: &str, _version: &str) {}

    /// PNG bytes of the document: `ids` (in page coordinates, their union) or, when empty, the
    /// current page's content. The default goes through `ui.render`, the PNG coming back as data.
    fn render_png(&mut self, ids: &[u64], scale: f64) -> Result<Vec<u8>, String> {
        let mut params = json!({"scale": scale});
        if !ids.is_empty() {
            params["ids"] = json!(ids);
        } else {
            // `ui.render` without ids exports the selection; pass the page's children instead.
            let kids = page_children(self)?;
            if kids.is_empty() {
                return Err("the current page is empty: nothing to render".into());
            }
            params["ids"] = json!(kids);
        }
        decode_data(&self.call("ui.render", params)?)
    }

    /// PNG bytes of what the user sees: the app window when attached, the current viewport when
    /// headless.
    fn screenshot_png(&mut self) -> Result<Vec<u8>, String> {
        capture(self, json!({}))
    }

    /// PNG bytes of the attached app's window `window` (an id from `window.list`).
    fn window_png(&mut self, window: u64) -> Result<Vec<u8>, String> {
        capture(self, json!({"window": window}))
    }
}

/// A window capture, the PNG coming back as data (no file: an app confined to automation roots
/// writes none outside them).
fn capture<B: Backend + ?Sized>(b: &mut B, params: Value) -> Result<Vec<u8>, String> {
    decode_data(&b.call("ui.screenshot", params)?)
}

/// The ids of the current page's top-level layers.
pub fn page_children<B: Backend + ?Sized>(b: &mut B) -> Result<Vec<u64>, String> {
    let doc = b.call("document.inspect", json!({"depth": 1}))?;
    let current = doc.get("currentPage").cloned().unwrap_or(Value::Null);
    let pages = doc.get("pages").and_then(Value::as_array).cloned().unwrap_or_default();
    let page = pages.iter().find(|p| p.get("id") == Some(&current)).or_else(|| pages.first());
    Ok(page
        .and_then(|p| p.get("children"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|c| c.get("id").and_then(Value::as_u64)).collect())
        .unwrap_or_default())
}

/// Decode a `data` field (base64, optionally a data: URL) from an export result.
pub fn decode_data(v: &Value) -> Result<Vec<u8>, String> {
    let d = v.get("data").and_then(Value::as_str).ok_or("the export returned no data")?;
    let d = d.split_once(',').map(|(_, b)| b).unwrap_or(d);
    base64::engine::general_purpose::STANDARD.decode(d.trim()).map_err(|e| format!("bad base64 from the app: {e}"))
}

/// A running FigCraft app, reached through its loopback control port (`figcraft --control PORT`).
pub struct Attached {
    addr: String,
    conn: Option<(BufReader<TcpStream>, TcpStream)>,
    next_id: u64,
    /// How long the handshake's hello may take.
    hello_timeout: Duration,
}

/// How long the app may take to answer a call. The app's own timeout is 60 s; leave headroom.
const READ_TIMEOUT: Duration = Duration::from_secs(90);

/// How long the handshake's `app.hello` may take: the client gives up long before the app's own
/// timeout, so a wedged app must not hold the handshake for 60 s. The app answers it at once (it
/// needs no frame), so a couple of seconds is generous.
const HELLO_TIMEOUT: Duration = Duration::from_secs(5);

impl Attached {
    /// Connect to `addr` (e.g. `127.0.0.1:7979`), failing fast when nothing is listening.
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let mut r = Self { addr: addr.to_string(), conn: None, next_id: 1, hello_timeout: HELLO_TIMEOUT };
        r.reconnect()?;
        Ok(r)
    }

    /// Connect to `127.0.0.1:port`.
    pub fn port(port: u16) -> std::io::Result<Self> {
        Self::connect(&format!("127.0.0.1:{port}"))
    }

    /// Give the handshake's hello `timeout` to be answered.
    pub fn with_hello_timeout(mut self, timeout: Duration) -> Self {
        self.hello_timeout = timeout;
        self
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    fn reconnect(&mut self) -> std::io::Result<()> {
        self.conn = None;
        let mut last = std::io::Error::new(std::io::ErrorKind::NotFound, format!("cannot resolve {}", self.addr));
        for sa in self.addr.to_socket_addrs()? {
            match TcpStream::connect_timeout(&sa, Duration::from_millis(800)) {
                Ok(s) => {
                    s.set_nodelay(true).ok();
                    s.set_read_timeout(Some(READ_TIMEOUT)).ok();
                    let read = s.try_clone()?;
                    self.conn = Some((BufReader::new(read), s));
                    return Ok(());
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    /// The app already closed this connection (it drops ones idle past its socket timeout), or
    /// left bytes on it nobody asked for. Checked before a request is written, when replacing the
    /// connection can't repeat anything.
    fn stale(&mut self) -> bool {
        let Some((reader, writer)) = self.conn.as_mut() else { return false };
        if !reader.buffer().is_empty() || writer.set_nonblocking(true).is_err() {
            return true;
        }
        // Nothing to read on a healthy idle connection: the peek would block.
        let idle = matches!(writer.peek(&mut [0]), Err(e) if e.kind() == std::io::ErrorKind::WouldBlock);
        writer.set_nonblocking(false).is_err() || !idle
    }

    fn roundtrip(&mut self, line: &str) -> std::io::Result<String> {
        self.roundtrip_with(line, None)
    }

    /// Send `line` and read the reply. With `timeout`, the read gives up after it: the read
    /// timeout lives on the socket, so it covers the reader too.
    fn roundtrip_with(&mut self, line: &str, timeout: Option<Duration>) -> std::io::Result<String> {
        let Some((reader, writer)) = self.conn.as_mut() else {
            return Err(std::io::Error::new(std::io::ErrorKind::NotConnected, "not connected to the app"));
        };
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if let Some(t) = timeout {
            reader.get_ref().set_read_timeout(Some(t)).ok();
        }
        let mut reply = String::new();
        let read = reader.read_line(&mut reply);
        if timeout.is_some() {
            reader.get_ref().set_read_timeout(Some(READ_TIMEOUT)).ok();
        }
        if read? == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "control channel closed"));
        }
        Ok(reply)
    }
}

impl Backend for Attached {
    fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let line = json!({"id": id, "method": method, "params": params}).to_string();
        if self.conn.is_none() || self.stale() {
            self.reconnect().map_err(|e| {
                format!("FigCraft app at {} is not reachable: {e} (request was not sent; start it with `figcraft --control PORT`)", self.addr)
            })?;
        }
        // A lost reply does not mean the app failed to apply the request.
        let reply = self.roundtrip(&line).map_err(|e| {
            self.conn = None;
            format!(
                "FigCraft control call `{method}` at {} failed: {e}; the operation may have completed. Inspect the document before retrying.",
                self.addr
            )
        })?;
        let v: Value = serde_json::from_str(reply.trim()).map_err(|e| format!("bad reply from the app: {e}"))?;
        if v.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(v.get("result").cloned().unwrap_or(Value::Null))
        } else {
            Err(v.get("error").and_then(Value::as_str).unwrap_or("unknown error").to_string())
        }
    }

    fn has_ui(&self) -> bool {
        true
    }

    fn describe(&self) -> String {
        format!("attached to the FigCraft app at {}", self.addr)
    }

    fn hello(&mut self, client: &str, version: &str) {
        // The handshake's hello must not hang the client: the app answers it at once (it needs no
        // frame), so a short deadline is enough. A late reply would desync the stream, so the
        // connection is dropped instead of waiting for it.
        let id = self.next_id;
        self.next_id += 1;
        let line = json!({"id": id, "method": "app.hello", "params": {"client": client, "version": version}}).to_string();
        if (self.conn.is_none() || self.stale())
            && let Err(e) = self.reconnect()
        {
            log::debug!("the app is not reachable for the hello: {e}");
            return;
        }
        match self.roundtrip_with(&line, Some(self.hello_timeout)) {
            Ok(reply) => {
                if let Ok(v) = serde_json::from_str::<Value>(reply.trim())
                    && v.get("ok").and_then(Value::as_bool) != Some(true)
                {
                    log::debug!("the app did not accept the hello: {reply}");
                }
            }
            Err(e) => {
                // Drop the connection: a reply that arrives later would be read as the answer to
                // the next request.
                self.conn = None;
                log::debug!("the app did not answer the hello in time: {e}");
            }
        }
    }
}
