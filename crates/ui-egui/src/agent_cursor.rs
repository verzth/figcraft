//! The agent cursor: an MCP client's pointer drawn on the canvas, so a person watching the window
//! sees where an agent driving the app is (as in a live session, where everyone's cursor shows).
//! The control channel's `ui.pointer` and `ui.input` report where it is (page coordinates); it
//! glides there like a remote cursor and fades a couple of seconds after the last report, so an
//! idle agent leaves the canvas alone. Its tag is the client that said `app.hello` ("Claude
//! Code"), or "An AI agent" for a script that never did.

use std::time::Duration;

use egui::{Align2, vec2};
use serde_json::Value;

use crate::FigApp;
use crate::theme::ACCENT;

/// Fully opaque this long after the last report (seconds).
pub(crate) const HOLD: f64 = 2.0;
/// How long it then takes to fade out.
pub(crate) const FADE: f64 = 1.0;
/// How fast it closes in on where it was last reported (a live cursor's glide, per second).
const GLIDE: f64 = 20.0;

/// Where an agent's pointer was last reported for this window (page coordinates), and where it is
/// drawn now, gliding there.
#[derive(Default)]
pub struct AgentCursor {
    /// Where it was last reported.
    target: Option<(f64, f64)>,
    /// Where it is drawn.
    shown: Option<(f64, f64)>,
    /// Where it was drawn last frame, and the UI clock then (its glide step).
    frame: Option<((f64, f64), f64)>,
    /// The UI clock of the last report.
    last: f64,
}

impl AgentCursor {
    /// Where it was last reported (`ui.inspect`, tests).
    pub fn target(&self) -> Option<(f64, f64)> {
        self.target
    }

    /// Where it is drawn (`ui.inspect`, tests).
    pub fn shown(&self) -> Option<(f64, f64)> {
        self.shown
    }
}

/// Note where a control request puts the agent's pointer, before it runs: `ui.pointer`'s `x, y`
/// (page) or `sx, sy` (screen), and `ui.input`'s move/down/up events (the last one wins). A
/// request that carries no usable position leaves the cursor where it was.
pub(crate) fn note(app: &mut FigApp, method: &str, params: &Value, now: f64) {
    let at = match method {
        "ui.pointer" => at(app, params),
        "ui.input" => params
            .get("events")
            .and_then(Value::as_array)
            .and_then(|es| es.iter().rev().find_map(|e| xy(e.get("move")).or_else(|| xy(e.get("down"))).or_else(|| xy(e.get("up"))))),
        _ => return,
    };
    let Some((x, y)) = at else { return };
    let ui = &mut app.ui.agent_cursor;
    ui.target = Some((x, y));
    ui.last = now;
}

/// `ui.pointer`'s position: `x, y` in page coordinates, or `sx, sy` converted from the screen.
fn at(app: &FigApp, p: &Value) -> Option<(f64, f64)> {
    let f = |k: &str| p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite());
    match (f("x"), f("y")) {
        (Some(x), Some(y)) => Some((x, y)),
        _ => {
            let s = app.to_page(egui::pos2(f("sx")? as f32, f("sy")? as f32));
            // A screen point far off any page can convert to a non-finite page point.
            let p = (s.x, s.y);
            (p.0.is_finite() && p.1.is_finite()).then_some(p)
        }
    }
}

/// `ui.input`'s `[x, y]` position.
fn xy(v: Option<&Value>) -> Option<(f64, f64)> {
    let a = v?.as_array()?;
    let (x, y) = (a.first()?.as_f64()?, a.get(1)?.as_f64()?);
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

/// The agent cursor on the canvas: the dart in the accent colour with the client's name beside
/// it, gliding to where the pointer last was and fading when the agent goes quiet.
pub fn draw(app: &mut FigApp, painter: &egui::Painter, ctx: &egui::Context) {
    let now = ctx.input(|i| i.time);
    let (target, last, from) = {
        let ui = &app.ui.agent_cursor;
        let Some(target) = ui.target else { return };
        (target, ui.last, ui.frame)
    };
    if !target.0.is_finite() || !target.1.is_finite() {
        let ui = &mut app.ui.agent_cursor;
        ui.target = None;
        ui.shown = None;
        ui.frame = None;
        return;
    }
    let age = now - last;
    if age > HOLD + FADE {
        let ui = &mut app.ui.agent_cursor;
        ui.shown = None;
        ui.frame = None;
        return;
    }
    let (mut sx, mut sy) = from.map(|(p, _)| p).unwrap_or(target);
    let dt = from.map_or(0.0, |(_, t)| (now - t).clamp(0.0, 0.1));
    let k = if app.reduce_motion() { 1.0 } else { 1.0 - (-dt * GLIDE).exp() };
    sx += (target.0 - sx) * k;
    sy += (target.1 - sy) * k;
    let zoom = app.session.ds().map(|d| d.view.zoom).unwrap_or(1.0).max(1e-9);
    let gliding = ((target.0 - sx) * zoom).hypot((target.1 - sy) * zoom) >= 0.5;
    if !gliding {
        (sx, sy) = target;
    }
    let alpha = if age < HOLD { 1.0 } else { ((HOLD + FADE - age) / FADE) as f32 };
    // Frames while it may look different next frame, and none once it's gone.
    if gliding {
        ctx.request_repaint();
    } else if age < HOLD {
        ctx.request_repaint_after(Duration::from_secs_f64((HOLD - age).max(1e-3)));
    } else if alpha > 0.0 {
        ctx.request_repaint();
    }
    {
        let ui = &mut app.ui.agent_cursor;
        ui.shown = Some((sx, sy));
        ui.frame = Some(((sx, sy), now));
    }
    if alpha <= 0.0 {
        return;
    }
    let name = app.ui.agents.client_name();
    let tip = app.to_screen(sx, sy);
    let fill = if alpha >= 1.0 { ACCENT } else { ACCENT.gamma_multiply(alpha) };
    crate::live::pointer(painter, tip, fill);
    crate::live::name_tag(painter, tip + vec2(12.0, 16.0), Align2::LEFT_TOP, &name, fill);
}
