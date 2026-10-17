//! The agent cursor drawn headless: the control channel's pointer and raw input move it on the
//! canvas, its tag is the client that said `app.hello`, and it's gone a couple of seconds after
//! the last report (so an idle agent leaves the canvas alone).

use std::time::Duration;

use serde_json::{Value, json};

use crate::live_tests::{Harness, filled, has_text};
use crate::theme::ACCENT;

/// An MCP client, as the Agents tab sees it.
fn hello(h: &mut Harness) {
    crate::control::handle(&mut h.app, &h.ctx, "app.hello", &json!({"client": "claude-code", "version": "2.1.0"})).unwrap();
}

#[test]
fn the_agent_cursor_follows_the_pointer_and_carries_the_client_name() {
    let (mut h, _) = Harness::new();
    h.frames(2);
    hello(&mut h);
    assert_eq!(h.app.ui.agent_cursor.target(), None, "nothing until the agent points");
    crate::control::handle(&mut h.app, &h.ctx, "ui.pointer", &json!({"kind": "move", "x": 200.0, "y": 150.0})).unwrap();
    assert_eq!(h.app.ui.agent_cursor.target(), Some((200.0, 150.0)));
    let (shapes, _delay) = h.frame(vec![]);
    assert_eq!(filled(&shapes, ACCENT), 2, "the dart's two halves");
    assert!(has_text(&shapes, "Claude Code"), "the client's name beside it");
    // Somewhere far off: the next frame glides there and asks for more frames until it arrives.
    crate::control::handle(&mut h.app, &h.ctx, "ui.pointer", &json!({"kind": "move", "x": 900.0, "y": 600.0})).unwrap();
    let (_, delay) = h.frame(vec![]);
    assert!(delay < Duration::from_secs(1), "a cursor still gliding asks for the next frame");
}

#[test]
fn the_agent_cursor_fades_when_the_agent_goes_quiet() {
    let (mut h, _) = Harness::new();
    h.frames(2);
    crate::control::handle(&mut h.app, &h.ctx, "ui.pointer", &json!({"kind": "click", "x": 300.0, "y": 200.0})).unwrap();
    assert_eq!(filled(&h.frames(1), ACCENT), 2);
    assert!(has_text(&h.frames(1), "An AI agent"), "a script that never said hello is still an agent");
    h.skip(crate::agent_cursor::HOLD + crate::agent_cursor::FADE + 0.1);
    let shapes = h.frames(2);
    assert_eq!(filled(&shapes, ACCENT), 0, "gone once it fades");
    assert_eq!(h.app.ui.agent_cursor.shown(), None);
}

#[test]
fn raw_input_moves_the_agent_cursor_to_the_last_position_reported() {
    let (mut h, _) = Harness::new();
    crate::control::handle(
        &mut h.app,
        &h.ctx,
        "ui.input",
        &json!({"events": [{"move": [100.0, 120.0]}, {"scroll": [0.0, -30.0]}, {"down": [400.0, 350.0]}, {"up": [400.0, 350.0]}]}),
    )
    .unwrap();
    assert_eq!(h.app.ui.agent_cursor.target(), Some((400.0, 350.0)), "down/up count too, the last wins");
}

#[test]
fn junk_positions_neither_move_the_cursor_nor_crash() {
    let (mut h, _) = Harness::new();
    h.frames(1);
    crate::control::handle(&mut h.app, &h.ctx, "ui.pointer", &json!({"kind": "move", "x": 200.0, "y": 150.0})).unwrap();
    let junk: [Value; 4] = [
        json!({"kind": "move"}),
        json!({"kind": "move", "x": "NaN", "y": 1.0}),
        json!({"kind": "move", "x": {}, "y": []}),
        json!({"kind": "move", "sx": 1e300, "sy": -1e300}),
    ];
    for j in junk {
        // The pointer itself may accept or reject them; what matters is that nothing crashes
        // and the cursor keeps the last good position.
        let _ = crate::control::handle(&mut h.app, &h.ctx, "ui.pointer", &j);
    }
    assert_eq!(h.app.ui.agent_cursor.target(), Some((200.0, 150.0)), "the last good position");
    h.frame(vec![]);
}
