//! What every hand-run live test shares (`--ignored`, never CI): the control port of the
//! app to design in, and one prompt that asks each backend for the same rectangle — so a
//! single canvas check covers all three, each drawing under its own name.

/// The running app's control port, from `FIGCRAFT_TEST_PORT`. `None` skips the test: a
/// live turn needs an app to attach to, and CI has none.
pub(crate) fn live_port() -> Option<u16> {
    match std::env::var("FIGCRAFT_TEST_PORT").ok().and_then(|p| p.parse::<u16>().ok()) {
        Some(port) => Some(port),
        None => {
            println!("skipped: set FIGCRAFT_TEST_PORT to the running app's control port");
            None
        }
    }
}

/// The turn that proves the whole loop: the real agent CLI calls our MCP tools and a
/// rectangle lands on the attached app's canvas. `who` names the rectangle, so the canvas
/// shows which backend drew it when the tests are run one after another.
pub(crate) fn live_design_prompt(who: &str) -> String {
    format!(
        "Draw exactly one rectangle on the canvas: name it \"{who} probe\", place it at \
         x=40 y=40, 120 wide and 80 tall, solid colour #3B82F6. Use the MCP tools for it. \
         Do not open or save any file, and touch nothing else. When it exists, reply with \
         exactly: siap."
    )
}
