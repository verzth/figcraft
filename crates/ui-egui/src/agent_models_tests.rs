//! The model picker's catalogs, name-shaping, the merged list and turn routing.

use crate::agent_models::*;
use crate::agents::{BackendKind, PanelBackend};

#[test]
fn claude_offers_pen_devs_list_with_sonnet_picked() {
    let models = claude_models();
    let labels: Vec<&str> = models.iter().map(|m| m.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "Claude Fable 5.1",
            "Claude Fable 5",
            "Claude Opus 5.5",
            "Claude Opus 5",
            "Claude Opus 4.8",
            "Claude Opus 4.7",
            "Claude Opus 4.6",
            "Claude Sonnet 5.5",
            "Claude Sonnet 5",
            "Claude Sonnet 4.6",
            "Claude Haiku 5.5",
            "Claude Haiku 4.5",
        ]
    );
    assert_eq!(models[CLAUDE_DEFAULT].id, "claude-sonnet-5-5");
    assert!(models.iter().all(|m| m.provider == "Anthropic"));
    // Claude Code runs them; OpenCode takes the same names under `anthropic/…`.
    assert!(models.iter().all(|m| m.runners == [BackendKind::ClaudeCode, BackendKind::OpenCode]));
    assert_eq!(models[0].open, "anthropic/claude-fable-5-1");
}

#[test]
fn codex_defaults_to_the_models_page_recommendation() {
    let models = codex_models();
    assert_eq!(models[0].id, "gpt-6.1-sol");
    assert_eq!(models[0].label, "GPT 6.1 Sol");
    assert!(models.iter().any(|m| m.id == "gpt-6-astra" && m.label == "GPT 6 Astra"));
    assert!(models.iter().all(|m| m.provider == "OpenAI"));
    assert!(models.iter().all(|m| m.runners == [BackendKind::Codex, BackendKind::OpenCode]));
    assert_eq!(models[0].open, "openai/gpt-6.1-sol");
}

#[test]
fn opencode_rows_read_like_pen_devs() {
    let rows = from_opencode_list(
        "anthropic/claude-opus-4-8-fast\naxi/axi-lite\nminimax-coding-plan/MiniMax-M3.1-Flash-Preview\nopencode/glm-5.3-flash-free\nanthropic/claude-haiku-4-5-20251001\n",
    );
    let labels: Vec<&str> = rows.iter().map(|m| m.label.as_str()).collect();
    assert_eq!(labels, ["Claude Opus 4.8 Fast", "AXI Lite", "MiniMax M3.1 Flash Preview", "GLM 5.3 Flash Free", "Claude Haiku 4.5"]);
    assert_eq!(rows[0].id, "claude-opus-4-8-fast");
    assert_eq!(rows[0].open, "anthropic/claude-opus-4-8-fast");
    assert_eq!(rows[0].provider, "Anthropic");
    // Anthropic's names go to Claude Code too; the rest run on OpenCode.
    assert_eq!(rows[0].runners, [BackendKind::ClaudeCode, BackendKind::OpenCode]);
    assert_eq!(rows[1].runners, [BackendKind::OpenCode]);
    assert_eq!(rows[2].provider, "MiniMax Coding Plan");
}

#[test]
fn the_real_opencode_list_parses_whole() {
    let rows = from_opencode_list(OPENCODE_MODELS);
    assert_eq!(rows.len(), 43);
    let labels: Vec<&str> = rows.iter().filter(|m| m.provider == "Anthropic").map(|m| m.label.as_str()).collect();
    assert!(labels.contains(&"Claude Sonnet 5.5"));
    assert!(labels.contains(&"Claude Fable 5.1"));
    assert!(rows.iter().any(|m| m.open == "opencode/mimo-v2.6-flash-free" && m.id == "mimo-v2.6-flash-free" && m.label == "MiMo V2.6 Flash Free"));
}

#[test]
fn junk_lines_are_dropped() {
    let rows = from_opencode_list("\nno-slash-here\n  spaced/model  \n");
    assert_eq!(
        rows,
        vec![ModelChoice {
            id: "model".into(),
            open: "spaced/model".into(),
            label: "Model".into(),
            provider: "Spaced".into(),
            runners: vec![BackendKind::OpenCode],
        }]
    );
}

#[test]
fn one_dropdown_lists_every_provider_with_the_curated_names_first() {
    let merged = merge(from_opencode_list(OPENCODE_MODELS));
    // Anthropic, then OpenAI as pen.dev orders them…
    assert_eq!(merged[0].provider, "Anthropic");
    assert_eq!(merged[claude_models().len()].provider, "OpenAI");
    // …then OpenCode's own providers, without the curated models repeating under dated ids.
    assert!(merged.iter().any(|m| m.provider == "DeepSeek" && m.label == "DeepSeek V4 Pro"));
    assert!(merged.iter().any(|m| m.provider == "MiniMax Coding Plan"));
    assert!(merged.iter().any(|m| m.id == "claude-opus-4-5" && m.label == "Claude Opus 4.5"));
    assert_eq!(merged.iter().filter(|m| m.label == "Claude Haiku 4.5").count(), 1);
    assert_eq!(merged.iter().filter(|m| m.id == "claude-sonnet-5-5").count(), 1);
}

#[test]
fn the_default_model_runs_on_the_selected_backend_when_it_can() {
    let mut models = Models::default();
    let backends = vec![
        PanelBackend { kind: BackendKind::ClaudeCode, bin: "/bin/claude".into() },
        PanelBackend { kind: BackendKind::Codex, bin: "/bin/codex".into() },
    ];
    models.ensure(&backends);
    // Pen.dev's bar opens on Sonnet, and Claude Code is selected: it runs it.
    assert_eq!(models.label(), "Claude Sonnet 5.5");
    assert_eq!(models.route(&backends, 0), Route::Model { kind: BackendKind::ClaudeCode, id: "claude-sonnet-5-5".into() });
    // Codex selected, but a Claude model goes to the CLI that takes it — the model wins.
    assert_eq!(models.route(&backends, 1), Route::Model { kind: BackendKind::ClaudeCode, id: "claude-sonnet-5-5".into() });
    // This computer without any CLI for it: the starting default falls back quietly.
    let codex_only = vec![PanelBackend { kind: BackendKind::Codex, bin: "/bin/codex".into() }];
    assert_eq!(models.route(&codex_only, 0), Route::Selected(BackendKind::Codex));
}

#[test]
fn a_picked_model_goes_to_a_backend_that_runs_it_or_says_so() {
    let mut models = Models::default();
    let backends = vec![
        PanelBackend { kind: BackendKind::ClaudeCode, bin: "/bin/claude".into() },
        PanelBackend { kind: BackendKind::OpenCode, bin: "/bin/opencode".into() },
    ];
    models.ensure(&backends);
    // An OpenAI model: Codex is missing, so OpenCode runs it under `openai/…`.
    let gpt = models.choices().iter().position(|m| m.id == "gpt-6.1-sol").unwrap();
    models.search = "gpt".into();
    models.pick(gpt);
    assert!(models.search.is_empty());
    assert_eq!(models.route(&backends, 0), Route::Model { kind: BackendKind::OpenCode, id: "openai/gpt-6.1-sol".into() });
    // Now a computer where nothing takes it: the hand-picked model says so by name.
    let claude_only = vec![PanelBackend { kind: BackendKind::ClaudeCode, bin: "/bin/claude".into() }];
    assert_eq!(models.route(&claude_only, 0), Route::Unsupported("GPT 6.1 Sol".into()));
    // No backend at all.
    assert_eq!(models.route(&[], 0), Route::NoBackend);
}

#[test]
fn a_codex_only_computer_starts_on_gpt() {
    let mut models = Models::default();
    let backends = vec![PanelBackend { kind: BackendKind::Codex, bin: "/bin/codex".into() }];
    models.ensure(&backends);
    assert_eq!(models.label(), "GPT 6.1 Sol");
    assert_eq!(models.route(&backends, 0), Route::Model { kind: BackendKind::Codex, id: "gpt-6.1-sol".into() });
}

/// The user's `opencode models` output, kept so the parser is checked against real lines.
const OPENCODE_MODELS: &str = "anthropic/claude-fable-5
anthropic/claude-fable-5-1
anthropic/claude-haiku-4-5
anthropic/claude-haiku-4-5-20251001
anthropic/claude-haiku-5-5
anthropic/claude-opus-4-5
anthropic/claude-opus-4-5-20251101
anthropic/claude-opus-4-6
anthropic/claude-opus-4-7
anthropic/claude-opus-4-8
anthropic/claude-opus-4-8-fast
anthropic/claude-opus-5
anthropic/claude-opus-5-5
anthropic/claude-opus-5-5-fast
anthropic/claude-opus-5-fast
anthropic/claude-sonnet-4-5
anthropic/claude-sonnet-4-5-20250929
anthropic/claude-sonnet-4-6
anthropic/claude-sonnet-5
anthropic/claude-sonnet-5-5
axi/axi-lite
axi/axi-premium
axi/axi-pro
deepseek/deepseek-flash
deepseek/deepseek-v4-pro
minimax-coding-plan/MiniMax-M2
minimax-coding-plan/MiniMax-M2.1
minimax-coding-plan/MiniMax-M2.5
minimax-coding-plan/MiniMax-M2.5-highspeed
minimax-coding-plan/MiniMax-M2.7
minimax-coding-plan/MiniMax-M2.7-highspeed
minimax-coding-plan/MiniMax-M3
minimax-coding-plan/MiniMax-M3.1-Flash-Preview
opencode/beam-free
opencode/big-pickle
opencode/deepseek-v4.1-flash-free
opencode/glm-5.3-flash-free
opencode/longcat-2.5-preview-free
opencode/mimo-v2.6-flash-free
opencode/nemotron-3-ultra-free
opencode/nemotron-3.5-lightning-free
opencode/nemotron-3.5-super-vl-free
opencode/space-bunny-free
";
