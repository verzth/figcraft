//! Which models the Agent tab can run, and how their names read.
//!
//! One list for the whole picker, grouped by provider like pen.dev's: Anthropic's and OpenAI's
//! curated full names first, then OpenCode's own `opencode models` output live (one spawn,
//! ~60 ms, on a thread), so every backend's models sit in the same dropdown. Each choice knows
//! which CLIs take it: a turn routes the model to a backend that runs it — the selected one
//! when it can, else the first installed that does.

use std::collections::HashSet;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use crate::agents::{BackendKind, PanelBackend};
use crate::i18n::tr;

/// One row of the model picker: the ids the CLIs take, the name it reads as, its group, and
/// which CLIs run it (the preferred one first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelChoice {
    /// The id Claude Code and Codex take (bare: `claude-opus-5-5`).
    pub id: String,
    /// The id OpenCode takes (`anthropic/claude-opus-5-5`).
    pub open: String,
    /// The name the picker shows: "Claude Opus 5.5".
    pub label: String,
    /// The provider group it sits under.
    pub provider: String,
    /// The CLIs that take this model, best first.
    pub runners: Vec<BackendKind>,
}

impl ModelChoice {
    fn new(id: &str, label: &str, provider: &str, runners: &[BackendKind]) -> Self {
        Self {
            id: id.to_string(),
            open: format!("{provider}/{}", id).to_lowercase(),
            label: label.to_string(),
            provider: provider.to_string(),
            runners: runners.to_vec(),
        }
    }

    /// The id for one CLI: OpenCode wants `provider/model`, the others the bare id.
    fn id_for(&self, kind: BackendKind) -> String {
        if kind == BackendKind::OpenCode { self.open.clone() } else { self.id.clone() }
    }
}

/// Anthropic's models as `claude --model` takes them by full name (pen.dev's own list).
const CLAUDE: &[(&str, &str)] = &[
    ("claude-fable-5-1", "Claude Fable 5.1"),
    ("claude-fable-5", "Claude Fable 5"),
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-opus-5", "Claude Opus 5"),
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("claude-opus-4-7", "Claude Opus 4.7"),
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("claude-sonnet-5", "Claude Sonnet 5"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-haiku-5-5", "Claude Haiku 5.5"),
    ("claude-haiku-4-5", "Claude Haiku 4.5"),
];

/// OpenAI's Codex models (developers.openai.com/codex/models); the first is the pick.
const CODEX: &[(&str, &str)] = &[
    ("gpt-6.1-sol", "GPT 6.1 Sol"),
    ("gpt-6-astra", "GPT 6 Astra"),
    ("gpt-6-sol", "GPT 6 Sol"),
    ("gpt-6-luna", "GPT 6 Luna"),
    ("gpt-5.6-sol", "GPT 5.6 Sol"),
    ("gpt-5.6-terra", "GPT 5.6 Terra"),
    ("gpt-5.6-luna", "GPT 5.6 Luna"),
];

/// The picker's default — the model pen.dev's bar opens on; also OpenCode's runner for it.
pub const CLAUDE_DEFAULT: usize = 7; // claude-sonnet-5-5

/// Anthropic's models: Claude Code runs them, OpenCode too (`anthropic/…`).
fn claude_runners() -> Vec<BackendKind> {
    vec![BackendKind::ClaudeCode, BackendKind::OpenCode]
}

/// OpenAI's models: Codex runs them, OpenCode when it has the provider configured.
fn codex_runners() -> Vec<BackendKind> {
    vec![BackendKind::Codex, BackendKind::OpenCode]
}

/// Claude Code's models.
pub fn claude_models() -> Vec<ModelChoice> {
    static_list(CLAUDE, "Anthropic", &claude_runners())
}

/// Codex's models.
pub fn codex_models() -> Vec<ModelChoice> {
    static_list(CODEX, "OpenAI", &codex_runners())
}

fn static_list(rows: &[(&str, &str)], provider: &str, runners: &[BackendKind]) -> Vec<ModelChoice> {
    rows.iter().map(|(id, label)| ModelChoice::new(id, label, provider, runners)).collect()
}

/// OpenCode's list (`opencode models`: `provider/model` per line), parsed into rows. These
/// run on OpenCode; Anthropic's also go to Claude Code, whose names are the same namespace.
pub fn from_opencode_list(raw: &str) -> Vec<ModelChoice> {
    raw.lines()
        .filter_map(|line| {
            let open = line.trim();
            let (prefix, id) = open.split_once('/')?;
            let runners = if prefix == "anthropic" { claude_runners() } else { vec![BackendKind::OpenCode] };
            Some(ModelChoice { id: id.to_string(), open: open.to_string(), label: prettify(id), provider: words(prefix), runners })
        })
        .collect()
}

/// The whole picker: the curated lists first (as pen.dev orders them — Anthropic, then
/// OpenAI), then OpenCode's own list behind them; a model both lists carry shows once, under
/// its curated name (dated ids like `claude-haiku-4-5-20251001` fold into it too).
pub fn merge(opencode: Vec<ModelChoice>) -> Vec<ModelChoice> {
    let mut all = claude_models();
    all.extend(codex_models());
    let mut ids: HashSet<String> = all.iter().map(|c| c.id.clone()).collect();
    let mut labels: HashSet<String> = all.iter().map(|c| c.label.clone()).collect();
    for choice in opencode {
        if ids.contains(&choice.id) || labels.contains(&choice.label) {
            continue;
        }
        ids.insert(choice.id.clone());
        labels.insert(choice.label.clone());
        all.push(choice);
    }
    all
}

/// The name the picker shows: `claude-opus-4-5-20251101` → "Claude Opus 4.5" — the date
/// stamp folded away, hyphens into words, runs of single digits into dotted numbers.
fn prettify(model: &str) -> String {
    let model = strip_date(model);
    let mut words: Vec<String> = Vec::new();
    let mut in_version = false;
    for token in model.split('-').filter(|t| !t.is_empty()) {
        let digit = token.len() == 1 && token.chars().all(|c| c.is_ascii_digit());
        // "5" followed by "5" is one number: "5.5", not "5 5".
        if digit
            && in_version
            && let Some(last) = words.last_mut()
        {
            last.push('.');
            last.push_str(token);
            continue;
        }
        in_version = digit;
        words.push(word(token));
    }
    words.join(" ")
}

/// Drop a trailing `-20251101`-style stamp (pen.dev's names never carry one).
fn strip_date(model: &str) -> &str {
    let Some((head, tail)) = model.rsplit_once('-') else { return model };
    if tail.len() == 8 && tail.chars().all(|c| c.is_ascii_digit()) { head } else { model }
}

/// A hyphen-joined id as a name: `minimax-coding-plan` → "MiniMax Coding Plan".
fn words(id: &str) -> String {
    id.split('-').filter(|w| !w.is_empty()).map(word).collect::<Vec<_>>().join(" ")
}

/// One token as its name: brands keep their spelling, other words capitalize.
fn word(token: &str) -> String {
    match token.to_ascii_lowercase().as_str() {
        "openai" => return "OpenAI".to_string(),
        "gpt" => return "GPT".to_string(),
        "axi" => return "AXI".to_string(),
        "deepseek" => return "DeepSeek".to_string(),
        "minimax" => return "MiniMax".to_string(),
        "glm" => return "GLM".to_string(),
        "vl" => return "VL".to_string(),
        "mimo" => return "MiMo".to_string(),
        _ => {}
    }
    let mut chars = token.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Where the next turn runs (see [`Models::route`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// The selected backend with the CLI's own default model.
    Selected(BackendKind),
    /// A picked model, on a backend that takes it.
    Model {
        /// The backend that runs it.
        kind: BackendKind,
        /// The id that backend takes.
        id: String,
    },
    /// A model no backend on this computer takes (only a hand-picked one lands here).
    Unsupported(String),
    /// This computer found no backend at all.
    NoBackend,
}

/// The picker's state: the whole list, what's picked, and the live list in flight.
#[derive(Default)]
pub struct Models {
    /// Every choice in the dropdown, grouped by provider in order.
    choices: Vec<ModelChoice>,
    /// What the next turn runs — an index into `choices`; `None` says "the CLI's own default".
    picked: Option<usize>,
    /// The user picked `picked` themselves (vs. the list's starting default): an unsupported
    /// pick then says so instead of quietly falling back.
    explicit: bool,
    /// The search box's text (kept between opens, cleared on a pick).
    pub search: String,
    /// `opencode models` on its way in.
    loading: Option<Receiver<Vec<ModelChoice>>>,
    /// Whether that one spawn has happened (a bool, so later draws don't respawn it).
    started: bool,
    /// The popup just opened — the search box takes the caret (as pen.dev's does).
    focused: bool,
}

impl Models {
    /// Whether the popup opened since the last check (its search box takes the caret).
    pub fn take_focus(&mut self) -> bool {
        std::mem::take(&mut self.focused)
    }

    /// The whole list: the curated names right away, OpenCode's own list merged in on a
    /// thread when it lands (`opencode models` reads its config, ~60 ms).
    pub fn ensure(&mut self, backends: &[PanelBackend]) {
        if self.choices.is_empty() {
            self.choices = merge(Vec::new());
            // pen.dev opens on Sonnet; a computer with only Codex starts on GPT — the first
            // pick of what's actually here.
            let claude_like = backends.iter().any(|b| matches!(b.kind, BackendKind::ClaudeCode | BackendKind::OpenCode));
            let codex_here = backends.iter().any(|b| b.kind == BackendKind::Codex);
            self.picked = if !claude_like && codex_here { Some(CLAUDE.len()) } else { Some(CLAUDE_DEFAULT) };
        }
        if !self.started {
            self.started = true;
            if let Some(bin) = backends.iter().find(|b| b.kind == BackendKind::OpenCode).map(|b| b.bin.clone()) {
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    let list = match std::process::Command::new(bin).arg("models").output() {
                        Ok(out) => from_opencode_list(&String::from_utf8_lossy(&out.stdout)),
                        Err(_) => Vec::new(),
                    };
                    let _ = tx.send(list);
                });
                self.loading = Some(rx);
            }
        }
        self.drain();
    }

    /// Take OpenCode's list once it lands (empty when the spawn failed): the curated rows
    /// keep their positions, the rest merge in behind them.
    fn drain(&mut self) {
        let Some(rx) = &self.loading else { return };
        match rx.try_recv() {
            Ok(list) => {
                self.choices = merge(list);
                self.loading = None;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.choices = merge(Vec::new());
                self.loading = None;
            }
        }
    }

    /// Every choice, grouped by provider in order.
    pub fn choices(&self) -> &[ModelChoice] {
        &self.choices
    }

    /// Which choice the next turn runs (its starting default when `None` was never set).
    pub fn picked(&self) -> Option<usize> {
        self.picked
    }

    /// The picked model's name for the trigger — "Default" until one is picked.
    pub fn label(&self) -> String {
        self.picked.and_then(|i| self.choices.get(i)).map(|c| c.label.clone()).unwrap_or_else(|| tr("Default").to_string())
    }

    /// Where the next turn runs: the picked model on a backend that takes it — the selected
    /// one when it can, else the first installed that does — or, without a usable pick, the
    /// selected backend's own default model.
    pub fn route(&self, backends: &[PanelBackend], selected: usize) -> Route {
        let Some(selected_kind) = backends.get(selected).map(|b| b.kind) else { return Route::NoBackend };
        let Some(choice) = self.picked.and_then(|i| self.choices.get(i)) else {
            return Route::Selected(selected_kind);
        };
        let runners: Vec<BackendKind> = choice.runners.iter().copied().filter(|k| backends.iter().any(|b| b.kind == *k)).collect();
        if runners.is_empty() {
            // The starting default quietly falls back to the selected CLI's own model; a model
            // the user picked says so instead.
            return if self.explicit { Route::Unsupported(choice.label.clone()) } else { Route::Selected(selected_kind) };
        }
        let kind = if runners.contains(&selected_kind) { selected_kind } else { runners[0] };
        Route::Model { kind, id: choice.id_for(kind) }
    }

    /// Pick a model (a click in the picker) and clear the search.
    pub fn pick(&mut self, index: usize) {
        self.picked = Some(index);
        self.explicit = true;
        self.search.clear();
    }

    /// The search box takes the caret the next time the popup draws.
    pub fn focus_search(&mut self) {
        self.focused = true;
    }
}
