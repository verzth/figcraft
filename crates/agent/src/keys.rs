//! The API keys for the direct-API agent backends, kept in one local prefs file
//! (`AgentKeys.json` beside `AgentSessions.json`). No keys are hard-coded anywhere: the user
//! pastes their own, and the file is written atomically with owner-only permissions so a
//! shared machine never shows them to anyone else.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::AgentError;

/// The prefs file name. The caller resolves it (the app passes its config dir beside this).
pub const FILE_NAME: &str = "AgentKeys.json";

/// One user's direct-API credentials. Every field is optional — a missing key simply means
/// that backend is not offered in the panel.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeys {
    /// An Anthropic API key (`sk-ant-…`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic: Option<String>,
    /// An Anthropic-compatible base URL, for proxies or regional endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_base: Option<String>,
    /// An OpenAI-compatible API key (`sk-…` or provider-specific).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai: Option<String>,
    /// An OpenAI-compatible base URL (OpenAI, a gateway, a local server).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_base: Option<String>,
}

impl ApiKeys {
    /// Read the prefs file. Missing or unparsable content is an empty keyring: a corrupt file
    /// must not stop the app, it just means no API backends until the user saves again.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else { return Self::default() };
        let mut keys = serde_json::from_str::<Self>(&text).unwrap_or_default();
        keys.normalize();
        keys
    }

    /// Write the prefs file atomically (a temp file renamed over the old one) with mode 0600
    /// where the platform has modes, so a crash mid-write never leaves a half-written or
    /// world-readable file.
    pub fn save(&self, path: &Path) -> Result<(), AgentError> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(self).map_err(|e| AgentError::Setup(e.to_string()))?;
        let mut tmp = path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let tmp = std::path::PathBuf::from(tmp);
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&tmp).map_err(|e| AgentError::Setup(format!("{}: {e}", tmp.display())))?;
        file.write_all(json.as_bytes())
            .and_then(|()| file.sync_all())
            .and_then(|()| std::fs::rename(&tmp, path))
            .map_err(|e| AgentError::Setup(format!("{}: {e}", path.display())))?;
        Ok(())
    }

    /// A pasted key that is only whitespace is no key at all.
    fn normalize(&mut self) {
        for field in [&mut self.anthropic, &mut self.anthropic_base, &mut self.openai, &mut self.openai_base] {
            if field.as_deref().is_some_and(|s| s.trim().is_empty()) {
                *field = None;
            }
        }
    }
}
