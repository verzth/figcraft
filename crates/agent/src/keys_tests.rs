//! The prefs file's guarantees: a round-trip that keeps every field, a corrupt or missing
//! file that quietly means "no keys", and 0600 wherever the platform has modes.

use crate::keys::{ApiKeys, FILE_NAME};

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("figcraft-keys-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(FILE_NAME)
}

#[test]
fn a_saved_keyring_loads_back_field_for_field() {
    let path = scratch("roundtrip");
    let keys = ApiKeys {
        anthropic: Some("sk-ant-test".into()),
        anthropic_base: Some("https://proxy.example/".into()),
        openai: Some("sk-test".into()),
        openai_base: None,
    };
    keys.save(&path).unwrap();
    assert_eq!(ApiKeys::load(&path), keys);
    // A field with no key is left out of the file entirely.
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.contains("openaiBase"), "{text}");
    assert!(text.contains("\"anthropicBase\": \"https://proxy.example/\""), "{text}");
    std::fs::remove_dir_all(path.parent().unwrap_or(&path)).unwrap();
}

#[test]
fn a_missing_or_unparsable_file_is_an_empty_keyring() {
    let missing = scratch("missing");
    assert_eq!(ApiKeys::load(&missing), ApiKeys::default());
    std::fs::write(&missing, b"\xff\xfe not json at all {").unwrap();
    assert_eq!(ApiKeys::load(&missing), ApiKeys::default());
    // Unknown fields from a future version are ignored, the known ones still load.
    std::fs::write(&missing, br#"{"anthropic": "sk-ant-future", "futureThing": 42}"#).unwrap();
    assert_eq!(ApiKeys::load(&missing).anthropic.as_deref(), Some("sk-ant-future"));
    // A pasted key that is only whitespace is no key at all.
    std::fs::write(&missing, br#"{"anthropic": "   ", "openai": "sk-ok"}"#).unwrap();
    let keys = ApiKeys::load(&missing);
    assert_eq!(keys.anthropic, None);
    assert_eq!(keys.openai.as_deref(), Some("sk-ok"));
    std::fs::remove_dir_all(missing.parent().unwrap_or(&missing)).unwrap();
}

#[test]
#[cfg(unix)]
fn the_file_is_owner_only_and_the_temp_file_never_lingers() {
    use std::os::unix::fs::PermissionsExt;

    let path = scratch("mode");
    ApiKeys { anthropic: Some("sk-ant-test".into()), ..ApiKeys::default() }.save(&path).unwrap();
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600, "mode was {mode:o}");
    let tmp = path.with_extension("json.tmp");
    assert!(!tmp.exists(), "the temp file was left behind: {}", tmp.display());
    std::fs::remove_dir_all(path.parent().unwrap_or(&path)).unwrap();
}
