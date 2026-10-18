//! Image attachments as a turn carries them: the panel stores a dropped image's bytes in
//! [`Turn::attachments`], and here a turn splits into the prompt text (naming every file,
//! exactly as the CLI backends phrase it) and the images that travel beside it as base64
//! image blocks. What the bytes *are* is decided by their magic numbers — never by the
//! file's name or declared type — so a panel drop and the API request can never disagree.

use base64::Engine as _;

use crate::{Attachment, Turn};

/// Anthropic's own ceiling for one image (5 MB); OpenAI's published one is higher, and
/// one cap keeps the panel's drops and both APIs in step. Anything larger stays named in
/// the prompt, the way a CLI backend would treat it.
pub const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

/// The image format behind these bytes, by their magic numbers: the four formats both
/// APIs accept as vision input, and `None` for anything else (a text file renamed
/// `.png`, a truncated download, an SVG).
pub fn image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// A turn split for an API that can see images.
pub struct Vision<'a> {
    /// What the model reads: the typed message, then every attachment's name — the same
    /// line the CLI backends add, so the model hears the same words on either route.
    pub prompt: String,
    /// The attachments that travel as image blocks, each with the media type its bytes
    /// proved. Over the size cap or not an image at all, and it stays named in the prompt.
    pub images: Vec<(&'a Attachment, &'static str)>,
}

impl<'a> Vision<'a> {
    /// Split one turn: name every attachment in the prompt, carry the images beside it.
    pub fn of(turn: &'a Turn) -> Self {
        let mut prompt = turn.text.trim().to_string();
        if !turn.attachments.is_empty() {
            let names: Vec<&str> = turn.attachments.iter().map(|a| a.name.as_str()).collect();
            prompt.push_str(&format!("\n\n(Attached files: {}.)", names.join(", ")));
        }
        let images =
            turn.attachments.iter().filter(|a| a.data.len() <= MAX_IMAGE_BYTES).filter_map(|a| Some((a, image_media_type(&a.data)?))).collect();
        Self { prompt, images }
    }
}

/// The bytes as standard base64: what both APIs want inside an image block.
pub fn encode(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}
