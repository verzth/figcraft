//! What [`crate::vision`] decides for a turn: which bytes count as an image, which
//! attachments travel as blocks, and that the prompt names every file either way.

use crate::Attachment;
use crate::Turn;
use crate::vision::{MAX_IMAGE_BYTES, Vision, encode, image_media_type};

/// PNG's eight magic bytes, then a few bytes to carry: the FakeApi tests need real
/// headers, not a real file — the sniffing reads the head and the API fakes never decode.
const PNG: &[u8] = b"\x89PNG\r\n\x1a\nfigcraft";
const JPEG: &[u8] = b"\xFF\xD8\xFF\xE0figcraft";
const GIF: &[u8] = b"GIF89afigcraft";
const WEBP: &[u8] = b"RIFF\x00\x00\x00\x00WEBPVP8 ";

fn attachment(name: &str, mime: &str, data: &[u8]) -> Attachment {
    Attachment { name: name.into(), mime: mime.into(), data: data.to_vec() }
}

#[test]
fn every_format_the_apis_take_sniffs_from_its_bytes_and_nothing_else_does() {
    assert_eq!(image_media_type(PNG), Some("image/png"));
    assert_eq!(image_media_type(JPEG), Some("image/jpeg"));
    assert_eq!(image_media_type(GIF), Some("image/gif"));
    assert_eq!(image_media_type(WEBP), Some("image/webp"));
    // A text file renamed .png, an SVG and the empty drop: none of them are images, so
    // the API is never sent bytes that claim to be one.
    assert_eq!(image_media_type(b"# Design\n"), None);
    assert_eq!(image_media_type(b"<svg viewBox=\"0 0 8 8\"/>"), None);
    assert_eq!(image_media_type(b""), None);
    // RIFF without WEBP at offset 8: some other RIFF container, not one of the four.
    assert_eq!(image_media_type(b"RIFF\x00\x00\x00\x00WAVEfmt "), None);
}

#[test]
fn a_turn_splits_into_a_prompt_naming_every_file_and_the_images_it_carries() {
    let turn = Turn {
        text: "  match this sketch  ".into(),
        attachments: vec![
            attachment("skema.png", "image/png", PNG),
            attachment("DESIGN.md", "text/plain", b"# tokens\n"),
            // The declared type lies (the panel sniffs too, but this end must not trust
            // it): bytes that are not an image stay named in the prompt.
            attachment("fake.png", "image/png", b"<html>"),
        ],
    };
    let vision = Vision::of(&turn);
    // The message is trimmed and every attachment is named, exactly as the CLIs phrase it.
    assert_eq!(vision.prompt, "match this sketch\n\n(Attached files: skema.png, DESIGN.md, fake.png.)");
    assert_eq!(vision.images.len(), 1, "only the bytes that are an image travel");
    let (carried, media_type) = (vision.images[0].0, vision.images[0].1);
    assert_eq!(carried.name, "skema.png");
    assert_eq!(media_type, "image/png");
}

#[test]
fn an_image_over_the_cap_stays_named_and_the_bytes_encode_as_base64() {
    // A real PNG head, stretched past the cap: it is the size that keeps it back, not
    // the bytes.
    let mut big = PNG.to_vec();
    big.resize(MAX_IMAGE_BYTES + 1, 0);
    let turn = Turn { text: "too big".into(), attachments: vec![attachment("huge.png", "image/png", &big)] };
    let vision = Vision::of(&turn);
    assert!(vision.images.is_empty(), "an oversize image is not carried");
    assert!(vision.prompt.contains("(Attached files: huge.png.)"), "{}", vision.prompt);
    // One byte over the cap is the panel's own ceiling: at the cap, it still goes.
    let mut at_cap = PNG.to_vec();
    at_cap.resize(MAX_IMAGE_BYTES, 0);
    let turn = Turn { text: "fits".into(), attachments: vec![attachment("at.png", "image/png", &at_cap)] };
    assert_eq!(Vision::of(&turn).images.len(), 1);
    assert_eq!(encode(b"\x89PNG\r\n\x1a\n"), "iVBORw0KGgo=");
}

#[test]
fn a_turn_without_attachments_is_the_message_alone() {
    let turn = Turn { text: "buat kotak".into(), attachments: Vec::new() };
    let vision = Vision::of(&turn);
    assert_eq!(vision.prompt, "buat kotak");
    assert!(vision.images.is_empty());
}
