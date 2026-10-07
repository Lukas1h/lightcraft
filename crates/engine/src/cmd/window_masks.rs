//! AI Window mask commands (see `crate::window`).

use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd, f64_or, has_selection};
use crate::{Result, Session};

/// The photos a command applies to: `ids` if given, else the selection, else the active photo.
pub(crate) fn target_ids(s: &Session, p: &Value) -> Vec<PhotoId> {
    if let Some(a) = p.get("ids").and_then(Value::as_array) {
        return a.iter().filter_map(Value::as_u64).map(PhotoId).collect();
    }
    if !s.selection.ids.is_empty() {
        return s.selection.ids.clone();
    }
    s.active().into_iter().collect()
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "mask.addWindow",
            "Window",
            [],
            None,
            "{ids?: [photoId, …] (default: the selection, else the active photo)} — mask the window glass with the AI Window model (a Gemini image model, called with your own Google Cloud API key; the photo's preview is sent to Google, so Settings ▸ AI Window must be on). Adds a `Window` mask with default adjustments (exposure −1 EV, highlights −100, temp +10, saturation +10, dehaze +5), or replaces the Window selection of the photo's existing Window mask; other masks are untouched. In the app this returns {started, total} at once (progress: window.status, cancel: window.cancel); otherwise it waits and returns the status {total, done, found, none, errors}",
            has_selection,
            |s, p| s.window_start(target_ids(s, p))
        ),
        cmd!(
            "window.setInset",
            "Window Mask Inset",
            [],
            None,
            "{inset: pixels of a 6000 px wide frame (0..40, default setting 4), id?: maskId, photo?: photoId (default: the active photo)} — pull the Window mask's boundary in from the glass edge (painting past the glass darkens the frame; stopping short is nearly invisible). Recomputed from the stored matte: no model call",
            has_selection,
            |s, p| {
                let c = "window.setInset";
                let photo = p.get("photo").and_then(Value::as_u64).map(PhotoId).or(s.active()).ok_or_else(|| bad(c, "no active photo"))?;
                let mask = p.get("id").and_then(Value::as_u64).map(|v| v as u32);
                let inset = f64_or(p, "inset", s.window.settings.inset);
                s.window_set_inset(photo, mask, inset)?;
                Ok(json!({"inset": inset}))
            }
        ),
        cmd!(
            query "window.cancel",
            "Cancel Window Masks",
            [],
            None,
            "{} — stop the running AI Window batch (photos not started are skipped) → {cancelled}",
            always,
            |s, _| Ok(json!({"cancelled": s.window_cancel()}))
        ),
        cmd!(
            query "window.status",
            "Window Mask Progress",
            [],
            None,
            "{} — the running (or last) AI Window batch → {running, total, done, current: [photoId], found, none (no window seen), errors: [{photo, error}], cancelled}",
            always,
            |s, _| Ok(json!(s.window.status()))
        ),
        cmd!(
            query "window.settings",
            "AI Window Settings",
            [],
            None,
            "{enabled?, noticeAccepted?, model?, boxesModel?, concurrency? (1..8), inset? (px at 6000 wide), plausibility?, recovery?, apiKey? (\"\" removes it)} — read or change the AI Window settings → the settings, {hasKey, keyStore, keyPersistent}. The key is never returned, logged or stored in the settings file. The feature is off until `enabled` and `noticeAccepted` are both true (the app shows the privacy notice first)",
            always,
            |s, p| s.window_configure(p)
        ),
    ]
}
