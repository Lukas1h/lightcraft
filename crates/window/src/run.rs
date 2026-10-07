//! One photo end to end: model call, optional recovery pass, cleanup.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::gemini::{self, Sleeper, Transport};
use crate::matte::Gray;
use crate::pipeline::{self, CleanOptions};
use crate::{Error, Result, Settings, panes};

/// The photo being masked: the neutral develop render the model sees.
pub trait Preview {
    /// Luminance of the preview (0..1), at the resolution the matte is computed at.
    fn luma(&self) -> &Gray;
    /// The preview (or a normalized crop `[x0, y0, x1, y1]` of it) as a JPEG, long side ≤ 2048.
    fn jpeg(&self, rect: Option<[f32; 4]>) -> Result<Vec<u8>>;
}

/// What came back.
#[derive(Debug)]
pub struct Found {
    /// The cleaned soft matte before the inset, at the preview's size; `None` = no window found.
    pub matte: Option<Gray>,
    /// Model calls made (segmentation + recovery).
    pub calls: usize,
}

/// Ask the model for the window glass in `src`.
pub fn find_windows(src: &dyn Preview, t: &dyn Transport, sleeper: &dyn Sleeper, s: &Settings, cancel: &AtomicBool) -> Result<Found> {
    let mut calls = 0;
    let luma = src.luma();
    let reply = gemini::call(t, sleeper, &s.model, &gemini::segmentation_body(&src.jpeg(None)?), cancel)?;
    calls += 1;
    let mut raw = match gemini::last_image(&reply)? {
        Some(png) => {
            let m = pipeline::decode_png_gray(&png)?;
            crate::matte::resize_bilinear(&m, luma.w, luma.h).ok_or_else(|| Error::BadResponse("unusable mask size".into()))?
        }
        None => Gray::new(luma.w, luma.h).ok_or_else(|| Error::Other("the preview has an unusable size".into()))?,
    };
    if s.recovery {
        calls += recover(src, t, sleeper, s, &mut raw, cancel)?;
    }
    let opts = CleanOptions { plausibility: s.plausibility, ..CleanOptions::default() };
    let cleaned = pipeline::clean(&raw, luma, &opts)?;
    Ok(Found { matte: (cleaned.coverage() > 0.0).then_some(cleaned), calls })
}

/// The multi-pane recovery pass; returns the model calls it made.
fn recover(src: &dyn Preview, t: &dyn Transport, sleeper: &dyn Sleeper, s: &Settings, raw: &mut Gray, cancel: &AtomicBool) -> Result<usize> {
    let mut calls = 0;
    let detect = gemini::call(t, sleeper, &s.boxes_model, &gemini::boxes_body(&src.jpeg(None)?), cancel)?;
    calls += 1;
    let windows = panes::group(&gemini::parse_boxes(&gemini::text(&detect)), 0.25);
    for w in panes::weak_windows(raw, &windows, 0.6).into_iter().take(12) {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let rect = panes::padded(w.bounds, 0.15);
        let reply = gemini::call(t, sleeper, &s.model, &gemini::segmentation_body(&src.jpeg(Some(rect))?), cancel)?;
        calls += 1;
        if let Some(png) = gemini::last_image(&reply)? {
            panes::union_region(raw, &pipeline::decode_png_gray(&png)?, rect);
        }
    }
    Ok(calls)
}

#[cfg(test)]
mod tests;
