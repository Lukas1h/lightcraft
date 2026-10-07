//! The AI Window mask: one call to an image model (Gemini on Vertex AI) returns a black-and-white
//! picture of the window glass in an interior photo; this crate cleans it up into a soft matte
//! the develop pipeline stores like any other mask.
//!
//! - [`matte`], [`panes`], [`pipeline`]: pure image processing (bilinear resampling, thresholding,
//!   components, the plausibility filter with neighbour rescue, the boundary inset, pane
//!   grouping). Unit-tested without a network.
//! - [`gemini`]: request bodies, response parsing, retry with backoff, behind [`gemini::Transport`].
//! - [`net`]: the real transport, pure-Rust HTTPS ([`http`]). **This crate and the SAM 3 model
//!   download are the only network code in LightCraft.** Nothing here runs unless the user turned
//!   the feature on ([`Settings::enabled`]); the engine checks that before any call.
//! - [`run`]: one photo end to end.
//!
//! No UI dependencies (L3). Photos leave the machine only as the JPEG preview in the request.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod b64;
pub mod gemini;
#[cfg(not(target_arch = "wasm32"))]
mod http;
pub mod matte;
#[cfg(not(target_arch = "wasm32"))]
pub mod net;
pub mod panes;
pub mod pipeline;
pub mod run;

pub use matte::Gray;
use serde::{Deserialize, Serialize};

/// Why a window request failed. Messages are for the user: no keys, no image data.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("no Google Cloud API key is set (Settings → AI Window)")]
    NoKey,
    #[error("cancelled")]
    Cancelled,
    #[error("network error: {0}")]
    Network(String),
    #[error("the server stopped answering")]
    Stalled,
    #[error("the service is rate limiting or unavailable (HTTP {status} {api_status}); gave up after {tries} tries: {message}")]
    RateLimited { status: u16, api_status: String, message: String, tries: usize },
    #[error("the service refused the request (HTTP {status}): {message}")]
    Api { status: u16, message: String },
    #[error("unexpected reply: {0}")]
    BadResponse(String),
    #[error("{0}")]
    Refused(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The feature's settings (everything except the key, which lives in the OS keychain).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Opt-in: the feature does nothing, and nothing is sent anywhere, until this is on.
    pub enabled: bool,
    /// The one-time privacy notice was shown and accepted.
    pub notice_accepted: bool,
    /// Segmentation model.
    pub model: String,
    /// Model for the multi-pane recovery pass.
    pub boxes_model: String,
    /// Photos in flight at once (1: the free tier rate-limits hard).
    pub concurrency: u32,
    /// Boundary inset in pixels of a 6000 px wide frame (scaled with the width).
    pub inset: f64,
    /// Drop regions that don't look like a bright window (with neighbour rescue).
    pub plausibility: bool,
    /// Re-run under-covered multi-pane windows on a crop.
    pub recovery: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            enabled: false,
            notice_accepted: false,
            model: "gemini-3.1-flash-lite-image".into(),
            boxes_model: "gemini-3.8-flash".into(),
            concurrency: 1,
            inset: 4.0,
            plausibility: true,
            recovery: false,
        }
    }
}

/// Most photos in flight at once.
pub const MAX_CONCURRENCY: u32 = 8;

impl Settings {
    /// Clamp hostile / out-of-range values (a damaged settings file) into the supported range.
    pub fn sanitized(mut self) -> Settings {
        let d = Settings::default();
        let ok_model = |m: &str| !m.is_empty() && m.len() <= 80 && m.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'));
        if !ok_model(&self.model) {
            self.model = d.model;
        }
        if !ok_model(&self.boxes_model) {
            self.boxes_model = d.boxes_model;
        }
        self.concurrency = self.concurrency.clamp(1, MAX_CONCURRENCY);
        self.inset = if self.inset.is_finite() { self.inset.clamp(0.0, 40.0) } else { d.inset };
        self
    }
}

/// The adjustments a new Window mask starts with, in LightCraft's local-adjustment units:
/// `(exposure EV, highlights, temp, saturation, dehaze)`. Exposure is in stops here (Lightroom
/// stores it as stops/4; the preset importer multiplies by 4, so this is −1 EV, not −4).
pub const DEFAULT_ADJUST: (f64, f64, f64, f64, f64) = (-1.0, -100.0, 10.0, 10.0, 5.0);

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
