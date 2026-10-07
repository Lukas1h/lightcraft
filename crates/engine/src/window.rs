//! The AI Window mask (`lightcraft-window`): one click masks the window glass of a photo, or of a
//! selection of photos.
//!
//! **Opt-in and isolated.** Nothing is sent anywhere until the user has turned the feature on and
//! accepted the privacy notice ([`Settings::enabled`] / [`Settings::notice_accepted`]); then the
//! only thing that leaves the machine is a ≤ 2048 px JPEG preview of the photo, to the Google
//! Cloud endpoint, authenticated with the user's own API key. The key is read through
//! [`KeyStore`] and is never written to the catalog, the settings file, a log or an error
//! message.
//!
//! **Where the mask lives.** The result is an ordinary mask (`Mask` with a
//! [`MaskShape::Window`] component) in the photo's develop settings, so it is saved in the
//! catalog / sidecar with the edit, copies and syncs to other photos as data (no recompute, no
//! network), and combines with other components (add / subtract / intersect, brush, invert,
//! Edge). The component stores the cleaned soft matte at the preview's resolution (a
//! [`SegMask`] logit grid, ≤ 2048 px a side) twice: before (`source`) and after the inset
//! (`seg`), so the Inset slider ([`Session::window_set_inset`]) never calls the model again.
//!
//! Photos run on worker threads (`Settings::concurrency` at a time, one by default) and
//! [`Session::window_poll`] applies the results on the UI thread, like AI mask requests. Without
//! [`WindowMasks::background`] (CLI, MCP, tests) the command waits for the batch.

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};

use lightcraft_catalog::PhotoId;
use lightcraft_codecs::encode::{ChromaSubsampling, EncodeImage, EncodeMeta, Samples, encode_jpeg};
use lightcraft_develop::{LocalAdjustments, Mask, MaskComponent, MaskOp, MaskShape, SegMask};
use lightcraft_window::gemini::{RealSleeper, Sleeper, Transport};
use lightcraft_window::run::Preview;
use lightcraft_window::{DEFAULT_ADJUST, Gray, Settings, matte, pipeline};
use serde_json::{Value, json};

use crate::Session;

/// Long edge of the preview the model sees (the spec's 2000 px).
pub const PREVIEW_EDGE: usize = 2000;
/// With the recovery pass on, previews are rendered larger so crops stay sharp.
pub const RECOVERY_EDGE: usize = 3000;
/// The model gets images up to this long edge.
const MODEL_EDGE: usize = 2048;
/// Most photos one command takes.
pub const MAX_BATCH: usize = 500;

/// Where the API key lives. Implementations must never log or return the key in an error.
pub trait KeyStore: Send + Sync {
    fn get(&self) -> Result<Option<String>, String>;
    fn set(&self, key: &str) -> Result<(), String>;
    fn clear(&self) -> Result<(), String>;
    /// Whether [`KeyStore::set`] survives a restart (the OS keychain does; the fallback doesn't).
    fn persistent(&self) -> bool;
    /// Where the key is kept, for the settings screen.
    fn describe(&self) -> &'static str;
}

/// The built-in store: the `GOOGLE_CLOUD_API_KEY` (or `LIGHTCRAFT_GOOGLE_API_KEY`) environment
/// variable, else a key typed in this session, which is kept in memory only. The desktop app
/// replaces it with the OS keychain.
#[derive(Default)]
pub struct EnvKeys {
    typed: Mutex<Option<String>>,
}

impl KeyStore for EnvKeys {
    fn get(&self) -> Result<Option<String>, String> {
        if let Some(k) = self.typed.lock().unwrap_or_else(PoisonError::into_inner).clone() {
            return Ok(Some(k));
        }
        Ok(["GOOGLE_CLOUD_API_KEY", "LIGHTCRAFT_GOOGLE_API_KEY"]
            .iter()
            .find_map(|n| std::env::var(n).ok())
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty()))
    }

    fn set(&self, key: &str) -> Result<(), String> {
        *self.typed.lock().unwrap_or_else(PoisonError::into_inner) = Some(key.to_string());
        Ok(())
    }

    fn clear(&self) -> Result<(), String> {
        *self.typed.lock().unwrap_or_else(PoisonError::into_inner) = None;
        Ok(())
    }

    fn persistent(&self) -> bool {
        false
    }

    fn describe(&self) -> &'static str {
        "this session only (or the GOOGLE_CLOUD_API_KEY environment variable)"
    }
}

/// A finished photo.
struct Outcome {
    photo: PhotoId,
    result: Result<Option<Gray>, String>,
}

/// One batch in progress.
struct Run {
    queue: VecDeque<PhotoId>,
    in_flight: HashSet<PhotoId>,
    total: usize,
    done: usize,
    errors: Vec<(PhotoId, String)>,
    found: Vec<PhotoId>,
    empty: Vec<PhotoId>,
    cancel: Arc<AtomicBool>,
    transport: Arc<dyn Transport>,
    settings: Settings,
}

/// What `window.status` reports.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Status {
    pub running: bool,
    pub total: usize,
    pub done: usize,
    pub current: Vec<u64>,
    pub found: Vec<u64>,
    /// Photos the model saw no window in.
    pub none: Vec<u64>,
    pub errors: Vec<PhotoError>,
    pub cancelled: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PhotoError {
    pub photo: u64,
    pub error: String,
}

/// The AI Window mask feature's state in a [`Session`].
pub struct WindowMasks {
    pub settings: Settings,
    /// Where the settings are saved (set by the app; `None`: not persisted).
    pub settings_file: Option<PathBuf>,
    /// Run photos in the background and apply them in [`Session::window_poll`] (the desktop
    /// app); otherwise commands wait for the batch (CLI, MCP, tests).
    pub background: bool,
    pub keys: Arc<dyn KeyStore>,
    /// Replaces the HTTPS transport (tests replay recorded answers).
    pub transport: Option<Arc<dyn Transport>>,
    pub sleeper: Arc<dyn Sleeper>,
    run: Option<Run>,
    last: Status,
    messages: Vec<String>,
    results: (Sender<Outcome>, Receiver<Outcome>),
}

impl Default for WindowMasks {
    fn default() -> Self {
        WindowMasks {
            settings: Settings::default(),
            settings_file: None,
            background: false,
            keys: Arc::new(EnvKeys::default()),
            transport: None,
            sleeper: Arc::new(RealSleeper),
            run: None,
            last: Status::default(),
            messages: Vec::new(),
            results: channel(),
        }
    }
}

impl WindowMasks {
    /// Point at the settings file and load it (a missing or damaged file means the defaults, i.e.
    /// the feature stays off; the damage is reported once).
    pub fn load_settings(&mut self, file: PathBuf) {
        match std::fs::read(&file) {
            Ok(bytes) => match serde_json::from_slice::<Settings>(&bytes) {
                Ok(s) => self.settings = s.sanitized(),
                Err(e) => self.messages.push(format!("AI Window settings couldn't be read ({e}); using the defaults (the feature is off).")),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => self.messages.push(format!("AI Window settings couldn't be read ({e}); using the defaults (the feature is off).")),
        }
        self.settings_file = Some(file);
    }

    fn save_settings(&self) -> Result<(), String> {
        let Some(file) = &self.settings_file else { return Ok(()) };
        let json = serde_json::to_vec_pretty(&self.settings).map_err(|e| e.to_string())?;
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("AI Window settings: {e}"))?;
        }
        let tmp = file.with_extension("json.tmp");
        std::fs::write(&tmp, json).and_then(|()| std::fs::rename(&tmp, file)).map_err(|e| format!("AI Window settings: {e}"))
    }

    /// Whether a batch is running.
    pub fn busy(&self) -> bool {
        self.run.is_some()
    }

    /// The running batch, or the last one finished.
    pub fn status(&self) -> Status {
        match &self.run {
            None => self.last.clone(),
            Some(r) => Status {
                running: true,
                total: r.total,
                done: r.done,
                current: r.in_flight.iter().map(|p| p.0).collect(),
                found: r.found.iter().map(|p| p.0).collect(),
                none: r.empty.iter().map(|p| p.0).collect(),
                errors: r.errors.iter().map(|(p, e)| PhotoError { photo: p.0, error: e.clone() }).collect(),
                cancelled: r.cancel.load(Ordering::Relaxed),
            },
        }
    }

    /// Settings as the `window.settings` command shows them.
    pub fn describe(&self) -> Value {
        let has_key = self.keys.get().ok().flatten().is_some();
        let mut v = json!({
            "enabled": self.settings.enabled,
            "noticeAccepted": self.settings.notice_accepted,
            "model": self.settings.model,
            "boxesModel": self.settings.boxes_model,
            "concurrency": self.settings.concurrency,
            "inset": self.settings.inset,
            "plausibility": self.settings.plausibility,
            "recovery": self.settings.recovery,
            "hasKey": has_key,
            "keyStore": self.keys.describe(),
            "keyPersistent": self.keys.persistent(),
            "available": cfg!(not(target_arch = "wasm32")),
        });
        if let Some(f) = &self.settings_file {
            v["file"] = json!(f.display().to_string());
        }
        v
    }
}

/// The photo as the model sees it.
struct Shot {
    /// Up to [`RECOVERY_EDGE`], for crops.
    full: Rgb,
    /// Up to [`MODEL_EDGE`].
    main: Rgb,
    luma: Gray,
}

#[derive(Clone)]
struct Rgb {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

impl Rgb {
    fn resized(&self, w: usize, h: usize) -> Rgb {
        let mut data = vec![0u8; w * h * 3];
        let (sx, sy) = (self.w as f32 / w as f32, self.h as f32 / h as f32);
        let at =
            |x: usize, y: usize, c: usize| f32::from(self.data.get((y.min(self.h - 1) * self.w + x.min(self.w - 1)) * 3 + c).copied().unwrap_or(0));
        for y in 0..h {
            let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
            let (y0, ty) = (fy as usize, fy.fract());
            for x in 0..w {
                let fx = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
                let (x0, tx) = (fx as usize, fx.fract());
                for c in 0..3 {
                    let top = at(x0, y0, c) * (1.0 - tx) + at(x0 + 1, y0, c) * tx;
                    let bot = at(x0, y0 + 1, c) * (1.0 - tx) + at(x0 + 1, y0 + 1, c) * tx;
                    if let Some(o) = data.get_mut((y * w + x) * 3 + c) {
                        *o = (top * (1.0 - ty) + bot * ty + 0.5) as u8;
                    }
                }
            }
        }
        Rgb { w, h, data }
    }

    fn fit(&self, edge: usize) -> Rgb {
        let long = self.w.max(self.h);
        if long <= edge {
            return self.clone();
        }
        let s = edge as f32 / long as f32;
        self.resized(((self.w as f32 * s).round() as usize).max(1), ((self.h as f32 * s).round() as usize).max(1))
    }

    fn crop(&self, r: [f32; 4]) -> Option<Rgb> {
        let px = |v: f32, n: usize| ((v.clamp(0.0, 1.0) * n as f32).round() as usize).min(n);
        let (x0, x1, y0, y1) = (px(r[0], self.w), px(r[2], self.w), px(r[1], self.h), px(r[3], self.h));
        if x1 <= x0 + 8 || y1 <= y0 + 8 {
            return None;
        }
        let (w, h) = (x1 - x0, y1 - y0);
        let mut data = Vec::with_capacity(w * h * 3);
        for y in y0..y1 {
            data.extend_from_slice(self.data.get((y * self.w + x0) * 3..(y * self.w + x1) * 3)?);
        }
        Some(Rgb { w, h, data })
    }

    fn jpeg(&self) -> lightcraft_window::Result<Vec<u8>> {
        let img = EncodeImage::new(self.w as u32, self.h as u32, 3, Samples::U8(&self.data));
        encode_jpeg(&img, 95, ChromaSubsampling::S444, &EncodeMeta::default())
            .map_err(|e| lightcraft_window::Error::Other(format!("couldn't encode the preview: {e}")))
    }
}

impl Preview for Shot {
    fn luma(&self) -> &Gray {
        &self.luma
    }

    fn jpeg(&self, rect: Option<[f32; 4]>) -> lightcraft_window::Result<Vec<u8>> {
        match rect {
            None => self.main.jpeg(),
            Some(r) => {
                let crop = self.full.crop(r).ok_or_else(|| lightcraft_window::Error::Other("the window crop is too small".into()))?;
                // a small crop is enlarged so the model has something to look at
                let long = crop.w.max(crop.h);
                let crop = if long < 1024 {
                    let s = 1024.0 / long as f32;
                    crop.resized(((crop.w as f32 * s) as usize).max(1), ((crop.h as f32 * s) as usize).max(1))
                } else {
                    crop.fit(MODEL_EDGE)
                };
                crop.jpeg()
            }
        }
    }
}

/// Everything a worker thread needs for one photo.
struct Work {
    photo: PhotoId,
    render: crate::media::RenderJob,
    missing: Option<String>,
    settings: Settings,
    transport: Arc<dyn Transport>,
    sleeper: Arc<dyn Sleeper>,
    cancel: Arc<AtomicBool>,
    reply: Sender<Outcome>,
}

fn run_work(w: Work) {
    let Work { photo, render, missing, settings, transport, sleeper, cancel, reply } = w;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Option<Gray>, String> {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let r = render.run();
        let img = match r.rendered {
            Ok(r) => r.image,
            Err(e) => {
                return Err(match missing {
                    Some(path) => format!("This photo's file is missing (moved or deleted): {path}. Library ▸ Find Missing Photos can relink it."),
                    None => format!("couldn't render the photo: {e}"),
                });
            }
        };
        let full = Rgb { w: img.width, h: img.height, data: img.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect() };
        let main = full.fit(PREVIEW_EDGE);
        let luma = Gray::from_vec(
            main.w,
            main.h,
            main.data.chunks_exact(3).map(|p| (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255.0).collect(),
        )
        .ok_or("the photo has an unusable size")?;
        let shot = Shot { full, main, luma };
        lightcraft_window::run::find_windows(&shot, transport.as_ref(), sleeper.as_ref(), &settings, &cancel)
            .map(|f| f.matte)
            .map_err(|e| e.to_string())
    }))
    .unwrap_or_else(|_| Err("the window mask worker failed unexpectedly".into()));
    let _ = reply.send(Outcome { photo, result });
}

/// A new mask for a Window component.
fn window_mask(next: u32, comp: MaskComponent) -> Mask {
    let (exposure, highlights, temp, saturation, dehaze) = DEFAULT_ADJUST;
    Mask {
        id: next,
        name: "Window".into(),
        adjust: LocalAdjustments { exposure, highlights, temp, saturation, dehaze, ..LocalAdjustments::default() },
        components: vec![comp],
        ..Mask::default()
    }
}

fn is_window(c: &MaskComponent) -> bool {
    matches!(c.shape, MaskShape::Window { .. })
}

/// The stored shape for a cleaned matte and an inset (pixels of a 6000 px wide frame).
fn window_shape(source: &Gray, inset: f64, edge: f64) -> MaskShape {
    let px = pipeline::inset_px(inset, source.w);
    let cut = matte::inset(source, px);
    MaskShape::Window {
        seg: Some(SegMask::from_grid(cut.w, cut.h, &matte::to_logits(&cut))),
        source: Some(SegMask::from_grid(source.w, source.h, &matte::to_logits(source))),
        inset,
        edge,
    }
}

impl Session {
    /// Why the feature can't run right now.
    fn window_ready(&self) -> Result<(), crate::EngineError> {
        let w = &self.window;
        if cfg!(target_arch = "wasm32") {
            return Err(crate::EngineError::Other("AI Window masks aren't available in the web build".into()));
        }
        if !w.settings.enabled {
            return Err(crate::EngineError::Other(
                "AI Window is off. It sends a preview of each photo to Google; turn it on in Settings ▸ AI Window (window.settings {enabled: true, noticeAccepted: true}).".into(),
            ));
        }
        if !w.settings.notice_accepted {
            return Err(crate::EngineError::Other(
                "Read and accept the AI Window privacy notice first: photo previews are sent to Google's Gemini API with your own API key (window.settings {noticeAccepted: true}).".into(),
            ));
        }
        Ok(())
    }

    /// Start masking the windows of `ids` (default: the selection, else the active photo). In
    /// background mode this returns at once; otherwise it waits for the batch.
    pub fn window_start(&mut self, ids: Vec<PhotoId>) -> Result<Value, crate::EngineError> {
        self.window_ready()?;
        if self.window.run.is_some() {
            return Err(crate::EngineError::Other("AI Window is already running; cancel it first (window.cancel)".into()));
        }
        let mut seen = HashSet::new();
        let ids: Vec<PhotoId> = ids.into_iter().filter(|i| self.catalog.photo(*i).is_some() && seen.insert(*i)).take(MAX_BATCH).collect();
        if ids.is_empty() {
            return Err(crate::EngineError::Other("no photo to mask".into()));
        }
        let transport: Arc<dyn Transport> = match &self.window.transport {
            Some(t) => t.clone(),
            None => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let key = self
                        .window
                        .keys
                        .get()
                        .map_err(crate::EngineError::Other)?
                        .ok_or(lightcraft_window::Error::NoKey)
                        .map_err(|e| crate::EngineError::Other(e.to_string()))?;
                    Arc::new(lightcraft_window::net::HttpTransport::new(&key).map_err(|e| crate::EngineError::Other(e.to_string()))?)
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err(crate::EngineError::Other("AI Window masks aren't available in the web build".into()));
                }
            }
        };
        while self.window.results.1.try_recv().is_ok() {}
        let total = ids.len();
        self.window.run = Some(Run {
            queue: ids.into(),
            in_flight: HashSet::new(),
            total,
            done: 0,
            errors: Vec::new(),
            found: Vec::new(),
            empty: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            transport,
            settings: self.window.settings.clone(),
        });
        self.window_pump();
        if self.window.background {
            return Ok(json!({"started": true, "total": total}));
        }
        while self.window.run.is_some() {
            match self.window.results.1.recv_timeout(std::time::Duration::from_secs(1)) {
                Ok(o) => self.window_apply(o),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(_) => break,
            }
            self.window_pump();
        }
        let s = self.window.last.clone();
        if s.found.is_empty() && !s.errors.is_empty() && s.errors.len() == s.total {
            let first = s.errors.first().map(|e| e.error.clone()).unwrap_or_default();
            return Err(crate::EngineError::Other(first));
        }
        Ok(json!(s))
    }

    /// Start photos from the queue while there is room, and finish the batch when it is done.
    fn window_pump(&mut self) {
        loop {
            let Some(run) = self.window.run.as_ref() else { return };
            let cancelled = run.cancel.load(Ordering::Relaxed);
            if cancelled && let Some(r) = self.window.run.as_mut() {
                r.done += r.queue.len();
                r.queue.clear();
            }
            let Some(run) = self.window.run.as_ref() else { return };
            if run.queue.is_empty() && run.in_flight.is_empty() {
                let status = Status {
                    running: false,
                    total: run.total,
                    done: run.done,
                    current: vec![],
                    found: run.found.iter().map(|p| p.0).collect(),
                    none: run.empty.iter().map(|p| p.0).collect(),
                    errors: run.errors.iter().map(|(p, e)| PhotoError { photo: p.0, error: e.clone() }).collect(),
                    cancelled,
                };
                self.window.last = status;
                self.window.run = None;
                return;
            }
            if run.in_flight.len() >= run.settings.concurrency.max(1) as usize {
                return;
            }
            let Some(photo) = self.window.run.as_mut().and_then(|r| r.queue.pop_front()) else { return };
            match self.window_work(photo) {
                Ok(work) => {
                    if let Some(r) = self.window.run.as_mut() {
                        r.in_flight.insert(photo);
                    }
                    if std::thread::Builder::new().name("window-mask".into()).spawn(move || run_work(work)).is_err() {
                        self.window_fail(photo, "couldn't start a worker thread".into());
                    }
                }
                Err(e) => self.window_fail(photo, e),
            }
        }
    }

    fn window_fail(&mut self, photo: PhotoId, error: String) {
        if let Some(r) = self.window.run.as_mut() {
            r.in_flight.remove(&photo);
            r.done += 1;
            r.errors.push((photo, error));
        }
    }

    /// The render job and settings for one photo: its look without masks, uncropped (mask
    /// coordinates live in the full oriented frame), so what the model sees matches the photo
    /// and does not change when the mask is adjusted.
    fn window_work(&mut self, photo: PhotoId) -> Result<Work, String> {
        let run = self.window.run.as_ref().ok_or("not running")?;
        let (settings, transport, cancel) = (run.settings.clone(), run.transport.clone(), run.cancel.clone());
        let p = self.catalog.photo(photo).ok_or("no such photo")?.clone();
        let mut d = (*p.develop).clone();
        d.masks.clear();
        let edge = if settings.recovery { RECOVERY_EDGE } else { PREVIEW_EDGE };
        let render = self.preview_job(photo, edge, edge, false, &d).ok_or("no such photo")?;
        let missing = match &p.source {
            lightcraft_catalog::Source::File { path } if !std::path::Path::new(path).exists() => Some(path.clone()),
            _ => None,
        };
        Ok(Work { photo, render, missing, settings, transport, sleeper: self.window.sleeper.clone(), cancel, reply: self.window.results.0.clone() })
    }

    /// Apply one finished photo.
    fn window_apply(&mut self, o: Outcome) {
        let Some(run) = self.window.run.as_mut() else { return };
        run.in_flight.remove(&o.photo);
        run.done += 1;
        match o.result {
            Err(e) if e == "cancelled" || e == lightcraft_window::Error::Cancelled.to_string() => {}
            Err(e) => run.errors.push((o.photo, e)),
            Ok(None) => run.empty.push(o.photo),
            Ok(Some(m)) => {
                let inset = run.settings.inset;
                match self.window_store(o.photo, &m, inset) {
                    Ok(()) => {
                        if let Some(r) = self.window.run.as_mut() {
                            r.found.push(o.photo);
                        }
                    }
                    Err(e) => {
                        if let Some(r) = self.window.run.as_mut() {
                            r.errors.push((o.photo, e));
                        }
                    }
                }
            }
        }
    }

    /// Put a matte on photo `photo`: replaces its previous Window component (keeping the mask's
    /// other components and adjustments), or creates the mask. Other masks are never touched.
    fn window_store(&mut self, photo: PhotoId, m: &Gray, inset: f64) -> Result<(), String> {
        let mut d = (*self.develop_of(photo).ok_or("the photo was removed")?).clone();
        let edge = d
            .masks
            .iter()
            .flat_map(|k| &k.components)
            .find_map(|c| if let MaskShape::Window { edge, .. } = c.shape { Some(edge) } else { None })
            .unwrap_or(0.0);
        let shape = window_shape(m, inset, edge);
        let existing = d.masks.iter_mut().find_map(|k| k.components.iter_mut().find(|c| is_window(c)));
        match existing {
            Some(c) => c.shape = shape,
            None => {
                let next = d.next_mask_id();
                d.masks.push(window_mask(next, MaskComponent { name: None, op: MaskOp::Add, invert: false, shape }));
                if self.active() == Some(photo) {
                    self.active_mask = Some(next);
                }
            }
        }
        self.set_develop(photo, d, "Window Mask").map_err(|e| e.to_string())
    }

    /// Redo a Window component's inset from its stored pre-inset matte (no model call).
    pub fn window_set_inset(&mut self, photo: PhotoId, mask: Option<u32>, inset: f64) -> Result<(), crate::EngineError> {
        let err = |m: &str| crate::EngineError::Other(m.to_string());
        let inset = if inset.is_finite() { inset.clamp(0.0, 40.0) } else { return Err(err("inset must be a number")) };
        let mut d = (*self.develop_of(photo).ok_or_else(|| err("no such photo"))?).clone();
        let comp = d
            .masks
            .iter_mut()
            .filter(|k| mask.is_none_or(|id| k.id == id))
            .flat_map(|k| k.components.iter_mut())
            .find(|c| is_window(c))
            .ok_or_else(|| err("the photo has no Window mask"))?;
        let MaskShape::Window { seg, source, inset: stored, .. } = &mut comp.shape else { return Err(err("not a Window component")) };
        let src = source.as_ref().ok_or_else(|| err("this Window mask has no stored source to re-inset; run mask.addWindow again"))?;
        let logits = src.logits().ok_or_else(|| err("the stored Window matte is damaged; run mask.addWindow again"))?;
        let g = matte::from_logits(src.side as usize, src.height(), &logits).ok_or_else(|| err("the stored Window matte is damaged"))?;
        let cut = matte::inset(&g, pipeline::inset_px(inset, g.w));
        *seg = Some(SegMask::from_grid(cut.w, cut.h, &matte::to_logits(&cut)));
        *stored = inset;
        self.set_develop(photo, d, "Window Inset")
    }

    /// Cancel the running batch: photos not started are skipped, calls in flight stop at the next
    /// network read or retry wait.
    pub fn window_cancel(&mut self) -> bool {
        match &self.window.run {
            Some(r) => {
                r.cancel.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    /// Apply finished photos and start the next ones; call every frame (cheap when idle). Returns
    /// the messages to show and whether any photo changed.
    pub fn window_poll(&mut self) -> WindowPolled {
        let mut polled = WindowPolled { messages: std::mem::take(&mut self.window.messages), changed: false };
        if self.window.run.is_none() {
            return polled;
        }
        let was = self.window.run.as_ref().map(|r| r.found.len()).unwrap_or(0);
        let done: Vec<Outcome> = self.window.results.1.try_iter().collect();
        for o in done {
            self.window_apply(o);
        }
        self.window_pump();
        let now = self.window.run.as_ref().map(|r| r.found.len()).unwrap_or(self.window.last.found.len());
        polled.changed = now != was || self.window.run.is_none();
        if self.window.run.is_none() {
            let s = &self.window.last;
            for e in &s.errors {
                polled.messages.push(format!("AI Window: {}", e.error));
            }
            if s.found.is_empty() && s.errors.is_empty() && !s.cancelled && s.total > 0 {
                polled.messages.push("AI Window: no window was found.".into());
            }
        }
        polled
    }

    /// Apply new settings from `window.settings` and save them.
    pub fn window_configure(&mut self, p: &Value) -> Result<Value, crate::EngineError> {
        let mut s = self.window.settings.clone();
        let flag = |k: &str| p.get(k).and_then(Value::as_bool);
        if let Some(v) = flag("enabled") {
            s.enabled = v;
        }
        if let Some(v) = flag("noticeAccepted") {
            s.notice_accepted = v;
        }
        if let Some(v) = p.get("model").and_then(Value::as_str) {
            s.model = v.trim().to_string();
        }
        if let Some(v) = p.get("boxesModel").and_then(Value::as_str) {
            s.boxes_model = v.trim().to_string();
        }
        if let Some(v) = p.get("concurrency").and_then(Value::as_u64) {
            s.concurrency = v.min(u64::from(u32::MAX)) as u32;
        }
        if let Some(v) = p.get("inset").and_then(Value::as_f64) {
            s.inset = v;
        }
        if let Some(v) = flag("plausibility") {
            s.plausibility = v;
        }
        if let Some(v) = flag("recovery") {
            s.recovery = v;
        }
        // an unusable value is an error, not a silent clamp, for the fields that name things
        let clean = s.clone().sanitized();
        if clean.model != s.model || clean.boxes_model != s.boxes_model {
            return Err(crate::EngineError::Other("model names may contain only letters, digits, `-`, `.` and `_`".into()));
        }
        self.window.settings = clean;
        if let Some(key) = p.get("apiKey").and_then(Value::as_str) {
            let key = key.trim();
            if key.is_empty() {
                self.window.keys.clear().map_err(crate::EngineError::Other)?;
            } else {
                self.window.keys.set(key).map_err(crate::EngineError::Other)?;
            }
        }
        self.window.save_settings().map_err(crate::EngineError::Other)?;
        Ok(self.window.describe())
    }
}

/// What [`Session::window_poll`] did.
#[derive(Debug, Default)]
pub struct WindowPolled {
    pub messages: Vec<String>,
    pub changed: bool,
}

#[cfg(test)]
mod tests;
