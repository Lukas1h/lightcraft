//! Synthetic interiors and "recorded" model replies for the tests (no network, no media files).
//!
//! A scene is drawn procedurally: wall, windows with a grid of lites separated by mullions, things
//! in front (a chair back, blinds). The model's answer is the ground-truth glass mask rendered at
//! 1264 px with a soft anti-aliased edge, wrapped in the JSON the service returns (the same image
//! twice, as the real model does).

use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde_json::{Value, json};

use crate::gemini::{Sleeper, Transport};
use crate::matte::Gray;
use crate::run::Preview;
use crate::{Error, b64};

pub struct Scene {
    pub luma: Gray,
    /// 1 on window glass.
    pub truth: Gray,
    /// Lite centres (x, y) in pixels.
    pub lites: Vec<(usize, usize)>,
    /// Points that must stay out of the mask (mullions, chair back, blinds).
    pub frame_points: Vec<(usize, usize)>,
}

pub struct Window {
    pub x: usize,
    pub y: usize,
    pub cols: usize,
    pub rows: usize,
    /// Lite size and mullion thickness.
    pub lite: (usize, usize),
    pub mullion: usize,
    /// Luminance behind each lite (row-major), cycled.
    pub outside: Vec<f32>,
}

pub fn scene(w: usize, h: usize, wall: f32, windows: &[Window], chair: Option<(usize, usize, usize, usize)>) -> Scene {
    let mut luma = Gray { w, h, data: vec![wall; w * h] };
    let mut truth = Gray { w, h, data: vec![0.0; w * h] };
    let (mut lites, mut frame_points) = (Vec::new(), Vec::new());
    let fill = |g: &mut Gray, x0: usize, y0: usize, x1: usize, y1: usize, v: f32| {
        for y in y0..y1.min(g.h) {
            for x in x0..x1.min(g.w) {
                g.data[y * g.w + x] = v;
            }
        }
    };
    for win in windows {
        let (tw, th) = (win.cols * win.lite.0 + (win.cols + 1) * win.mullion, win.rows * win.lite.1 + (win.rows + 1) * win.mullion);
        // the white frame / mullions are bright too (the model must not be fooled by them)
        fill(&mut luma, win.x, win.y, win.x + tw, win.y + th, 0.97);
        for r in 0..win.rows {
            for c in 0..win.cols {
                let x0 = win.x + win.mullion + c * (win.lite.0 + win.mullion);
                let y0 = win.y + win.mullion + r * (win.lite.1 + win.mullion);
                let v = win.outside[(r * win.cols + c) % win.outside.len()];
                fill(&mut luma, x0, y0, x0 + win.lite.0, y0 + win.lite.1, v);
                fill(&mut truth, x0, y0, x0 + win.lite.0, y0 + win.lite.1, 1.0);
                lites.push((x0 + win.lite.0 / 2, y0 + win.lite.1 / 2));
                frame_points.push((x0 + win.lite.0 + win.mullion / 2, y0 + win.lite.1 / 2));
                frame_points.push((x0 + win.lite.0 / 2, y0 + win.lite.1 + win.mullion / 2));
            }
        }
    }
    if let Some((x0, y0, x1, y1)) = chair {
        // in front of the glass: dark, and not part of the truth
        fill(&mut luma, x0, y0, x1, y1, 0.1);
        fill(&mut truth, x0, y0, x1, y1, 0.0);
        frame_points.push(((x0 + x1) / 2, (y0 + y1) / 2));
        lites.retain(|(x, y)| !(*x >= x0 && *x < x1 && *y >= y0 && *y < y1));
    }
    Scene { luma, truth, lites, frame_points }
}

/// The model's picture of `truth`: resampled to `side` px wide with a soft edge (box blur of ~1 px).
pub fn model_matte(truth: &Gray, side: usize) -> Gray {
    let h = (truth.h * side).div_ceil(truth.w);
    let mut out = Gray { w: side, h, data: vec![0.0; side * h] };
    let (sx, sy) = (truth.w as f32 / side as f32, truth.h as f32 / h as f32);
    for y in 0..h {
        for x in 0..side {
            // area average of the covered source pixels = an anti-aliased edge ramp
            let (x0, x1) = ((x as f32 * sx) as usize, (((x + 1) as f32 * sx).ceil() as usize).min(truth.w));
            let (y0, y1) = ((y as f32 * sy) as usize, (((y + 1) as f32 * sy).ceil() as usize).min(truth.h));
            let (mut s, mut n) = (0.0, 0.0);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    s += truth.data[yy * truth.w + xx];
                    n += 1.0;
                }
            }
            out.data[y * side + x] = if n > 0.0 { s / n } else { 0.0 };
        }
    }
    out
}

pub fn png_of(m: &Gray) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, m.w as u32, m.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().unwrap();
        let data: Vec<u8> = m.data.iter().flat_map(|v| {
            let b = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            [b, b, b]
        }).collect();
        w.write_image_data(&data).unwrap();
    }
    out
}

/// A recorded answer: text, the picture mid-thought, more text, the picture again as the answer.
pub fn reply_with_image(png: &[u8]) -> Value {
    let img = json!({ "inlineData": { "mimeType": "image/png", "data": b64::encode(png) } });
    json!({ "candidates": [{ "content": { "role": "model", "parts": [
        { "text": "thinking about the windows", "thought": true }, img.clone(),
        { "text": "Here is the mask." }, img ] }, "finishReason": "STOP" }] })
}

pub fn reply_without_image() -> Value {
    json!({ "candidates": [{ "content": { "role": "model", "parts": [{ "text": "There is no window in this photo." }] }, "finishReason": "STOP" }] })
}

pub fn error_reply(code: u16, status: &str) -> Value {
    json!({ "error": { "code": code, "message": "Resource has been exhausted", "status": status } })
}

/// Replays queued answers (status, JSON) in order; the last one repeats. Counts the calls.
pub struct Replay {
    answers: Mutex<Vec<(u16, Value)>>,
    pub calls: Mutex<Vec<(String, Vec<u8>)>>,
}

impl Replay {
    pub fn new(answers: Vec<(u16, Value)>) -> Replay {
        Replay { answers: Mutex::new(answers), calls: Mutex::new(Vec::new()) }
    }
}

impl Transport for Replay {
    fn post(&self, model: &str, body: &[u8], _cancel: &AtomicBool) -> Result<(u16, Vec<u8>), Error> {
        self.calls.lock().unwrap().push((model.to_string(), body.to_vec()));
        let mut a = self.answers.lock().unwrap();
        let (status, v) = if a.len() > 1 { a.remove(0) } else { a[0].clone() };
        Ok((status, serde_json::to_vec(&v).unwrap()))
    }
}

/// Records the requested waits instead of sleeping.
#[derive(Default)]
pub struct NoSleep(pub Mutex<Vec<Duration>>);

impl Sleeper for NoSleep {
    fn sleep(&self, d: Duration, _cancel: &AtomicBool) -> bool {
        self.0.lock().unwrap().push(d);
        true
    }
}

pub struct TestPreview(pub Gray);

impl Preview for TestPreview {
    fn luma(&self) -> &Gray {
        &self.0
    }
    fn jpeg(&self, _rect: Option<[f32; 4]>) -> Result<Vec<u8>, Error> {
        Ok(vec![0xff, 0xd8, 0xff, 0xd9])
    }
}
