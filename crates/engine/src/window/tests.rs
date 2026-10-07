use std::sync::atomic::AtomicBool;

use lightcraft_window::Error;
use lightcraft_window::gemini::{Sleeper, Transport};

use super::*;

/// Answers every call with a mask picture: the left 60 % of the frame is "window".
struct Fake {
    calls: Mutex<usize>,
    status: u16,
}

fn reply() -> Vec<u8> {
    let (w, h) = (200u32, 120u32);
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().unwrap();
        let data: Vec<u8> = (0..w * h).map(|i| if (i % w) < 120 { 255 } else { 0 }).collect();
        wr.write_image_data(&data).unwrap();
    }
    let b64 = {
        // standard base64
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in out.chunks(3) {
            let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
            for i in 0..4 {
                s.push(if i <= c.len() { char::from(A[((n >> (18 - 6 * i)) & 63) as usize]) } else { '=' });
            }
        }
        s
    };
    serde_json::to_vec(
        &json!({"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": b64}}]}, "finishReason": "STOP"}]}),
    )
    .unwrap()
}

impl Transport for Fake {
    fn post(&self, _model: &str, _body: &[u8], _c: &AtomicBool) -> Result<(u16, Vec<u8>), Error> {
        *self.calls.lock().unwrap() += 1;
        if self.status == 200 {
            Ok((200, reply()))
        } else {
            Ok((self.status, br#"{"error":{"status":"PERMISSION_DENIED","message":"nope"}}"#.to_vec()))
        }
    }
}

struct Instant;
impl Sleeper for Instant {
    fn sleep(&self, _d: std::time::Duration, _c: &AtomicBool) -> bool {
        true
    }
}

fn session(status: u16) -> (Session, Arc<Fake>) {
    let mut s = Session::with_demo();
    let t = Arc::new(Fake { calls: Mutex::new(0), status });
    s.window.transport = Some(t.clone());
    s.window.sleeper = Arc::new(Instant);
    s.execute("window.settings", &json!({"enabled": true, "noticeAccepted": true, "plausibility": false, "inset": 0.0})).unwrap();
    (s, t)
}

#[test]
fn it_refuses_to_run_until_the_user_opted_in() {
    let mut s = Session::with_demo();
    let t = Arc::new(Fake { calls: Mutex::new(0), status: 200 });
    s.window.transport = Some(t.clone());
    let e = s.execute("mask.addWindow", &json!({})).unwrap_err().to_string();
    assert!(e.contains("AI Window is off"), "{e}");
    s.execute("window.settings", &json!({"enabled": true})).unwrap();
    let e = s.execute("mask.addWindow", &json!({})).unwrap_err().to_string();
    assert!(e.contains("privacy notice"), "{e}");
    assert_eq!(*t.calls.lock().unwrap(), 0, "nothing was sent");
}

#[test]
fn a_window_mask_is_an_ordinary_mask_with_the_default_adjustments() {
    let (mut s, t) = session(200);
    let id = s.active().unwrap();
    let r = s.execute("mask.addWindow", &json!({})).unwrap();
    assert_eq!(r["found"], json!([id.0]));
    assert_eq!(*t.calls.lock().unwrap(), 1);
    let d = s.develop_of(id).unwrap();
    assert_eq!(d.masks.len(), 1);
    let m = &d.masks[0];
    assert_eq!((m.adjust.exposure, m.adjust.highlights, m.adjust.temp, m.adjust.saturation, m.adjust.dehaze), (-1.0, -100.0, 10.0, 10.0, 5.0));
    let MaskShape::Window { seg: Some(seg), source: Some(_), .. } = &m.components[0].shape else { panic!("a Window component") };
    // the stored matte lines up with the picture: left side in, right side out
    let g = lightcraft_window::matte::from_logits(seg.side as usize, seg.height(), &seg.logits().unwrap()).unwrap();
    assert!(g.get(g.w / 4, g.h / 2) > 0.9 && g.get(g.w * 9 / 10, g.h / 2) < 0.1);
    // it survives the settings round trip (catalog / sidecar / sync)
    let back: lightcraft_develop::DevelopSettings = serde_json::from_value(serde_json::to_value(&*d).unwrap()).unwrap();
    assert_eq!(back.masks, d.masks);
}

#[test]
fn rerunning_replaces_only_the_window_selection() {
    let (mut s, _t) = session(200);
    let id = s.active().unwrap();
    s.execute("mask.add", &json!({"kind": "radial"})).unwrap();
    s.execute("mask.addWindow", &json!({})).unwrap();
    assert_eq!(s.develop_of(id).unwrap().masks.len(), 2);
    // the user tunes the window mask, then runs it again
    let mut d = (*s.develop_of(id).unwrap()).clone();
    d.masks[1].adjust.exposure = -2.0;
    s.set_develop(id, d, "tune").unwrap();
    let before = s.develop_of(id).unwrap().masks[0].clone();
    s.execute("mask.addWindow", &json!({})).unwrap();
    let d = s.develop_of(id).unwrap();
    assert_eq!(d.masks.len(), 2, "no second Window mask");
    assert_eq!(d.masks[0], before, "the other mask is untouched");
    assert_eq!(d.masks[1].adjust.exposure, -2.0, "adjustments are kept");
}

#[test]
fn the_inset_is_redone_locally_from_the_stored_matte() {
    let (mut s, t) = session(200);
    let id = s.active().unwrap();
    s.execute("mask.addWindow", &json!({})).unwrap();
    let get = |s: &Session| match &s.develop_of(id).unwrap().masks[0].components[0].shape {
        MaskShape::Window { seg, source, inset, .. } => (seg.clone().unwrap(), source.clone().unwrap(), *inset),
        _ => panic!(),
    };
    let (seg0, src0, _) = get(&s);
    s.execute("window.setInset", &json!({"inset": 30.0})).unwrap();
    let (seg1, src1, inset) = get(&s);
    assert_eq!(inset, 30.0);
    assert_eq!(src0, src1, "the source matte is kept");
    assert_ne!(seg0, seg1, "the visible matte shrank");
    assert_eq!(*t.calls.lock().unwrap(), 1, "no second model call");
}

#[test]
fn a_batch_reports_per_photo_errors_and_never_adds_a_mask_for_them() {
    let (mut s, _t) = session(403);
    let ids: Vec<u64> = s.catalog.photos().take(2).map(|p| p.id.0).collect();
    let e = s.execute("mask.addWindow", &json!({"ids": ids})).unwrap_err().to_string();
    assert!(e.contains("API key was not accepted"), "{e}");
    assert!(!e.contains("AQ."), "no key in messages");
    assert!(s.catalog.photos().all(|p| p.develop.masks.is_empty()));
    let st = s.window.status();
    assert!(!st.running && st.errors.len() == 2);
}

#[test]
fn settings_are_sanitized_and_the_key_is_never_reported() {
    let (mut s, _t) = session(200);
    let r = s.execute("window.settings", &json!({"concurrency": 99, "inset": 1e9, "apiKey": "secret-value"})).unwrap();
    assert_eq!(r["concurrency"], 8);
    assert_eq!(r["inset"], 40.0);
    assert_eq!(r["hasKey"], true);
    assert!(!r.to_string().contains("secret-value"));
    assert!(s.execute("window.settings", &json!({"model": "../x"})).is_err());
}
