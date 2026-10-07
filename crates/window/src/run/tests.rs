use std::sync::atomic::AtomicBool;

use serde_json::json;

use super::*;
use crate::fixtures::*;

fn bay() -> (Scene, Vec<Window>) {
    // three 3×3 windows; dark trees behind some lites, bright sky behind others
    let outside = vec![0.95, 0.3, 0.9, 0.25, 0.97, 0.28, 0.93, 0.92, 0.3];
    let mk = |x| Window { x, y: 60, cols: 3, rows: 3, lite: (60, 70), mullion: 8, outside: outside.clone() };
    let ws = vec![mk(40), mk(40 + 3 * 60 + 4 * 8 + 40), mk(40 + 2 * (3 * 60 + 4 * 8 + 40))];
    (scene(840, 460, 0.55, &ws, None), ws)
}

fn run(scene: &Scene, replies: Vec<(u16, serde_json::Value)>, s: &Settings) -> (Result<Found>, Replay) {
    let t = Replay::new(replies);
    let found = find_windows(&TestPreview(scene.luma.clone()), &t, &NoSleep::default(), s, &AtomicBool::new(false));
    (found, t)
}

fn settings() -> Settings {
    Settings { inset: 0.0, ..Settings::default() }
}

#[test]
fn a_three_window_bay_is_masked_lite_by_lite() {
    let (sc, _) = bay();
    let answer = reply_with_image(&png_of(&model_matte(&sc.truth, 400)));
    let (found, t) = run(&sc, vec![(200, answer)], &settings());
    let m = found.unwrap().matte.expect("windows found");
    for (x, y) in &sc.lites {
        assert!(m.get(*x, *y) > 0.9, "lite at {x},{y} is missing");
    }
    for (x, y) in &sc.frame_points {
        assert!(m.get(*x, *y) < 0.1, "frame at {x},{y} is in the mask");
    }
    // coverage matches the glass area within a percent
    let want = sc.truth.coverage();
    assert!((m.coverage() - want).abs() < want * 0.03, "{} vs {want}", m.coverage());
    assert_eq!(t.calls.lock().unwrap().len(), 1, "one call, recovery is off");
    assert_eq!(t.calls.lock().unwrap()[0].0, "gemini-3.1-flash-lite-image");
}

#[test]
fn dim_lites_survive_through_neighbour_rescue() {
    // a dark, contrasty render: every lite but one is dim; none of them reaches 200/255 alone
    let outside = vec![0.2, 0.22, 0.25, 0.97, 0.21, 0.24];
    let w = Window { x: 40, y: 50, cols: 3, rows: 2, lite: (70, 90), mullion: 4, outside };
    let sc = scene(400, 300, 0.2, &[w], None);
    let answer = reply_with_image(&png_of(&model_matte(&sc.truth, 300)));
    let (found, _) = run(&sc, vec![(200, answer.clone())], &settings());
    let m = found.unwrap().matte.unwrap();
    assert!(sc.lites.iter().all(|(x, y)| m.get(*x, *y) > 0.9), "a dim lite was dropped");
    // without the filter everything is kept too; with no bright lite at all, the filter rejects all
    let dark = scene(400, 300, 0.2, &[Window { x: 40, y: 50, cols: 3, rows: 2, lite: (70, 90), mullion: 4, outside: vec![0.2] }], None);
    let answer = reply_with_image(&png_of(&model_matte(&dark.truth, 300)));
    assert!(run(&dark, vec![(200, answer.clone())], &settings()).0.unwrap().matte.is_none());
    let off = Settings { plausibility: false, ..settings() };
    assert!(run(&dark, vec![(200, answer)], &off).0.unwrap().matte.is_some());
}

#[test]
fn a_hallucinated_window_on_an_appliance_is_rejected() {
    // no window: a brushed-steel fridge (luma 0.55) on a flat wall; the model "masks" the fridge
    let mut sc = scene(400, 300, 0.5, &[], None);
    let mut fridge = crate::Gray::new(400, 300).unwrap();
    for y in 40..260 {
        for x in 150..250 {
            sc.luma.data[y * 400 + x] = 0.55;
            fridge.data[y * 400 + x] = 1.0;
        }
    }
    let answer = reply_with_image(&png_of(&model_matte(&fridge, 300)));
    assert!(run(&sc, vec![(200, answer)], &settings()).0.unwrap().matte.is_none());
    // and the model saying "no window" is simply an empty result, not an error
    let (found, _) = run(&sc, vec![(200, reply_without_image())], &settings());
    let found = found.unwrap();
    assert!(found.matte.is_none() && found.calls == 1);
}

#[test]
fn the_recovery_pass_fills_an_under_covered_window_and_only_adds() {
    let (sc, ws) = bay();
    // the first answer misses the whole middle window
    let mut partial = sc.truth.clone();
    let x0 = ws[1].x;
    for y in 0..partial.h {
        for x in x0..x0 + 3 * 60 + 4 * 8 {
            partial.data[y * partial.w + x] = 0.0;
        }
    }
    // the detector reports the middle window's panes
    let mut boxes = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            let px = x0 + 8 + c * 68;
            let py = 60 + 8 + r * 78;
            boxes.push(json!({ "box_2d": [py * 1000 / 460, px * 1000 / 840, (py + 70) * 1000 / 460, (px + 60) * 1000 / 840], "label": "window" }));
        }
    }
    let detect = json!({ "candidates": [{ "content": { "parts": [{ "text": serde_json::to_string(&boxes).unwrap() }] }, "finishReason": "STOP" }] });
    let first = reply_with_image(&png_of(&model_matte(&partial, 400)));
    let s = Settings { recovery: true, ..settings() };
    // the rect the pass crops is the window's bounds + 15 %, so serve a crop answer that matches it
    let b = crate::panes::group(&crate::gemini::parse_boxes(&serde_json::to_string(&boxes).unwrap()), 0.25);
    let rect = crate::panes::padded(b[0].bounds, 0.15);
    let mut crop = crate::Gray::new(300, 300).unwrap();
    for y in 0..300 {
        for x in 0..300 {
            let u = rect[0] + (x as f32 + 0.5) / 300.0 * (rect[2] - rect[0]);
            let v = rect[1] + (y as f32 + 0.5) / 300.0 * (rect[3] - rect[1]);
            crop.data[y * 300 + x] = sc.truth.sample(u * sc.truth.w as f32, v * sc.truth.h as f32);
        }
    }
    let (found, t) = run(&sc, vec![(200, first.clone()), (200, detect.clone()), (200, reply_with_image(&png_of(&crop)))], &s);
    let m = found.unwrap().matte.unwrap();
    assert!(sc.lites.iter().all(|(x, y)| m.get(*x, *y) > 0.9), "recovery left a lite out");
    let calls = t.calls.lock().unwrap();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[1].0, "gemini-3.8-flash");
    // without recovery the middle window stays missing
    let (found, _) = run(&sc, vec![(200, first)], &settings());
    let m = found.unwrap().matte.unwrap();
    assert!(m.get(ws[1].x + 38, 100) < 0.1);
}
