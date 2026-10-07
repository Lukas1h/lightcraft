use std::sync::atomic::AtomicBool;

use super::*;
use crate::fixtures::*;

#[test]
fn the_request_has_the_documented_shape() {
    let body = segmentation_body(&[1, 2, 3]);
    let text = body.to_string();
    assert_eq!(body["generationConfig"]["temperature"], 0);
    assert_eq!(body["generationConfig"]["topP"], 0.95);
    assert_eq!(body["generationConfig"]["maxOutputTokens"], 32768);
    assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH");
    assert_eq!(body["generationConfig"]["imageConfig"]["imageSize"], "1K");
    assert!(!text.contains("aspectRatio"), "\"auto\" is rejected, so it must be omitted");
    assert_eq!(body["contents"][0]["parts"][1]["text"], "Mask this image.");
    assert_eq!(body["safetySettings"].as_array().unwrap().len(), 4);
    assert!(body["safetySettings"].as_array().unwrap().iter().all(|s| s["threshold"] == "OFF"));
    assert!(body["systemInstruction"]["parts"][0]["text"].as_str().unwrap().starts_with("You are a real estate photography editor assistant."));
}

#[test]
fn the_last_image_is_used() {
    let first = png_of(&crate::Gray::from_vec(1, 1, vec![0.0]).unwrap());
    let last = png_of(&crate::Gray::from_vec(2, 2, vec![1.0; 4]).unwrap());
    let img = |p: &[u8]| json!({ "inlineData": { "mimeType": "image/png", "data": crate::b64::encode(p) } });
    let v = json!({ "candidates": [{ "content": { "parts": [img(&first), { "text": "x" }, img(&last)] }, "finishReason": "STOP" }] });
    assert_eq!(last_image(&v).unwrap().unwrap(), last);
    assert_eq!(last_image(&reply_without_image()).unwrap(), None);
    assert_eq!(last_image(&json!({})).unwrap(), None);
    let blocked = json!({ "promptFeedback": { "blockReason": "SAFETY" } });
    assert!(matches!(last_image(&blocked), Err(Error::Refused(_))));
    let damaged = json!({ "candidates": [{ "content": { "parts": [{ "inlineData": { "mimeType": "image/png", "data": "@@@" } }] } }] });
    assert!(matches!(last_image(&damaged), Err(Error::BadResponse(_))));
}

#[test]
fn rate_limits_are_retried_with_backoff_and_then_succeed() {
    let ok = reply_with_image(&[1]);
    let t = Replay::new(vec![(429, error_reply(429, "RESOURCE_EXHAUSTED")), (503, error_reply(503, "UNAVAILABLE")), (200, ok.clone())]);
    let s = NoSleep::default();
    let got = call(&t, &s, "m", &json!({}), &AtomicBool::new(false)).unwrap();
    assert_eq!(got, ok);
    assert_eq!(*s.0.lock().unwrap(), vec![Duration::from_secs(10), Duration::from_secs(20)]);
    assert_eq!(t.calls.lock().unwrap().len(), 3);
}

#[test]
fn a_rate_limit_that_never_clears_is_a_clear_error_after_the_last_try() {
    let t = Replay::new(vec![(429, error_reply(429, "RESOURCE_EXHAUSTED"))]);
    let s = NoSleep::default();
    let err = call(&t, &s, "m", &json!({}), &AtomicBool::new(false)).unwrap_err();
    let Error::RateLimited { tries, status, .. } = &err else { panic!("{err:?}") };
    assert_eq!((*tries, *status), (6, 429));
    assert_eq!(t.calls.lock().unwrap().len(), 6);
    assert_eq!(s.0.lock().unwrap().iter().map(|d| d.as_secs()).collect::<Vec<_>>(), vec![10, 20, 30, 40, 50]);
    let msg = err.to_string();
    assert!(msg.contains("rate limiting") && msg.contains("6 tries"), "{msg}");
}

#[test]
fn other_failures_are_not_retried_and_never_echo_the_key() {
    let bad = json!({ "error": { "code": 403, "message": "API key not valid", "status": "PERMISSION_DENIED" } });
    let t = Replay::new(vec![(403, bad)]);
    let err = call(&t, &NoSleep::default(), "m", &json!({}), &AtomicBool::new(false)).unwrap_err();
    assert_eq!(t.calls.lock().unwrap().len(), 1);
    assert!(err.to_string().contains("API key was not accepted"));
    // the blocked-key service error is retried (it clears)
    let blocked = json!({ "error": { "code": 403, "status": "PERMISSION_DENIED", "message": "x", "details": [{ "reason": "API_KEY_SERVICE_BLOCKED" }] } });
    let t = Replay::new(vec![(403, blocked), (200, json!({}))]);
    call(&t, &NoSleep::default(), "m", &json!({}), &AtomicBool::new(false)).unwrap();
    assert_eq!(t.calls.lock().unwrap().len(), 2);
}

#[test]
fn cancelling_stops_retries() {
    let t = Replay::new(vec![(429, error_reply(429, "RESOURCE_EXHAUSTED"))]);
    let cancel = AtomicBool::new(true);
    assert_eq!(call(&t, &NoSleep::default(), "m", &json!({}), &cancel).unwrap_err(), Error::Cancelled);
    assert!(t.calls.lock().unwrap().is_empty());
}

#[test]
fn boxes_are_parsed_leniently() {
    let b = parse_boxes("```json\n[{\"box_2d\":[100,200,300,400],\"label\":\"window\"},{\"box_2d\":[1,2],\"label\":\"x\"},{\"label\":\"y\"}]\n```");
    assert_eq!(b, vec![PaneBox([0.2, 0.1, 0.4, 0.3])]);
    assert!(parse_boxes("not json").is_empty());
    assert!(parse_boxes("[]").is_empty());
}
