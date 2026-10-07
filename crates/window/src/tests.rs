use super::*;

#[test]
fn settings_default_to_off_and_sanitize() {
    let d = Settings::default();
    assert!(!d.enabled && !d.notice_accepted);
    assert_eq!((d.concurrency, d.inset, d.plausibility, d.recovery), (1, 4.0, true, false));
    let bad = Settings { model: "../../etc".into(), concurrency: 99, inset: f64::NAN, boxes_model: String::new(), ..Settings::default() }.sanitized();
    assert_eq!((bad.model.as_str(), bad.concurrency), ("gemini-3.1-flash-lite-image", MAX_CONCURRENCY));
    assert_eq!((bad.inset, bad.boxes_model.as_str()), (4.0, "gemini-3.8-flash"));
    // a damaged file with missing fields keeps the defaults
    let s: Settings = serde_json::from_str(r#"{"enabled":true}"#).unwrap();
    assert!(s.enabled && s.model == d.model);
}

#[test]
fn default_adjustments_use_stops_not_quarter_stops() {
    assert_eq!(DEFAULT_ADJUST.0, -1.0);
}

#[test]
fn the_transport_refuses_an_empty_key_and_hides_it_in_debug() {
    assert_eq!(net::HttpTransport::new("  ").unwrap_err(), Error::NoKey);
    let t = net::HttpTransport::new("secret-key-123").unwrap();
    assert!(!format!("{t:?}").contains("secret-key-123"));
    assert!(net::HttpTransport::new("a b").is_err());
}
