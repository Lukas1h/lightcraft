//! The model calls: request bodies, response parsing, and retry with backoff. The network is
//! behind [`Transport`] so tests replay recorded responses.
//!
//! Never logged or put in an error message: the API key, image data, response bodies (the
//! server's own `error.message` is kept, it carries no request data).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::Error;
use crate::b64;

/// Instructions for the segmentation model (verbatim from the reference tool).
pub const SYSTEM_PROMPT: &str = "You are a real estate photography editor assistant.\n\
You will assist with mundane editing tasks like tediously masking out windows for perfect HDR window pulls.\n\
\n\
The user will submit an interior photo with windows.\n\
You will generate a perfect black and white mask of all portions of the image showing the outside, perfectly masking the window glass panes.\n\
Don't let bright area's on the window frame fool you.\n\
Don't include anything in the mask that is inside, including the window mullion, grilles, sill or frame.\n\
Don't include curtains or semi-transparent blinds in the mask, the must be solid black.\n\
\n\
Outside sections should be solid white.\n\
\n\
Mask out things that may be in front of windows or sitting on the window sill like plants, light fixtures or faucets.";

/// The user turn's text.
pub const USER_PROMPT: &str = "Mask this image.";

/// The recovery pass's detection prompt.
pub const BOX_PROMPT: &str = "Segment the window glass in this photo. Include only the glazed opening. Exclude the frame, sash, mullion bar, blinds, curtain and sill. Treat each pane as its own instance. Only count windows set into a wall that a person looks through from inside a room. Do NOT count glass in doors, sidelights next to an entrance, skylights, or glass in furniture. If a photo contains no such window, return an empty list. Output a JSON list where each entry has box_2d and label. Do not output polygons or masks.";

/// Longest response accepted (a 1K PNG as base64 is well under this).
pub const MAX_RESPONSE: usize = 48 << 20;

/// One round trip: the model name and the JSON request body in, the HTTP status and body out.
pub trait Transport: Send + Sync {
    fn post(&self, model: &str, body: &[u8], cancel: &AtomicBool) -> Result<(u16, Vec<u8>), Error>;
}

/// Backoff between retries; injectable so tests don't wait.
pub trait Sleeper: Send + Sync {
    /// Sleep `d`, returning `false` when `cancel` was raised.
    fn sleep(&self, d: Duration, cancel: &AtomicBool) -> bool;
}

/// Sleeps in 100 ms steps so a cancel is noticed quickly.
pub struct RealSleeper;

impl Sleeper for RealSleeper {
    fn sleep(&self, d: Duration, cancel: &AtomicBool) -> bool {
        let end = std::time::Instant::now() + d;
        while std::time::Instant::now() < end {
            if cancel.load(Ordering::Relaxed) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        !cancel.load(Ordering::Relaxed)
    }
}

/// Tries per call, and the wait before retry `n` (10 s, 20 s, 30 s …).
pub const MAX_TRIES: usize = 6;

pub fn backoff(retry: usize) -> Duration {
    Duration::from_secs(10 * (retry as u64 + 1))
}

/// The segmentation request body for a JPEG.
pub fn segmentation_body(jpeg: &[u8]) -> Value {
    json!({
        "systemInstruction": { "parts": [{ "text": SYSTEM_PROMPT }] },
        "contents": [{ "role": "user", "parts": [
            { "inlineData": { "mimeType": "image/jpeg", "data": b64::encode(jpeg) } },
            { "text": USER_PROMPT },
        ]}],
        "generationConfig": {
            "temperature": 0,
            "topP": 0.95,
            "maxOutputTokens": 32768,
            "responseModalities": ["TEXT", "IMAGE"],
            "thinkingConfig": { "thinkingLevel": "HIGH" },
            "mediaResolution": "MEDIA_RESOLUTION_HIGH",
            // no aspectRatio: "auto" is rejected; omitted, the output follows the input
            "imageConfig": { "imageSize": "1K", "imageOutputOptions": { "mimeType": "image/png" } },
        },
        "safetySettings": [
            { "category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "OFF" },
            { "category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "OFF" },
            { "category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "OFF" },
            { "category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "OFF" },
        ],
    })
}

/// The box-detection request body for a JPEG (JSON output).
pub fn boxes_body(jpeg: &[u8]) -> Value {
    json!({
        "contents": [{ "role": "user", "parts": [
            { "inlineData": { "mimeType": "image/jpeg", "data": b64::encode(jpeg) } },
            { "text": BOX_PROMPT },
        ]}],
        "generationConfig": { "temperature": 0, "responseMimeType": "application/json" },
    })
}

/// The status string of an error body (`RESOURCE_EXHAUSTED`, …) and the server's message.
fn api_error(body: &[u8]) -> (String, String) {
    let v: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let e = v.get("error").unwrap_or(&Value::Null);
    let status = e.get("status").and_then(Value::as_str).unwrap_or("").to_string();
    let mut message = e.get("message").and_then(Value::as_str).unwrap_or("").to_string();
    // the server may echo the offending key fragment: keep messages short and cut at char boundaries
    if message.chars().count() > 300 {
        message = message.chars().take(300).collect::<String>() + "…";
    }
    let reason = e
        .get("details")
        .and_then(Value::as_array)
        .and_then(|d| d.iter().find_map(|x| x.get("reason").and_then(Value::as_str)))
        .unwrap_or("");
    (if status.is_empty() { reason.to_string() } else { status }, message)
}

/// Whether a failed attempt is worth repeating.
fn retryable(status: u16, api_status: &str, reason_hint: &str) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504)
        || matches!(api_status, "RESOURCE_EXHAUSTED" | "UNAVAILABLE" | "DEADLINE_EXCEEDED" | "INTERNAL" | "API_KEY_SERVICE_BLOCKED")
        || reason_hint.contains("API_KEY_SERVICE_BLOCKED")
}

/// POST `body` to `model`, retrying with backoff on rate limits and transient failures (up to
/// [`MAX_TRIES`] tries). Returns the parsed JSON response.
pub fn call(t: &dyn Transport, sleeper: &dyn Sleeper, model: &str, body: &Value, cancel: &AtomicBool) -> Result<Value, Error> {
    let bytes = serde_json::to_vec(body).map_err(|e| Error::Other(e.to_string()))?;
    let mut last = Error::Other("no attempt was made".into());
    for attempt in 0..MAX_TRIES {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        if attempt > 0 && !sleeper.sleep(backoff(attempt - 1), cancel) {
            return Err(Error::Cancelled);
        }
        match t.post(model, &bytes, cancel) {
            Ok((200, resp)) => return serde_json::from_slice(&resp).map_err(|_| Error::BadResponse("the reply is not JSON".into())),
            Ok((status, resp)) => {
                let (api_status, message) = api_error(&resp);
                let hint = String::from_utf8_lossy(resp.get(..resp.len().min(4096)).unwrap_or_default()).into_owned();
                if retryable(status, &api_status, &hint) {
                    last = Error::RateLimited { status, api_status, message, tries: attempt + 1 };
                    continue;
                }
                return Err(match status {
                    400 => Error::Api { status, message: format!("the request was rejected: {message}") },
                    401 | 403 => Error::Api { status, message: format!("the API key was not accepted ({api_status}): {message}") },
                    404 => Error::Api { status, message: format!("model `{model}` was not found: {message}") },
                    _ => Error::Api { status, message },
                });
            }
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e @ (Error::Network(_) | Error::Stalled)) => {
                last = e;
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    Err(last)
}

/// The finish reason / block reason when the model refused, for a clear message.
fn refusal(v: &Value) -> Option<String> {
    if let Some(r) = v.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Some(format!("the photo was blocked by the model ({r})"));
    }
    let r = v.pointer("/candidates/0/finishReason").and_then(Value::as_str)?;
    (!matches!(r, "STOP" | "MAX_TOKENS")).then(|| format!("the model stopped with `{r}`"))
}

/// The LAST image part of a response (the model returns the picture twice: mid-thought, then as
/// the answer). `Ok(None)` = no image = no window found.
pub fn last_image(v: &Value) -> Result<Option<Vec<u8>>, Error> {
    let parts = v.pointer("/candidates/0/content/parts").and_then(Value::as_array);
    let image = parts.and_then(|p| {
        p.iter()
            .filter_map(|part| part.get("inlineData").or_else(|| part.get("inline_data")))
            .filter(|d| d.get("mimeType").or_else(|| d.get("mime_type")).and_then(Value::as_str).is_some_and(|m| m.starts_with("image/")))
            .filter_map(|d| d.get("data").and_then(Value::as_str))
            .next_back()
    });
    match image {
        Some(data) => b64::decode(data).map(Some).ok_or_else(|| Error::BadResponse("the image in the reply is damaged".into())),
        None => match refusal(v) {
            Some(why) => Err(Error::Refused(why)),
            None => Ok(None),
        },
    }
}

/// The concatenated text parts of a response.
pub fn text(v: &Value) -> String {
    v.pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(|p| p.iter().filter(|x| x.get("thought").and_then(Value::as_bool) != Some(true)).filter_map(|x| x.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""))
        .unwrap_or_default()
}

/// A detected pane/window box, normalized 0..1 (`[x0, y0, x1, y1]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaneBox(pub [f32; 4]);

/// Parse the detection reply: a JSON list of `{box_2d: [ymin, xmin, ymax, xmax] (0..1000), label}`.
/// Damaged entries are skipped.
pub fn parse_boxes(reply: &str) -> Vec<PaneBox> {
    let reply = reply.trim();
    // tolerate a ```json fence
    let reply = reply.trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(reply) else { return Vec::new() };
    items
        .iter()
        .take(256)
        .filter_map(|it| {
            let b = it.get("box_2d")?.as_array()?;
            let n = |i: usize| b.get(i).and_then(Value::as_f64).filter(|v| v.is_finite());
            let (y0, x0, y1, x1) = (n(0)?, n(1)?, n(2)?, n(3)?);
            let c = |v: f64| (v / 1000.0).clamp(0.0, 1.0) as f32;
            let r = [c(x0.min(x1)), c(y0.min(y1)), c(x0.max(x1)), c(y0.max(y1))];
            (r[2] - r[0] > 0.002 && r[3] - r[1] > 0.002).then_some(PaneBox(r))
        })
        .collect()
}

#[cfg(test)]
mod tests;
