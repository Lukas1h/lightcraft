//! The real transport: HTTPS to Vertex AI with the API key in a header (never in the URL).

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use crate::gemini::{MAX_RESPONSE, Transport};
use crate::http::{self, HttpError, Limits, Url};
use crate::{Error, Result};

/// Where requests go: `{base}/v1beta1/publishers/google/models/{model}:generateContent`.
pub const DEFAULT_BASE: &str = "https://aiplatform.googleapis.com";

pub struct HttpTransport {
    key: String,
    base: String,
}

impl HttpTransport {
    pub fn new(key: &str) -> Result<HttpTransport> {
        Self::with_base(key, DEFAULT_BASE)
    }

    /// For tests against a local server.
    pub fn with_base(key: &str, base: &str) -> Result<HttpTransport> {
        let key = key.trim();
        if key.is_empty() {
            return Err(Error::NoKey);
        }
        if key.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err(Error::Other("the API key contains spaces or control characters".into()));
        }
        Ok(HttpTransport { key: key.to_string(), base: base.trim_end_matches('/').to_string() })
    }
}

// no key in debug output
impl std::fmt::Debug for HttpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpTransport").field("base", &self.base).finish_non_exhaustive()
    }
}

fn convert(e: HttpError) -> Error {
    match e {
        HttpError::Cancelled => Error::Cancelled,
        HttpError::Stalled => Error::Stalled,
        other => Error::Network(other.to_string()),
    }
}

impl Transport for HttpTransport {
    fn post(&self, model: &str, body: &[u8], cancel: &AtomicBool) -> Result<(u16, Vec<u8>)> {
        if model.is_empty() || !model.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_')) {
            return Err(Error::Other(format!("`{model}` is not a usable model name")));
        }
        let url = Url::parse(&format!("{}/v1beta1/publishers/google/models/{model}:generateContent", self.base)).map_err(convert)?;
        // the model thinks for up to ~30 s before the first byte
        let limits = Limits { connect: Duration::from_secs(20), stall: Duration::from_secs(180), cancel };
        let mut resp = http::post(&url, &[("x-goog-api-key", self.key.clone())], body, &limits).map_err(convert)?;
        let status = resp.status;
        let bytes = resp.read_to_end(MAX_RESPONSE, &limits).map_err(convert)?;
        Ok((status, bytes))
    }
}
