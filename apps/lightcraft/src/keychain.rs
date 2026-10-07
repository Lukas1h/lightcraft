//! The Google Cloud API key for AI Window masks, in the operating system's credential store
//! (macOS Keychain, Windows Credential Manager, the Secret Service on Linux) through the
//! `keyring` crate. Never in the library, a settings file or a log. If the store can't be reached
//! (a headless Linux box without a Secret Service) the environment variable
//! `GOOGLE_CLOUD_API_KEY` still works and typing a key keeps it for the session only.

use std::sync::Mutex;

use lightcraft_engine::window::KeyStore;

const SERVICE: &str = "LightCraft";
const ACCOUNT: &str = "google-cloud-api-key";

#[derive(Default)]
pub struct Keychain {
    /// Typed while the OS store is unavailable (this session only).
    fallback: Mutex<Option<String>>,
}

fn entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE, ACCOUNT).map_err(|e| format!("the system keychain is not available: {e}"))
}

impl KeyStore for Keychain {
    fn get(&self) -> Result<Option<String>, String> {
        if let Some(k) = self.fallback.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone() {
            return Ok(Some(k));
        }
        if let Ok(e) = entry() {
            match e.get_password() {
                Ok(k) if !k.trim().is_empty() => return Ok(Some(k.trim().to_string())),
                Ok(_) | Err(keyring::Error::NoEntry) => {}
                // locked or unreachable store: fall through to the environment
                Err(e) => log::warn!("keychain: {e}"),
            }
        }
        Ok(["GOOGLE_CLOUD_API_KEY", "LIGHTCRAFT_GOOGLE_API_KEY"]
            .iter()
            .find_map(|n| std::env::var(n).ok())
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty()))
    }

    fn set(&self, key: &str) -> Result<(), String> {
        match entry().and_then(|e| e.set_password(key).map_err(|e| e.to_string())) {
            Ok(()) => {
                *self.fallback.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
                Ok(())
            }
            Err(why) => {
                // (the message names the failure, never the key)
                log::warn!("keychain: couldn't store the key ({why}); keeping it for this session");
                *self.fallback.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(key.to_string());
                Ok(())
            }
        }
    }

    fn clear(&self) -> Result<(), String> {
        *self.fallback.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        match entry().map(|e| e.delete_credential()) {
            Ok(Ok(())) | Ok(Err(keyring::Error::NoEntry)) | Err(_) => Ok(()),
            Ok(Err(e)) => Err(format!("couldn't remove the key from the keychain: {e}")),
        }
    }

    fn persistent(&self) -> bool {
        self.fallback.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none() && entry().is_ok()
    }

    fn describe(&self) -> &'static str {
        if self.persistent() { "the system keychain" } else { "this session only (the system keychain isn't available)" }
    }
}
