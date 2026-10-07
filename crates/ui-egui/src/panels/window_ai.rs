//! Settings ▸ AI Window: the one-time privacy notice, the Google Cloud API key, model names,
//! concurrency, inset and the cleanup options. Everything goes through `window.settings`, so the
//! window is only a view of the engine's settings (and works the same from the control channel).

use egui::{Align2, RichText, vec2};
use lightcraft_window::Settings;
use serde_json::json;

use crate::LightcraftApp;
use crate::theme::Tokens;
use crate::widgets::register;

/// What the window edits between "open" and "Save".
#[derive(Clone, Default)]
pub struct Draft {
    pub settings: Settings,
    /// Typed key; never serialized, cleared as soon as it is saved or the window closes.
    pub key: String,
    pub accept: bool,
}

// (the typed key must not show up in debug output)
impl std::fmt::Debug for Draft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Draft").field("settings", &self.settings).field("key_typed", &!self.key.is_empty()).finish_non_exhaustive()
    }
}

/// Open the window (loads the current settings).
pub fn open(app: &mut LightcraftApp) {
    let settings = app.session.window.settings.clone();
    let accept = settings.notice_accepted;
    app.ui.window_setup = Some(Draft { settings, key: String::new(), accept });
}

/// The notice shown before the feature can be turned on.
pub const NOTICE: &str = "AI Window sends a preview of each photo you run it on (a JPEG of up to 2048 px) to Google's Gemini API, \
using your own Google Cloud API key. Nothing else leaves your computer, and LightCraft has no telemetry. Google's terms and \
pricing apply to those requests. The feature is off until you turn it on here. The key is not stored in your library, settings \
files or logs.";

pub fn show(app: &mut LightcraftApp, ctx: &egui::Context) {
    let Some(mut draft) = app.ui.window_setup.take() else { return };
    let t = Tokens::get(ctx);
    let mut close = false;
    let mut save = false;
    let mut remove_key = false;
    let described = app.session.window.describe();
    let has_key = described["hasKey"].as_bool().unwrap_or(false);
    let store = described["keyStore"].as_str().unwrap_or("").to_string();
    let frame = egui::Frame::window(&ctx.global_style()).inner_margin(egui::Margin::symmetric(18, 14));
    egui::Window::new(crate::i18n::tr("AI Window"))
        .id(egui::Id::new("ai-window-settings"))
        .order(egui::Order::Foreground)
        .collapsible(false)
        .resizable(false)
        .frame(frame)
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.set_width(460.0);
            ui.spacing_mut().item_spacing.y = 8.0;
            ui.add(egui::Label::new(RichText::new(crate::i18n::tr(NOTICE)).color(t.text_dim)).wrap());
            let r = ui.checkbox(&mut draft.accept, crate::i18n::tr("I understand that photo previews are sent to Google"));
            register(ctx, "check:windowAccept", r.rect);
            let r = ui.add_enabled(draft.accept, egui::Checkbox::new(&mut draft.settings.enabled, crate::i18n::tr("Turn on AI Window masks")));
            register(ctx, "check:windowEnabled", r.rect);
            ui.separator();
            ui.label(RichText::new(crate::i18n::tr("Google Cloud API key")).font(t.semibold(12.5)).color(t.text_label));
            ui.horizontal(|ui| {
                let r = ui.add(
                    egui::TextEdit::singleline(&mut draft.key)
                        .password(true)
                        .hint_text(if has_key { "key set" } else { "paste your key" })
                        .desired_width(280.0),
                );
                register(ctx, "field:windowKey", r.rect);
                let r = ui.add_enabled(has_key, egui::Button::new(crate::i18n::tr("Remove")));
                register(ctx, "button:windowKeyRemove", r.rect);
                remove_key = r.clicked();
            });
            ui.label(RichText::new(format!("{}: {store}", crate::i18n::tr("Kept"))).color(t.text_dim));
            ui.separator();
            egui::Grid::new("ai-window-grid").num_columns(2).spacing(vec2(10.0, 6.0)).show(ui, |ui| {
                ui.label(crate::i18n::tr("Model"));
                let r = ui.add(egui::TextEdit::singleline(&mut draft.settings.model).desired_width(280.0));
                register(ctx, "field:windowModel", r.rect);
                ui.end_row();
                ui.label(crate::i18n::tr("Pane-detection model"));
                let r = ui.add(egui::TextEdit::singleline(&mut draft.settings.boxes_model).desired_width(280.0));
                register(ctx, "field:windowBoxesModel", r.rect);
                ui.end_row();
                ui.label(crate::i18n::tr("Photos at once"));
                let mut n = draft.settings.concurrency;
                let r = ui.add(egui::DragValue::new(&mut n).range(1..=lightcraft_window::MAX_CONCURRENCY));
                register(ctx, "field:windowConcurrency", r.rect);
                draft.settings.concurrency = n;
                ui.end_row();
                ui.label(crate::i18n::tr("Inset (px at 6000 wide)"));
                let r = ui.add(egui::Slider::new(&mut draft.settings.inset, 0.0..=20.0));
                register(ctx, "slider:windowInset", r.rect);
                ui.end_row();
            });
            let r = ui.checkbox(&mut draft.settings.plausibility, crate::i18n::tr("Drop regions that don't look like a bright window"));
            register(ctx, "check:windowPlausibility", r.rect);
            let r = ui.checkbox(&mut draft.settings.recovery, crate::i18n::tr("Re-check multi-pane windows on a crop (slower)"));
            register(ctx, "check:windowRecovery", r.rect);
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let r = ui.add(egui::Button::new(crate::i18n::tr("Save")).min_size(vec2(70.0, 26.0)));
                register(ctx, "button:windowSave", r.rect);
                save = r.clicked();
                let r = ui.add(egui::Button::new(crate::i18n::tr("Close")).min_size(vec2(70.0, 26.0)));
                register(ctx, "button:windowClose", r.rect);
                close = r.clicked();
            });
        });
    if remove_key {
        let _ = app.run("window.settings", json!({"apiKey": ""}));
    }
    if save {
        let mut p = json!({
            "noticeAccepted": draft.accept,
            "enabled": draft.accept && draft.settings.enabled,
            "model": draft.settings.model,
            "boxesModel": draft.settings.boxes_model,
            "concurrency": draft.settings.concurrency,
            "inset": draft.settings.inset,
            "plausibility": draft.settings.plausibility,
            "recovery": draft.settings.recovery,
        });
        if !draft.key.trim().is_empty() {
            p["apiKey"] = json!(draft.key.trim());
        }
        match app.run("window.settings", p) {
            Ok(_) => {
                draft.key.clear();
                close = true;
            }
            Err(e) => app.toast_error(ctx, e.to_string()),
        }
    }
    if !close {
        app.ui.window_setup = Some(draft);
    }
}
