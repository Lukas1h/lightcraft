//! Engine-level tests for the AI Window mask. TODO(next session): replay recorded answers through
//! `Session::window.transport` (see `lightcraft-window`'s `fixtures.rs`) and check that
//! `mask.addWindow` adds one `Window` mask with the default adjustments, that a second run replaces
//! only its component, that the feature refuses to run until `window.settings` enables it, and that
//! `window.setInset` changes `seg` but not `source`.
