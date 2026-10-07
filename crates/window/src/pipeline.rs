//! From the model's picture to the stored matte.

use crate::matte::{self, Gray};
use crate::{Error, Result};

/// Decode the model's PNG to a 0..1 greyscale plane (RGB averaged; alpha ignored).
pub fn decode_png_gray(bytes: &[u8]) -> Result<Gray> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_limits(png::Limits { bytes: 128 << 20 });
    let mut reader = dec.read_info().map_err(|e| Error::BadResponse(format!("the mask image can't be read: {e}")))?;
    let (w, h) = (reader.info().width as usize, reader.info().height as usize);
    if w == 0 || h == 0 || w.saturating_mul(h) > matte::MAX_PIXELS {
        return Err(Error::BadResponse("the mask image has an unusable size".into()));
    }
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or_else(|| Error::BadResponse("the mask image is too large".into()))?];
    let info = reader.next_frame(&mut buf).map_err(|e| Error::BadResponse(format!("the mask image can't be decoded: {e}")))?;
    let (channels, bytes_per) = (info.color_type.samples(), if info.bit_depth == png::BitDepth::Sixteen { 2 } else { 1 });
    if !matches!(info.bit_depth, png::BitDepth::Eight | png::BitDepth::Sixteen) {
        return Err(Error::BadResponse("the mask image has an unsupported bit depth".into()));
    }
    let colour = match info.color_type {
        png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => 1,
        png::ColorType::Rgb | png::ColorType::Rgba => 3,
        png::ColorType::Indexed => return Err(Error::BadResponse("the mask image is palette based".into())),
    };
    let max = if bytes_per == 2 { 65535.0 } else { 255.0 };
    let sample = |px: &[u8], c: usize| -> f32 {
        if bytes_per == 2 {
            f32::from(u16::from_be_bytes([px.get(c * 2).copied().unwrap_or(0), px.get(c * 2 + 1).copied().unwrap_or(0)]))
        } else {
            f32::from(px.get(c).copied().unwrap_or(0))
        }
    };
    let data: Vec<f32> = buf
        .chunks_exact(channels * bytes_per)
        .take(w * h)
        .map(|px| (0..colour).map(|c| sample(px, c)).sum::<f32>() / colour as f32 / max)
        .collect();
    Gray::from_vec(w, h, data).ok_or_else(|| Error::BadResponse("the mask image is incomplete".into()))
}

/// Cleanup settings.
#[derive(Clone, Copy, Debug)]
pub struct CleanOptions {
    /// Run the plausibility filter.
    pub plausibility: bool,
    /// Accept a region on its own when its 99th-percentile luminance is at least this (0..1).
    pub min_p99: f32,
    /// Neighbour-rescue distance as a fraction of the width.
    pub reach: f32,
}

impl Default for CleanOptions {
    fn default() -> Self {
        CleanOptions { plausibility: true, min_p99: 200.0 / 255.0, reach: 0.015 }
    }
}

/// Resize the model's soft matte to the preview's size (bilinear, *before* any threshold), remove
/// specks, and apply the plausibility filter against the preview's luminance. The result is the
/// soft matte before the inset.
pub fn clean(model: &Gray, luma: &Gray, o: &CleanOptions) -> Result<Gray> {
    let soft = matte::resize_bilinear(model, luma.w, luma.h).ok_or_else(|| Error::BadResponse("unusable mask size".into()))?;
    let labels = matte::label_components(&soft);
    let min_area = matte::min_region_area(soft.w);
    // specks go first: they are neither accepted nor able to rescue a neighbour
    let mut keep: Vec<bool> = labels.area.iter().enumerate().map(|(i, a)| i != 0 && *a >= min_area).collect();
    if o.plausibility {
        let pruned = labels.retain(&keep);
        keep = matte::plausible_regions(&pruned, luma, o.min_p99, (o.reach * soft.w as f32).max(2.0));
    }
    Ok(matte::keep_regions(&soft, &labels, &keep))
}

/// Pixels of inset at a frame `w` pixels wide, given the setting in pixels of a 6000 px frame.
pub fn inset_px(setting: f64, w: usize) -> f32 {
    (setting.max(0.0) * w as f64 / 6000.0) as f32
}

#[cfg(test)]
mod tests;
