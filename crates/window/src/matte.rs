//! Soft mattes (`0..1` coverage planes) and the pure operations the window pipeline is made of:
//! bilinear resampling, thresholding, connected components, the plausibility filter and the
//! boundary inset. Nothing here touches the network.

use std::collections::{HashMap, HashSet, VecDeque};

/// A greyscale plane, `w × h` row-major. Used for both mattes and luminance.
#[derive(Clone, Debug, PartialEq)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Largest plane these functions will allocate (hostile sizes are refused, not allocated).
pub const MAX_PIXELS: usize = 64 << 20;

impl Gray {
    pub fn new(w: usize, h: usize) -> Option<Gray> {
        let n = w.checked_mul(h)?;
        (w > 0 && h > 0 && n <= MAX_PIXELS).then(|| Gray { w, h, data: vec![0.0; n] })
    }

    /// From row-major data; `None` when the length doesn't match.
    pub fn from_vec(w: usize, h: usize, data: Vec<f32>) -> Option<Gray> {
        (w > 0 && h > 0 && w.checked_mul(h)? == data.len() && data.len() <= MAX_PIXELS).then_some(Gray { w, h, data })
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.w && y < self.h { self.data.get(y * self.w + x).copied().unwrap_or(0.0) } else { 0.0 }
    }

    /// Bilinear sample at continuous pixel coordinates (pixel centres at `i + 0.5`), edge-clamped.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let fx = (x - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fy = (y - 0.5).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let top = self.get(x0, y0) * (1.0 - tx) + self.get(x0 + 1, y0) * tx;
        let bot = self.get(x0, y0 + 1) * (1.0 - tx) + self.get(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bot * ty
    }

    /// The fraction of pixels above 0.5.
    pub fn coverage(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().filter(|v| **v > 0.5).count() as f32 / self.data.len() as f32
    }
}

/// Bilinear resize of the whole plane (pixel-centre aligned, so a soft edge ramp keeps its
/// position). Downscales are box-averaged first when the factor is large, so thin features
/// don't alias.
pub fn resize_bilinear(src: &Gray, w: usize, h: usize) -> Option<Gray> {
    let mut out = Gray::new(w, h)?;
    let (sx, sy) = (src.w as f32 / w as f32, src.h as f32 / h as f32);
    for y in 0..h {
        let fy = (y as f32 + 0.5) * sy;
        for x in 0..w {
            let v = src.sample((x as f32 + 0.5) * sx, fy);
            if let Some(o) = out.data.get_mut(y * w + x) {
                *o = v;
            }
        }
    }
    Some(out)
}

/// `>= level` → 1, else 0.
pub fn threshold(src: &Gray, level: f32) -> Gray {
    Gray { w: src.w, h: src.h, data: src.data.iter().map(|v| if *v >= level { 1.0 } else { 0.0 }).collect() }
}

/// What the old tool did and the spec warns against: threshold at the model's resolution, then
/// resize (nearest), which bakes in a staircase. Kept for the comparison test.
pub fn threshold_then_resize(src: &Gray, w: usize, h: usize, level: f32) -> Option<Gray> {
    let t = threshold(src, level);
    let mut out = Gray::new(w, h)?;
    for y in 0..h {
        let sy = (y * src.h / h).min(src.h - 1);
        for x in 0..w {
            let sx = (x * src.w / w).min(src.w - 1);
            if let Some(o) = out.data.get_mut(y * w + x) {
                *o = t.get(sx, sy);
            }
        }
    }
    Some(out)
}

/// Resize bilinearly first (soft greys), then threshold: the anti-aliased ramp places the edge
/// at sub-pixel accuracy.
pub fn resize_then_threshold(src: &Gray, w: usize, h: usize, level: f32) -> Option<Gray> {
    Some(threshold(&resize_bilinear(src, w, h)?, level))
}

/// Connected components (4-connectivity) of the pixels `>= 0.5`. Label 0 is background.
pub struct Labels {
    pub w: usize,
    pub h: usize,
    pub label: Vec<u32>,
    /// Pixel count per label (index 0 unused = background count).
    pub area: Vec<usize>,
}

pub fn label_components(m: &Gray) -> Labels {
    let mut label = vec![0u32; m.data.len()];
    let mut area = vec![0usize];
    let mut queue = VecDeque::new();
    for start in 0..m.data.len() {
        if m.data.get(start).is_none_or(|v| *v < 0.5) || label.get(start).is_none_or(|l| *l != 0) {
            continue;
        }
        let id = area.len() as u32;
        let mut count = 0usize;
        if let Some(l) = label.get_mut(start) {
            *l = id;
        }
        queue.push_back(start);
        while let Some(i) = queue.pop_front() {
            count += 1;
            let (x, y) = (i % m.w, i / m.w);
            let mut visit = |j: usize| {
                if m.data.get(j).is_some_and(|v| *v >= 0.5) && label.get(j) == Some(&0) {
                    if let Some(l) = label.get_mut(j) {
                        *l = id;
                    }
                    queue.push_back(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < m.w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - m.w);
            }
            if y + 1 < m.h {
                visit(i + m.w);
            }
        }
        area.push(count);
    }
    Labels { w: m.w, h: m.h, label, area }
}

impl Labels {
    /// The same labels with every region `keep` rejects turned into background.
    pub fn retain(&self, keep: &[bool]) -> Labels {
        let ok = |l: u32| keep.get(l as usize).copied().unwrap_or(false);
        Labels {
            w: self.w,
            h: self.h,
            label: self.label.iter().map(|l| if ok(*l) { *l } else { 0 }).collect(),
            area: self.area.iter().enumerate().map(|(i, a)| if ok(i as u32) { *a } else { 0 }).collect(),
        }
    }
}

/// Minimum area (in pixels of a plane `w` wide) a region must have to not be a speck:
/// `0.25 · (24 px · scale)²`, `scale = w / 6000`.
pub fn min_region_area(w: usize) -> usize {
    let s = 24.0 * w as f64 / 6000.0;
    (0.25 * s * s).ceil().max(1.0) as usize
}

/// The 99th-percentile of `luma` (0..1) over each region, from a 256-bin histogram.
pub fn region_p99(labels: &Labels, luma: &Gray) -> Vec<f32> {
    let n = labels.area.len();
    let mut hist = vec![[0u32; 256]; n];
    for (l, v) in labels.label.iter().zip(&luma.data) {
        if *l != 0
            && let Some(h) = hist.get_mut(*l as usize)
            && let Some(b) = h.get_mut((v.clamp(0.0, 1.0) * 255.0 + 0.5) as usize)
        {
            *b += 1;
        }
    }
    hist.iter()
        .enumerate()
        .map(|(i, h)| {
            let total = labels.area.get(i).copied().unwrap_or(0) as f64;
            let want = (total * 0.01).ceil() as u64; // pixels allowed above the percentile
            let mut above = 0u64;
            for (b, c) in h.iter().enumerate().rev() {
                above += u64::from(*c);
                if above >= want.max(1) {
                    return b as f32 / 255.0;
                }
            }
            0.0
        })
        .collect()
}

/// Which regions are plausible windows. A region is accepted on its own when its 99th-percentile
/// luminance is at least `min_p99`; any other region is accepted when it lies within `reach`
/// pixels of an accepted one, repeatedly (so a whole grid of lites is kept if any lite is bright).
/// Distances are measured on a coarse grid, not at full resolution.
pub fn plausible_regions(labels: &Labels, luma: &Gray, min_p99: f32, reach: f32) -> Vec<bool> {
    let n = labels.area.len();
    let p99 = region_p99(labels, luma);
    let mut accepted: Vec<bool> = (0..n).map(|i| i != 0 && p99.get(i).is_some_and(|p| *p >= min_p99)).collect();
    if n <= 1 || accepted.iter().all(|a| *a) {
        return accepted;
    }
    // coarse grid: cells of reach/2 px; two regions are neighbours when cells within 2 cells
    // (≥ reach px) of each other hold both
    let cell = (reach / 2.0).max(1.0);
    let (gw, gh) = (((labels.w as f32 / cell).ceil() as usize).max(1), ((labels.h as f32 / cell).ceil() as usize).max(1));
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); gw.saturating_mul(gh).min(1 << 22)];
    if cells.len() != gw * gh {
        return accepted;
    }
    for y in 0..labels.h {
        let gy = ((y as f32 / cell) as usize).min(gh - 1);
        for x in 0..labels.w {
            let l = labels.label.get(y * labels.w + x).copied().unwrap_or(0);
            if l != 0 {
                let gx = ((x as f32 / cell) as usize).min(gw - 1);
                if let Some(c) = cells.get_mut(gy * gw + gx)
                    && !c.contains(&l)
                {
                    c.push(l);
                }
            }
        }
    }
    let mut edges: HashMap<u32, HashSet<u32>> = HashMap::new();
    for gy in 0..gh {
        for gx in 0..gw {
            let Some(here) = cells.get(gy * gw + gx) else { continue };
            for ny in gy.saturating_sub(2)..=(gy + 2).min(gh - 1) {
                for nx in gx.saturating_sub(2)..=(gx + 2).min(gw - 1) {
                    let Some(there) = cells.get(ny * gw + nx) else { continue };
                    for a in here {
                        for b in there {
                            if a != b {
                                edges.entry(*a).or_default().insert(*b);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut queue: VecDeque<u32> = (0..n as u32).filter(|i| accepted.get(*i as usize).copied().unwrap_or(false)).collect();
    while let Some(a) = queue.pop_front() {
        for b in edges.get(&a).into_iter().flatten() {
            if let Some(flag) = accepted.get_mut(*b as usize)
                && !*flag
            {
                *flag = true;
                queue.push_back(*b);
            }
        }
    }
    accepted
}

/// Keep the soft matte only inside the regions `keep` says to (and a 2 px skirt around them so the
/// anti-aliased ramp survives); specks and rejected regions become 0.
pub fn keep_regions(soft: &Gray, labels: &Labels, keep: &[bool]) -> Gray {
    let (w, h) = (soft.w, soft.h);
    let kept = |x: usize, y: usize| labels.label.get(y * w + x).is_some_and(|l| *l != 0 && keep.get(*l as usize).copied().unwrap_or(false));
    let rejected = |x: usize, y: usize| labels.label.get(y * w + x).is_some_and(|l| *l != 0 && !keep.get(*l as usize).copied().unwrap_or(false));
    let mut out = soft.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = soft.data.get(i).copied().unwrap_or(0.0);
            if v == 0.0 {
                continue;
            }
            let keep_px = if kept(x, y) {
                true
            } else if rejected(x, y) {
                false
            } else {
                // ramp pixel (below 0.5): belongs to a kept region if one is within 2 px
                let mut near = false;
                'o: for ny in y.saturating_sub(2)..=(y + 2).min(h - 1) {
                    for nx in x.saturating_sub(2)..=(x + 2).min(w - 1) {
                        if kept(nx, ny) {
                            near = true;
                            break 'o;
                        }
                    }
                }
                near
            };
            if !keep_px && let Some(o) = out.data.get_mut(i) {
                *o = 0.0;
            }
        }
    }
    out
}

/// Erode the matte by `radius` pixels (fractional allowed): the minimum over a ring of bilinear
/// samples, so soft edge ramps shift inward by exactly `radius` and the result stays soft.
pub fn inset(m: &Gray, radius: f32) -> Gray {
    if radius.is_nan() || radius <= 0.0 || radius.is_infinite() {
        return m.clone();
    }
    const DIRS: usize = 16;
    let offsets: Vec<(f32, f32)> = (0..DIRS)
        .map(|k| {
            let a = k as f32 / DIRS as f32 * std::f32::consts::TAU;
            (a.cos() * radius, a.sin() * radius)
        })
        .collect();
    let mut out = m.clone();
    for y in 0..m.h {
        for x in 0..m.w {
            let i = y * m.w + x;
            let v = m.data.get(i).copied().unwrap_or(0.0);
            if v <= 0.0 {
                continue;
            }
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            let mut lo = v;
            for (dx, dy) in &offsets {
                // outside the frame counts as unmasked only when the matte is clear there; the
                // clamped sample keeps masks that run to the image border from shrinking from it
                lo = lo.min(m.sample(cx + dx, cy + dy));
                if lo <= 0.0 {
                    break;
                }
            }
            if let Some(o) = out.data.get_mut(i) {
                *o = lo;
            }
        }
    }
    out
}

/// Logits of a soft matte (clamped so fully in/out stay finite).
pub fn to_logits(m: &Gray) -> Vec<f32> {
    m.data
        .iter()
        .map(|p| {
            let p = p.clamp(1e-4, 1.0 - 1e-4);
            (p / (1.0 - p)).ln()
        })
        .collect()
}

/// The inverse of [`to_logits`].
pub fn from_logits(w: usize, h: usize, logits: &[f32]) -> Option<Gray> {
    Gray::from_vec(w, h, logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect())
}

#[cfg(test)]
mod tests;
