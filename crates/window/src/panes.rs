//! Grouping detected pane boxes into windows, and finding the ones the mask under-covers.

use crate::Gray;
use crate::gemini::PaneBox;

/// A window: the panes grouped into it and their joint bounds (normalized `[x0, y0, x1, y1]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub panes: Vec<PaneBox>,
    pub bounds: [f32; 4],
}

/// Group boxes that nearly touch: the gap between two boxes is under `gap_frac` (0.25) of the
/// smaller box's size along that axis.
pub fn group(boxes: &[PaneBox], gap_frac: f32) -> Vec<Window> {
    let n = boxes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p.get(i).is_some_and(|q| *q != i) {
            let up = p.get(i).copied().unwrap_or(i);
            let gp = p.get(up).copied().unwrap_or(up);
            if let Some(x) = p.get_mut(i) {
                *x = gp;
            }
            i = gp;
        }
        i
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (Some(a), Some(b)) = (boxes.get(i), boxes.get(j)) else { continue };
            let (a, b) = (a.0, b.0);
            let gap_x = (a[0].max(b[0]) - a[2].min(b[2])).max(0.0);
            let gap_y = (a[1].max(b[1]) - a[3].min(b[3])).max(0.0);
            let size_x = (a[2] - a[0]).min(b[2] - b[0]);
            let size_y = (a[3] - a[1]).min(b[3] - b[1]);
            if gap_x <= gap_frac * size_x && gap_y <= gap_frac * size_y {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if let Some(p) = parent.get_mut(ri) {
                    *p = rj;
                }
            }
        }
    }
    let mut groups: Vec<(usize, Vec<PaneBox>)> = Vec::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        let Some(b) = boxes.get(i) else { continue };
        match groups.iter_mut().find(|(r, _)| *r == root) {
            Some((_, v)) => v.push(*b),
            None => groups.push((root, vec![*b])),
        }
    }
    groups
        .into_iter()
        .map(|(_, panes)| {
            let bounds = panes.iter().fold([1.0f32, 1.0, 0.0, 0.0], |a, p| [a[0].min(p.0[0]), a[1].min(p.0[1]), a[2].max(p.0[2]), a[3].max(p.0[3])]);
            Window { panes, bounds }
        })
        .collect()
}

/// Fraction of the box covered by the matte (`>= 0.5`).
pub fn box_coverage(m: &Gray, b: &PaneBox) -> f32 {
    let px = |v: f32, n: usize| ((v * n as f32).round().max(0.0) as usize).min(n);
    let (x0, x1) = (px(b.0[0], m.w), px(b.0[2], m.w));
    let (y0, y1) = (px(b.0[1], m.h), px(b.0[3], m.h));
    let mut total = 0usize;
    let mut on = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            total += 1;
            if m.get(x, y) >= 0.5 {
                on += 1;
            }
        }
    }
    if total == 0 { 1.0 } else { on as f32 / total as f32 }
}

/// Windows with a pane covered less than `min_cover` (0.6): the ones to segment again.
pub fn weak_windows(m: &Gray, windows: &[Window], min_cover: f32) -> Vec<Window> {
    windows.iter().filter(|w| w.panes.iter().any(|p| box_coverage(m, p) < min_cover)).cloned().collect()
}

/// A window's bounds grown by `pad` (0.15) of its size, clamped to the frame.
pub fn padded(bounds: [f32; 4], pad: f32) -> [f32; 4] {
    let (w, h) = (bounds[2] - bounds[0], bounds[3] - bounds[1]);
    [(bounds[0] - pad * w).max(0.0), (bounds[1] - pad * h).max(0.0), (bounds[2] + pad * w).min(1.0), (bounds[3] + pad * h).min(1.0)]
}

/// Union `crop` (a matte of the normalized region `rect`) into `full`: per-pixel maximum, never
/// removing anything.
pub fn union_region(full: &mut Gray, crop: &Gray, rect: [f32; 4]) {
    let (x0, y0) = ((rect[0] * full.w as f32).round() as usize, (rect[1] * full.h as f32).round() as usize);
    let (x1, y1) = (((rect[2] * full.w as f32).round() as usize).min(full.w), ((rect[3] * full.h as f32).round() as usize).min(full.h));
    for y in y0..y1 {
        for x in x0..x1 {
            let u = (x as f32 + 0.5 - x0 as f32) / (x1 - x0).max(1) as f32 * crop.w as f32;
            let v = (y as f32 + 0.5 - y0 as f32) / (y1 - y0).max(1) as f32 * crop.h as f32;
            let c = crop.sample(u, v);
            if let Some(o) = full.data.get_mut(y * full.w + x) {
                *o = o.max(c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearby_panes_form_one_window() {
        let boxes = [PaneBox([0.1, 0.1, 0.2, 0.3]), PaneBox([0.21, 0.1, 0.31, 0.3]), PaneBox([0.1, 0.31, 0.2, 0.5]), PaneBox([0.7, 0.1, 0.8, 0.3])];
        let w = group(&boxes, 0.25);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].panes.len(), 3);
        assert_eq!(w[0].bounds, [0.1, 0.1, 0.31, 0.5]);
    }

    #[test]
    fn under_covered_panes_are_found() {
        let mut m = Gray::new(100, 100).unwrap();
        for y in 10..30 {
            for x in 10..20 {
                m.data[y * 100 + x] = 1.0;
            }
        }
        let boxes = [PaneBox([0.1, 0.1, 0.2, 0.3]), PaneBox([0.21, 0.1, 0.31, 0.3])];
        let w = group(&boxes, 0.25);
        assert!((box_coverage(&m, &boxes[0]) - 1.0).abs() < 1e-3);
        assert_eq!(weak_windows(&m, &w, 0.6).len(), 1);
    }

    #[test]
    fn union_only_adds() {
        let mut full = Gray::new(10, 10).unwrap();
        full.data[0] = 1.0;
        let crop = Gray::from_vec(2, 2, vec![0.0; 4]).unwrap();
        union_region(&mut full, &crop, [0.0, 0.0, 0.5, 0.5]);
        assert_eq!(full.data[0], 1.0);
        let crop = Gray::from_vec(2, 2, vec![1.0; 4]).unwrap();
        union_region(&mut full, &crop, [0.5, 0.5, 1.0, 1.0]);
        assert_eq!(full.get(7, 7), 1.0);
        assert_eq!(full.get(2, 2), 0.0);
    }
}
