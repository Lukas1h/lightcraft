use super::*;

/// A straight (slanted) edge, drawn at `w × h` and as a soft low-resolution picture of it.
fn slanted_edge(w: usize, h: usize) -> Gray {
    // coverage = area average of "x*0.37 + y*0.93 < c" sampled 8×8 per pixel
    let mut g = Gray::new(w, h).unwrap();
    for y in 0..h {
        for x in 0..w {
            let mut n = 0;
            for sy in 0..8 {
                for sx in 0..8 {
                    let (fx, fy) = ((x as f32 + (sx as f32 + 0.5) / 8.0) / w as f32, (y as f32 + (sy as f32 + 0.5) / 8.0) / h as f32);
                    if fx * 0.8 + fy * 0.6 < 0.55 {
                        n += 1;
                    }
                }
            }
            g.data[y * w + x] = n as f32 / 64.0;
        }
    }
    g
}

fn disagreement(a: &Gray, b: &Gray) -> usize {
    a.data.iter().zip(&b.data).filter(|(p, q)| (**p >= 0.5) != (**q >= 0.5)).count()
}

#[test]
fn bilinear_then_threshold_beats_threshold_then_resize() {
    let (low, hi) = ((60, 40), (600, 400));
    let model = slanted_edge(low.0, low.1);
    let truth = slanted_edge(hi.0, hi.1);
    let smooth = resize_then_threshold(&model, hi.0, hi.1, 0.5).unwrap();
    let stair = threshold_then_resize(&model, hi.0, hi.1, 0.5).unwrap();
    let (e_smooth, e_stair) = (disagreement(&smooth, &truth), disagreement(&stair, &truth));
    assert!(e_smooth * 3 < e_stair, "smooth {e_smooth} vs staircase {e_stair}");
    // the staircase is jagged: its boundary moves in whole 10 px steps, so along a row the edge
    // position takes few distinct values compared to the smooth one
    let edge_x = |g: &Gray, y: usize| (0..g.w).find(|x| g.get(*x, y) < 0.5).unwrap_or(g.w);
    let distinct = |g: &Gray| (0..g.h).map(|y| edge_x(g, y)).collect::<std::collections::BTreeSet<_>>().len();
    assert!(distinct(&smooth) > distinct(&stair) * 2, "{} vs {}", distinct(&smooth), distinct(&stair));
}

#[test]
fn resize_keeps_a_constant_plane_and_refuses_absurd_sizes() {
    let g = Gray::from_vec(4, 3, vec![0.7; 12]).unwrap();
    let r = resize_bilinear(&g, 9, 7).unwrap();
    assert!(r.data.iter().all(|v| (v - 0.7).abs() < 1e-5));
    assert!(Gray::new(usize::MAX, 2).is_none());
    assert!(Gray::new(0, 5).is_none());
    assert!(Gray::from_vec(3, 3, vec![0.0; 8]).is_none());
}

#[test]
fn components_and_specks() {
    let mut g = Gray::new(60, 40).unwrap();
    for y in 5..15 {
        for x in 5..20 {
            g.data[y * 60 + x] = 1.0;
        }
    }
    g.data[30 * 60 + 40] = 1.0;
    let l = label_components(&g);
    assert_eq!(l.area.len(), 3);
    assert_eq!(l.area[1], 150);
    assert_eq!(l.area[2], 1);
    assert!(min_region_area(6000) >= 144);
}

#[test]
fn inset_moves_a_soft_edge_in_by_the_radius() {
    // a vertical edge with a 2 px linear ramp, centred on x = 30
    let mut g = Gray::new(60, 10).unwrap();
    for y in 0..10 {
        for x in 0..60 {
            g.data[y * 60 + x] = ((30.0 - (x as f32 + 0.5)) / 2.0 + 0.5).clamp(0.0, 1.0);
        }
    }
    let cross = |m: &Gray| (0..59).find(|x| m.get(*x, 5) >= 0.5 && m.get(*x + 1, 5) < 0.5).unwrap() as f32;
    let before = cross(&g);
    let after = cross(&inset(&g, 4.0));
    assert!((before - after - 4.0).abs() <= 1.0, "edge {before} → {after}");
    assert_eq!(inset(&g, 0.0), g);
    assert_eq!(inset(&g, f32::NAN), g);
    // soft stays soft: values strictly between 0 and 1 survive
    assert!(inset(&g, 1.5).data.iter().any(|v| *v > 0.05 && *v < 0.95));
}

#[test]
fn a_whole_grid_is_kept_if_one_lite_is_bright() {
    // three lites in a row (gap 6 px), only the left one bright; a far dim blob alone
    let (w, h) = (400, 100);
    let mut m = Gray::new(w, h).unwrap();
    let mut luma = Gray::new(w, h).unwrap();
    let boxes = [(10, 40), (50, 90), (100, 140), (300, 340)];
    for (i, (x0, x1)) in boxes.iter().enumerate() {
        for y in 20..60 {
            for x in *x0..*x1 {
                m.data[y * w + x] = 1.0;
                luma.data[y * w + x] = if i == 0 { 0.95 } else { 0.3 };
            }
        }
    }
    let labels = label_components(&m);
    let ok = plausible_regions(&labels, &luma, 200.0 / 255.0, 12.0);
    assert_eq!(&ok[1..], &[true, true, true, false], "the chain 1→2→3 is rescued, the far blob is not");
}

#[test]
fn logits_round_trip() {
    let g = Gray::from_vec(3, 1, vec![0.0, 0.5, 1.0]).unwrap();
    let back = from_logits(3, 1, &to_logits(&g)).unwrap();
    assert!(back.data[0] < 0.001 && (back.data[1] - 0.5).abs() < 1e-3 && back.data[2] > 0.999);
}
