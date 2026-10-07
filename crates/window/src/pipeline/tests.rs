use super::*;
use crate::fixtures::*;

#[test]
fn decodes_grey_rgb_and_rejects_garbage() {
    let m = crate::Gray::from_vec(3, 2, vec![0.0, 0.5, 1.0, 1.0, 0.0, 0.25]).unwrap();
    let d = decode_png_gray(&png_of(&m)).unwrap();
    assert_eq!((d.w, d.h), (3, 2));
    assert!(d.data.iter().zip(&m.data).all(|(a, b)| (a - b).abs() < 0.01));
    assert!(decode_png_gray(b"not a png").is_err());
    assert!(decode_png_gray(&[]).is_err());
}

#[test]
fn inset_scales_with_the_frame_width() {
    assert!((inset_px(4.0, 6000) - 4.0).abs() < 1e-6);
    assert!((inset_px(4.0, 2000) - 4.0 / 3.0).abs() < 1e-5);
    assert_eq!(inset_px(-3.0, 2000), 0.0);
}
