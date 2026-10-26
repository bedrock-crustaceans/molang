//! `clamp`, `lerp`, `inverse_lerp`, `hermite_blend`, `lerprotate` and `min_angle`.

use super::formula;
use crate::numeric::{Platform, PostOp, arith};

/// `math.clamp(v, lo, hi)`: `v > hi ? hi : max(v, lo)`, so `clamp(3, 2, 1)` = 1.
///
/// A NaN `v` gives `lo`, a NaN `lo` is returned, and a NaN `hi` never matches.
#[inline]
pub fn clamp(platform: Platform, v: f32, lo: f32, hi: f32, post: PostOp) -> f32 {
    post.apply(platform, if v > hi { hi } else { arith::max(v, lo) })
}

/// `math.hermite_blend(t)`: `3t² − 2t³`, computed as `(3·t)·t − ((t + t)·t)·t`.
#[inline]
pub fn hermite_blend(platform: Platform, t: f32, post: PostOp) -> f32 {
    post.apply(platform, formula::hermite_blend(platform, t))
}

/// `math.lerp(a, b, t)`: `a + t·(b − a)`.
#[inline]
pub fn lerp(platform: Platform, a: f32, b: f32, t: f32, post: PostOp) -> f32 {
    post.apply(platform, formula::lerp(platform, a, b, t))
}

/// `math.lerprotate(a, b, t)`: `a + t·wrap(b − a)`, not wrapped again (`lerprotate(350, 10, 0.5)`
/// = 360).
#[inline]
pub fn lerprotate(platform: Platform, a: f32, b: f32, t: f32, post: PostOp) -> f32 {
    post.apply(platform, formula::lerprotate(platform, a, b, t))
}

/// `math.inverse_lerp(a, b, v)`: `(v − a) / (b − a)` with no zero guard.
#[inline]
pub fn inverse_lerp(platform: Platform, a: f32, b: f32, v: f32, post: PostOp) -> f32 {
    post.apply(platform, formula::inverse_lerp(platform, a, b, v))
}

/// `math.min_angle(x)`: `x` wrapped into `[-180, 180)`, so `min_angle(180)` = −180.
#[inline]
pub fn min_angle(platform: Platform, x: f32, post: PostOp) -> f32 {
    post.apply(platform, formula::wrap_angle(platform, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{PostOp, test_support::*};

    const P: Platform = LINUX;

    #[test]
    fn lerp_and_lerprotate_take_the_first_of_two_quiet_nans() {
        let (neg, pos) = (f32::from_bits(0xffc0_0000), f32::from_bits(0x7fc0_0000));
        for p in PLATFORMS {
            for f in [lerp, lerprotate] {
                for (a, b, t, want) in [
                    (neg, pos, 0.5, pos),
                    (pos, neg, 0.5, neg),
                    (0.5, neg, pos, neg),
                    (0.5, pos, neg, pos),
                    (neg, 0.5, pos, neg),
                    (pos, 0.5, neg, pos),
                ] {
                    assert_eq!(
                        f(p, a, b, t, PostOp::IDENTITY).to_bits(),
                        want.to_bits(),
                        "{p:?} {a} {b} {t}"
                    );
                }
            }
        }
        assert_eq!(lerp(P, 1.0, 3.0, 0.25, PostOp::IDENTITY), 1.5);
        assert_eq!(lerprotate(P, 350.0, 10.0, 0.5, PostOp::IDENTITY), 360.0);
    }

    #[test]
    fn clamp_on_ordinary_values_and_bounds() {
        assert_eq!(clamp(P, 0.5, 0.0, 1.0, ID), 0.5);
        assert_eq!(clamp(P, -3.0, 0.0, 1.0, ID), 0.0);
        assert_eq!(clamp(P, 3.0, 0.0, 1.0, ID), 1.0);
        assert_eq!(clamp(P, 0.0, 0.0, 1.0, ID), 0.0);
        assert_eq!(clamp(P, 1.0, 0.0, 1.0, ID), 1.0);
        assert_eq!(clamp(P, 7.0, 2.0, 2.0, ID), 2.0);
        assert_eq!(clamp(P, -7.0, 2.0, 2.0, ID), 2.0);
        assert_eq!(clamp(P, 5.0, -INF, INF, ID), 5.0);
        assert_eq!(clamp(P, INF, 0.0, 1.0, ID), 1.0);
        assert_eq!(clamp(P, -INF, 0.0, 1.0, ID), 0.0);
        assert_eq!(clamp(P, INF, 0.0, INF, ID), INF);
        assert_eq!(clamp(P, -INF, -INF, 1.0, ID), -INF);
    }

    #[test]
    fn clamp_with_inverted_bounds_tests_the_upper_bound_first() {
        assert_eq!(clamp(P, 3.0, 2.0, 1.0, ID), 1.0);
        assert_eq!(clamp(P, 1.5, 2.0, 1.0, ID), 1.0);
        assert_eq!(clamp(P, 0.0, 2.0, 1.0, ID), 2.0);
        assert_eq!(clamp(P, 1.0, 2.0, 1.0, ID), 2.0);
        assert_eq!(clamp(P, 2.0, 2.0, 1.0, ID), 1.0);
        assert_eq!(clamp(P, 100.0, 5.0, -5.0, ID), -5.0);
        assert_eq!(clamp(P, -100.0, 5.0, -5.0, ID), 5.0);
    }

    #[test]
    fn clamp_of_signed_zeros_returns_the_second_operand_of_max() {
        for p in PLATFORMS {
            assert_bits(clamp(p, -0.0, 0.0, 1.0, ID), 0.0);
            assert_bits(clamp(p, 0.0, -0.0, 1.0, ID), -0.0);
            assert_bits(clamp(p, 0.0, -1.0, -0.0, ID), 0.0);
            assert_bits(clamp(p, -0.0, -1.0, 0.0, ID), -0.0);
        }
    }

    #[test]
    fn clamp_applies_the_post_op() {
        assert_eq!(clamp(P, 5.0, 0.0, 3.0, AFFINE), 7.0);
        assert_eq!(clamp(P, -5.0, 0.0, 3.0, AFFINE), 1.0);
        assert_eq!(clamp(P, 2.0, 0.0, 3.0, AFFINE), 5.0);
        assert_eq!(clamp(P, INF, 0.0, INF, AFFINE), INF);
        assert_eq!(clamp(P, NAN, 0.0, 3.0, AFFINE), 1.0);
        assert!(clamp(P, NAN, NAN, 1.0, AFFINE).is_nan());
        assert_bits(clamp(P, 2.1, 0.0, 1.1, AFFINE), 1.1 * 2.0 + 1.0);
    }

    #[test]
    fn hermite_blend_on_key_values_agrees_where_the_arithmetic_is_exact() {
        for (t, expected) in [
            (0.0_f32, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.25, 0.156_25),
            (0.75, 0.843_75),
            (-1.0, 5.0),
            (2.0, -4.0),
            (-0.5, 1.0),
        ] {
            assert_eq!(hermite_blend(P, t, ID), expected, "t = {t}");
        }
        assert_bits(hermite_blend(P, 0.0, ID), 0.0);
        assert!(hermite_blend(P, NAN, ID).is_nan());
    }

    #[test]
    fn hermite_blend_of_an_infinity() {
        for p in PLATFORMS {
            // ∞ − ∞.
            assert!(hermite_blend(p, INF, ID).is_nan());
            assert_eq!(hermite_blend(p, -INF, ID), INF);
        }
    }

    #[test]
    fn hermite_blend_applies_the_post_op() {
        assert_eq!(hermite_blend(P, 0.5, AFFINE), 2.0);
        assert_eq!(hermite_blend(P, 1.0, PostOp::new(-1.0, 3.0)), 2.0);
        assert_bits(hermite_blend(P, -0.0, PostOp::new(2.0, 0.0)), 0.0);
    }

    #[test]
    fn lerp_on_key_values() {
        assert_eq!(lerp(P, 2.0, 6.0, 0.0, ID), 2.0);
        assert_eq!(lerp(P, 2.0, 6.0, 1.0, ID), 6.0);
        assert_eq!(lerp(P, 2.0, 6.0, 0.5, ID), 4.0);
        assert_eq!(lerp(P, 2.0, 6.0, 0.25, ID), 3.0);
        assert_eq!(lerp(P, 2.0, 6.0, 2.0, ID), 10.0);
        assert_eq!(lerp(P, 2.0, 6.0, -1.0, ID), -2.0);
        assert_eq!(lerp(P, 3.0, 3.0, 100.0, ID), 3.0);
    }

    #[test]
    fn lerp_with_non_finite_operands() {
        assert!(lerp(P, 0.0, 1.0, NAN, ID).is_nan());
        assert!(lerp(P, NAN, 1.0, 0.5, ID).is_nan());
        assert!(lerp(P, 0.0, NAN, 0.5, ID).is_nan());
        assert_eq!(lerp(P, 0.0, INF, 0.5, ID), INF);
        assert!(lerp(P, 0.0, INF, 0.0, ID).is_nan());
        assert!(lerp(P, INF, INF, 0.5, ID).is_nan());
        assert_eq!(lerp(P, 0.0, 1.0, INF, ID), INF);
        assert_eq!(lerp(P, 1.0, 0.0, INF, ID), -INF);
    }

    #[test]
    fn lerp_applies_the_post_op() {
        assert_eq!(lerp(P, 0.0, 10.0, 0.5, AFFINE), 11.0);
        assert_eq!(lerp(P, 0.0, 10.0, 0.5, PostOp::new(-1.0, 0.0)), -5.0);
        assert_bits(lerp(P, 0.0, 0.0, 0.5, PostOp::new(-1.0, 0.0)), 0.0);
    }

    #[test]
    fn inverse_lerp_on_key_values() {
        assert_eq!(inverse_lerp(P, 2.0, 6.0, 2.0, ID), 0.0);
        assert_eq!(inverse_lerp(P, 2.0, 6.0, 6.0, ID), 1.0);
        assert_eq!(inverse_lerp(P, 2.0, 6.0, 4.0, ID), 0.5);
        assert_eq!(inverse_lerp(P, 2.0, 6.0, 10.0, ID), 2.0);
        assert_eq!(inverse_lerp(P, 2.0, 6.0, -2.0, ID), -1.0);
        assert_eq!(inverse_lerp(P, 6.0, 2.0, 5.0, ID), 0.25);
    }

    #[test]
    fn inverse_lerp_has_no_zero_guard() {
        assert_eq!(inverse_lerp(P, 5.0, 5.0, 6.0, ID), INF);
        assert_eq!(inverse_lerp(P, 5.0, 5.0, 4.0, ID), -INF);
        assert!(inverse_lerp(P, 5.0, 5.0, 5.0, ID).is_nan());
        assert!(inverse_lerp(P, NAN, 5.0, 5.0, ID).is_nan());
        assert!(inverse_lerp(P, 0.0, 1.0, NAN, ID).is_nan());
        assert!(inverse_lerp(P, 0.0, INF, INF, ID).is_nan());
        assert_bits(inverse_lerp(P, 0.0, INF, 1.0, ID), 0.0);
    }

    #[test]
    fn inverse_lerp_applies_the_post_op() {
        assert_eq!(inverse_lerp(P, 0.0, 4.0, 2.0, AFFINE), 2.0);
        assert_bits(inverse_lerp(P, 0.0, 4.0, 0.0, PostOp::new(2.0, 0.0)), 0.0);
        assert_eq!(inverse_lerp(P, 5.0, 5.0, 6.0, AFFINE), INF);
    }

    fn wrap_angle(x: f32) -> f32 {
        formula::wrap_angle(P, x)
    }

    #[test]
    fn wrap_angle_chooses_its_nans() {
        let negative = f32::from_bits(0xff80_0005);
        assert_eq!(wrap_angle(negative).to_bits(), 0xffc0_0005);
        assert_eq!(
            wrap_angle(f32::from_bits(0x7fc0_0006)).to_bits(),
            0x7fc0_0006
        );
        assert_eq!(wrap_angle(INF).to_bits(), 0xffc0_0000);
        assert_eq!(wrap_angle(-INF).to_bits(), 0xffc0_0000);
    }

    #[test]
    fn wrap_angle_maps_into_the_half_open_range() {
        for (x, expected) in [
            (0.0_f32, 0.0),
            (90.0, 90.0),
            (179.0, 179.0),
            (180.0, -180.0),
            (181.0, -179.0),
            (-180.0, -180.0),
            (-181.0, 179.0),
            (360.0, 0.0),
            (-360.0, 0.0),
            (540.0, -180.0),
            (-540.0, -180.0),
            (370.0, 10.0),
            (-370.0, -10.0),
            (720.0, 0.0),
        ] {
            assert_eq!(wrap_angle(x), expected, "wrap_angle({x})");
        }
        assert!(wrap_angle(NAN).is_nan());
        assert!(wrap_angle(INF).is_nan());
        assert!(wrap_angle(-INF).is_nan());
    }

    #[test]
    fn wrap_angle_signs_of_the_zero_results() {
        assert_bits(wrap_angle(0.0), 0.0);
        assert_bits(wrap_angle(-0.0), 0.0);
        assert_bits(wrap_angle(-360.0), 0.0);
    }

    /// `1180 + 2^-14` is a tie on the `f32` grid there, so shifting `1000 + 2^-14` by +180 drops
    /// the step.
    #[test]
    fn wrap_angle_rounds_the_argument_shifted_by_plus_180() {
        let step = 2.0_f32.powi(-14);
        assert_bits(wrap_angle(1000.0 + step), -80.0);
        assert_bits(wrap_angle(-1000.0 - step), 79.999_94);
        assert_bits(min_angle(P, 1000.0 + step, ID), -80.0);
        assert_bits(min_angle(P, -1000.0 - step, ID), 79.999_94);
    }

    #[test]
    fn lerprotate_takes_the_short_way_round() {
        assert_eq!(lerprotate(P, 350.0, 10.0, 0.5, ID), 360.0);
        assert_eq!(lerprotate(P, 10.0, 350.0, 0.5, ID), 0.0);
        assert_eq!(lerprotate(P, 0.0, 90.0, 0.5, ID), 45.0);
        assert_eq!(lerprotate(P, 90.0, 0.0, 0.5, ID), 45.0);
        assert_eq!(lerprotate(P, 0.0, 180.0, 0.5, ID), -90.0);
        assert_eq!(lerprotate(P, 5.0, 5.0, 0.5, ID), 5.0);
        assert_eq!(lerprotate(P, 0.0, 90.0, 2.0, ID), 180.0);
        assert_eq!(lerprotate(P, 0.0, 90.0, 0.0, ID), 0.0);
        assert_eq!(lerprotate(P, 0.0, 90.0, 1.0, ID), 90.0);
    }

    #[test]
    fn lerprotate_with_non_finite_operands_and_the_post_op() {
        assert!(lerprotate(P, 0.0, NAN, 0.5, ID).is_nan());
        assert!(lerprotate(P, NAN, 0.0, 0.5, ID).is_nan());
        assert!(lerprotate(P, 0.0, 90.0, NAN, ID).is_nan());
        assert!(lerprotate(P, 0.0, INF, 0.5, ID).is_nan());
        assert_eq!(lerprotate(P, 0.0, 90.0, 0.5, AFFINE), 91.0);
        assert_eq!(
            lerprotate(P, 350.0, 10.0, 0.5, PostOp::new(-1.0, 0.0)),
            -360.0
        );
    }

    #[test]
    fn lerprotate_rounds_the_product_and_the_sum_separately() {
        let (a, b, t) = (3.3_f32, 100.1_f32, 0.3_f32);
        let expected = t * wrap_angle(b - a) + a;
        assert_bits(lerprotate(P, a, b, t, ID), expected);
    }

    #[test]
    fn min_angle_wraps_into_the_half_open_range() {
        for (x, wrapped) in [
            (180.0_f32, -180.0),
            (-180.0, -180.0),
            (540.0, -180.0),
            (-540.0, -180.0),
            (370.0, 10.0),
            (-370.0, -10.0),
            (0.0, 0.0),
            (179.0, 179.0),
            (359.0, -1.0),
            (360.0, 0.0),
            (-359.0, 1.0),
        ] {
            assert_eq!(min_angle(P, x, ID), wrapped, "min_angle(P, {x})");
        }
        assert!(min_angle(P, NAN, ID).is_nan());
        assert!(min_angle(P, INF, ID).is_nan());
        assert!(min_angle(P, -INF, ID).is_nan());
        assert_bits(min_angle(P, -0.0, ID), 0.0);
    }

    #[test]
    fn min_angle_applies_the_post_op() {
        assert_eq!(min_angle(P, 190.0, AFFINE), -339.0);
        assert_eq!(min_angle(P, 180.0, PostOp::new(-1.0, 0.0)), 180.0);
        assert_bits(min_angle(P, 360.0, PostOp::new(2.0, 0.0)), 0.0);
    }

    #[test]
    fn hermite_blend_is_computed_term_by_term() {
        let t = f32::from_bits(0x3fde_6363);
        for p in PLATFORMS {
            assert_eq!(hermite_blend(p, t, ID), (3.0 * t) * t - ((t + t) * t) * t);
            assert_eq!(hermite_blend(p, t, ID).to_bits(), 0xbfb7_7588);
        }
    }

    #[test]
    fn lerp_lerprotate_inverse_lerp_and_min_angle_on_key_values() {
        assert_eq!(lerp(P, 0.0, 10.0, 0.5, ID), 5.0);
        assert_eq!(lerp(P, 0.0, 10.0, 2.0, ID), 20.0);
        assert_eq!(lerp(P, 0.0, 10.0, 0.5, PostOp::new(2.0, 1.0)), 11.0);
        assert_eq!(lerprotate(P, 350.0, 10.0, 0.5, ID), 360.0);
        assert_eq!(lerprotate(P, 10.0, 350.0, 0.5, ID), 0.0);
        assert_eq!(inverse_lerp(P, 1.0, 5.0, 3.0, ID), 0.5);
        assert_eq!(inverse_lerp(P, 1.0, 5.0, 9.0, ID), 2.0);
        assert_eq!(inverse_lerp(P, 5.0, 5.0, 4.0, ID), f32::NEG_INFINITY);
        assert!(inverse_lerp(P, 5.0, 5.0, 5.0, ID).is_nan());
        for (x, wrapped) in [
            (180.0, -180.0),
            (-180.0, -180.0),
            (540.0, -180.0),
            (370.0, 10.0),
            (-370.0, -10.0),
            (0.0, 0.0),
            (179.0, 179.0),
        ] {
            assert_eq!(min_angle(P, x, ID), wrapped, "min_angle(P, {x})");
        }
    }

    #[test]
    fn lerp_rounds_each_operation() {
        let la = 1.1_f32;
        let lb = la * 3.3;
        let t = third();
        let stepwise = la + t * (lb - la);
        assert_ne!(stepwise, t.mul_add(lb - la, la));
        for p in PLATFORMS {
            assert_eq!(lerp(p, la, lb, t, ID), stepwise);
        }
    }

    #[test]
    fn clamp_with_nan() {
        for p in PLATFORMS {
            assert_eq!(clamp(p, NAN, 1.0, 2.0, ID), 1.0);
            assert!(clamp(p, 4.0, NAN, 5.0, ID).is_nan());
            assert_eq!(clamp(p, 4.0, 1.0, NAN, ID), 4.0);
        }
    }
}
