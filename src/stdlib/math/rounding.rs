//! `ceil`, `floor`, `round` and `trunc`.

use super::offset_first;
use crate::numeric::{Platform, PostOp, fpu};

/// `math.ceil(x)`: round toward +∞ (`ceil(-0.5)` = −0).
#[inline]
pub fn ceil(platform: Platform, x: f32, post: PostOp) -> f32 {
    offset_first(platform, fpu::ceil(x), post)
}

/// `math.floor(x)`: round toward −∞.
#[inline]
pub fn floor(platform: Platform, x: f32, post: PostOp) -> f32 {
    offset_first(platform, fpu::floor(x), post)
}

/// `math.round(x)`: round half away from zero (`round(-2.5)` = −3).
#[inline]
pub fn round(platform: Platform, x: f32, post: PostOp) -> f32 {
    offset_first(platform, fpu::round(x), post)
}

/// `math.trunc(x)`: round toward zero.
#[inline]
pub fn trunc(platform: Platform, x: f32, post: PostOp) -> f32 {
    offset_first(platform, fpu::trunc(x), post)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{PostOp, test_support::*};

    const P: Platform = NATIVE;

    #[test]
    fn ceil_floor_round_and_trunc_on_ordinary_values() {
        // (x, ceil, floor, round, trunc)
        let table: [(f32, f32, f32, f32, f32); 10] = [
            (1.1, 2.0, 1.0, 1.0, 1.0),
            (-1.1, -1.0, -2.0, -1.0, -1.0),
            (1.5, 2.0, 1.0, 2.0, 1.0),
            (-1.5, -1.0, -2.0, -2.0, -1.0),
            (2.5, 3.0, 2.0, 3.0, 2.0),
            (-2.5, -2.0, -3.0, -3.0, -2.0),
            (0.499_999_97, 1.0, 0.0, 0.0, 0.0),
            (3.0, 3.0, 3.0, 3.0, 3.0),
            (-3.0, -3.0, -3.0, -3.0, -3.0),
            (
                8_388_609.0,
                8_388_609.0,
                8_388_609.0,
                8_388_609.0,
                8_388_609.0,
            ),
        ];
        for (x, c, f, r, t) in table {
            assert_bits(ceil(P, x, ID), c);
            assert_bits(floor(P, x, ID), f);
            assert_bits(round(P, x, ID), r);
            assert_bits(trunc(P, x, ID), t);
        }
    }

    #[test]
    fn rounding_keeps_zeros_infinities_and_nan() {
        for f in [ceil, floor, round, trunc] {
            assert_bits(f(P, 0.0, ID), 0.0);
            assert_bits(f(P, -0.0, ID), -0.0);
            assert_eq!(f(P, INF, ID), INF);
            assert_eq!(f(P, -INF, ID), -INF);
            assert!(f(P, NAN, ID).is_nan());
        }
    }

    #[test]
    fn rounding_quietens_a_nan_on_every_platform() {
        for f in [ceil, floor, round, trunc] {
            for p in PLATFORMS {
                assert_eq!(f(p, f32::from_bits(0xff80_0001), ID).to_bits(), 0xffc0_0001);
                assert_eq!(f(p, f32::from_bits(0x7fc0_0002), ID).to_bits(), 0x7fc0_0002);
            }
        }
    }

    #[test]
    fn rounding_a_small_negative_value_gives_negative_zero() {
        assert_bits(ceil(P, -0.5, ID), -0.0);
        assert_bits(round(P, -0.4, ID), -0.0);
        assert_bits(trunc(P, -0.9, ID), -0.0);
        assert_bits(floor(P, 0.9, ID), 0.0);
        assert_bits(ceil(P, -1.0e-45, ID), -0.0);
    }

    #[test]
    fn rounding_with_a_real_post_op_normalises_negative_zero() {
        let post = PostOp::new(2.0, 0.0);
        assert_bits(ceil(P, -0.5, post), 0.0);
        assert_bits(round(P, -0.4, post), 0.0);
        assert_bits(trunc(P, -0.9, post), 0.0);
        assert_bits(floor(P, 0.9, post), 0.0);
    }

    #[test]
    fn rounding_applies_the_post_op_to_the_rounded_value() {
        assert_eq!(ceil(P, 1.1, AFFINE), 5.0);
        assert_eq!(floor(P, 1.9, AFFINE), 3.0);
        assert_eq!(round(P, 1.5, AFFINE), 5.0);
        assert_eq!(round(P, -1.5, AFFINE), -3.0);
        assert_eq!(trunc(P, -1.9, AFFINE), -1.0);
        assert_eq!(ceil(P, INF, AFFINE), INF);
        assert!(ceil(P, NAN, AFFINE).is_nan());
    }

    #[test]
    fn ceil_floor_round_trunc_on_halves_and_with_a_post_op() {
        assert_eq!(ceil(P, -0.5, ID).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(ceil(P, 1.1, ID), 2.0);
        assert_eq!(floor(P, -0.5, ID), -1.0);
        assert_eq!(floor(P, 1.9, ID), 1.0);
        assert_eq!(round(P, 0.5, ID), 1.0);
        assert_eq!(round(P, -0.5, ID), -1.0);
        assert_eq!(round(P, -1.5, ID), -2.0);
        assert_eq!(round(P, -2.5, ID), -3.0);
        assert_eq!(round(P, 2.4, ID), 2.0);
        assert_eq!(trunc(P, 1.7, ID), 1.0);
        assert_eq!(trunc(P, -1.7, ID), -1.0);
        assert_eq!(ceil(P, 1.1, PostOp::new(-2.0, -1.0)), -5.0);
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn rounding_adds_the_offset_first_on_windows() {
        let post = PostOp::new(NAN, -NAN);
        for f in [ceil, floor, round, trunc] {
            assert_bits(f(LINUX, 2.5, post), NAN);
            assert_bits(f(WINDOWS, 2.5, post), -NAN);
            assert_bits(f(ANDROID, 2.5, post), NAN);
        }
    }
}
