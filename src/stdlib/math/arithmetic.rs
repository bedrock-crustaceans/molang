//! `abs`, `copy_sign`, `exp`, `ln`, `max`, `min`, `mod`, `pow`, `sign` and `sqrt`.

use super::{offset_first, transcendental};
use crate::numeric::{Platform, PostOp, arith, fpu, lt};

/// `math.abs(x)`: `x` with its sign bit cleared.
#[inline]
pub fn abs(platform: Platform, x: f32, post: PostOp) -> f32 {
    offset_first(platform, x.abs(), post)
}

/// `math.copy_sign(a, b)`: the magnitude of `a` with the sign bit of `b`.
#[inline]
pub fn copy_sign(platform: Platform, a: f32, b: f32, post: PostOp) -> f32 {
    post.apply(platform, a.copysign(b))
}

/// `math.exp(x)`: e^x.
#[inline]
pub fn exp(platform: Platform, x: f32, post: PostOp) -> f32 {
    post.apply(platform, transcendental::exp(platform, x))
}

/// `math.ln(x)`: the natural logarithm (`ln(0)` = −∞).
///
/// A negative argument (−∞ included) gives the platform's NaN (see [`Platform`]); a NaN argument
/// comes back quietened.
#[inline]
pub fn ln(platform: Platform, x: f32, post: PostOp) -> f32 {
    post.apply(platform, transcendental::ln(platform, x))
}

/// `math.max(a, b)`: see [`arith::max`].
#[inline]
pub fn max(platform: Platform, a: f32, b: f32, post: PostOp) -> f32 {
    post.apply(platform, arith::max(a, b))
}

/// `math.min(a, b)`: see [`arith::min`].
#[inline]
pub fn min(platform: Platform, a: f32, b: f32, post: PostOp) -> f32 {
    post.apply(platform, arith::min(a, b))
}

/// `math.mod(a, b)` with a non-literal divisor: the truncated remainder, or the node's offset when
/// `b == 0`.
///
/// The post-op is applied even when it is `(1, 0)`, so `math.mod(-3, 3)` is `+0`.
#[inline]
pub fn mod_runtime(platform: Platform, a: f32, b: f32, post: PostOp) -> f32 {
    if b == 0.0 {
        post.offset
    } else {
        mod_const(platform, a, b, post)
    }
}

/// `math.mod(a, b)` with a literal divisor: the truncated remainder with no zero test, so a literal
/// 0 gives the platform's NaN. Always applies the post-op.
#[inline]
pub fn mod_const(platform: Platform, a: f32, b: f32, post: PostOp) -> f32 {
    arith::mul_add(platform, fpu::rem(platform, a, b), post.scale, post.offset)
}

/// `math.pow(a, b)`: `a` to the power `b`. A NaN `a` with an odd integer `b` gives that NaN with
/// its sign bit cleared and its quiet bit set; an invalid operation gives the platform's NaN.
#[inline]
pub fn pow(platform: Platform, a: f32, b: f32, post: PostOp) -> f32 {
    post.apply(platform, transcendental::pow(platform, a, b))
}

/// `math.sign(x)`: `−(S + O)` when `x < 0`, else `S + O`; `sign(0)` = `sign(−0)` = `sign(NaN)` =
/// 1.
///
/// The post-op is not `raw·S + O`: `v.x = -1; math.sign(v.x) + 1` is −2.
#[inline]
pub fn sign(platform: Platform, x: f32, post: PostOp) -> f32 {
    let k = post.truthy_value(platform);
    if lt(x, 0.0) { -k } else { k }
}

/// `math.sqrt(x)`: IEEE square root; a negative argument gives the platform's NaN.
#[inline]
pub fn sqrt(platform: Platform, x: f32, post: PostOp) -> f32 {
    post.apply(platform, fpu::sqrt(platform, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::{PostOp, test_support::*};
    use crate::stdlib::math::{clamp, hermite_blend};

    const P: Platform = LINUX;

    #[test]
    fn ln_of_a_negative_number_is_the_nan_of_the_platform() {
        for (p, nan) in [
            (LINUX, 0xffc0_0000),
            (WINDOWS, 0xffc0_0000),
            (ANDROID, 0x7fc0_0000),
        ] {
            for x in [
                -1.0_f32,
                -0.5,
                -1.0e-30,
                -1.0e-45,
                -f32::MAX,
                f32::NEG_INFINITY,
            ] {
                assert_eq!(ln(p, x, ID).to_bits(), nan, "{p:?} {x}");
                assert_eq!(ln(p, x, AFFINE).to_bits(), nan, "{p:?} {x} with a post-op");
            }
            assert_eq!(
                ln(p, f32::from_bits(0xffc0_0000), ID).to_bits(),
                0xffc0_0000
            );
            assert_eq!(ln(p, 0.0, ID), f32::NEG_INFINITY);
            assert_eq!(ln(p, -0.0, ID), f32::NEG_INFINITY);
            assert_eq!(ln(p, 1.0, ID).to_bits(), 0);
        }
    }

    #[test]
    fn abs_clears_the_sign_of_every_value() {
        for (x, expected) in [
            (-2.0_f32, 2.0),
            (2.0, 2.0),
            (-0.0, 0.0),
            (0.0, 0.0),
            (-INF, INF),
            (INF, INF),
            (-1.0e-45, 1.0e-45),
            (f32::MIN, f32::MAX),
        ] {
            assert_bits(abs(P, x, ID), expected);
        }
        assert!(abs(P, NAN, ID).is_nan());
        assert!(abs(P, -NAN, ID).is_nan());
        assert!(abs(P, -NAN, ID).is_sign_positive());
    }

    #[test]
    fn abs_applies_the_post_op_after_the_absolute_value() {
        assert_eq!(abs(P, -3.0, AFFINE), 7.0);
        assert_eq!(abs(P, 3.0, AFFINE), 7.0);
        assert_eq!(abs(P, -3.0, PostOp::new(-1.0, 0.0)), -3.0);
        assert_bits(abs(P, -0.0, PostOp::new(-1.0, 0.0)), 0.0);
        assert_eq!(abs(P, -INF, AFFINE), INF);
    }

    #[test]
    fn copy_sign_takes_the_sign_bit_of_the_second_argument() {
        assert_bits(copy_sign(P, 2.0, 3.0, ID), 2.0);
        assert_bits(copy_sign(P, 2.0, -3.0, ID), -2.0);
        assert_bits(copy_sign(P, -2.0, 3.0, ID), 2.0);
        assert_bits(copy_sign(P, -2.0, -3.0, ID), -2.0);
        assert_bits(copy_sign(P, 2.0, 0.0, ID), 2.0);
        assert_bits(copy_sign(P, 2.0, -0.0, ID), -2.0);
        assert_bits(copy_sign(P, 0.0, -1.0, ID), -0.0);
        assert_bits(copy_sign(P, -0.0, 1.0, ID), 0.0);
        assert_bits(copy_sign(P, INF, -1.0, ID), -INF);
        assert_bits(copy_sign(P, -INF, 1.0, ID), INF);
        assert_bits(copy_sign(P, 2.0, -INF, ID), -2.0);
    }

    #[test]
    fn copy_sign_with_nan_operands() {
        let negative = copy_sign(P, NAN, -1.0, ID);
        assert!(negative.is_nan() && negative.is_sign_negative());
        let positive = copy_sign(P, NAN, 1.0, ID);
        assert!(positive.is_nan() && positive.is_sign_positive());
        assert_bits(copy_sign(P, 2.0, -NAN, ID), -2.0);
        assert_bits(copy_sign(P, 2.0, NAN, ID), 2.0);
    }

    #[test]
    fn copy_sign_applies_the_post_op() {
        assert_eq!(copy_sign(P, 3.0, -1.0, AFFINE), -5.0);
        assert_eq!(copy_sign(P, -3.0, 1.0, AFFINE), 7.0);
        assert_bits(copy_sign(P, 0.0, -1.0, PostOp::new(2.0, 0.0)), 0.0);
    }

    #[test]
    fn exp_and_ln_on_key_values() {
        assert_bits(exp(P, 0.0, ID), 1.0);
        assert_bits(exp(P, -0.0, ID), 1.0);
        close(exp(P, 1.0, ID), std::f32::consts::E, 1e-6);
        close(exp(P, -1.0, ID), 1.0 / std::f32::consts::E, 1e-7);
        assert_eq!(exp(P, INF, ID), INF);
        assert_bits(exp(P, -INF, ID), 0.0);
        assert_eq!(exp(P, 100.0, ID), INF);
        assert_bits(exp(P, -200.0, ID), 0.0);
        assert!(exp(P, NAN, ID).is_nan());
        assert_bits(ln(P, 1.0, ID), 0.0);
        close(ln(P, std::f32::consts::E, ID), 1.0, 1e-6);
        close(ln(P, 10.0, ID), std::f32::consts::LN_10, 1e-6);
        assert_eq!(ln(P, 0.0, ID), -INF);
        assert_eq!(ln(P, -0.0, ID), -INF);
        assert_eq!(ln(P, INF, ID), INF);
        assert!(ln(P, -1.0, ID).is_nan());
        assert!(ln(P, -INF, ID).is_nan());
        assert!(ln(P, NAN, ID).is_nan());
    }

    #[test]
    fn exp_and_ln_apply_the_post_op() {
        assert_eq!(exp(P, 0.0, AFFINE), 3.0);
        assert_eq!(ln(P, 1.0, AFFINE), 1.0);
        assert_eq!(exp(P, INF, AFFINE), INF);
        assert_eq!(ln(P, 0.0, AFFINE), -INF);
        assert_eq!(exp(P, -INF, PostOp::new(5.0, 7.0)), 7.0);
    }

    #[test]
    fn pow_on_key_values() {
        assert_eq!(pow(P, 2.0, 10.0, ID), 1024.0);
        assert_eq!(pow(P, 2.0, -1.0, ID), 0.5);
        assert_eq!(pow(P, -2.0, 3.0, ID), -8.0);
        assert_eq!(pow(P, -2.0, 2.0, ID), 4.0);
        assert_eq!(pow(P, 9.0, 0.5, ID), 3.0);
        assert_bits(pow(P, 7.0, 0.0, ID), 1.0);
        assert_bits(pow(P, NAN, 0.0, ID), 1.0);
        assert_bits(pow(P, 1.0, NAN, ID), 1.0);
        assert_bits(pow(P, 0.0, 0.0, ID), 1.0);
        assert!(pow(P, -8.0, 1.0 / 3.0, ID).is_nan());
        assert!(pow(P, NAN, 1.0, ID).is_nan());
        assert!(pow(P, 2.0, NAN, ID).is_nan());
        assert_eq!(pow(P, 0.0, -1.0, ID), INF);
        assert_eq!(pow(P, -0.0, -1.0, ID), -INF);
        assert_eq!(pow(P, -0.0, -2.0, ID), INF);
        assert_bits(pow(P, 0.0, 3.0, ID), 0.0);
        assert_bits(pow(P, -0.0, 3.0, ID), -0.0);
        assert_eq!(pow(P, INF, 2.0, ID), INF);
        assert_bits(pow(P, INF, -2.0, ID), 0.0);
        assert_eq!(pow(P, 2.0, INF, ID), INF);
        assert_bits(pow(P, 0.5, INF, ID), 0.0);
        assert_eq!(pow(P, 10.0, 100.0, ID), INF);
        assert_bits(pow(P, 2.0, 10.0, ID), 1024.0);
    }

    #[test]
    fn pow_of_a_nan_to_an_odd_integer_clears_the_sign_bit() {
        let negative_nan = f32::from_bits(0xffc0_0000);
        for b in [1.0, -1.0, 3.0, -5.0] {
            assert_eq!(
                pow(P, negative_nan, b, ID).to_bits(),
                0x7fc0_0000,
                "pow(P, -NaN, {b})"
            );
        }
        for b in [2.0, 0.5, -2.0, 16_777_217.0] {
            assert_eq!(
                pow(P, negative_nan, b, ID).to_bits(),
                0xffc0_0000,
                "pow(P, -NaN, {b})"
            );
        }
        assert_bits(pow(P, negative_nan, 0.0, ID), 1.0);
    }

    #[test]
    fn pow_never_returns_a_signalling_nan() {
        assert_eq!(
            pow(P, f32::from_bits(0x7f80_0001), 1.0, ID).to_bits(),
            0x7fc0_0001
        );
        assert_eq!(
            pow(P, f32::from_bits(0xff80_0001), -1.0, ID).to_bits(),
            0x7fc0_0001
        );
        assert_eq!(
            pow(P, f32::from_bits(0xff80_0001), 2.0, ID).to_bits(),
            0xffc0_0001
        );
        assert_eq!(
            pow(P, 2.0, f32::from_bits(0x7f80_0001), ID).to_bits(),
            0x7fc0_0001
        );
    }

    #[test]
    fn pow_applies_the_post_op() {
        assert_eq!(pow(P, 2.0, 3.0, AFFINE), 17.0);
        assert_eq!(pow(P, 2.0, 3.0, PostOp::new(-1.0, 0.0)), -8.0);
        assert_eq!(pow(P, 0.0, -1.0, AFFINE), INF);
    }

    #[test]
    fn sqrt_on_key_values() {
        assert_eq!(sqrt(P, 16.0, ID), 4.0);
        assert_eq!(sqrt(P, 2.25, ID), 1.5);
        assert_bits(sqrt(P, 0.0, ID), 0.0);
        assert_bits(sqrt(P, -0.0, ID), -0.0);
        assert_eq!(sqrt(P, INF, ID), INF);
        assert!(sqrt(P, -1.0, ID).is_nan());
        assert!(sqrt(P, -INF, ID).is_nan());
        assert!(sqrt(P, NAN, ID).is_nan());
        assert_bits(sqrt(P, 2.0, ID), 2.0_f32.sqrt());
        assert_bits(sqrt(P, 1.0e-45, ID), 1.0e-45_f32.sqrt());
        assert_eq!(sqrt(P, 16.0, AFFINE), 9.0);
        assert_bits(sqrt(P, -0.0, PostOp::new(2.0, 0.0)), 0.0);
    }

    #[test]
    fn max_and_min_on_ordinary_values() {
        assert_eq!(max(P, 0.0, 1.0, ID), 1.0);
        assert_eq!(max(P, 1.0, 0.0, ID), 1.0);
        assert_eq!(min(P, 0.0, 1.0, ID), 0.0);
        assert_eq!(min(P, 1.0, 0.0, ID), 0.0);
        assert_eq!(max(P, -INF, INF, ID), INF);
        assert_eq!(min(P, -INF, INF, ID), -INF);
        assert_eq!(max(P, 2.0, 2.0, ID), 2.0);
        assert_eq!(min(P, 2.0, 2.0, ID), 2.0);
        assert_eq!(max(P, 1.0, 2.0, PostOp::new(3.0, 1.0)), 7.0);
        assert_eq!(min(P, 1.0, 2.0, PostOp::new(3.0, 1.0)), 4.0);
    }

    #[test]
    fn max_and_min_with_a_real_post_op_normalise_negative_zero() {
        assert_bits(max(P, -0.0, -0.0, ID), -0.0);
        assert_bits(max(P, -0.0, -0.0, PostOp::new(2.0, 0.0)), 0.0);
        assert_bits(min(P, -0.0, -0.0, PostOp::new(2.0, 0.0)), 0.0);
    }

    #[test]
    fn mod_runtime_and_mod_const_agree_for_a_non_zero_divisor() {
        for (a, b) in [
            (7.5_f32, 2.0),
            (-7.5, 2.0),
            (7.5, -2.0),
            (-7.5, -2.0),
            (0.5, 3.0),
            (3.0, 7.0),
            (INF, 1.0 + 1.0),
            (5.0, INF),
            (5.0, -INF),
        ] {
            for post in [ID, AFFINE, PostOp::new(0.5, -3.0)] {
                assert_bits(mod_runtime(P, a, b, post), mod_const(P, a, b, post));
            }
        }
    }

    #[test]
    fn mod_follows_the_sign_of_the_dividend() {
        assert_eq!(mod_runtime(P, 7.5, 2.0, ID), 1.5);
        assert_eq!(mod_runtime(P, -7.5, 2.0, ID), -1.5);
        assert_eq!(mod_runtime(P, 7.5, -2.0, ID), 1.5);
        assert_eq!(mod_runtime(P, -7.5, -2.0, ID), -1.5);
        assert_eq!(mod_runtime(P, 5.0, INF, ID), 5.0);
        assert_eq!(mod_runtime(P, -5.0, -INF, ID), -5.0);
        assert!(mod_runtime(P, INF, 2.0, ID).is_nan());
        assert!(mod_runtime(P, -INF, 2.0, ID).is_nan());
        assert!(mod_runtime(P, NAN, 2.0, ID).is_nan());
        assert!(mod_runtime(P, 2.0, NAN, ID).is_nan());
        assert_eq!(mod_runtime(P, 1.0, 3.0, ID), 1.0);
        assert_eq!(mod_runtime(P, 3.0, 3.0, ID), 0.0);
    }

    #[test]
    fn mod_always_applies_the_post_op_even_for_the_identity() {
        assert_bits(mod_runtime(P, -0.0, 3.0, ID), 0.0);
        assert_bits(mod_const(P, -0.0, 3.0, ID), 0.0);
        assert_bits(mod_runtime(P, -6.0, 3.0, ID), 0.0);
        assert_bits(mod_const(P, -6.0, 3.0, ID), 0.0);
        assert_bits(mod_const(P, -6.0, 3.0, PostOp::new(1.0, -0.0)), -0.0);
    }

    #[test]
    fn mod_runtime_returns_the_offset_for_a_zero_divisor() {
        assert_eq!(mod_runtime(P, 5.0, 0.0, ID), 0.0);
        assert_bits(mod_runtime(P, 5.0, 0.0, ID), 0.0);
        assert_bits(mod_runtime(P, 5.0, -0.0, ID), 0.0);
        assert_eq!(mod_runtime(P, 5.0, 0.0, PostOp::new(2.0, 0.125)), 0.125);
        assert_eq!(mod_runtime(P, NAN, 0.0, PostOp::new(2.0, 0.125)), 0.125);
        assert_eq!(mod_runtime(P, INF, 0.0, PostOp::new(2.0, 0.125)), 0.125);
        assert_bits(mod_runtime(P, 5.0, 0.0, PostOp::new(2.0, -0.0)), -0.0);
    }

    #[test]
    fn mod_const_by_zero_is_nan() {
        assert!(mod_const(P, 5.0, 0.0, ID).is_nan());
        assert!(mod_const(P, 5.0, -0.0, ID).is_nan());
        assert!(mod_const(P, 0.0, 0.0, ID).is_nan());
        assert!(mod_const(P, 5.0, 0.0, AFFINE).is_nan());
        assert!(mod_const(P, INF, 0.0, ID).is_nan());
    }

    #[test]
    fn mod_rounds_the_post_op_twice_on_every_platform() {
        let post = PostOp::new(0.7, 0.3);
        let r = 5.3_f32 % 1.1;
        for p in PLATFORMS {
            assert_bits(mod_runtime(p, 5.3, 1.1, post), r * 0.7 + 0.3);
            assert_bits(mod_const(p, 5.3, 1.1, post), r * 0.7 + 0.3);
        }
    }

    #[test]
    fn sign_on_ordinary_values_and_zeros() {
        assert_bits(sign(P, 3.0, ID), 1.0);
        assert_bits(sign(P, -3.0, ID), -1.0);
        assert_bits(sign(P, 1.0e-45, ID), 1.0);
        assert_bits(sign(P, -1.0e-45, ID), -1.0);
        assert_bits(sign(P, INF, ID), 1.0);
        assert_bits(sign(P, -INF, ID), -1.0);
        assert_bits(sign(P, 0.0, ID), 1.0);
        assert_bits(sign(P, -0.0, ID), 1.0);
    }

    #[test]
    fn sign_of_nan_is_positive() {
        assert_bits(sign(P, NAN, ID), 1.0);
        assert_bits(sign(P, -NAN, ID), 1.0);
        assert_bits(sign(P, NAN, PostOp::new(2.0, 3.0)), 5.0);
    }

    #[test]
    fn sign_returns_the_precomputed_constant_not_raw_times_scale_plus_offset() {
        assert_eq!(sign(P, 4.0, PostOp::new(2.0, 3.0)), 5.0);
        assert_eq!(sign(P, -4.0, PostOp::new(2.0, 3.0)), -5.0);
        assert_eq!(sign(P, 0.0, PostOp::new(2.0, 3.0)), 5.0);
        assert_bits(sign(P, 4.0, PostOp::new(0.0, 0.0)), 0.0);
        assert_bits(sign(P, -4.0, PostOp::new(0.0, 0.0)), -0.0);
        assert_bits(sign(P, 1.0, PostOp::new(0.1, 0.2)), 0.1_f32 + 0.2);
        assert_bits(sign(P, -1.0, PostOp::new(0.1, 0.2)), -(0.1_f32 + 0.2));
    }

    #[test]
    fn sign_post_op_quirk() {
        assert_eq!(sign(P, -1.0, PostOp::new(1.0, 1.0)), -2.0);
        assert_eq!(sign(P, -1.0, PostOp::new(-2.0, 1.0)), 1.0);
        assert_eq!(sign(P, 1.0, PostOp::new(1.0, 1.0)), 2.0);
        assert_eq!(sign(P, 1.0, PostOp::new(-2.0, 1.0)), -1.0);
    }

    #[test]
    fn abs_clamp_copy_sign_exp_ln_sqrt_pow_hermite_max_min_on_key_values() {
        assert_eq!(abs(P, -2.0, ID), 2.0);
        assert_eq!(abs(P, -0.0, ID).to_bits(), 0.0_f32.to_bits());
        assert_eq!(clamp(P, 3.0, 2.0, 1.0, ID), 1.0);
        assert_eq!(clamp(P, -1.0, -2.0, -3.0, ID), -3.0);
        assert_eq!(clamp(P, 1.0, 2.0, 3.0, ID), 2.0);
        assert_eq!(
            clamp(P, 2.1, 0.0, 1.1, PostOp::new(2.0, 1.0)),
            arith::mul_add(P, 1.1, 2.0, 1.0)
        );
        assert_eq!(copy_sign(P, -1.1, 3.1, ID), 1.1);
        assert_eq!(copy_sign(P, 2.0, -0.0, ID), -2.0);
        assert_eq!(copy_sign(P, 0.0, -2.0, ID).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(exp(P, 0.0, ID), 1.0);
        assert_eq!(ln(P, 1.0, ID), 0.0);
        assert_eq!(ln(P, 0.0, ID), f32::NEG_INFINITY);
        assert!(sqrt(P, -1.0, ID).is_nan());
        assert_eq!(sqrt(P, 16.0, ID), 4.0);
        assert_eq!(pow(P, 2.0, 10.0, ID), 1024.0);
        assert_eq!(hermite_blend(P, 0.5, ID), 0.5);
        assert_eq!(hermite_blend(P, 2.0, ID), -4.0);
        assert_eq!(hermite_blend(P, 0.0, ID), 0.0);
        assert_eq!(hermite_blend(P, 1.0, ID), 1.0);
        assert_eq!(max(P, 0.0, 1.0, ID), 1.0);
        assert_eq!(min(P, 0.0, 1.0, ID), 0.0);
        assert_eq!(max(P, 1.0, 2.0, PostOp::new(3.0, 1.0)), 7.0);
    }

    #[test]
    fn min_max_return_the_second_operand_on_every_platform() {
        for p in PLATFORMS {
            assert_eq!(max(p, NAN, 4.0, ID), 4.0);
            assert!(max(p, 4.0, NAN, ID).is_nan());
            assert!(max(p, NAN, NAN, ID).is_nan());
            assert_eq!(min(p, NAN, 4.0, ID), 4.0);
            assert!(min(p, 4.0, NAN, ID).is_nan());
            assert_bits(max(p, -0.0, 0.0, ID), 0.0);
            assert_bits(max(p, 0.0, -0.0, ID), -0.0);
        }
    }

    #[test]
    fn abs_adds_the_offset_first_on_windows() {
        let post = PostOp::new(NAN, -NAN);
        assert_bits(abs(LINUX, 2.0, post), NAN);
        assert_bits(abs(WINDOWS, 2.0, post), -NAN);
        assert_bits(abs(ANDROID, 2.0, post), NAN);
    }

    #[test]
    fn sign_of_nan_and_of_zero() {
        assert_eq!(sign(P, NAN, ID), 1.0);
        assert_eq!(sign(P, 0.0, ID), 1.0);
        assert_eq!(sign(P, -0.0, ID), 1.0);
        assert_eq!(sign(P, -3.0, ID), -1.0);
        assert_eq!(sign(P, 3.0, ID), 1.0);
    }

    #[test]
    fn mod_zero_rules() {
        assert!(mod_const(P, 1.0, 0.0, ID).is_nan());
        assert!(mod_const(P, 5.0, 0.0, ID).is_nan());
        assert_eq!(mod_runtime(P, 1.0, 0.0, ID), 0.0);
        assert_eq!(mod_runtime(P, 1.0, -0.0, ID), 0.0);
        assert_eq!(mod_runtime(P, 1.0, 0.0, PostOp::new(2.0, 0.125)), 0.125);
        assert_eq!(mod_runtime(P, -5.1, 3.0, ID), -5.1_f32 % 3.0);
        assert!((mod_runtime(P, -5.1, 3.0, ID) - -2.1).abs() <= 1e-6);
        assert_eq!(mod_const(P, 7.5, 2.0, PostOp::new(2.0, 1.0)), 4.0);
        assert!(mod_runtime(P, 1.0, NAN, ID).is_nan());
        assert_eq!(-3.0_f32 % 3.0, 0.0);
        assert!((-3.0_f32 % 3.0).is_sign_negative());
        assert_eq!(mod_const(P, -3.0, 3.0, ID).to_bits(), 0.0_f32.to_bits());
        assert_eq!(mod_runtime(P, -3.0, 3.0, ID).to_bits(), 0.0_f32.to_bits());
    }
}
