//! Truthiness, comparisons, `add`, `negate`, `mul`, `div` and `array_index`.

use super::{Platform, PostOp, fpu};

/// `x != 0.0`: `−0.0` is false and NaN is **true**.
#[inline]
pub fn truthy(x: f32) -> bool {
    x != 0.0
}

/// `!x`: the node's "false" constant when `x` is truthy (NaN included), else its "true" constant.
#[inline]
pub fn not(platform: Platform, x: f32, post: PostOp) -> f32 {
    post.select(platform, !truthy(x))
}

/// `a < b`: false with a NaN.
#[inline]
pub fn lt(a: f32, b: f32) -> bool {
    a < b
}

/// `a <= b`: false with a NaN.
#[inline]
pub fn le(a: f32, b: f32) -> bool {
    a <= b
}

/// `a > b`: false with a NaN.
#[inline]
pub fn gt(a: f32, b: f32) -> bool {
    a > b
}

/// `a >= b`: false with a NaN.
#[inline]
pub fn ge(a: f32, b: f32) -> bool {
    a >= b
}

/// `a == b`: exact, no epsilon; false with a NaN.
#[inline]
pub fn eq(a: f32, b: f32) -> bool {
    a == b
}

/// `a != b`: exact, no epsilon; true with a NaN.
#[inline]
pub fn ne(a: f32, b: f32) -> bool {
    a != b
}

/// The last step of an n-ary `+`: `(top + acc)·S + O`, with `top` the sum of the earlier terms, so
/// in `a + b` the left operand is the first.
#[inline]
pub fn add(platform: Platform, acc: f32, top: f32, post: PostOp) -> f32 {
    post.apply(platform, fpu::add(platform, top, acc))
}

/// Unary minus: `O − x·S`, or the sign flip `−x` in the plain form.
///
/// `negate(+0)` is −0, unlike `0 − x`. The optimiser folds every negation into a post-op, so the
/// evaluator never runs this.
#[inline]
pub fn negate(platform: Platform, x: f32, post: PostOp) -> f32 {
    if post.is_identity() {
        -x
    } else {
        fpu::sub(platform, post.offset, fpu::mul(platform, x, post.scale))
    }
}

/// `acc * top` with the node's post-op: `(top·acc)·S + O`, `top` the left operand of `a * b`.
#[inline]
pub fn mul(platform: Platform, acc: f32, top: f32, post: PostOp) -> f32 {
    post.apply(platform, fpu::mul(platform, top, acc))
}

/// The division guard, run on the divisor before the numerator is evaluated.
///
/// `None` when `|d| < f32::EPSILON`: the division is `0.0`, with no post-op, and the numerator is
/// **not** evaluated. Otherwise the divisor: `d` when `signed` (`MolangVersion` ≥ 7), `|d|`
/// otherwise. A NaN divisor passes.
#[inline]
pub fn div_guard(signed: bool, d: f32) -> Option<f32> {
    if d.abs() < f32::EPSILON {
        None
    } else {
        Some(if signed { d } else { d.abs() })
    }
}

/// `a / b` for a divisor that passed [`div_guard`], with the node's post-op: `(a/b)·S + O`.
#[inline]
pub fn div(platform: Platform, a: f32, b: f32, post: PostOp) -> f32 {
    post.apply(platform, fpu::div(platform, a, b))
}

/// The multiplier a literal divisor folds into: `x / c` becomes `x · (1/c)`, or `x · 0` when
/// `|c| < f32::EPSILON` or `c` is NaN. The sign is kept in every version.
#[inline]
pub fn fold_const_divisor(c: f32) -> f32 {
    if c.abs() >= f32::EPSILON {
        1.0 / c
    } else {
        0.0
    }
}

/// An all-constant `a / b`: `0` when `|b| < f32::EPSILON` or `b` is NaN.
#[inline]
pub fn fold_const_div(platform: Platform, a: f32, b: f32) -> f32 {
    if b.abs() >= f32::EPSILON {
        fpu::div(platform, a, b)
    } else {
        0.0
    }
}

/// The element index of `array.x[index]`: `max(0, trunc(index)) mod len`, or `None` for an empty
/// array.
///
/// NaN gives element 0; past the `i32` range the index is as
/// [`arith::to_int`](super::arith::to_int) converts it.
#[inline]
pub fn array_index(platform: Platform, index: f32, len: usize) -> Option<usize> {
    (fpu::to_int(platform, index).max(0) as usize).checked_rem(len)
}

#[cfg(test)]
mod tests {
    // Expected values keep nine significant digits.
    #![allow(clippy::excessive_precision)]

    use super::*;
    use crate::numeric::test_support::*;
    use proptest::prelude::*;

    const P: Platform = NATIVE;

    #[test]
    fn truthiness_is_a_float_compare_with_zero() {
        assert!(truthy(NAN));
        assert!(!truthy(0.0));
        assert!(!truthy(-0.0));
        assert!(truthy(-0.5));
        assert!(truthy(0.000_000_1));
        assert_eq!(not(P, NAN, ID), 0.0);
        assert_eq!(not(P, -0.0, ID), 1.0);
        assert_eq!(not(P, 3.0, ID), 0.0);
        assert_eq!(not(P, 0.0, PostOp::new(2.0, 1.0)), 3.0);
        assert_eq!(not(P, 5.0, PostOp::new(2.0, 1.0)), 1.0);
    }

    #[test]
    fn truthy_at_the_extremes() {
        for v in [
            1.0_f32,
            -1.0,
            1.0e-45,
            -1.0e-45,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            INF,
            -INF,
            NAN,
        ] {
            assert!(truthy(v), "{v:e}");
        }
        assert!(!truthy(0.0));
        assert!(!truthy(-0.0));
    }

    proptest! {
        #[test]
        fn truthy_is_not_equal_to_zero(x in proptest::num::f32::ANY) {
            prop_assert_eq!(truthy(x), x != 0.0);
        }

        #[test]
        fn not_selects_the_opposite_constant(x in proptest::num::f32::ANY, scale in -8.0_f32..8.0, offset in -8.0_f32..8.0) {
            let post = PostOp::new(scale, offset);
            let expected = if truthy(x) { post.falsy_value() } else { post.truthy_value(P) };
            prop_assert_eq!(not(P, x, post).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn not_returns_the_post_op_constants() {
        let post = PostOp::new(-2.0, 0.5);
        assert_eq!(not(P, 0.0, post), -1.5);
        assert_eq!(not(P, -0.0, post), -1.5);
        assert_eq!(not(P, 7.0, post), 0.5);
        assert_eq!(not(P, NAN, post), 0.5);
        assert_eq!(not(P, INF, post), 0.5);
        assert_eq!(not(P, -INF, ID), 0.0);
        assert_eq!(not(P, 1.0e-45, ID), 0.0);
    }

    #[test]
    fn every_comparison_with_a_nan_is_false() {
        for (a, b) in [(NAN, 4.0), (4.0, NAN), (NAN, NAN)] {
            assert!(!lt(a, b));
            assert!(!le(a, b));
            assert!(!gt(a, b));
            assert!(!ge(a, b));
            assert!(!eq(a, b));
            assert!(ne(a, b));
        }
    }

    #[test]
    fn ordinary_comparisons_are_the_float_comparisons() {
        let samples = [
            -2.0_f32,
            -0.0,
            0.0,
            1.0,
            1.000_000_1,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for a in samples {
            for b in samples {
                assert_eq!(lt(a, b), a < b);
                assert_eq!(le(a, b), a <= b);
                assert_eq!(gt(a, b), a > b);
                assert_eq!(ge(a, b), a >= b);
            }
        }
    }

    #[test]
    fn comparisons_at_equal_and_adjacent_values() {
        let next = f32::from_bits(1.0_f32.to_bits() + 1);
        assert!(!lt(1.0, 1.0));
        assert!(le(1.0, 1.0));
        assert!(!gt(1.0, 1.0));
        assert!(ge(1.0, 1.0));
        assert!(lt(1.0, next));
        assert!(!lt(next, 1.0));
        assert!(le(1.0, next));
        assert!(!le(next, 1.0));
        assert!(gt(next, 1.0));
        assert!(ge(next, 1.0));
        assert!(!ge(1.0, next));
        assert!(eq(1.0, 1.0));
        assert!(!eq(1.0, next));
        assert!(ne(1.0, next));
        assert!(!ne(1.0, 1.0));
    }

    #[test]
    fn comparisons_treat_signed_zeros_and_infinities_as_ordered_values() {
        assert!(!lt(-0.0, 0.0));
        assert!(le(-0.0, 0.0));
        assert!(!gt(0.0, -0.0));
        assert!(ge(0.0, -0.0));
        assert!(lt(-INF, INF));
        assert!(gt(INF, -INF));
        assert!(le(INF, INF));
        assert!(ge(-INF, -INF));
        assert!(lt(f32::MAX, INF));
        assert!(eq(-0.0, 0.0));
        assert!(!ne(-0.0, 0.0));
        assert!(eq(INF, INF));
        assert!(ne(INF, -INF));
    }

    proptest! {
        #[test]
        fn gt_and_ge_are_lt_and_le_swapped(a in proptest::num::f32::ANY, b in proptest::num::f32::ANY) {
            prop_assert_eq!(gt(a, b), lt(b, a));
            prop_assert_eq!(ge(a, b), le(b, a));
        }
    }

    #[test]
    fn negate_and_add_apply_the_post_op() {
        for p in PLATFORMS {
            assert_eq!(negate(p, 2.5, ID), -2.5);
            assert_eq!(negate(p, 0.0, ID).to_bits(), (-0.0_f32).to_bits());
            assert_eq!(negate(p, 2.5, PostOp::new(2.0, 1.0)), -4.0);
            assert_eq!(add(p, 0.1, 0.2, ID), 0.300_000_01);
            assert_eq!(add(p, 1.5, 2.5, PostOp::new(2.0, 1.0)), 9.0);
            let x = third();
            assert_eq!(negate(p, x, PostOp::new(3.0, 1.0)), 1.0 - x * 3.0);
        }
    }

    #[test]
    fn negate_plain_form_is_a_sign_flip() {
        assert_bits(negate(P, 0.0, ID), -0.0);
        assert_bits(negate(P, -0.0, ID), 0.0);
        assert_bits(negate(P, 1.5, ID), -1.5);
        assert_eq!(negate(P, INF, ID), -INF);
        assert_eq!(negate(P, -INF, ID), INF);
        assert!(negate(P, NAN, ID).is_nan());
        assert_bits(negate(P, 0.0, PostOp::new(1.0, -0.0)), -0.0);
    }

    #[test]
    fn negate_with_a_post_op_is_offset_minus_x_times_scale() {
        assert_eq!(negate(P, 3.0, PostOp::new(1.0, 10.0)), 7.0);
        assert_eq!(negate(P, 3.0, PostOp::new(-2.0, 0.5)), 6.5);
        assert_eq!(negate(P, 0.0, PostOp::new(1.0, 0.5)), 0.5);
        assert!(negate(P, NAN, PostOp::new(2.0, 1.0)).is_nan());
        assert_eq!(negate(P, INF, PostOp::new(2.0, 1.0)), -INF);
        assert_eq!(negate(P, -INF, PostOp::new(2.0, 1.0)), INF);
        assert!(negate(P, INF, PostOp::new(0.0, 1.0)).is_nan());
    }

    #[test]
    fn add_plain_form_is_a_single_f32_addition() {
        assert_bits(add(P, -0.0, -0.0, ID), -0.0);
        assert_bits(add(P, 0.0, -0.0, ID), 0.0);
        assert_bits(add(P, 0.1, 0.2, ID), 0.1_f32 + 0.2);
        assert_eq!(add(P, f32::MAX, f32::MAX, ID), INF);
        assert!(add(P, INF, -INF, ID).is_nan());
        assert!(add(P, NAN, 1.0, ID).is_nan());
        assert_eq!(add(P, INF, 1.0, ID), INF);
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn add_and_mul_keep_the_left_of_two_quiet_nans() {
        let (neg, pos) = (f32::from_bits(0xffc0_0000), f32::from_bits(0x7fc0_0000));
        for p in EMULATED {
            for post in [ID, AFFINE] {
                assert_bits(add(p, pos, neg, post), neg);
                assert_bits(add(p, neg, pos, post), pos);
                assert_bits(mul(p, pos, neg, post), neg);
                assert_bits(mul(p, neg, pos, post), pos);
            }
        }
        assert_bits(add(LINUX, INF, -INF, ID), neg);
        assert_bits(mul(LINUX, 0.0, INF, ID), neg);
        assert_bits(add(ANDROID, INF, -INF, ID), pos);
        assert_bits(mul(ANDROID, 0.0, INF, ID), pos);
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn a_signalling_nan_wins_on_android() {
        let (quiet, signalling) = (f32::from_bits(0xffc0_0001), f32::from_bits(0x7f80_0002));
        assert_eq!(add(LINUX, signalling, quiet, ID).to_bits(), 0xffc0_0001);
        assert_eq!(add(ANDROID, signalling, quiet, ID).to_bits(), 0x7fc0_0002);
        assert_eq!(mul(ANDROID, signalling, quiet, ID).to_bits(), 0x7fc0_0002);
        assert_eq!(div(ANDROID, quiet, signalling, ID).to_bits(), 0x7fc0_0002);
        assert_eq!(div(LINUX, quiet, signalling, ID).to_bits(), 0xffc0_0001);
    }

    #[test]
    fn add_applies_the_post_op_to_the_sum() {
        for p in PLATFORMS {
            assert_eq!(add(p, 1.0, 2.0, PostOp::new(3.0, 1.0)), 10.0);
            assert_eq!(add(p, 1.0, 2.0, PostOp::new(1.0, 1.0)), 4.0);
            assert_eq!(add(p, 1.0, -1.0, PostOp::new(5.0, 2.0)), 2.0);
            assert_bits(add(p, -0.0, -0.0, PostOp::new(2.0, 0.0)), 0.0);
            assert_bits(add(p, -0.0, -0.0, PostOp::new(2.0, -0.0)), -0.0);
            assert_bits(add(p, 0.0, third(), PostOp::new(3.0, -1.0)), 0.0);
        }
    }

    #[test]
    fn mul_plain_form_is_a_single_f32_multiplication() {
        assert_bits(mul(P, 3.0, 0.1, ID), 3.0_f32 * 0.1);
        assert_bits(mul(P, -0.0, 5.0, ID), -0.0);
        assert_bits(mul(P, 0.0, -5.0, ID), -0.0);
        assert!(mul(P, INF, 0.0, ID).is_nan());
        assert_eq!(mul(P, INF, -2.0, ID), -INF);
        assert!(mul(P, NAN, 1.0, ID).is_nan());
        assert_eq!(mul(P, f32::MAX, 2.0, ID), INF);
        assert_bits(mul(P, -0.0, 5.0, PostOp::new(1.0, -0.0)), -0.0);
    }

    #[test]
    fn multiplication_scales_the_product() {
        // v.k * v.j * 7
        let (k, j) = (1.4_f32, 0.75_f32);
        let product_then_scale = (k * j) * 7.0;
        assert_ne!(product_then_scale, k * (j * 7.0));
        for p in PLATFORMS {
            assert_eq!(mul(p, k, j, PostOp::new(7.0, 0.0)), product_then_scale);
            assert_eq!(
                mul(p, 3.0, 0.1, PostOp::new(7.0, 0.0)),
                (3.0_f32 * 0.1) * 7.0
            );
            assert_eq!(
                mul(p, k, j, PostOp::new(7.0, 0.25)),
                product_then_scale + 0.25
            );
            assert_eq!(mul(p, k, j, ID), k * j);
            assert_eq!(mul(p, 1.0e30, 1.0e30, PostOp::new(1.0e-30, 0.0)), INF);
        }
    }

    #[test]
    fn mul_with_an_exact_post_op() {
        assert_eq!(mul(P, 3.0, 4.0, PostOp::new(2.0, 1.0)), 25.0);
        assert_eq!(mul(P, -3.0, 4.0, PostOp::new(0.5, 0.25)), -5.75);
        assert_eq!(mul(P, 3.0, 0.0, PostOp::new(2.0, 1.5)), 1.5);
    }

    #[test]
    fn mul_normalises_negative_zero_only_through_a_real_post_op() {
        assert_bits(mul(P, -0.0, 1.0, PostOp::new(2.0, 0.0)), 0.0);
        assert_bits(mul(P, -0.0, 1.0, PostOp::new(2.0, -0.0)), -0.0);
    }

    #[test]
    fn div_guard_and_its_version_switch() {
        assert_eq!(f32::EPSILON, 1.192_092_9e-7);
        assert_eq!(div_guard(true, 0.000_000_1), None);
        assert_eq!(div_guard(false, 0.000_000_1), None);
        assert_eq!(div_guard(true, -0.000_000_1), None);
        assert_eq!(div_guard(true, 0.0), None);
        assert_eq!(div_guard(true, f32::EPSILON), Some(f32::EPSILON));
        assert_eq!(
            div_guard(true, f32::from_bits(f32::EPSILON.to_bits() - 1)),
            None
        );
        let signed = div_guard(true, -1.0).unwrap();
        assert_eq!(div(P, 5.0, signed, ID), -5.0);
        let unsigned = div_guard(false, -1.0).unwrap();
        assert_eq!(div(P, 5.0, unsigned, ID), 5.0);
        assert_eq!(div(P, -10.0, div_guard(false, -2.0).unwrap(), ID), -5.0);
        assert_eq!(div(P, -10.0, div_guard(true, -2.0).unwrap(), ID), 5.0);
        for signed in [true, false] {
            let divisor = div_guard(signed, NAN).unwrap();
            assert!(divisor.is_nan());
            for p in PLATFORMS {
                assert!(div(p, 1.0, divisor, ID).is_nan());
                assert!(div(p, 1.0, divisor, PostOp::new(2.0, 1.0)).is_nan());
            }
        }
    }

    #[test]
    fn div_guard_negative_threshold_and_signed_zero() {
        for signed in [true, false] {
            assert_eq!(div_guard(signed, -0.0), None);
            assert_eq!(div_guard(signed, -f32::EPSILON * 0.5), None);
            assert_eq!(div_guard(signed, 1.0e-45), None);
            assert_eq!(div_guard(signed, -1.0e-45), None);
            assert_eq!(div_guard(signed, f32::MIN_POSITIVE), None);
        }
        assert_eq!(div_guard(true, -f32::EPSILON), Some(-f32::EPSILON));
        assert_eq!(div_guard(false, -f32::EPSILON), Some(f32::EPSILON));
        assert_eq!(
            div_guard(true, f32::from_bits(f32::EPSILON.to_bits() + 1)),
            Some(f32::from_bits(f32::EPSILON.to_bits() + 1))
        );
    }

    #[test]
    fn div_guard_passes_infinities_and_ordinary_divisors() {
        assert_eq!(div_guard(true, INF), Some(INF));
        assert_eq!(div_guard(true, -INF), Some(-INF));
        assert_eq!(div_guard(false, INF), Some(INF));
        assert_eq!(div_guard(false, -INF), Some(INF));
        assert_eq!(div_guard(true, 4.0), Some(4.0));
        assert_eq!(div_guard(false, 4.0), Some(4.0));
        assert_eq!(div_guard(true, -4.0), Some(-4.0));
        assert_eq!(div_guard(false, -4.0), Some(4.0));
    }

    proptest! {
        #[test]
        fn div_guard_fires_exactly_below_epsilon_for_non_nan_divisors(d in proptest::num::f32::NORMAL | proptest::num::f32::SUBNORMAL | proptest::num::f32::ZERO | proptest::num::f32::INFINITE, signed in any::<bool>()) {
            let guarded = div_guard(signed, d);
            prop_assert_eq!(guarded.is_none(), d.abs() < f32::EPSILON);
            if let Some(divisor) = guarded {
                prop_assert_eq!(divisor.to_bits(), if signed { d } else { d.abs() }.to_bits());
            }
        }
    }

    #[test]
    fn division_scales_after_dividing() {
        let (a, b) = (1000.0_f32, 13.0_f32);
        let divide_first = (a / b) * 3.0;
        assert_ne!(divide_first, (a * 3.0) / b);
        for p in PLATFORMS {
            assert_eq!(div(p, a, b, PostOp::new(3.0, 0.0)), divide_first);
            assert_eq!(div(p, a, b, PostOp::new(3.0, 0.5)), divide_first + 0.5);
            assert_eq!(div(p, a, b, ID), a / b);
            assert_bits(
                div(p, 1.0e30, 1.0e10, PostOp::new(1.0e10, 0.0)),
                (1.0e30_f32 / 1.0e10) * 1.0e10,
            );
        }
    }

    #[test]
    fn div_plain_form_is_a_single_division() {
        assert_bits(div(P, -0.0, 5.0, ID), -0.0);
        assert_bits(div(P, 0.0, -5.0, ID), -0.0);
        assert_eq!(div(P, 1.0, 0.0, ID), INF);
        assert_eq!(div(P, -1.0, 0.0, ID), -INF);
        assert!(div(P, 0.0, 0.0, ID).is_nan());
        assert_bits(div(P, 1.0, INF, ID), 0.0);
        assert!(div(P, INF, INF, ID).is_nan());
        assert_eq!(div(P, 6.0, 4.0, ID), 1.5);
    }

    #[test]
    fn div_with_an_exact_post_op() {
        assert_eq!(div(P, 8.0, 2.0, PostOp::new(3.0, 1.0)), 13.0);
        assert_eq!(div(P, -8.0, 4.0, PostOp::new(0.5, 0.25)), -0.75);
        assert_bits(div(P, -0.0, 4.0, PostOp::new(2.0, 0.0)), 0.0);
    }

    #[test]
    fn literal_divisors_fold_to_a_reciprocal() {
        assert_eq!(fold_const_divisor(4.0), 0.25);
        assert_eq!(fold_const_divisor(-1.0), -1.0);
        assert_eq!(fold_const_divisor(0.0), 0.0);
        assert_eq!(fold_const_divisor(1.0e-8), 0.0);
        assert_eq!(fold_const_divisor(NAN), 0.0);
        // The numerator is still evaluated, so a NaN one stays NaN.
        assert_eq!(1.0 * fold_const_divisor(0.0), 0.0);
        assert!((NAN * fold_const_divisor(0.0)).is_nan());
        assert_eq!(fold_const_div(P, 1.0, 0.0), 0.0);
        assert_eq!(fold_const_div(P, 0.0, 0.0), 0.0);
        assert_eq!(fold_const_div(P, 1.0, NAN), 0.0);
        assert_eq!(fold_const_div(P, -10.0, -2.0), 5.0);
        // `7 * 3 / 9` groups the division first.
        let grouped = 7.0 * fold_const_div(P, 3.0, 9.0);
        assert_eq!(grouped, 7.0 * 0.333_333_34_f32);
        assert_eq!(grouped, 2.333_333_5);
        assert_eq!(
            PostOp::new(fold_const_div(P, 3.0, 9.0), 0.0).apply(P, 1.0),
            0.333_333_34
        );
    }

    #[test]
    fn fold_const_divisor_threshold_and_infinities() {
        assert_bits(fold_const_divisor(f32::EPSILON), 1.0 / f32::EPSILON);
        assert_bits(fold_const_divisor(-f32::EPSILON), -(1.0 / f32::EPSILON));
        assert_bits(
            fold_const_divisor(f32::from_bits(f32::EPSILON.to_bits() - 1)),
            0.0,
        );
        assert_bits(
            fold_const_divisor(-f32::from_bits(f32::EPSILON.to_bits() - 1)),
            0.0,
        );
        assert_bits(fold_const_divisor(-0.0), 0.0);
        assert_bits(fold_const_divisor(1.0e-45), 0.0);
        assert_bits(fold_const_divisor(INF), 0.0);
        assert_bits(fold_const_divisor(-INF), -0.0);
        assert_bits(fold_const_divisor(2.0), 0.5);
        assert_bits(fold_const_divisor(3.0), 1.0 / 3.0);
    }

    #[test]
    fn fold_const_div_threshold_and_non_finite_operands() {
        assert_bits(fold_const_div(P, 1.0, f32::EPSILON), 1.0 / f32::EPSILON);
        assert_bits(fold_const_div(P, 1.0, -f32::EPSILON), -(1.0 / f32::EPSILON));
        assert_bits(
            fold_const_div(P, 1.0, f32::from_bits(f32::EPSILON.to_bits() - 1)),
            0.0,
        );
        assert_bits(fold_const_div(P, -1.0, 0.0), 0.0);
        assert_bits(fold_const_div(P, INF, 0.0), 0.0);
        assert_bits(fold_const_div(P, NAN, 0.0), 0.0);
        assert!(fold_const_div(P, NAN, 2.0).is_nan());
        assert_eq!(fold_const_div(P, INF, 2.0), INF);
        assert_bits(fold_const_div(P, 1.0, INF), 0.0);
        assert_bits(fold_const_div(P, -1.0, INF), -0.0);
        assert!(fold_const_div(P, INF, INF).is_nan());
        assert_bits(fold_const_div(P, 0.0, -4.0), -0.0);
        assert_bits(fold_const_div(P, 1.0, NAN), 0.0);
        assert_bits(fold_const_div(P, NAN, NAN), 0.0);
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn fold_const_div_of_an_invalid_division_is_the_default_nan() {
        assert_eq!(fold_const_div(LINUX, INF, INF).to_bits(), 0xffc0_0000);
        assert_eq!(fold_const_div(WINDOWS, INF, INF).to_bits(), 0xffc0_0000);
        assert_eq!(fold_const_div(ANDROID, INF, INF).to_bits(), 0x7fc0_0000);
    }

    proptest! {
        #[test]
        fn fold_const_div_matches_the_division_wherever_the_guard_does_not_fire(a in proptest::num::f32::NORMAL, b in proptest::num::f32::NORMAL | proptest::num::f32::ZERO | proptest::num::f32::SUBNORMAL) {
            let folded = fold_const_div(P, a, b);
            if b.abs() >= f32::EPSILON {
                prop_assert_eq!(folded.to_bits(), (a / b).to_bits());
            } else {
                prop_assert_eq!(folded.to_bits(), 0.0_f32.to_bits());
            }
        }
    }

    #[test]
    fn array_index_wraps_and_never_goes_out_of_range() {
        for p in PLATFORMS {
            assert_eq!(array_index(p, 0.0, 3), Some(0));
            assert_eq!(array_index(p, 2.9, 3), Some(2));
            assert_eq!(array_index(p, 3.0, 3), Some(0));
            assert_eq!(array_index(p, 7.5, 3), Some(1));
            assert_eq!(array_index(p, -1.0, 3), Some(0));
            assert_eq!(array_index(p, NAN, 3), Some(0));
            assert_eq!(array_index(p, f32::NEG_INFINITY, 3), Some(0));
            for index in [
                0.0,
                2.0,
                -1.0,
                NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1.0e30,
            ] {
                assert_eq!(array_index(p, index, 0), None);
            }
            assert_eq!(array_index(p, 1.0e30, 1), Some(0));
        }
    }

    #[test]
    fn array_index_truncates_negative_fractions_to_the_first_element() {
        for index in [-0.0, -0.5, -0.999, -1.0, -2.5, -1.0e9, -1.0e30] {
            for p in PLATFORMS {
                assert_eq!(array_index(p, index, 5), Some(0), "{index}");
            }
        }
    }

    #[test]
    fn array_index_of_a_single_element_array_is_always_zero() {
        for index in [0.0, 0.5, 1.0, 7.9, -3.0, NAN, INF, -INF, 1.0e10] {
            for p in PLATFORMS {
                assert_eq!(array_index(p, index, 1), Some(0), "{index}");
            }
        }
    }

    #[test]
    fn array_index_wraps_with_the_remainder_by_the_length() {
        for (index, len, expected) in [
            (4.0_f32, 4, 0),
            (5.0, 4, 1),
            (11.99, 4, 3),
            (12.0, 4, 0),
            (100.0, 7, 2),
            (6.0, 7, 6),
            (7.0, 7, 0),
        ] {
            assert_eq!(
                array_index(P, index, len),
                Some(expected),
                "{index} % {len}"
            );
        }
    }

    #[test]
    fn array_index_beyond_the_i32_range_saturates_natively() {
        for (index, len) in [(3.0e9, 7), (f32::MAX, 10), (INF, 3)] {
            assert_eq!(
                array_index(NATIVE, index, len),
                Some((i32::MAX as usize) % len)
            );
        }
        for p in PLATFORMS {
            assert_eq!(
                array_index(p, 2_147_483_520.0, 1000),
                Some(2_147_483_520 % 1000)
            );
        }
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn array_index_beyond_the_i32_range_is_zero_on_the_servers_and_saturates_on_android() {
        for p in [LINUX, WINDOWS] {
            assert_eq!(array_index(p, 3.0e9, 7), Some(0));
            assert_eq!(array_index(p, f32::MAX, 10), Some(0));
            assert_eq!(array_index(p, INF, 3), Some(0));
        }
        assert_eq!(
            array_index(ANDROID, 3.0e9, 7),
            Some((i32::MAX as usize) % 7)
        );
        assert_eq!(
            array_index(ANDROID, f32::MAX, 10),
            Some((i32::MAX as usize) % 10)
        );
        assert_eq!(array_index(ANDROID, INF, 3), Some((i32::MAX as usize) % 3));
    }

    #[test]
    fn array_index_with_a_huge_length_returns_the_truncated_index() {
        assert_eq!(array_index(P, 5.7, usize::MAX), Some(5));
        assert_eq!(array_index(P, -5.7, usize::MAX), Some(0));
    }

    proptest! {
        #[test]
        fn array_index_is_always_in_range_for_a_non_empty_array(index in proptest::num::f32::ANY, len in 1_usize..1000, p in proptest::sample::select(PLATFORMS.to_vec())) {
            let i = array_index(p, index, len).unwrap();
            prop_assert!(i < len);
        }
    }
}
