//! `math.random`, `math.random_integer`, `math.die_roll` and `math.die_roll_integer`.
//!
//! Each operator is a pure function of its bounds and of the raw samples it draws, each clamped by
//! [`clamp_sample`]; the number and order of draws are part of the contract.

use super::formula::{Sort, integer_draw, integer_reach, interpolate, sorted_bounds};
use crate::numeric::{Platform, PostOp, arith};

mod dice;

pub use dice::{DieRoll, die_roll, die_roll_count, die_roll_integer};

/// Clamps a raw sample to `[0, 1]`; NaN and `−0` become `+0`, on every platform.
#[inline]
pub fn clamp_sample(_platform: Platform, raw: f32) -> f32 {
    if raw > 1.0 {
        1.0
    } else if raw > 0.0 {
        raw
    } else {
        0.0
    }
}

/// `math.random(a, b)` with run-time bounds, over the closed interval of the sorted bounds:
/// `hi·r + (1 − r)·lo`, NaN for a NaN bound.
#[inline]
pub fn random(platform: Platform, a: f32, b: f32, sample: f32, post: PostOp) -> f32 {
    let r = clamp_sample(platform, sample);
    let (lo, hi) = sorted_bounds(Sort::RunTime, a, b);
    post.apply(platform, interpolate(platform, lo, hi, r))
}

/// The post-op `math.random(a, b)` with two literal bounds folds into: `S' = (hi − lo)·S`,
/// `O' = lo·S + O`, evaluated by [`random_folded`].
#[inline]
pub fn random_const_bounds(platform: Platform, a: f32, b: f32, post: PostOp) -> PostOp {
    let (lo, hi) = sorted_bounds(Sort::Literal, a, b);
    PostOp {
        scale: arith::mul(platform, arith::sub(platform, hi, lo), post.scale),
        offset: arith::mul_add(platform, lo, post.scale, post.offset),
    }
}

/// `math.random` with literal bounds: `r·S' + O'` for a [`random_const_bounds`] post-op.
///
/// Unlike [`PostOp::apply`] this always multiplies and adds, even for `(1, 0)`.
#[inline]
pub fn random_folded(platform: Platform, sample: f32, folded: PostOp) -> f32 {
    arith::mul_add(
        platform,
        clamp_sample(platform, sample),
        folded.scale,
        folded.offset,
    )
}

/// `math.random_integer(a, b)` with run-time bounds: `floor((1 − r)·lo + r·top)` with
/// `top = (hi·(−ε) + 1) + hi`, then clamped to `[lo, hi]`. The bounds are not rounded, so
/// `random_integer(0.1, 1000001)` at `r = 0` is 0.1.
#[inline]
pub fn random_integer(platform: Platform, a: f32, b: f32, sample: f32, post: PostOp) -> f32 {
    let r = clamp_sample(platform, sample);
    let (lo, hi) = sorted_bounds(Sort::RunTime, a, b);
    let reach = integer_reach(platform, hi);
    post.apply(platform, integer_draw(platform, lo, hi, reach, r))
}

/// `math.random_integer(a, b)` with two literal bounds: [`random_integer`] with the literal
/// sorting of a NaN bound.
#[inline]
pub fn random_integer_const_bounds(
    platform: Platform,
    a: f32,
    b: f32,
    sample: f32,
    post: PostOp,
) -> f32 {
    let r = clamp_sample(platform, sample);
    let (lo, hi) = sorted_bounds(Sort::Literal, a, b);
    let reach = integer_reach(platform, hi);
    post.apply(platform, integer_draw(platform, lo, hi, reach, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;
    use crate::rng::FixedRng;
    use crate::stdlib::math::formula::clamp_draw;

    const P: Platform = NATIVE;
    use proptest::prelude::*;

    #[test]
    fn clamp_sample_keeps_values_inside_the_unit_interval() {
        for raw in [
            0.25_f32,
            0.5,
            0.999_999_9,
            1.0e-30,
            1.0e-45,
            f32::MIN_POSITIVE,
        ] {
            assert_bits(clamp_sample(P, raw), raw);
        }
        assert_bits(clamp_sample(P, 0.0), 0.0);
        assert_bits(clamp_sample(P, 1.0), 1.0);
    }

    #[test]
    fn clamp_sample_clamps_everything_else() {
        for raw in [1.000_000_1, 2.0, 1.0e30, f32::MAX, INF] {
            assert_bits(clamp_sample(P, raw), 1.0);
        }
        for raw in [-1.0e-45, -0.5, -1.0, -1.0e30, f32::MIN, -INF] {
            assert_bits(clamp_sample(P, raw), 0.0);
        }
        assert_bits(clamp_sample(P, -0.0), 0.0);
        assert_bits(clamp_sample(P, NAN), 0.0);
        assert_bits(clamp_sample(P, -NAN), 0.0);
    }

    proptest! {
        #[test]
        fn clamp_sample_is_always_inside_the_closed_unit_interval(raw in proptest::num::f32::ANY) {
            let clamped = clamp_sample(P, raw);
            prop_assert!((0.0..=1.0).contains(&clamped));
            prop_assert!(clamped.is_sign_positive());
            if (0.0..=1.0).contains(&raw) && raw != 0.0 {
                prop_assert_eq!(clamped.to_bits(), raw.to_bits());
            }
        }
    }

    #[test]
    fn sorted_bounds_orders_ordinary_bounds() {
        for sort in [Sort::RunTime, Sort::Literal] {
            assert_eq!(sorted_bounds(sort, 1.0, 3.0), (1.0, 3.0));
            assert_eq!(sorted_bounds(sort, 3.0, 1.0), (1.0, 3.0));
            assert_eq!(sorted_bounds(sort, 2.0, 2.0), (2.0, 2.0));
            assert_eq!(sorted_bounds(sort, -INF, INF), (-INF, INF));
            assert_eq!(sorted_bounds(sort, INF, -INF), (-INF, INF));
            assert_eq!(sorted_bounds(sort, -5.0, -2.0), (-5.0, -2.0));
        }
    }

    #[test]
    fn sorted_bounds_with_a_nan_bound_depends_on_the_sort() {
        let nan_pair = |(lo, hi): (f32, f32)| (lo.is_nan(), hi.is_nan());
        assert_eq!(
            nan_pair(sorted_bounds(Sort::RunTime, NAN, 4.0)),
            (true, false)
        );
        assert_eq!(sorted_bounds(Sort::RunTime, NAN, 4.0).1, 4.0);
        assert_eq!(
            nan_pair(sorted_bounds(Sort::RunTime, 4.0, NAN)),
            (false, true)
        );
        assert_eq!(sorted_bounds(Sort::RunTime, 4.0, NAN).0, 4.0);
        assert_eq!(
            nan_pair(sorted_bounds(Sort::Literal, NAN, 4.0)),
            (true, true)
        );
        assert_eq!(sorted_bounds(Sort::Literal, 4.0, NAN), (4.0, 4.0));
        for sort in [Sort::RunTime, Sort::Literal] {
            assert_eq!(nan_pair(sorted_bounds(sort, NAN, NAN)), (true, true));
        }
    }

    #[test]
    fn sorted_bounds_of_equal_zeros_of_opposite_sign() {
        let (lo, hi) = sorted_bounds(Sort::RunTime, 0.0, -0.0);
        assert_bits(lo, 0.0);
        assert_bits(hi, -0.0);
        let (lo, hi) = sorted_bounds(Sort::Literal, 0.0, -0.0);
        assert_bits(lo, 0.0);
        assert_bits(hi, 0.0);
    }

    #[test]
    fn clamp_draw_tests_the_upper_bound_first() {
        assert_bits(clamp_draw(5.0, 1.0, 3.0), 3.0);
        assert_bits(clamp_draw(0.0, 1.0, 3.0), 1.0);
        assert_bits(clamp_draw(2.0, 1.0, 3.0), 2.0);
        assert_bits(clamp_draw(3.0, 1.0, 3.0), 3.0);
        assert_bits(clamp_draw(2.0, 3.0, 1.0), 1.0);
        assert_bits(clamp_draw(0.5, 3.0, 1.0), 3.0);
        assert_bits(clamp_draw(NAN, 1.0, 3.0), 1.0);
    }

    #[test]
    fn interpolate_hits_the_bounds_at_samples_zero_and_one() {
        assert_bits(interpolate(P, 2.0, 7.0, 0.0), 2.0);
        assert_bits(interpolate(P, 2.0, 7.0, 1.0), 7.0);
        assert_bits(interpolate(P, 2.0, 7.0, 0.5), 4.5);
        assert_bits(interpolate(P, -3.0, 3.0, 0.5), 0.0);
        assert_bits(interpolate(P, 5.0, 5.0, 0.37), 5.0);
    }

    #[test]
    fn interpolate_rounds_each_operation() {
        let (lo, hi, r) = (
            third(),
            f32::from_bits(0x3f91_bc0d),
            f32::from_bits(0x3ecf_a978),
        );
        assert_bits(interpolate(P, lo, hi, r), hi * r + (1.0 - r) * lo);
    }

    #[test]
    fn interpolate_on_inexact_operands() {
        let (lo, hi, r) = (1.0_f32 / 7.0, 1.0_f32 / 7.0 + 1.0 / 3.0, 3.0_f32 / 19.0);
        assert_bits(interpolate(P, lo, hi, r), hi * r + (1.0 - r) * lo);
        assert_eq!(interpolate(P, lo, hi, r).to_bits(), 0x3e48_2e32);
    }

    #[test]
    fn interpolate_with_infinite_bounds_never_forms_the_span() {
        assert_eq!(
            interpolate(P, f32::NEG_INFINITY, -180.0, 0.0),
            f32::NEG_INFINITY
        );
        assert!(interpolate(P, -INF, INF, 0.5).is_nan());
        assert!(interpolate(P, 1.0, INF, 0.0).is_nan());
    }

    #[test]
    fn integer_reach_is_just_below_hi_plus_one() {
        let hi = 7.0_f32;
        assert_bits(integer_reach(P, hi), (-f32::EPSILON * hi + 1.0) + hi);
        let reach = integer_reach(P, hi);
        assert!(reach < 8.0 && reach > 7.999_998, "{reach}");
        close_reach(integer_reach(P, 2.5), 3.5);
    }

    #[track_caller]
    fn close_reach(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{actual} is not near {expected}"
        );
    }

    #[test]
    fn integer_draw_floors_and_clamps_to_the_bounds() {
        let (lo, hi) = (2.0_f32, 7.0_f32);
        let reach = integer_reach(P, hi);
        assert_bits(integer_draw(P, lo, hi, reach, 0.0), 2.0);
        assert_bits(integer_draw(P, lo, hi, reach, 1.0), 7.0);
        assert_bits(integer_draw(P, lo, hi, reach, 0.5), 4.0);
        assert_bits(integer_draw(P, lo, hi, reach, 0.999), 7.0);
        assert_bits(integer_draw(P, lo, hi, reach, 0.1), 2.0);
        assert_bits(integer_draw(P, lo, hi, reach, 0.17), 3.0);
        assert_bits(integer_draw(P, lo, hi, 100.0, 1.0), 7.0);
        assert_bits(integer_draw(P, lo, hi, -100.0, 1.0), 2.0);
    }

    #[test]
    fn random_at_samples_zero_half_and_one() {
        assert_bits(random(P, 2.0, 7.0, 0.0, ID), 2.0);
        assert_bits(random(P, 2.0, 7.0, 0.5, ID), 4.5);
        assert_bits(random(P, 2.0, 7.0, 1.0, ID), 7.0);
        assert_bits(random(P, -4.0, 4.0, 0.5, ID), 0.0);
        assert_bits(random(P, -4.0, -2.0, 0.5, ID), -3.0);
        for sample in [0.0, 0.3, 1.0] {
            assert_bits(random(P, 3.0, 3.0, sample, ID), 3.0);
        }
    }

    #[test]
    fn random_with_inverted_bounds_equals_the_sorted_call() {
        for sample in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_bits(
                random(P, 7.0, 2.0, sample, ID),
                random(P, 2.0, 7.0, sample, ID),
            );
            assert_bits(
                random(P, 7.0, 2.0, sample, AFFINE),
                random(P, 2.0, 7.0, sample, AFFINE),
            );
        }
    }

    #[test]
    fn random_clamps_the_sample() {
        assert_bits(random(P, 2.0, 7.0, 3.0, ID), 7.0);
        assert_bits(random(P, 2.0, 7.0, INF, ID), 7.0);
        assert_bits(random(P, 2.0, 7.0, -3.0, ID), 2.0);
        assert_bits(random(P, 2.0, 7.0, -INF, ID), 2.0);
        assert_bits(random(P, 2.0, 7.0, NAN, ID), 2.0);
        assert_bits(random(P, 2.0, 7.0, -0.0, ID), 2.0);
        for (a, b, sample, run_time, literal) in [
            (
                16_777_216.0,
                -1_274.555_4,
                1_695.562_9,
                0x4b7f_ffff,
                0x4b7f_ffff,
            ),
            (3.866_985e-26, -INF, 417.559_8, 0xff80_0000, 0xff80_0000),
            (
                712.738_95,
                4.589_961_6e8,
                16_777_216.0,
                0x4dda_ddcc,
                0x4dda_ddcc,
            ),
        ] {
            assert_eq!(
                random_integer(P, a, b, sample, ID).to_bits(),
                run_time,
                "{a} {b} {sample}"
            );
            assert_eq!(
                random_integer_const_bounds(P, a, b, sample, ID).to_bits(),
                literal,
                "{a} {b} {sample}"
            );
        }
    }

    /// `floor` can land past either bound; the draw clamps it back.
    #[test]
    fn an_integer_draw_is_clamped_to_the_bounds() {
        assert_bits(integer_draw(P, 0.0, 2.5, integer_reach(P, 2.5), 1.0), 2.5);
        assert_bits(integer_draw(P, 0.5, 3.0, integer_reach(P, 3.0), 0.0), 0.5);
    }

    #[test]
    fn random_applies_the_post_op() {
        assert_bits(random(P, 2.0, 7.0, 0.5, AFFINE), 10.0);
        assert_bits(random(P, 2.0, 7.0, 0.0, PostOp::new(-1.0, 0.0)), -2.0);
        assert_bits(random(P, 2.0, 7.0, 1.0, PostOp::new(0.0, 5.0)), 5.0);
        assert_bits(random(P, 0.0, 0.0, 0.5, PostOp::new(-1.0, 0.0)), 0.0);
    }

    #[test]
    fn random_bounds_with_nan() {
        assert!(random(P, NAN, 4.0, 0.5, ID).is_nan());
        for sample in [0.0, 0.5, 1.0] {
            assert!(random(P, NAN, 4.0, sample, ID).is_nan());
            assert!(random(P, 4.0, NAN, sample, ID).is_nan());
            assert!(random_integer(P, NAN, 4.0, sample, ID).is_nan());
            assert_eq!(random_integer(P, 4.0, NAN, sample, ID), 4.0);
        }
        assert!(random_folded(P, 0.5, random_const_bounds(P, NAN, 4.0, ID)).is_nan());
        assert_eq!(
            random_folded(P, 0.5, random_const_bounds(P, 4.0, NAN, ID)),
            4.0
        );
    }

    #[test]
    fn random_with_two_nan_bounds_is_nan() {
        for sample in [0.0, 0.5, 1.0] {
            assert!(random(P, NAN, NAN, sample, ID).is_nan());
            assert!(random_integer(P, NAN, NAN, sample, ID).is_nan());
        }
    }

    #[test]
    fn random_with_infinite_bounds() {
        assert_eq!(random(P, -INF, 5.0, 0.0, ID), -INF);
        assert_eq!(random(P, 5.0, INF, 1.0, ID), INF);
        assert!(random(P, -INF, 5.0, 1.0, ID).is_nan());
        assert!(random(P, -INF, INF, 0.5, ID).is_nan());
        assert_eq!(random(P, 5.0, INF, 0.5, ID), INF);
    }

    #[test]
    fn random_of_negative_zero_bounds_is_negative_zero() {
        for p in PLATFORMS {
            assert_bits(random(p, -0.0, -0.0, 0.5, ID), -0.0);
        }
    }

    #[test]
    fn random_interpolation_rounds_each_operation() {
        let (lo, hi, r) = (
            third(),
            f32::from_bits(0x3f91_bc0d),
            f32::from_bits(0x3ecf_a978),
        );
        assert_eq!(random(P, lo, hi, r, ID), hi * r + (1.0 - r) * lo);
        assert_eq!(random(P, lo, hi, r, ID).to_bits(), 0x3f28_f09f);
        assert_eq!(((hi - lo) * r + lo).to_bits(), 0x3f28_f09e);
        assert_eq!(
            random(P, -180.0, f32::NEG_INFINITY, 0.0, ID),
            f32::NEG_INFINITY
        );
        let (lo, hi, r) = (-3.0_f32, 2.0_f32, 0.5_f32);
        let top = (-f32::EPSILON * hi + 1.0) + hi;
        assert_eq!(
            random_integer(P, lo, hi, r, ID),
            ((1.0 - r) * lo + top * r).floor()
        );
        assert_eq!(
            random_integer_const_bounds(P, lo, hi, r, ID),
            ((1.0 - r) * lo + top * r).floor()
        );
        // 0.5·(−3) + 0.5·2.9999998 is just below 0.
        assert_eq!(random_integer(P, lo, hi, r, ID), -1.0);
        let mut half = FixedRng::HALF;
        assert_eq!(
            die_roll_integer(P, 1.0, lo, hi, 1, &mut half, ID),
            Some(-1.0)
        );
        let mut rng = FixedRng::from_sample(third()).expect("a word's sample");
        assert_eq!(
            die_roll(P, 1.0, 1.1, 7.3, 1, &mut rng, ID),
            Some(0.0 + (third() * 7.0 + (1.0 - third()) * 1.0))
        );
    }

    #[test]
    fn random_const_bounds_folds_the_bounds_into_the_post_op() {
        assert_eq!(random_const_bounds(P, 2.0, 7.0, ID), PostOp::new(5.0, 2.0));
        assert_eq!(random_const_bounds(P, 7.0, 2.0, ID), PostOp::new(5.0, 2.0));
        assert_eq!(
            random_const_bounds(P, 7.0, 2.0, AFFINE),
            PostOp::new(10.0, 5.0)
        );
        assert_eq!(
            random_const_bounds(P, 3.0, 3.0, AFFINE),
            PostOp::new(0.0, 7.0)
        );
        assert_eq!(
            random_const_bounds(P, -4.0, 4.0, PostOp::new(-1.0, 0.5)),
            PostOp::new(-8.0, 4.5)
        );
        assert_eq!(random_const_bounds(P, -INF, 1.0, ID).scale, INF);
    }

    #[test]
    fn random_const_bounds_rounds_the_offset_twice() {
        let (lo, post) = (1.1_f32, PostOp::new(3.3, 0.7));
        assert_bits(random_const_bounds(P, lo, 9.0, post).offset, lo * 3.3 + 0.7);
        assert_bits(
            random_const_bounds(P, lo, 9.0, post).scale,
            (9.0_f32 - lo) * 3.3,
        );
    }

    #[test]
    fn random_const_bounds_offset_is_unfused_for_an_inexact_product() {
        let (lo, post) = (0.1_f32, PostOp::new(3.1, 1.1));
        assert_ne!(
            (lo * 3.1 + 1.1).to_bits(),
            lo.mul_add(3.1, 1.1).to_bits(),
            "the operands must tell fused from unfused"
        );
        assert_eq!(
            random_const_bounds(P, lo, 0.5, post).offset.to_bits(),
            0x3fb4_7ae2
        );
        let folded = random_const_bounds(P, lo, 0.5, post);
        assert_bits(
            random_folded(P, 0.3, folded),
            0.3 * folded.scale + folded.offset,
        );
    }

    #[test]
    fn random_const_bounds_with_a_nan_bound() {
        let folded = random_const_bounds(P, NAN, 4.0, ID);
        assert!(folded.scale.is_nan() && folded.offset.is_nan());
        assert_eq!(random_const_bounds(P, 4.0, NAN, ID), PostOp::new(0.0, 4.0));
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn random_const_bounds_chooses_its_nans() {
        let (q1, q2) = (f32::from_bits(0x7fc0_0001), f32::from_bits(0xffc0_0002));
        assert_eq!(
            random_const_bounds(LINUX, q1, 4.0, PostOp::new(q2, 0.0))
                .scale
                .to_bits(),
            0x7fc0_0001
        );
        assert_eq!(
            random_const_bounds(LINUX, INF, INF, ID).scale.to_bits(),
            0xffc0_0000
        );
        assert_eq!(
            random_const_bounds(ANDROID, INF, INF, ID).scale.to_bits(),
            0x7fc0_0000
        );
    }

    #[test]
    fn random_folded_always_multiplies_and_adds() {
        assert_bits(random_folded(P, 0.0, PostOp::new(5.0, 2.0)), 2.0);
        assert_bits(random_folded(P, 0.5, PostOp::new(5.0, 2.0)), 4.5);
        assert_bits(random_folded(P, 1.0, PostOp::new(5.0, 2.0)), 7.0);
        assert_bits(random_folded(P, 9.0, PostOp::new(5.0, 2.0)), 7.0);
        assert_bits(random_folded(P, -9.0, PostOp::new(5.0, 2.0)), 2.0);
        assert_bits(random_folded(P, NAN, PostOp::new(5.0, 2.0)), 2.0);
        // (1, −0) is still multiplied and added, so the −0 offset is normalised.
        assert_bits(random_folded(P, 0.0, PostOp::new(1.0, -0.0)), 0.0);
        assert_bits(random_folded(P, 0.25, PostOp::IDENTITY), 0.25);
        assert_bits(random_folded(P, 1.0, PostOp::new(0.0, 3.0)), 3.0);
        let folded = PostOp::new(3.0, -1.0);
        assert_bits(random_folded(P, third(), folded), 0.0);
    }

    #[test]
    fn folded_form_agrees_with_the_run_time_form_for_exact_bounds() {
        for (a, b) in [(2.0_f32, 7.0), (7.0, 2.0), (-4.0, 4.0), (0.0, 1.0)] {
            for sample in [0.0, 0.25, 0.5, 1.0] {
                for post in [ID, AFFINE, PostOp::new(0.5, -3.0)] {
                    let folded = random_const_bounds(P, a, b, post);
                    assert_bits(
                        random_folded(P, sample, folded),
                        random(P, a, b, sample, post),
                    );
                }
            }
        }
    }

    #[test]
    fn random_integer_at_samples_zero_half_and_one() {
        assert_bits(random_integer(P, 2.0, 7.0, 0.0, ID), 2.0);
        assert_bits(random_integer(P, 2.0, 7.0, 0.5, ID), 4.0);
        assert_bits(random_integer(P, 2.0, 7.0, 1.0, ID), 7.0);
        assert_bits(random_integer(P, 0.0, 1.0, 0.0, ID), 0.0);
        assert_bits(random_integer(P, 0.0, 1.0, 0.5, ID), 0.0);
        assert_bits(random_integer(P, 0.0, 1.0, 1.0, ID), 1.0);
        assert_bits(random_integer(P, -3.0, 2.0, 0.0, ID), -3.0);
        assert_bits(random_integer(P, -3.0, 2.0, 1.0, ID), 2.0);
        for sample in [0.0, 0.5, 1.0] {
            assert_bits(random_integer(P, 3.0, 3.0, sample, ID), 3.0);
        }
    }

    #[test]
    fn random_integer_with_inverted_bounds_equals_the_sorted_call() {
        for sample in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_bits(
                random_integer(P, 7.0, 2.0, sample, ID),
                random_integer(P, 2.0, 7.0, sample, ID),
            );
            assert_bits(
                random_integer_const_bounds(P, 7.0, 2.0, sample, ID),
                random_integer_const_bounds(P, 2.0, 7.0, sample, ID),
            );
        }
    }

    #[test]
    fn random_integer_covers_every_integer_between_the_bounds() {
        let mut seen = [false; 6];
        for k in 0..=1000 {
            let value = random_integer(P, 2.0, 7.0, k as f32 / 1000.0, ID);
            assert_eq!(value, value.floor());
            assert!((2.0..=7.0).contains(&value));
            seen[value as usize - 2] = true;
        }
        assert_eq!(seen, [true; 6]);
    }

    #[test]
    fn random_integer_does_not_round_its_bounds() {
        assert_bits(random_integer(P, 0.5, 2.5, 0.0, ID), 0.5);
        assert_bits(random_integer(P, 0.5, 2.5, 1.0, ID), 2.5);
        assert_bits(random_integer_const_bounds(P, 0.5, 2.5, 0.0, ID), 0.5);
        assert_bits(random_integer_const_bounds(P, 0.5, 2.5, 1.0, ID), 2.5);
        assert_eq!(random_integer(P, 0.1, 1_000_001.0, 0.0, ID), 0.1);
    }

    #[test]
    fn random_integer_clamps_the_sample_and_applies_the_post_op() {
        assert_bits(random_integer(P, 2.0, 7.0, 9.0, ID), 7.0);
        assert_bits(random_integer(P, 2.0, 7.0, -9.0, ID), 2.0);
        assert_bits(random_integer(P, 2.0, 7.0, NAN, ID), 2.0);
        assert_bits(random_integer(P, 0.0, 3.0, 1.0, AFFINE), 7.0);
        assert_bits(random_integer(P, 0.0, 3.0, 0.0, AFFINE), 1.0);
        assert_bits(random_integer_const_bounds(P, 0.0, 3.0, 1.0, AFFINE), 7.0);
        assert_bits(
            random_integer(P, 0.0, 0.0, 0.5, PostOp::new(-1.0, 0.0)),
            0.0,
        );
    }

    #[test]
    fn random_integer_with_nan_bounds() {
        for sample in [0.0, 0.5, 1.0] {
            assert!(random_integer(P, NAN, 4.0, sample, ID).is_nan());
            assert_bits(random_integer(P, 4.0, NAN, sample, ID), 4.0);
        }
    }

    #[test]
    fn random_integer_const_bounds_sorts_nan_with_the_literal_rule() {
        for sample in [0.0, 0.5, 1.0] {
            assert!(random_integer_const_bounds(P, NAN, 4.0, sample, ID).is_nan());
            assert_bits(random_integer_const_bounds(P, 4.0, NAN, sample, ID), 4.0);
        }
    }

    #[test]
    fn random_integer_const_bounds_matches_the_run_time_form() {
        for lo in -8..8 {
            for hi in lo..16 {
                for k in 0..=20 {
                    let (lo, hi, r) = (lo as f32, hi as f32, k as f32 / 20.0);
                    assert_bits(
                        random_integer_const_bounds(P, lo, hi, r, ID),
                        random_integer(P, lo, hi, r, ID),
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn random_lies_between_the_sorted_bounds_for_finite_bounds(a in -1000.0_f32..1000.0, b in -1000.0_f32..1000.0, sample in 0.0_f32..=1.0) {
            let (lo, hi) = (a.min(b), a.max(b));
            let slack = 1e-3;
            let value = random(P, a, b, sample, ID);
            prop_assert!(value >= lo - slack && value <= hi + slack, "{} not in [{}, {}]", value, lo, hi);
            let integer = random_integer(P, a, b, sample, ID);
            prop_assert!(integer >= lo && integer <= hi);
        }
    }

    #[test]
    fn random_samples_are_clamped() {
        assert_eq!(clamp_sample(P, 0.25), 0.25);
        assert_eq!(clamp_sample(P, 2.0), 1.0);
        assert_eq!(clamp_sample(P, -1.0), 0.0);
        assert_eq!(clamp_sample(P, -0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(clamp_sample(P, NAN), 0.0);
        assert_eq!(clamp_sample(P, 1.0), 1.0);
        assert_eq!(random(P, 2.0, 7.0, 9.0, ID), 7.0);
        assert_eq!(random(P, 2.0, 7.0, -9.0, ID), 2.0);
    }

    #[test]
    fn random_and_random_integer_on_key_samples_and_bounds() {
        for (sample, real, integer) in [(0.0, 2.0, 2.0), (0.5, 4.5, 4.0), (1.0, 7.0, 7.0)] {
            assert_eq!(random(P, 2.0, 7.0, sample, ID), real);
            assert_eq!(random(P, 7.0, 2.0, sample, ID), real, "inverted bounds");
            assert_eq!(random_integer(P, 2.0, 7.0, sample, ID), integer);
            assert_eq!(
                random_integer(P, 7.0, 2.0, sample, ID),
                integer,
                "inverted bounds"
            );
            assert_eq!(
                random_integer_const_bounds(P, 2.0, 7.0, sample, ID),
                integer
            );
            assert_eq!(
                random(P, 2.0, 7.0, sample, PostOp::new(2.0, 1.0)),
                real * 2.0 + 1.0
            );
            let folded = random_const_bounds(P, 7.0, 2.0, PostOp::new(2.0, 1.0));
            assert_eq!(folded, PostOp::new(10.0, 5.0));
            assert_eq!(random_folded(P, sample, folded), real * 2.0 + 1.0);
            assert_eq!(
                random_folded(P, sample, random_const_bounds(P, 2.0, 7.0, ID)),
                real
            );
        }
        assert_eq!(random_integer(P, 0.1, 1_000_001.0, 0.0, ID), 0.1);
        assert_eq!(
            random_integer_const_bounds(P, 0.1, 1_000_001.0, 0.0, ID),
            0.1
        );
        assert_eq!(random_integer(P, 0.0, 3.0, 1.0, PostOp::new(2.0, 1.0)), 7.0);
    }
}
