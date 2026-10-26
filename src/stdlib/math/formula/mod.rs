//! The standard functions' formulas: every operation rounded on its own, in formula order, through
//! the platform's arithmetic, so the platform chooses every NaN; the transcendentals come from
//! [`cmath`](super::cmath).

mod ease;

pub(crate) use ease::*;

use super::{INVERSE_TRIG_TOLERANCE, RAD_TO_DEG, cmath};
use crate::numeric::fpu::{self, add, div, max, mul, sub, to_int};
use crate::numeric::{Platform, PostOp};

/// 65536/2π.
const SIN_INDEX_SCALE: f32 = f32::from_bits(0x4622_f983);

const HALF_PI: f32 = f32::from_bits(0x3fc9_0fdb);

/// The bounce curve's `n1`.
const BOUNCE_N1: f32 = 7.5625;
/// The bounce thresholds `1/2.75`, `2/2.75`, `2.5/2.75`.
const BOUNCE_T1: f32 = f32::from_bits(0x3eba_2e8c);
const BOUNCE_T2: f32 = f32::from_bits(0x3f3a_2e8c);
const BOUNCE_T3: f32 = f32::from_bits(0x3f68_ba2f);
/// The bounce centres `1.5/2.75`, `2.25/2.75`, `2.625/2.75`.
const BOUNCE_C2: f32 = f32::from_bits(0x3f0b_a2e9);
const BOUNCE_C3: f32 = f32::from_bits(0x3f51_745d);
const BOUNCE_C4: f32 = f32::from_bits(0x3f74_5d17);

/// The back constants `c1 = 1.70158`, `c3 = c1 + 1`, `c2 = c1·1.525` and `c2 + 1`.
const BACK_C1: f32 = f32::from_bits(0x3fd9_cd60);
const BACK_C3: f32 = f32::from_bits(0x402c_e6b0);
const BACK_C2: f32 = f32::from_bits(0x4026_12ff);
const BACK_C2_PLUS_1: f32 = f32::from_bits(0x4066_12ff);

/// How random bounds are sorted, which decides where a NaN bound ends up.
#[derive(Copy, Clone)]
pub(crate) enum Sort {
    /// A NaN in either position survives in one of the two bounds.
    RunTime,
    /// Two literal bounds: a NaN in either position gives the first operand for both.
    Literal,
}

/// `sin(i / SIN_INDEX_SCALE)`.
#[inline]
pub(crate) fn sin_table_entry(p: Platform, index: i32) -> f32 {
    cmath::sin(p, (index & 0xffff) as f32 / SIN_INDEX_SCALE)
}

/// Entry `trunc(x·SIN_INDEX_SCALE) mod 65536`.
#[inline]
pub(crate) fn table_sin(p: Platform, radians: f32) -> f32 {
    sin_table_entry(p, to_int(p, radians * SIN_INDEX_SCALE))
}

/// Entry `trunc(x·SIN_INDEX_SCALE + 16384) mod 65536`, rounded twice.
#[inline]
pub(crate) fn table_cos(p: Platform, radians: f32) -> f32 {
    let index = add(p, mul(p, radians, SIN_INDEX_SCALE), 16384.0);
    sin_table_entry(p, to_int(p, index))
}

#[inline]
pub(crate) fn sorted_bounds(sort: Sort, a: f32, b: f32) -> (f32, f32) {
    let lo = if b < a { b } else { a };
    let hi = match sort {
        Sort::RunTime => {
            if a > b {
                a
            } else {
                b
            }
        }
        Sort::Literal => {
            if b > a {
                b
            } else {
                a
            }
        }
    };
    (lo, hi)
}

/// `hi·r + (1 − r)·lo`.
#[inline]
pub(crate) fn interpolate(p: Platform, lo: f32, hi: f32, r: f32) -> f32 {
    add(p, mul(p, hi, r), mul(p, sub(p, 1.0, r), lo))
}

/// The upper interpolation end `(hi·(−ε) + 1) + hi`.
#[inline]
pub(crate) fn integer_reach(p: Platform, hi: f32) -> f32 {
    add(p, add(p, mul(p, -f32::EPSILON, hi), 1.0), hi)
}

/// `v > hi ? hi : max(v, lo)`.
#[inline]
pub(crate) fn clamp_draw(v: f32, lo: f32, hi: f32) -> f32 {
    if v > hi { hi } else { max(v, lo) }
}

/// `floor((1 − r)·lo + r·reach)`, clamped to `[lo, hi]` upper bound first.
#[inline]
pub(crate) fn integer_draw(p: Platform, lo: f32, hi: f32, reach: f32, r: f32) -> f32 {
    let v = fpu::floor(add(p, mul(p, sub(p, 1.0, r), lo), mul(p, reach, r)));
    clamp_draw(v, lo, hi)
}

/// `sum + integer_draw(…)`.
#[inline]
pub(crate) fn integer_roll(p: Platform, sum: f32, lo: f32, hi: f32, reach: f32, r: f32) -> f32 {
    add(p, sum, integer_draw(p, lo, hi, reach, r))
}

/// `sum + (r·hi + (1 − r)·lo)`.
#[inline]
pub(crate) fn roll(p: Platform, sum: f32, lo: f32, hi: f32, r: f32) -> f32 {
    add(p, sum, add(p, mul(p, r, hi), mul(p, sub(p, 1.0, r), lo)))
}

/// Clamps to `[-1, 1]` unless `|x| > 1.0005`; a NaN passes through.
#[inline]
pub(crate) fn inverse_trig_argument(x: f32) -> f32 {
    if x.abs() > INVERSE_TRIG_TOLERANCE || x.is_nan() {
        x
    } else if x > 1.0 {
        1.0
    } else {
        max(x, -1.0)
    }
}

/// `((rad·k)·S) + O`.
#[inline]
pub(crate) fn degrees(p: Platform, rad: f32, post: PostOp) -> f32 {
    post.apply(p, mul(p, rad, RAD_TO_DEG))
}

/// `(3·t)·t − ((t + t)·t)·t`.
#[inline]
pub(crate) fn hermite_blend(p: Platform, t: f32) -> f32 {
    sub(
        p,
        mul(p, mul(p, 3.0, t), t),
        mul(p, mul(p, add(p, t, t), t), t),
    )
}

/// `(b − a)·t + a`; of two NaNs, `b − a` wins over `t` and the product over `a`.
#[inline]
pub(crate) fn lerp(p: Platform, a: f32, b: f32, t: f32) -> f32 {
    add(p, mul(p, sub(p, b, a), t), a)
}

/// `wrap(b − a)·t + a`; of two NaNs, the wrapped difference wins over `t` and the product over `a`.
#[inline]
pub(crate) fn lerprotate(p: Platform, a: f32, b: f32, t: f32) -> f32 {
    add(p, mul(p, wrap_angle(p, sub(p, b, a)), t), a)
}

/// `(v − a) / (b − a)`.
#[inline]
pub(crate) fn inverse_lerp(p: Platform, a: f32, b: f32, v: f32) -> f32 {
    div(p, sub(p, v, a), sub(p, b, a))
}

/// Wraps degrees into `[-180, 180)`.
#[inline]
pub(crate) fn wrap_angle(p: Platform, x: f32) -> f32 {
    let r = cmath::rem(p, add(p, x, 180.0), 360.0);
    sub(p, if r < 0.0 { add(p, r, 360.0) } else { r }, 180.0)
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::numeric::{Platform, PostOp};

    /// An easing body.
    pub(crate) type Body = fn(Platform, f32, f32, f32) -> f32;
    /// A public `math::ease_*` function.
    pub(crate) type Public = fn(Platform, f32, f32, f32, PostOp) -> f32;

    /// The 30 easing bodies in op order, each with its name and its public function.
    pub(crate) fn easing_bodies() -> [(&'static str, Body, Public); 30] {
        use super as m;
        use crate::stdlib::math;
        [
            ("in_quad", m::in_quad, math::ease_in_quad),
            ("out_quad", m::out_quad, math::ease_out_quad),
            ("in_out_quad", m::in_out_quad, math::ease_in_out_quad),
            ("in_cubic", m::in_cubic, math::ease_in_cubic),
            ("out_cubic", m::out_cubic, math::ease_out_cubic),
            ("in_out_cubic", m::in_out_cubic, math::ease_in_out_cubic),
            ("in_quart", m::in_quart, math::ease_in_quart),
            ("out_quart", m::out_quart, math::ease_out_quart),
            ("in_out_quart", m::in_out_quart, math::ease_in_out_quart),
            ("in_quint", m::in_quint, math::ease_in_quint),
            ("out_quint", m::out_quint, math::ease_out_quint),
            ("in_out_quint", m::in_out_quint, math::ease_in_out_quint),
            ("in_sine", m::in_sine, math::ease_in_sine),
            ("out_sine", m::out_sine, math::ease_out_sine),
            ("in_out_sine", m::in_out_sine, math::ease_in_out_sine),
            ("in_expo", m::in_expo, math::ease_in_expo),
            ("out_expo", m::out_expo, math::ease_out_expo),
            ("in_out_expo", m::in_out_expo, math::ease_in_out_expo),
            ("in_circ", m::in_circ, math::ease_in_circ),
            ("out_circ", m::out_circ, math::ease_out_circ),
            ("in_out_circ", m::in_out_circ, math::ease_in_out_circ),
            ("in_bounce", m::in_bounce, math::ease_in_bounce),
            ("out_bounce", m::out_bounce, math::ease_out_bounce),
            ("in_out_bounce", m::in_out_bounce, math::ease_in_out_bounce),
            ("in_back", m::in_back, math::ease_in_back),
            ("out_back", m::out_back, math::ease_out_back),
            ("in_out_back", m::in_out_back, math::ease_in_out_back),
            ("in_elastic", m::in_elastic, math::ease_in_elastic),
            ("out_elastic", m::out_elastic, math::ease_out_elastic),
            (
                "in_out_elastic",
                m::in_out_elastic,
                math::ease_in_out_elastic,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::easing_bodies;
    use super::*;
    use crate::numeric::test_support::*;
    use crate::stdlib::math::PI;

    #[test]
    fn a_nan_bound_survives_the_sort() {
        assert!(sorted_bounds(Sort::RunTime, NAN, 1.0).0.is_nan());
        assert!(sorted_bounds(Sort::RunTime, 1.0, NAN).1.is_nan());
        assert!(
            sorted_bounds(Sort::Literal, NAN, 1.0).0.is_nan()
                && sorted_bounds(Sort::Literal, NAN, 1.0).1.is_nan()
        );
        assert_eq!(sorted_bounds(Sort::Literal, 1.0, NAN), (1.0, 1.0));
    }

    #[test]
    fn ordinary_bounds_sort_and_interpolate_alike_on_every_platform() {
        for sort in [Sort::RunTime, Sort::Literal] {
            assert_eq!(sorted_bounds(sort, 5.0, 1.0), (1.0, 5.0));
            assert_eq!(sorted_bounds(sort, -1.0, 2.0), (-1.0, 2.0));
        }
        for p in PLATFORMS {
            assert_bits(interpolate(p, 1.0, 5.0, 0.0), 1.0);
            assert_bits(interpolate(p, 1.0, 5.0, 0.5), 3.0);
            assert_bits(interpolate(p, 1.0, 5.0, 1.0), 5.0);
            let reach = integer_reach(p, 6.0);
            assert_bits(integer_draw(p, 1.0, 6.0, reach, 0.0), 1.0);
            assert_bits(integer_draw(p, 1.0, 6.0, reach, 1.0), 6.0);
            assert_bits(integer_draw(p, 1.0, 6.0, reach, 0.5), 3.0);
            assert_bits(roll(p, 2.0, 1.0, 5.0, 0.5), 5.0);
        }
    }

    #[test]
    fn the_formulas_agree_on_exact_operands() {
        for p in PLATFORMS {
            assert_bits(inverse_trig_argument(1.0004), 1.0);
            assert_bits(inverse_trig_argument(-1.0004), -1.0);
            assert_bits(inverse_trig_argument(1.001), 1.001);
            assert_bits(degrees(p, 0.0, PostOp::IDENTITY), 0.0);
            assert_bits(degrees(p, 0.0, AFFINE), 1.0);
            assert_bits(hermite_blend(p, 0.5), 0.5);
            assert_bits(hermite_blend(p, 1.0), 1.0);
            assert_bits(lerp(p, 1.0, 5.0, 0.5), 3.0);
            assert_bits(lerprotate(p, 350.0, 10.0, 0.5), 360.0);
            assert_bits(sin_table_entry(p, 0), 0.0);
            assert_bits(table_sin(p, 0.0), 0.0);
            close(table_cos(p, 0.0), 1.0, 1e-6);
        }
    }

    #[test]
    fn degrees_keeps_the_sign_and_payload_of_a_nan() {
        let nan = f32::from_bits(0xffc0_0001);
        for p in PLATFORMS {
            assert_eq!(degrees(p, nan, PostOp::IDENTITY).to_bits(), 0xffc0_0001);
            assert_eq!(degrees(p, nan, AFFINE).to_bits(), 0xffc0_0001);
            assert_eq!(
                degrees(p, f32::from_bits(0xff80_0001), PostOp::IDENTITY).to_bits(),
                0xffc0_0001
            );
        }
    }

    #[test]
    fn a_nan_passes_the_inverse_trig_clamp() {
        assert!(inverse_trig_argument(NAN).is_nan());
        assert_bits(inverse_trig_argument(-1.0004), -1.0);
    }

    #[test]
    fn each_operation_chooses_its_nan() {
        let (q1, q2, q3) = (
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc0_0002),
            f32::from_bits(0x7fc0_0003),
        );
        for p in PLATFORMS {
            // The left operand of each step first.
            assert_eq!(interpolate(p, q1, q2, q3).to_bits(), 0xffc0_0002);
            assert_eq!(interpolate(p, q1, 1.0, q3).to_bits(), 0x7fc0_0003);
            let (lo, hi) = (f32::from_bits(0xffc0_0001), f32::from_bits(0x7fc0_0002));
            assert_eq!(interpolate(p, lo, hi, 0.5).to_bits(), 0x7fc0_0002);
            assert_eq!(roll(p, 0.0, lo, hi, 0.5).to_bits(), 0x7fc0_0002);
            assert_eq!(integer_reach(p, q2).to_bits(), 0xffc0_0002);
            // The clamp returns `lo` for a NaN draw.
            assert_eq!(integer_draw(p, q1, 6.0, q2, q3).to_bits(), 0x7fc0_0001);
            assert_eq!(
                integer_roll(p, q2, q1, 6.0, 6.0, 0.5).to_bits(),
                0xffc0_0002
            );
            assert_eq!(roll(p, q2, q1, q3, 0.5).to_bits(), 0xffc0_0002);
            assert_eq!(roll(p, 1.0, q1, q3, 0.5).to_bits(), 0x7fc0_0003);
            assert_eq!(hermite_blend(p, q2).to_bits(), 0xffc0_0002);
            assert_eq!(inverse_lerp(p, q1, q2, q3).to_bits(), 0x7fc0_0003);
            assert_eq!(inverse_lerp(p, q1, q2, 1.0).to_bits(), 0x7fc0_0001);
            assert_eq!(lerp(p, q1, q2, 0.5).to_bits(), 0xffc0_0002);
            // Invalid operations on numbers.
            let invalid = servers_or_android(p, 0xffc0_0000, 0x7fc0_0000);
            assert_eq!(interpolate(p, 1.0, INF, 0.0).to_bits(), invalid);
            assert_eq!(integer_reach(p, -INF).to_bits(), invalid);
            assert_eq!(integer_draw(p, -INF, 6.0, INF, 0.5), -INF);
            assert_eq!(
                integer_roll(p, -INF, 1.0, 6.0, 6.0, 0.5).to_bits(),
                0xff80_0000
            );
            assert_eq!(roll(p, -INF, 1.0, INF, 0.5).to_bits(), invalid);
            assert_eq!(hermite_blend(p, INF).to_bits(), invalid);
            assert_eq!(inverse_lerp(p, 5.0, 5.0, 5.0).to_bits(), invalid);
        }
    }

    #[test]
    fn a_signalling_nan_wins_on_android_only() {
        let (quiet, signalling) = (f32::from_bits(0xffc0_0001), f32::from_bits(0x7f80_0002));
        assert_eq!(lerp(LINUX, 1.0, quiet, signalling).to_bits(), 0xffc0_0001);
        assert_eq!(lerp(ANDROID, 1.0, quiet, signalling).to_bits(), 0x7fc0_0002);
    }

    #[test]
    fn the_formulas_round_each_step() {
        let x = 0.3_f32;
        for p in PLATFORMS {
            assert_bits(hermite_blend(p, x), (3.0 * x) * x - ((x + x) * x) * x);
            assert_bits(interpolate(p, -3.0, 7.5, x), 7.5 * x + (1.0 - x) * -3.0);
            assert_bits(
                roll(p, 1.0, -3.0, 7.5, x),
                1.0 + (x * 7.5 + (1.0 - x) * -3.0),
            );
            assert_bits(
                degrees(p, x, PostOp::new(3.0, -1.0)),
                (x * RAD_TO_DEG) * 3.0 - 1.0,
            );
        }
    }

    #[test]
    fn every_easing_starts_at_the_start_and_ends_near_the_end() {
        for p in PLATFORMS {
            for (_, body, _) in easing_bodies() {
                close(body(p, 1.0, 5.0, 0.0), 1.0, 5e-3);
                close(body(p, 1.0, 5.0, 1.0), 5.0, 5e-3);
            }
        }
    }

    #[test]
    fn the_public_easings_call_the_bodies() {
        for p in PLATFORMS {
            for (name, body, public) in easing_bodies() {
                for (start, end) in [
                    (1.0_f32, 5.0),
                    (-2.5, 7.25),
                    (0.0, 0.0),
                    (-0.0, 1.0e-45),
                    (3.0, -4.0),
                ] {
                    for t in [
                        0.0_f32, -0.0, 0.1, 0.25, 0.3, 0.5, 0.75, 0.999, 1.0, 1.5, -0.5, 10.0, NAN,
                        INF,
                    ] {
                        let (via_public, direct) =
                            (public(p, start, end, t, ID), body(p, start, end, t));
                        assert_eq!(
                            via_public.to_bits(),
                            direct.to_bits(),
                            "{p:?} ease_{name}({start}, {end}, {t})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn table_constants_are_the_documented_bit_patterns() {
        assert_eq!(SIN_INDEX_SCALE.to_bits(), 0x4622_f983);
        assert_eq!(HALF_PI.to_bits(), 0x3fc9_0fdb);
        assert_eq!(PI.to_bits(), 0x4049_0fdb);
        close(SIN_INDEX_SCALE, 10_430.378, 1e-3);
    }

    #[test]
    fn sin_table_entry_is_sin_of_the_index_divided_by_the_scale() {
        for index in [0_i32, 1, 100, 4096, 16_384, 32_768, 49_152, 65_535] {
            let i = index as f32;
            for p in PLATFORMS {
                assert_bits(
                    sin_table_entry(p, index),
                    cmath::sin(p, i / SIN_INDEX_SCALE),
                );
            }
        }
    }

    #[test]
    fn the_glibc_sine_table_differs_from_the_rounded_one_in_893_entries() {
        let differing = (0..=0xffff)
            .filter(|&i| sin_table_entry(LINUX, i) != sin_table_entry(WINDOWS, i))
            .count();
        assert_eq!(differing, 893);
        for i in [1, 0x1234, 0xfffe] {
            assert_bits(sin_table_entry(ANDROID, i), sin_table_entry(LINUX, i));
        }
    }

    #[test]
    fn sin_table_entry_at_the_quadrant_boundaries() {
        let p = LINUX;
        assert_bits(sin_table_entry(p, 0), 0.0);
        close(sin_table_entry(p, 16_384), 1.0, 1e-6);
        close(sin_table_entry(p, 32_768), 0.0, 2e-5);
        close(sin_table_entry(p, 49_152), -1.0, 1e-6);
        close(
            sin_table_entry(p, 8_192),
            std::f32::consts::FRAC_1_SQRT_2,
            1e-5,
        );
        close(sin_table_entry(p, 65_535), 0.0, 2e-4);
    }

    #[test]
    fn sin_table_entry_masks_the_index_to_sixteen_bits() {
        let p = LINUX;
        for index in [0_i32, 1, 5, 1000, 40_000, 65_535] {
            assert_bits(
                sin_table_entry(p, index + 65_536),
                sin_table_entry(p, index),
            );
            assert_bits(
                sin_table_entry(p, index + 3 * 65_536),
                sin_table_entry(p, index),
            );
            assert_bits(
                sin_table_entry(p, index - 65_536),
                sin_table_entry(p, index),
            );
        }
        assert_bits(sin_table_entry(p, -1), sin_table_entry(p, 65_535));
        assert_bits(sin_table_entry(p, -16_384), sin_table_entry(p, 49_152));
        assert_bits(sin_table_entry(p, i32::MIN), sin_table_entry(p, 0));
        assert_bits(sin_table_entry(p, i32::MAX), sin_table_entry(p, 65_535));
    }

    #[test]
    fn table_sin_truncates_the_scaled_angle() {
        let p = LINUX;
        assert_bits(table_sin(p, 0.0), 0.0);
        let angle = 1.5 / SIN_INDEX_SCALE;
        assert_bits(table_sin(p, angle), sin_table_entry(p, 1));
        close(table_sin(p, HALF_PI), 1.0, 1e-4);
        close(table_sin(p, PI), 0.0, 2e-4);
        let small = 3.5 / SIN_INDEX_SCALE;
        assert_bits(table_sin(p, -small), sin_table_entry(p, -3));
        close(table_sin(p, -HALF_PI), -1.0, 1e-4);
    }

    #[test]
    fn table_sin_beyond_the_int_range_is_entry_zero_on_the_servers_and_saturates_on_android() {
        for p in [LINUX, LINUX_1_26_36, WINDOWS] {
            assert_bits(table_sin(p, 1.0e6), sin_table_entry(p, 0));
            assert_bits(table_sin(p, INF), sin_table_entry(p, 0));
            assert_bits(table_cos(p, INF), sin_table_entry(p, 0));
        }
        assert_bits(table_sin(ANDROID, 1.0e6), sin_table_entry(ANDROID, 65_535));
        assert_bits(table_sin(ANDROID, INF), sin_table_entry(ANDROID, 65_535));
        assert_bits(table_cos(ANDROID, INF), sin_table_entry(ANDROID, 65_535));
        assert_bits(table_cos(ANDROID, -INF), sin_table_entry(ANDROID, 0));
        for p in PLATFORMS {
            assert_bits(table_sin(p, -INF), sin_table_entry(p, 0));
            assert_bits(table_sin(p, NAN), 0.0);
            assert_bits(table_cos(p, NAN), 0.0);
        }
    }

    #[test]
    fn table_cos_is_the_table_shifted_by_a_quarter_turn() {
        let p = LINUX;
        assert_bits(table_cos(p, 0.0), sin_table_entry(p, 16_384));
        close(table_cos(p, 0.0), 1.0, 1e-6);
        close(table_cos(p, PI), -1.0, 1e-4);
        close(table_cos(p, HALF_PI), 0.0, 2e-4);
        close(table_cos(p, -PI), -1.0, 1e-4);
        let angle = 1.5 / SIN_INDEX_SCALE;
        assert_bits(table_cos(p, angle), sin_table_entry(p, 16_385));
        assert_bits(table_cos(p, -5.0 * HALF_PI), table_cos(p, -HALF_PI));
    }
}
