//! The easings: the Penner formulas in order, every operation rounded on its own.
//!
//! The first operand of each arithmetic call is the one whose NaN wins (of two quiet ones); the
//! start value is the second operand of the final addition except in the elastic curves and the
//! second half of `in_out_bounce`, and on [`Platform::BdsWindows`] in the ten easings that call
//! [`plus_start`].

use super::{
    BACK_C1, BACK_C2, BACK_C2_PLUS_1, BACK_C3, BOUNCE_C2, BOUNCE_C3, BOUNCE_C4, BOUNCE_N1,
    BOUNCE_T1, BOUNCE_T2, BOUNCE_T3, HALF_PI, table_cos, table_sin,
};
use crate::numeric::Platform;
use crate::numeric::fpu::{self, add, div, mul, sub};
use crate::stdlib::math::{PI, transcendental};

/// 0.075, 2π and 0.3.
const ELASTIC_SHIFT: f32 = f32::from_bits(0x3d99_999a);
const TWO_PI: f32 = f32::from_bits(0x40c9_0fdb);
const ELASTIC_PERIOD: f32 = f32::from_bits(0x3e99_999a);

/// `x + 0`, turning a `−0` product into `+0` where the formulas need it.
#[inline]
fn plus_zero(p: Platform, x: f32) -> f32 {
    add(p, x, 0.0)
}

/// `x + start`; on [`Platform::BdsWindows`] `start + x`, which differs only in the NaN of two.
#[inline]
fn plus_start(p: Platform, x: f32, start: f32) -> f32 {
    if p.reorders_operands() {
        add(p, start, x)
    } else {
        add(p, x, start)
    }
}

/// `x·h`; on [`Platform::BdsWindows`] `h·x`, which differs only in the NaN of two.
#[inline]
fn times_half_range(p: Platform, x: f32, h: f32) -> f32 {
    if p.reorders_operands() {
        mul(p, h, x)
    } else {
        mul(p, x, h)
    }
}

/// `in_quad`: `c·t·t + b`.
#[inline]
pub(crate) fn in_quad(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    add(p, mul(p, mul(p, sub(p, end, start), t), t), start)
}

/// `out_quad`: `−c·t·(t − 2) + b`, computed as `b − (t − 2)·(c·t)`.
#[inline]
pub(crate) fn out_quad(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let x = mul(p, sub(p, end, start), t);
    sub(p, start, mul(p, add(p, t, -2.0), x))
}

/// `in_out_quad`: `1 > u ? c/2·u·u + b : −c/2·((u − 1)·((u − 1) − 2) − 1) + b`.
#[inline]
pub(crate) fn in_out_quad(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let u = add(p, t, t);
    let d = sub(p, end, start);
    if 1.0 > u {
        plus_start(p, mul(p, mul(p, mul(p, d, 0.5), u), u), start)
    } else {
        let v = add(p, u, -1.0);
        let inner = add(p, mul(p, add(p, -2.0, v), v), -1.0);
        plus_start(p, times_half_range(p, inner, mul(p, d, -0.5)), start)
    }
}

/// `in_cubic`: `c·t·t·t + b`.
#[inline]
pub(crate) fn in_cubic(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    add(
        p,
        mul(p, mul(p, mul(p, sub(p, end, start), t), t), t),
        start,
    )
}

/// `out_cubic`: `c·((t − 1)³ + 1) + b`.
#[inline]
pub(crate) fn out_cubic(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let v = add(p, t, -1.0);
    add(
        p,
        mul(p, add(p, mul(p, mul(p, v, v), v), 1.0), sub(p, end, start)),
        start,
    )
}

/// `in_out_cubic`: `1 > u ? c/2·u·u·u + b : c/2·((u − 2)³ + 2) + b`.
#[inline]
pub(crate) fn in_out_cubic(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let u = add(p, t, t);
    let d = sub(p, end, start);
    if 1.0 > u {
        plus_start(p, mul(p, mul(p, mul(p, mul(p, d, 0.5), u), u), u), start)
    } else {
        let w = add(p, u, -2.0);
        let inner = add(p, mul(p, mul(p, w, w), w), 2.0);
        plus_start(p, times_half_range(p, inner, mul(p, d, 0.5)), start)
    }
}

/// `in_quart`: `c·t·t·t·t + b`.
#[inline]
pub(crate) fn in_quart(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    add(
        p,
        mul(p, mul(p, mul(p, mul(p, sub(p, end, start), t), t), t), t),
        start,
    )
}

/// `out_quart`: `−c·((t − 1)⁴ − 1) + b`.
#[inline]
pub(crate) fn out_quart(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let v = add(p, t, -1.0);
    sub(
        p,
        start,
        mul(
            p,
            add(p, mul(p, mul(p, mul(p, v, v), v), v), -1.0),
            sub(p, end, start),
        ),
    )
}

/// `in_out_quart`: `1 > u ? c/2·u⁴ + b : −c/2·((u − 2)⁴ − 2) + b`.
#[inline]
pub(crate) fn in_out_quart(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let u = add(p, t, t);
    let d = sub(p, end, start);
    if 1.0 > u {
        plus_start(
            p,
            mul(p, mul(p, mul(p, mul(p, mul(p, d, 0.5), u), u), u), u),
            start,
        )
    } else {
        let w = add(p, u, -2.0);
        let inner = add(p, mul(p, mul(p, mul(p, w, w), w), w), -2.0);
        plus_start(p, times_half_range(p, inner, mul(p, d, -0.5)), start)
    }
}

/// `in_quint`: `c·t·t·t·t·t + b`.
#[inline]
pub(crate) fn in_quint(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    add(
        p,
        mul(
            p,
            mul(p, mul(p, mul(p, mul(p, sub(p, end, start), t), t), t), t),
            t,
        ),
        start,
    )
}

/// `out_quint`: `c·((t − 1)⁵ + 1) + b`.
#[inline]
pub(crate) fn out_quint(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let v = add(p, t, -1.0);
    add(
        p,
        mul(
            p,
            add(p, mul(p, mul(p, mul(p, mul(p, v, v), v), v), v), 1.0),
            sub(p, end, start),
        ),
        start,
    )
}

/// `in_out_quint`: `1 > u ? c/2·u⁵ + b : c/2·((u − 2)⁵ + 2) + b`.
#[inline]
pub(crate) fn in_out_quint(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let u = add(p, t, t);
    let d = sub(p, end, start);
    if 1.0 > u {
        let x = mul(
            p,
            mul(p, mul(p, mul(p, mul(p, mul(p, d, 0.5), u), u), u), u),
            u,
        );
        plus_start(p, x, start)
    } else {
        let w = add(p, u, -2.0);
        let inner = add(p, mul(p, mul(p, mul(p, mul(p, w, w), w), w), w), 2.0);
        plus_start(p, times_half_range(p, inner, mul(p, d, 0.5)), start)
    }
}

/// `in_sine`: `c − c·cos(t·π/2) + b`.
#[inline]
pub(crate) fn in_sine(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let c = table_cos(p, mul(p, t, HALF_PI));
    let d = sub(p, end, start);
    add(p, sub(p, d, mul(p, c, d)), start)
}

/// `out_sine`: `c·sin(t·π/2) + b`.
#[inline]
pub(crate) fn out_sine(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    add(
        p,
        mul(p, sub(p, end, start), table_sin(p, mul(p, t, HALF_PI))),
        start,
    )
}

/// `in_out_sine`: `−c/2·(cos(π·t) − 1) + b`.
#[inline]
pub(crate) fn in_out_sine(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let h = mul(p, sub(p, end, start), -0.5);
    let c = table_cos(p, mul(p, t, PI));
    add(p, mul(p, add(p, c, -1.0), h), start)
}

/// `in_expo`: `c·2^(10·(t − 1)) + b`.
#[inline]
pub(crate) fn in_expo(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let d = sub(p, end, start);
    add(
        p,
        mul(p, transcendental::exp2(p, mul(p, add(p, t, -1.0), 10.0)), d),
        start,
    )
}

/// `out_expo`: `c·(1 − 2^(−10t)) + b`.
#[inline]
pub(crate) fn out_expo(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let d = sub(p, end, start);
    add(
        p,
        mul(p, sub(p, 1.0, transcendental::exp2(p, mul(p, t, -10.0))), d),
        start,
    )
}

/// `in_out_expo`: `1 > u ? c/2·2^(10·(u − 1)) + b : c/2·(2 − 2^(−10·(u − 1))) + b`.
#[inline]
pub(crate) fn in_out_expo(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let u = add(p, t, t);
    let h = mul(p, sub(p, end, start), 0.5);
    let v = add(p, u, -1.0);
    if 1.0 > u {
        plus_start(
            p,
            mul(p, transcendental::exp2(p, mul(p, v, 10.0)), h),
            start,
        )
    } else {
        plus_start(
            p,
            mul(p, sub(p, 2.0, transcendental::exp2(p, mul(p, v, -10.0))), h),
            start,
        )
    }
}

/// `in_circ`: `−c·(sqrt(1 − t·t) − 1) + b`; negating `c` flips a NaN's sign too. On
/// [`Platform::BdsWindows`] `b − c·(sqrt(1 − t·t) − 1)`, the same value with `c`'s NaN unflipped.
#[inline]
pub(crate) fn in_circ(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let d = sub(p, end, start);
    let q = add(p, fpu::sqrt(p, sub(p, 1.0, mul(p, t, t))), -1.0);
    if p.reorders_operands() {
        sub(p, start, mul(p, d, q))
    } else {
        add(p, mul(p, q, -d), start)
    }
}

/// `out_circ`: `c·sqrt(1 − (t − 1)²) + b`.
#[inline]
pub(crate) fn out_circ(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let v = add(p, t, -1.0);
    let x = mul(
        p,
        sub(p, end, start),
        fpu::sqrt(p, sub(p, 1.0, mul(p, v, v))),
    );
    plus_start(p, x, start)
}

/// `in_out_circ`: `1 > u ? −c/2·(sqrt(1 − u²) − 1) + b : c/2·(sqrt(1 − (u − 2)²) + 1) + b`, with
/// the halved range as the first factor.
#[inline]
pub(crate) fn in_out_circ(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let u = add(p, t, t);
    let d = sub(p, end, start);
    let (x, k, factor) = if 1.0 > u {
        (u, -1.0, mul(p, d, -0.5))
    } else {
        (add(p, u, -2.0), 1.0, mul(p, d, 0.5))
    };
    let y = mul(
        p,
        factor,
        add(p, k, fpu::sqrt(p, sub(p, 1.0, mul(p, x, x)))),
    );
    plus_start(p, y, start)
}

/// `n1·y·y + k` with `y = x − centre`; a NaN takes the last piece.
#[inline]
fn bounce(p: Platform, x: f32) -> f32 {
    if BOUNCE_T1 > x {
        mul(p, mul(p, BOUNCE_N1, x), x)
    } else if BOUNCE_T2 > x {
        let y = add(p, x, -BOUNCE_C2);
        add(p, mul(p, mul(p, BOUNCE_N1, y), y), 0.75)
    } else if BOUNCE_T3 > x {
        let y = add(p, x, -BOUNCE_C3);
        add(p, mul(p, mul(p, BOUNCE_N1, y), y), 0.9375)
    } else {
        let y = add(p, x, -BOUNCE_C4);
        add(p, mul(p, mul(p, BOUNCE_N1, y), y), 0.984_375)
    }
}

/// `in_bounce`: `c − out_bounce(1 − t)·c + b`.
#[inline]
pub(crate) fn in_bounce(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let d = sub(p, end, start);
    let x = plus_zero(p, mul(p, bounce(p, sub(p, 1.0, t)), d));
    add(p, sub(p, d, x), start)
}

/// `out_bounce`: `c·out_bounce(t) + b`.
#[inline]
pub(crate) fn out_bounce(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    add(p, mul(p, sub(p, end, start), bounce(p, t)), start)
}

/// `in_out_bounce`: `0.5 > t ? in_bounce(2t)·0.5 + b : out_bounce(2t − 1)·0.5 + c·0.5 + b`,
/// each half over `(0, c)`.
#[inline]
pub(crate) fn in_out_bounce(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let d = sub(p, end, start);
    if 0.5 > t {
        let x = plus_zero(p, mul(p, bounce(p, sub(p, 1.0, add(p, t, t))), d));
        plus_start(p, mul(p, plus_zero(p, sub(p, d, x)), 0.5), start)
    } else {
        let x = plus_zero(p, mul(p, bounce(p, add(p, add(p, t, t), -1.0)), d));
        let (x, d) = (mul(p, x, 0.5), mul(p, d, 0.5));
        let half = if p.reorders_operands() {
            add(p, d, x)
        } else {
            add(p, x, d)
        };
        add(p, start, half)
    }
}

/// `in_back`: `c·t·t·(c3·t − c1) + b`.
#[inline]
pub(crate) fn in_back(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let x = mul(p, mul(p, sub(p, end, start), t), t);
    add(p, mul(p, add(p, mul(p, t, BACK_C3), -BACK_C1), x), start)
}

/// `out_back`: `c·((t − 1)²·(c3·(t − 1) + c1) + 1) + b`.
#[inline]
pub(crate) fn out_back(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let v = add(p, t, -1.0);
    let y = add(p, mul(p, v, BACK_C3), BACK_C1);
    add(
        p,
        mul(p, add(p, mul(p, mul(p, v, v), y), 1.0), sub(p, end, start)),
        start,
    )
}

/// `in_out_back`: `1 > u ? c/2·(u²·((c2 + 1)·u − c2)) + b :
/// c/2·((u − 2)²·((c2 + 1)·(u − 2) + c2) + 2) + b`.
#[inline]
pub(crate) fn in_out_back(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    let h = mul(p, sub(p, end, start), 0.5);
    let u = add(p, t, t);
    if 1.0 > u {
        let x = mul(p, add(p, mul(p, u, BACK_C2_PLUS_1), -BACK_C2), mul(p, u, u));
        plus_start(p, mul(p, x, h), start)
    } else {
        let w = add(p, u, -2.0);
        let x = add(
            p,
            mul(p, add(p, mul(p, w, BACK_C2_PLUS_1), BACK_C2), mul(p, w, w)),
            2.0,
        );
        plus_start(p, mul(p, x, h), start)
    }
}

/// `((x − 0.075)·2π)/0.3`, rounded three times.
#[inline]
fn elastic_angle(p: Platform, x: f32) -> f32 {
    div(p, mul(p, add(p, x, -ELASTIC_SHIFT), TWO_PI), ELASTIC_PERIOD)
}

/// `in_elastic`: `t == 0` → `b`, `t == 1` → `b + c`, otherwise
/// `b − c·2^(10·(t − 1))·sin(((t − 1) − 0.075)·2π/0.3)`.
#[inline]
pub(crate) fn in_elastic(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    if t == 0.0 {
        return start;
    }
    let d = sub(p, end, start);
    if t == 1.0 {
        return add(p, start, d);
    }
    let v = add(p, t, -1.0);
    let scale = transcendental::exp2(p, mul(p, 10.0, v));
    sub(
        p,
        start,
        mul(p, mul(p, d, scale), table_sin(p, elastic_angle(p, v))),
    )
}

/// `out_elastic`: `t == 0` → `b`, `t == 1` → `b + c`, otherwise
/// `c·2^(−10t)·sin((t − 0.075)·2π/0.3) + c + b`.
#[inline]
pub(crate) fn out_elastic(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    if t == 0.0 {
        return start;
    }
    let d = sub(p, end, start);
    if t == 1.0 {
        return add(p, start, d);
    }
    let x = mul(p, transcendental::exp2(p, mul(p, -10.0, t)), d);
    add(
        p,
        start,
        add(p, mul(p, x, table_sin(p, elastic_angle(p, t))), d),
    )
}

/// `in_out_elastic`: `t == 0` → `b`, `2t == 2` → `b + c`; with `v = 2t − 1` and
/// `s = sin((v − 0.075)·2π/0.3)`, `1 > 2t ? b + −0.5·(c·2^(10v)·s) : b + (c·2^(−10v)·s·0.5 + c)`.
#[inline]
pub(crate) fn in_out_elastic(p: Platform, start: f32, end: f32, t: f32) -> f32 {
    if t == 0.0 {
        return start;
    }
    let d = sub(p, end, start);
    let u = add(p, t, t);
    if u == 2.0 {
        return add(p, start, d);
    }
    let v = add(p, u, -1.0);
    let s = table_sin(p, elastic_angle(p, v));
    if 1.0 > u {
        add(
            p,
            start,
            mul(
                p,
                mul(p, mul(p, transcendental::exp2(p, mul(p, v, 10.0)), d), s),
                -0.5,
            ),
        )
    } else {
        add(
            p,
            start,
            add(
                p,
                mul(
                    p,
                    mul(p, mul(p, transcendental::exp2(p, mul(p, v, -10.0)), d), s),
                    0.5,
                ),
                d,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;
    use crate::stdlib::math::transcendental::with_strict;
    use crate::stdlib::math::{self, formula::test_support::easing_bodies};

    /// The sign of the NaN each body returns (`+` clear, `-` set) with two NaNs of opposite sign in
    /// `(start, end)`, `(end, start)`, `(end, t)`, `(t, end)`, `(start, t)`, `(t, start)`, the
    /// first of each pair negative and the third argument 0.5; on [`Platform::BdsWindows`] ten
    /// easings take the start's NaN first.
    #[test]
    fn every_body_returns_its_nan_of_two() {
        let (n, p) = (f32::from_bits(0xffc0_0000), f32::from_bits(0x7fc0_0000));
        #[rustfmt::skip]
        let nans: [(&str, &str); 30] = [
            ("in_quad", "+--+-+"), ("out_quad", "-++--+"), ("in_out_quad", "+-+-+-"),
            ("in_cubic", "+--+-+"), ("out_cubic", "+-+-+-"), ("in_out_cubic", "+-+-+-"),
            ("in_quart", "+--+-+"), ("out_quart", "-++--+"), ("in_out_quart", "+-+-+-"),
            ("in_quint", "+--+-+"), ("out_quint", "+-+-+-"), ("in_out_quint", "+-+-+-"),
            ("in_sine", "+--+-+"), ("out_sine", "+--+-+"), ("in_out_sine", "+--+-+"),
            ("in_expo", "+-+-+-"), ("out_expo", "+-+-+-"), ("in_out_expo", "+-+-+-"),
            ("in_circ", "-++-+-"), ("out_circ", "+--+-+"), ("in_out_circ", "+--+-+"),
            ("in_bounce", "+--+-+"), ("out_bounce", "+--+-+"), ("in_out_bounce", "-++--+"),
            ("in_back", "+-+-+-"), ("out_back", "+-+-+-"), ("in_out_back", "+-+-+-"),
            ("in_elastic", "-+-+-+"), ("out_elastic", "-++--+"), ("in_out_elastic", "-++--+"),
        ];
        // Exactly ten easings; the other 20 keep the Linux choice.
        #[rustfmt::skip]
        let windows: [(&str, &str); 10] = [
            ("in_out_quad", "-+-+-+"), ("in_out_cubic", "-+-+-+"), ("in_out_quart", "-+-+-+"),
            ("in_out_quint", "-+-+-+"), ("in_out_expo", "-++--+"), ("in_circ", "-+-+-+"),
            ("out_circ", "-+-+-+"), ("in_out_circ", "-+-+-+"), ("in_out_bounce", "-+-+-+"),
            ("in_out_back", "-++--+"),
        ];
        let rows = [
            (n, p, 0.5),
            (p, n, 0.5),
            (0.5, n, p),
            (0.5, p, n),
            (n, 0.5, p),
            (p, 0.5, n),
        ];
        for platform in PLATFORMS {
            for ((name, body, _), (signs_name, signs)) in easing_bodies().into_iter().zip(nans) {
                assert_eq!(name, signs_name);
                let signs = match windows.iter().find(|(other, _)| *other == name) {
                    Some((_, other)) if platform == WINDOWS => other,
                    _ => signs,
                };
                for ((start, end, t), sign) in rows.into_iter().zip(signs.chars()) {
                    let got = body(platform, start, end, t);
                    let at = format!("{platform:?} {name}({start}, {end}, {t})");
                    assert!(got.is_nan(), "{at}");
                    let want = if sign == '+' { p } else { n };
                    assert_eq!(got.to_bits(), want.to_bits(), "{at}");
                }
            }
        }
    }

    /// Of the platforms' NaN rules only the order of the operands differs on Windows: the values
    /// are those of Linux.
    #[test]
    fn windows_swaps_operands_without_changing_a_value() {
        for (name, body, _) in easing_bodies() {
            assert_eq!(
                sweep_hash(|s, e, t| body(WINDOWS, s, e, t)),
                sweep_hash(|s, e, t| body(LINUX, s, e, t)),
                "{name}"
            );
        }
    }

    #[test]
    fn plus_zero_turns_negative_zero_into_positive_zero_and_keeps_everything_else() {
        for p in PLATFORMS {
            assert_bits(plus_zero(p, -0.0), 0.0);
            assert_bits(plus_zero(p, 0.0), 0.0);
            assert_bits(plus_zero(p, -1.5), -1.5);
            assert_bits(plus_zero(p, 1.0e-45), 1.0e-45);
            assert_bits(plus_zero(p, -1.0e-45), -1.0e-45);
            assert_eq!(plus_zero(p, INF), INF);
            assert!(plus_zero(p, NAN).is_nan());
        }
    }

    /// With `start = −0` and `d·0.5` underflowing to −0, the `+0` from `plus_zero` decides the
    /// sign.
    #[test]
    fn in_out_bounce_second_half_normalises_the_negative_zero_product() {
        let tiny = f32::from_bits(1);
        for p in PLATFORMS {
            assert_bits(in_out_bounce(p, -0.0, -tiny, 0.5), 0.0);
            assert_bits(math::ease_in_out_bounce(p, -0.0, -tiny, 0.5, ID), 0.0);
        }
    }

    #[test]
    fn in_out_quad_at_the_midpoint_is_start_plus_half_the_range() {
        for p in PLATFORMS {
            assert_bits(in_out_quad(p, 1.0, 5.0, 0.5), 3.0);
            assert_bits(in_out_quad(p, -1.0, 0.0, 0.5), -0.5);
            assert_bits(in_out_quad(p, 2.0, 2.0, 0.5), 2.0);
            assert_bits(in_out_quad(p, -0.0, 0.0, 16.5), -0.0);
        }
    }

    /// At `u = 1` the branches round differently: for `c = 2^-149` the first gives `fl(c/2) = 0`
    /// and the second `−0 + c = c`.
    #[test]
    fn in_out_elastic_at_the_midpoint_takes_the_second_branch() {
        let tiny = f32::from_bits(1);
        let three = f32::from_bits(3);
        for p in PLATFORMS {
            assert_bits(table_sin(p, elastic_angle(p, 0.0)), -1.0);
            assert_bits(in_out_elastic(p, 0.0, tiny, 0.5), tiny);
            assert_bits(in_out_elastic(p, 0.0, -tiny, 0.5), -tiny);
            assert_bits(in_out_elastic(p, 0.0, three, 0.5), tiny);
            assert_bits(in_out_elastic(p, 1.0, 1.0 + 4.0, 0.5), 3.0);
            assert_bits(math::ease_in_out_elastic(p, 0.0, tiny, 0.5, ID), tiny);
        }
    }

    /// At these `t` a regrouped angle or a period one step off moves the table index.
    #[test]
    fn the_elastic_curves_read_the_sine_table_at_the_index_of_the_three_step_angle() {
        let curves: [(super::super::test_support::Body, f32, u32); 6] = [
            (in_elastic, 0.2235, 0xbb80_080a),
            (in_elastic, 0.4485, 0x3c3c_cdfe),
            (out_elastic, 0.0375, 0x3ee8_d482),
            (out_elastic, 0.1875, 0x3f98_ad80),
            (in_out_elastic, 0.0125, 0x2e5f_42b2),
            (in_out_elastic, 0.0875, 0xb429_1280),
        ];
        for (ease, t, bits) in curves {
            for p in PLATFORMS {
                assert_eq!(ease(p, 0.0, 1.0, t).to_bits(), bits, "{p:?} t = {t}");
            }
        }
    }

    /// [`sweep_hash`] of each body on the `libm` crate, and the transcendental functions it calls:
    /// on the host's C library it holds within [`HOST_ULPS`] per call of the hashed values.
    #[rustfmt::skip]
    const SWEEP_HASHES: [(&str, u64, u32); 30] = [
        ("in_quad", 0xa740_e060_0249_41dc, 0),
        ("out_quad", 0x6e0c_6154_a7e9_c7a7, 0),
        ("in_out_quad", 0x909d_b438_fa55_cc6e, 0),
        ("in_cubic", 0x2675_f5d3_048b_14b3, 0),
        ("out_cubic", 0xe72e_5c82_74e7_6e21, 0),
        ("in_out_cubic", 0x6efe_e071_ff47_524f, 0),
        ("in_quart", 0x5263_6d29_b0e0_43ea, 0),
        ("out_quart", 0x0d5e_0527_cbcf_b19c, 0),
        ("in_out_quart", 0x5daa_5531_b8de_0d2a, 0),
        ("in_quint", 0xde79_6eb2_6dc0_3eee, 0),
        ("out_quint", 0x0ee9_154b_f207_c53f, 0),
        ("in_out_quint", 0x5850_494c_7ea1_be3a, 0),
        ("in_sine", 0x06f1_143b_6b2b_7f9c, 1),
        ("out_sine", 0x4039_7102_2baa_a718, 1),
        ("in_out_sine", 0xea44_6b19_d375_b6cd, 1),
        ("in_expo", 0xc96e_606d_4d2b_276a, 1),
        ("out_expo", 0x9b20_79bb_01f3_8c8c, 1),
        ("in_out_expo", 0x3658_a08a_433e_3afd, 1),
        ("in_circ", 0xc1a7_1232_d230_380f, 0),
        ("out_circ", 0xc0d4_b980_a419_ea7c, 0),
        ("in_out_circ", 0x99f6_d001_0c7e_a0cb, 0),
        ("in_bounce", 0x516e_7ceb_aeba_2db8, 0),
        ("out_bounce", 0x1647_68d2_6ac7_8d43, 0),
        ("in_out_bounce", 0x8950_180f_031f_c6dc, 0),
        ("in_back", 0x6352_1748_d047_74d5, 0),
        ("out_back", 0x4f45_06bc_aced_9fb8, 0),
        ("in_out_back", 0x5529_758c_e25a_e093, 0),
        ("in_elastic", 0x8a25_6ec9_78f8_7500, 2),
        ("out_elastic", 0x7432_06cd_de1f_3d64, 2),
        ("in_out_elastic", 0x31a9_561c_7193_d82e, 2),
    ];

    #[test]
    fn every_easing_keeps_the_bits_of_its_sweep_on_every_platform() {
        let mut failures = Vec::new();
        for p in PLATFORMS {
            for ((name, body, _), (hashed, expected, calls)) in
                easing_bodies().into_iter().zip(SWEEP_HASHES)
            {
                assert_eq!(name, hashed);
                let strict = with_strict(|| sweep_hash(|s, e, t| body(p, s, e, t)));
                if strict != expected {
                    failures.push(format!("{p:?} {name}: {strict:#018x}"));
                }
                if calls == 0 {
                    assert_eq!(
                        sweep_hash(|s, e, t| body(p, s, e, t)),
                        strict,
                        "{p:?} {name}"
                    );
                    continue;
                }
                for (s, e) in SWEEP_RANGES {
                    for t in sweep_ts() {
                        let (host, strict) = (body(p, s, e, t), with_strict(|| body(p, s, e, t)));
                        if !within_host_ulps_of(host, strict, s.abs().max(e.abs()), calls) {
                            failures
                                .push(format!("{p:?} {name}({s}, {e}, {t}): {host:e}, {strict:e}"));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
