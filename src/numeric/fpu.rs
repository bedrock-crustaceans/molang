//! The primitive operations every formula is built from, with the NaN rules and the float → int
//! conversion of the platform's processor ([`Fpu`]).
//!
//! An operation returns the host's result unless it is NaN; then the platform chooses which NaN.
//! Rust's operators leave that choice to the compiler and the host.

use super::Platform;
use super::platform::Fpu;

const QUIET_BIT: u32 = 0x0040_0000;

/// The NaN of an invalid operation on x86-64: sign bit set, quiet, no payload.
pub(crate) const X86_64_DEFAULT_NAN: f32 = f32::from_bits(0xffc0_0000);

/// The NaN of an invalid operation on arm64: sign bit clear, quiet, no payload.
pub(crate) const ARM64_DEFAULT_NAN: f32 = f32::from_bits(0x7fc0_0000);

#[inline]
pub(crate) fn quiet(x: f32) -> f32 {
    f32::from_bits(x.to_bits() | QUIET_BIT)
}

#[inline]
fn is_signalling(x: f32) -> bool {
    x.is_nan() && x.to_bits() & QUIET_BIT == 0
}

/// The NaN of an invalid operation on numbers.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) const fn default_nan(p: Platform) -> f32 {
    p.fpu().default_nan()
}

impl Fpu {
    const fn default_nan(self) -> f32 {
        match self {
            Self::X86_64 => X86_64_DEFAULT_NAN,
            Self::Arm64 => ARM64_DEFAULT_NAN,
        }
    }
}

/// The NaN result of an operation on `a` and `b`, `a` the left operand.
#[inline]
pub(crate) fn nan_of(fpu: Fpu, a: f32, b: f32) -> f32 {
    let kept = match fpu {
        Fpu::X86_64 if a.is_nan() => a,
        Fpu::Arm64 if is_signalling(a) || (a.is_nan() && !is_signalling(b)) => a,
        Fpu::X86_64 | Fpu::Arm64 if b.is_nan() => b,
        Fpu::X86_64 | Fpu::Arm64 => return fpu.default_nan(),
    };
    quiet(kept)
}

/// `r`, or the platform's NaN of `a` and `b` when `r` is NaN.
#[inline]
fn chosen(p: Platform, r: f32, a: f32, b: f32) -> f32 {
    if r.is_nan() { nan_of(p.fpu(), a, b) } else { r }
}

/// `a + b`.
#[inline]
pub(crate) fn add(p: Platform, a: f32, b: f32) -> f32 {
    chosen(p, a + b, a, b)
}

/// `a − b`.
#[inline]
pub(crate) fn sub(p: Platform, a: f32, b: f32) -> f32 {
    chosen(p, a - b, a, b)
}

/// `a·b`.
#[inline]
pub(crate) fn mul(p: Platform, a: f32, b: f32) -> f32 {
    chosen(p, a * b, a, b)
}

/// `a / b`.
#[inline]
pub(crate) fn div(p: Platform, a: f32, b: f32) -> f32 {
    chosen(p, a / b, a, b)
}

/// `a·b + c`, rounded twice.
#[inline]
pub(crate) fn mul_add(p: Platform, a: f32, b: f32, c: f32) -> f32 {
    add(p, mul(p, a, b), c)
}

/// `(a > b) ? a : b`: `b` when either is NaN or both are zeros.
#[inline]
pub(crate) fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `(a < b) ? a : b`: `b` when either is NaN or both are zeros.
#[inline]
pub(crate) fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `x` rounded by `f`; a NaN comes back quietened, as every platform's rounding gives it.
#[cfg(feature = "stdlib")]
#[inline]
fn rounded(x: f32, f: fn(f32) -> f32) -> f32 {
    if x.is_nan() { quiet(x) } else { f(x) }
}

/// Rounds toward −∞.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn floor(x: f32) -> f32 {
    rounded(x, f32::floor)
}

/// Rounds toward +∞.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn ceil(x: f32) -> f32 {
    rounded(x, f32::ceil)
}

/// Rounds half away from zero.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn round(x: f32) -> f32 {
    rounded(x, f32::round)
}

/// Rounds toward zero.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn trunc(x: f32) -> f32 {
    rounded(x, f32::trunc)
}

/// The square root; a negative `x` gives the platform's default NaN.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn sqrt(p: Platform, x: f32) -> f32 {
    let r = x.sqrt();
    if r.is_nan() { first_nan(x, x, p) } else { r }
}

/// `a % b`, the remainder truncated toward zero; a zero `b` or an infinite `a` gives the
/// platform's default NaN.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn rem(p: Platform, a: f32, b: f32) -> f32 {
    let r = a % b;
    if r.is_nan() { first_nan(a, b, p) } else { r }
}

/// The first NaN of `a` and `b` quietened, on every platform; else the default NaN.
#[cfg(feature = "stdlib")]
#[cold]
fn first_nan(a: f32, b: f32, p: Platform) -> f32 {
    if a.is_nan() {
        quiet(a)
    } else if b.is_nan() {
        quiet(b)
    } else {
        default_nan(p)
    }
}

/// Truncating float-to-int. Out of range or NaN: `i32::MIN` on x86-64; saturated, NaN to 0, on
/// arm64.
#[inline]
pub(crate) fn to_int(p: Platform, v: f32) -> i32 {
    match p.fpu() {
        Fpu::X86_64 if !(-2_147_483_648.0..2_147_483_648.0).contains(&v) => i32::MIN,
        // `as` saturates and turns NaN into 0, as arm64's conversion does.
        Fpu::X86_64 | Fpu::Arm64 => v as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;

    const QUIET: f32 = f32::from_bits(0x7fc0_0001);
    const NEGATIVE_QUIET: f32 = f32::from_bits(0xffc0_0002);
    const SIGNALLING: f32 = f32::from_bits(0x7f80_0003);
    const NEGATIVE_SIGNALLING: f32 = f32::from_bits(0xff80_0004);

    fn bits(x: f32) -> u32 {
        x.to_bits()
    }

    #[test]
    fn numbers_are_the_plain_operations_on_every_platform() {
        let values = [
            0.0_f32,
            -0.0,
            1.0,
            -2.5,
            0.1,
            3.0e38,
            -1.0e-45,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for p in PLATFORMS {
            for a in values {
                for b in values {
                    for (got, want) in [
                        (add(p, a, b), a + b),
                        (sub(p, a, b), a - b),
                        (mul(p, a, b), a * b),
                        (div(p, a, b), a / b),
                    ] {
                        if want.is_nan() {
                            assert_eq!(bits(got), bits(default_nan(p)), "{p:?} {a} {b}");
                        } else {
                            assert_eq!(bits(got), bits(want), "{p:?} {a} {b}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn x86_64_returns_the_left_nan_operand_quietened() {
        for p in [LINUX, LINUX_1_26_36, WINDOWS] {
            for op in [add, sub, mul, div] {
                assert_eq!(bits(op(p, NEGATIVE_QUIET, QUIET)), 0xffc0_0002);
                assert_eq!(bits(op(p, QUIET, NEGATIVE_QUIET)), 0x7fc0_0001);
                assert_eq!(bits(op(p, 1.0, QUIET)), 0x7fc0_0001);
                assert_eq!(bits(op(p, QUIET, SIGNALLING)), 0x7fc0_0001);
                assert_eq!(bits(op(p, 2.0, NEGATIVE_SIGNALLING)), 0xffc0_0004);
            }
            assert_eq!(bits(add(p, f32::INFINITY, -f32::INFINITY)), 0xffc0_0000);
        }
    }

    #[test]
    fn arm64_prefers_a_signalling_nan_and_defaults_to_the_positive_nan() {
        let p = ANDROID;
        for op in [add, sub, mul, div] {
            assert_eq!(bits(op(p, NEGATIVE_QUIET, QUIET)), 0xffc0_0002);
            assert_eq!(bits(op(p, QUIET, SIGNALLING)), 0x7fc0_0003);
            assert_eq!(bits(op(p, NEGATIVE_SIGNALLING, SIGNALLING)), 0xffc0_0004);
            assert_eq!(bits(op(p, 1.0, NEGATIVE_QUIET)), 0xffc0_0002);
        }
        assert_eq!(bits(add(p, f32::INFINITY, -f32::INFINITY)), 0x7fc0_0000);
        assert_eq!(bits(mul(p, 0.0, f32::INFINITY)), 0x7fc0_0000);
    }

    #[test]
    fn a_multiply_add_rounds_twice_and_chooses_its_nans_step_by_step() {
        // (1 + 2^-12)² = 1 + 2^-11 + 2^-24: the product rounds to 1 + 2^-11 first.
        let a = 1.0 + 2.0_f32.powi(-12);
        let c = -(1.0 + 2.0_f32.powi(-11));
        for p in PLATFORMS {
            assert_bits(mul_add(p, a, a, c), 0.0);
        }
        // The product's NaN, then the addend: a signalling addend wins on arm64 only.
        assert_eq!(bits(mul_add(LINUX, QUIET, 1.0, SIGNALLING)), 0x7fc0_0001);
        assert_eq!(bits(mul_add(ANDROID, QUIET, 1.0, SIGNALLING)), 0x7fc0_0003);
    }

    #[test]
    fn max_and_min_return_the_second_operand_on_nan_and_equal_zeros() {
        assert_bits(max(NAN, 4.0), 4.0);
        assert!(max(4.0, NAN).is_nan());
        assert_bits(min(NAN, 4.0), 4.0);
        assert!(min(4.0, NAN).is_nan());
        assert_bits(max(0.0, -0.0), -0.0);
        assert_bits(min(-0.0, 0.0), 0.0);
    }

    #[cfg(feature = "stdlib")]
    #[test]
    fn sqrt_and_the_remainder_choose_their_nans() {
        for p in PLATFORMS {
            let invalid = default_nan(p);
            for x in [-1.0, -1.0e-45, -INF] {
                assert_eq!(bits(sqrt(p, x)), bits(invalid));
            }
            assert_bits(sqrt(p, -0.0), -0.0);
            assert_eq!(bits(sqrt(p, NEGATIVE_SIGNALLING)), 0xffc0_0004);
            for (a, b) in [
                (5.0, 0.0),
                (1.0, -0.0),
                (INF, 3.0),
                (-INF, INF),
                (-INF, 0.0),
            ] {
                assert_eq!(bits(rem(p, a, b)), bits(invalid), "{a} % {b}");
            }
            for (first, second, expected) in [
                (QUIET, NEGATIVE_QUIET, 0x7fc0_0001),
                (NEGATIVE_SIGNALLING, QUIET, 0xffc0_0004),
                (QUIET, SIGNALLING, 0x7fc0_0001),
                (SIGNALLING, 0.0, 0x7fc0_0003),
                (INF, NEGATIVE_SIGNALLING, 0xffc0_0004),
            ] {
                assert_eq!(bits(rem(p, first, second)), expected, "{p:?}");
            }
            assert_bits(rem(p, -4.0, 2.0), -0.0);
            assert_bits(rem(p, 7.5, -2.0), 1.5);
            assert_bits(rem(p, -3.0, INF), -3.0);
        }
    }

    #[test]
    fn float_to_int_out_of_range() {
        for p in PLATFORMS {
            assert_eq!(to_int(p, 2.9), 2);
            assert_eq!(to_int(p, -2.9), -2);
            assert_eq!(to_int(p, -0.0), 0);
            assert_eq!(to_int(p, 2_147_483_520.0), 2_147_483_520);
            assert_eq!(to_int(p, -2_147_483_648.0), i32::MIN);
        }
        for v in [NAN, INF, -INF, 2_147_483_648.0, -3.0e9, f32::MAX] {
            assert_eq!(to_int(LINUX, v), i32::MIN, "{v}");
            assert_eq!(to_int(WINDOWS, v), i32::MIN, "{v}");
        }
        assert_eq!(to_int(ANDROID, NAN), 0);
        assert_eq!(to_int(ANDROID, INF), i32::MAX);
        assert_eq!(to_int(ANDROID, 2_147_483_648.0), i32::MAX);
        assert_eq!(to_int(ANDROID, -INF), i32::MIN);
    }
}
