//! The primitive operations every formula is built from.
//!
//! An operation returns the host's result unless it is NaN or a float → int conversion out of
//! range. On [`Platform::Native`] those stay the host's and Rust's; with the `platform-quirks`
//! feature an emulated platform chooses the NaN and the integer by its processor's rules.

use super::Platform;

#[cfg(any(feature = "stdlib", feature = "platform-quirks"))]
const QUIET_BIT: u32 = 0x0040_0000;

#[cfg(any(feature = "stdlib", feature = "platform-quirks"))]
#[inline]
pub(crate) fn quiet(x: f32) -> f32 {
    f32::from_bits(x.to_bits() | QUIET_BIT)
}

#[cfg(feature = "platform-quirks")]
use quirks as rules;

#[cfg(not(feature = "platform-quirks"))]
use native as rules;

pub(crate) use rules::{chosen, to_int};
#[cfg(feature = "stdlib")]
pub(crate) use rules::{inverse_trig, pow_nan};

/// The processor rules of the emulated platforms.
#[cfg(feature = "platform-quirks")]
mod quirks {
    use super::{Platform, QUIET_BIT, quiet};
    use crate::numeric::platform::Fpu;

    /// The NaN of an invalid operation on x86-64: sign bit set, quiet, no payload.
    pub(crate) const X86_64_DEFAULT_NAN: f32 = f32::from_bits(0xffc0_0000);

    /// The NaN of an invalid operation on arm64: sign bit clear, quiet, no payload.
    pub(crate) const ARM64_DEFAULT_NAN: f32 = f32::from_bits(0x7fc0_0000);

    #[inline]
    fn is_signalling(x: f32) -> bool {
        x.is_nan() && x.to_bits() & QUIET_BIT == 0
    }

    impl Fpu {
        pub(crate) const fn default_nan(self) -> f32 {
            match self {
                Self::X86_64 => X86_64_DEFAULT_NAN,
                Self::Arm64 => ARM64_DEFAULT_NAN,
            }
        }

        /// The NaN result of an operation on `a` and `b`, `a` the left operand.
        #[inline]
        fn nan_of(self, a: f32, b: f32) -> f32 {
            let kept = match self {
                Self::X86_64 if a.is_nan() => a,
                Self::Arm64 if is_signalling(a) || (a.is_nan() && !is_signalling(b)) => a,
                Self::X86_64 | Self::Arm64 if b.is_nan() => b,
                Self::X86_64 | Self::Arm64 => return self.default_nan(),
            };
            quiet(kept)
        }
    }

    /// `r`, or the platform's NaN of `a` and `b` when `r` is NaN.
    #[inline]
    pub(crate) fn chosen(p: Platform, r: f32, a: f32, b: f32) -> f32 {
        match p.fpu() {
            Some(fpu) if r.is_nan() => fpu.nan_of(a, b),
            _ => r,
        }
    }

    /// `r`, or when it is NaN the first NaN of `a` and `b` quietened, on every emulated
    /// platform, else the default NaN.
    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) fn first_nan(p: Platform, r: f32, a: f32, b: f32) -> f32 {
        match p.fpu() {
            Some(_) if r.is_nan() && a.is_nan() => quiet(a),
            Some(_) if r.is_nan() && b.is_nan() => quiet(b),
            Some(fpu) if r.is_nan() => fpu.default_nan(),
            _ => r,
        }
    }

    /// `r` for `asin(x)` or `acos(x)`; when it is NaN, the quietened `x` or else `0x7fc00000`.
    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) fn inverse_trig(p: Platform, r: f32, x: f32) -> f32 {
        match p.fpu() {
            Some(_) if r.is_nan() && x.is_nan() => quiet(x),
            Some(_) if r.is_nan() => ARM64_DEFAULT_NAN,
            _ => r,
        }
    }

    /// `r` for `pow(a, b)`; a NaN `a` with an odd integer `b` gives that NaN quietened with its
    /// sign bit cleared.
    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) fn pow_nan(p: Platform, r: f32, a: f32, b: f32) -> f32 {
        if p.fpu().is_some() && r.is_nan() && a.is_nan() && (b % 2.0).abs() == 1.0 {
            quiet(a.abs())
        } else {
            chosen(p, r, a, b)
        }
    }

    /// Truncating float-to-int. Out of range or NaN: `i32::MIN` on x86-64; saturated, NaN to 0,
    /// on arm64 and natively.
    #[inline]
    pub(crate) fn to_int(p: Platform, v: f32) -> i32 {
        match p.fpu() {
            Some(Fpu::X86_64) if !(-2_147_483_648.0..2_147_483_648.0).contains(&v) => i32::MIN,
            _ => v as i32,
        }
    }
}

/// The host's own results.
#[cfg(not(feature = "platform-quirks"))]
mod native {
    use super::Platform;

    #[inline]
    pub(crate) fn chosen(_: Platform, r: f32, _: f32, _: f32) -> f32 {
        r
    }

    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) fn first_nan(_: Platform, r: f32, _: f32, _: f32) -> f32 {
        r
    }

    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) fn inverse_trig(_: Platform, r: f32, _: f32) -> f32 {
        r
    }

    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) fn pow_nan(_: Platform, r: f32, _: f32, _: f32) -> f32 {
        r
    }

    /// Truncating float-to-int: saturated, NaN to 0.
    #[inline]
    pub(crate) fn to_int(_: Platform, v: f32) -> i32 {
        v as i32
    }
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

/// `x` rounded by `f`; a NaN comes back quietened, as every processor's rounding gives it.
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

/// The square root; a negative `x` gives NaN.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn sqrt(p: Platform, x: f32) -> f32 {
    rules::first_nan(p, x.sqrt(), x, x)
}

/// `a % b`, the remainder truncated toward zero; a zero `b` or an infinite `a` gives NaN.
#[cfg(feature = "stdlib")]
#[inline]
pub(crate) fn rem(p: Platform, a: f32, b: f32) -> f32 {
    rules::first_nan(p, a % b, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;

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
                            assert!(got.is_nan(), "{p:?} {a} {b}");
                        } else {
                            assert_eq!(bits(got), bits(want), "{p:?} {a} {b}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn natively_an_operation_is_the_host_operation() {
        let nan = f32::from_bits(0xffc0_0002);
        for (a, b) in [(nan, 1.0), (1.0, nan), (INF, -INF), (0.0, INF), (0.0, 0.0)] {
            assert_eq!(bits(add(NATIVE, a, b)), bits(a + b));
            assert_eq!(bits(sub(NATIVE, a, b)), bits(a - b));
            assert_eq!(bits(mul(NATIVE, a, b)), bits(a * b));
            assert_eq!(bits(div(NATIVE, a, b)), bits(a / b));
        }
    }

    #[test]
    fn a_multiply_add_rounds_twice() {
        // (1 + 2^-12)² = 1 + 2^-11 + 2^-24: the product rounds to 1 + 2^-11 first.
        let a = 1.0 + 2.0_f32.powi(-12);
        let c = -(1.0 + 2.0_f32.powi(-11));
        for p in PLATFORMS {
            assert_bits(mul_add(p, a, a, c), 0.0);
        }
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
    fn sqrt_and_the_remainder_are_nan_where_invalid() {
        for p in PLATFORMS {
            for x in [-1.0, -1.0e-45, -INF, NAN] {
                assert!(sqrt(p, x).is_nan());
            }
            assert_bits(sqrt(p, -0.0), -0.0);
            for (a, b) in [
                (5.0, 0.0),
                (1.0, -0.0),
                (INF, 3.0),
                (-INF, INF),
                (-INF, 0.0),
                (NAN, 1.0),
                (1.0, NAN),
            ] {
                assert!(rem(p, a, b).is_nan(), "{a} % {b}");
            }
            assert_bits(rem(p, -4.0, 2.0), -0.0);
            assert_bits(rem(p, 7.5, -2.0), 1.5);
            assert_bits(rem(p, -3.0, INF), -3.0);
        }
    }

    #[test]
    fn float_to_int_in_range_truncates() {
        for p in PLATFORMS {
            assert_eq!(to_int(p, 2.9), 2);
            assert_eq!(to_int(p, -2.9), -2);
            assert_eq!(to_int(p, -0.0), 0);
            assert_eq!(to_int(p, 2_147_483_520.0), 2_147_483_520);
            assert_eq!(to_int(p, -2_147_483_648.0), i32::MIN);
        }
    }

    #[test]
    fn natively_float_to_int_saturates() {
        assert_eq!(to_int(NATIVE, NAN), 0);
        assert_eq!(to_int(NATIVE, INF), i32::MAX);
        assert_eq!(to_int(NATIVE, 2_147_483_648.0), i32::MAX);
        assert_eq!(to_int(NATIVE, -INF), i32::MIN);
    }

    #[cfg(feature = "platform-quirks")]
    mod quirks {
        use super::*;
        use crate::numeric::platform::Fpu;

        const QUIET: f32 = f32::from_bits(0x7fc0_0001);
        const NEGATIVE_QUIET: f32 = f32::from_bits(0xffc0_0002);
        const SIGNALLING: f32 = f32::from_bits(0x7f80_0003);
        const NEGATIVE_SIGNALLING: f32 = f32::from_bits(0xff80_0004);

        fn default_nan(p: Platform) -> f32 {
            p.fpu().map_or(NAN, Fpu::default_nan)
        }

        #[test]
        fn an_invalid_operation_gives_the_default_nan() {
            for p in EMULATED {
                for got in [
                    add(p, INF, -INF),
                    sub(p, INF, INF),
                    mul(p, 0.0, INF),
                    div(p, 0.0, 0.0),
                    div(p, INF, -INF),
                ] {
                    assert_eq!(bits(got), bits(default_nan(p)), "{p:?}");
                }
            }
        }

        #[test]
        fn x86_64_returns_the_left_nan_operand_quietened() {
            for p in [LINUX, WINDOWS] {
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
        fn a_multiply_add_chooses_its_nans_step_by_step() {
            // The product's NaN, then the addend: a signalling addend wins on arm64 only.
            assert_eq!(bits(mul_add(LINUX, QUIET, 1.0, SIGNALLING)), 0x7fc0_0001);
            assert_eq!(bits(mul_add(ANDROID, QUIET, 1.0, SIGNALLING)), 0x7fc0_0003);
        }

        #[cfg(feature = "stdlib")]
        #[test]
        fn sqrt_and_the_remainder_choose_their_nans() {
            for p in EMULATED {
                let invalid = default_nan(p);
                for x in [-1.0, -1.0e-45, -INF] {
                    assert_eq!(bits(sqrt(p, x)), bits(invalid));
                }
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
            }
        }

        #[test]
        fn float_to_int_out_of_range() {
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
}
