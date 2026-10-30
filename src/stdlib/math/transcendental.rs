//! The transcendental functions: `std`'s `f32` methods, which call the host's C library, or with
//! the `strict-math` feature the `libm` crate, the same bits on every host.
//!
//! The platform chooses every NaN they return: a NaN argument quietened (of two, by its
//! [`nan_of`](fpu::nan_of) rule), else its default NaN. Two exceptions hold on every platform:
//! `asin` and `acos` beyond ±1 give the positive NaN, and `pow` of a NaN to an odd integer gives
//! that NaN quietened with its sign bit cleared. Both are measured on BDS Linux only; that BDS
//! Windows and Android give the same is assumed.

use crate::numeric::{Platform, fpu};

const POSITIVE_NAN: f32 = f32::from_bits(0x7fc0_0000);

/// `r`, or when it is NaN the platform's NaN of the arguments `a` and `b`.
#[inline]
fn chosen(p: Platform, r: f32, a: f32, b: f32) -> f32 {
    if r.is_nan() {
        fpu::nan_of(p.fpu(), a, b)
    } else {
        r
    }
}

/// The host's C library, through `std`.
#[cfg(any(test, not(feature = "strict-math")))]
mod host {
    #[inline]
    pub(super) fn sin(x: f32) -> f32 {
        x.sin()
    }

    #[inline]
    pub(super) fn cos(x: f32) -> f32 {
        x.cos()
    }

    #[inline]
    pub(super) fn asin(x: f32) -> f32 {
        x.asin()
    }

    #[inline]
    pub(super) fn acos(x: f32) -> f32 {
        x.acos()
    }

    #[inline]
    pub(super) fn atan(x: f32) -> f32 {
        x.atan()
    }

    #[inline]
    pub(super) fn atan2(y: f32, x: f32) -> f32 {
        y.atan2(x)
    }

    #[inline]
    pub(super) fn exp(x: f32) -> f32 {
        x.exp()
    }

    #[inline]
    pub(super) fn exp2(x: f32) -> f32 {
        x.exp2()
    }

    #[inline]
    pub(super) fn ln(x: f32) -> f32 {
        x.ln()
    }

    #[inline]
    pub(super) fn pow(a: f32, b: f32) -> f32 {
        a.powf(b)
    }
}

/// The `libm` crate.
#[cfg(any(test, feature = "strict-math"))]
mod strict {
    pub(super) use libm::{
        acosf as acos, asinf as asin, atan2f as atan2, atanf as atan, cosf as cos, exp2f as exp2,
        expf as exp, logf as ln, powf as pow, sinf as sin,
    };
}

#[cfg(all(not(test), not(feature = "strict-math")))]
use host as library;
#[cfg(all(not(test), feature = "strict-math"))]
use strict as library;

/// The unit tests can run a computation on the `libm` crate in either mode ([`with_strict`]).
#[cfg(test)]
mod library {
    use super::{host, strict};
    use std::cell::Cell;

    thread_local! {
        pub(super) static STRICT: Cell<bool> = const { Cell::new(cfg!(feature = "strict-math")) };
    }

    macro_rules! switched {
        ($($name:ident($($arg:ident),*);)*) => {$(
            pub(super) fn $name($($arg: f32),*) -> f32 {
                if STRICT.get() { strict::$name($($arg),*) } else { host::$name($($arg),*) }
            }
        )*};
    }

    switched! {
        sin(x); cos(x); asin(x); acos(x); atan(x); atan2(y, x); exp(x); exp2(x); ln(x); pow(a, b);
    }
}

/// `f` on the `libm` crate, whichever mode the crate is built in.
#[cfg(test)]
pub(crate) fn with_strict<T>(f: impl FnOnce() -> T) -> T {
    let before = library::STRICT.replace(true);
    let result = f();
    library::STRICT.set(before);
    result
}

/// The sine of `x` radians.
#[inline]
pub(crate) fn sin(p: Platform, x: f32) -> f32 {
    chosen(p, library::sin(x), x, x)
}

/// The cosine of `x` radians.
#[inline]
pub(crate) fn cos(p: Platform, x: f32) -> f32 {
    chosen(p, library::cos(x), x, x)
}

/// `r`, or when it is NaN the quietened `x` or else the positive NaN.
#[inline]
fn inverse_trig(r: f32, x: f32) -> f32 {
    match (r.is_nan(), x.is_nan()) {
        (false, _) => r,
        (true, true) => fpu::quiet(x),
        (true, false) => POSITIVE_NAN,
    }
}

/// The arcsine in radians; beyond ±1 the positive NaN.
#[inline]
pub(crate) fn asin(x: f32) -> f32 {
    inverse_trig(library::asin(x), x)
}

/// The arccosine in radians; beyond ±1 the positive NaN.
#[inline]
pub(crate) fn acos(x: f32) -> f32 {
    inverse_trig(library::acos(x), x)
}

/// The arctangent in radians.
#[inline]
pub(crate) fn atan(p: Platform, x: f32) -> f32 {
    chosen(p, library::atan(x), x, x)
}

/// The angle of `(x, y)` in radians.
#[inline]
pub(crate) fn atan2(p: Platform, y: f32, x: f32) -> f32 {
    chosen(p, library::atan2(y, x), y, x)
}

/// e^x.
#[inline]
pub(crate) fn exp(p: Platform, x: f32) -> f32 {
    chosen(p, library::exp(x), x, x)
}

/// 2^x.
#[inline]
pub(crate) fn exp2(p: Platform, x: f32) -> f32 {
    chosen(p, library::exp2(x), x, x)
}

/// The natural logarithm.
#[inline]
pub(crate) fn ln(p: Platform, x: f32) -> f32 {
    chosen(p, library::ln(x), x, x)
}

/// `a` to the power `b`; a NaN `a` with an odd integer `b` gives that NaN quietened with its sign
/// bit cleared.
#[inline]
pub(crate) fn pow(p: Platform, a: f32, b: f32) -> f32 {
    let r = library::pow(a, b);
    if r.is_nan() && a.is_nan() && (b % 2.0).abs() == 1.0 {
        fpu::quiet(a.abs())
    } else {
        chosen(p, r, a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;

    const SIGNALLING: f32 = f32::from_bits(0x7f80_0001);

    fn bits(x: f32) -> u32 {
        x.to_bits()
    }

    #[test]
    fn each_platform_chooses_the_nan_of_each_invalid_argument() {
        for (p, nan) in [
            (LINUX, 0xffc0_0000),
            (WINDOWS, 0xffc0_0000),
            (ANDROID, 0x7fc0_0000),
        ] {
            let got = [
                sin(p, INF),
                cos(p, -INF),
                ln(p, -1.0),
                ln(p, -INF),
                pow(p, -8.0, 1.0 / 3.0),
            ];
            assert_eq!(got.map(bits), [nan; 5], "{p:?}");
        }
    }

    #[test]
    fn asin_and_acos_beyond_one_give_the_positive_nan() {
        for x in [1.000_001, -1.000_001, 2.0, INF, -INF] {
            assert_eq!(bits(asin(x)), 0x7fc0_0000, "asin({x})");
            assert_eq!(bits(acos(x)), 0x7fc0_0000, "acos({x})");
        }
        assert_eq!(bits(asin(f32::from_bits(0xff80_0003))), 0xffc0_0003);
        assert_eq!(bits(acos(f32::from_bits(0xffc0_0000))), 0xffc0_0000);
    }

    #[test]
    fn pow_of_a_nan_to_an_odd_integer_clears_its_sign() {
        for p in PLATFORMS {
            for b in [1.0, -1.0, 3.0, -5.0] {
                assert_eq!(bits(pow(p, f32::from_bits(0xffc0_0000), b)), 0x7fc0_0000);
                assert_eq!(bits(pow(p, f32::from_bits(0xff80_0002), b)), 0x7fc0_0002);
            }
            for b in [2.0, 0.5, -2.0, 16_777_217.0] {
                assert_eq!(bits(pow(p, f32::from_bits(0xffc0_0000), b)), 0xffc0_0000);
            }
        }
    }

    #[test]
    fn a_nan_argument_comes_back_quiet_on_every_platform() {
        let negative_signalling = f32::from_bits(0xff80_0003);
        for p in PLATFORMS {
            for got in [
                sin(p, negative_signalling),
                cos(p, negative_signalling),
                asin(negative_signalling),
                acos(negative_signalling),
                atan(p, negative_signalling),
                exp(p, negative_signalling),
                exp2(p, negative_signalling),
                ln(p, negative_signalling),
                pow(p, negative_signalling, 2.0),
                pow(p, 2.0, negative_signalling),
                atan2(p, negative_signalling, 1.0),
                atan2(p, 1.0, negative_signalling),
            ] {
                assert_eq!(bits(got), 0xffc0_0003, "{p:?}");
            }
        }
    }

    #[test]
    fn of_two_nan_arguments_arm64_takes_the_signalling_one() {
        let quiet = f32::from_bits(0xffc0_0005);
        for (p, expected) in [(LINUX, 0xffc0_0005), (ANDROID, 0x7fc0_0001)] {
            assert_eq!(bits(atan2(p, quiet, SIGNALLING)), expected, "{p:?}");
            assert_eq!(bits(pow(p, quiet, SIGNALLING)), expected, "{p:?}");
        }
    }

    #[test]
    fn pow_keeps_its_special_cases() {
        for p in PLATFORMS {
            assert_bits(pow(p, NAN, 0.0), 1.0);
            assert_bits(pow(p, 1.0, NAN), 1.0);
            assert_bits(pow(p, -INF, 0.5), INF);
            assert_bits(pow(p, -2.0, 3.0), -8.0);
        }
    }

    #[test]
    fn ordinary_arguments_are_not_nan() {
        for p in PLATFORMS {
            assert_eq!(ln(p, 0.0), -INF);
            assert_eq!(ln(p, -0.0), -INF);
            assert_bits(atan2(p, 0.0, 0.0), 0.0);
            assert_bits(asin(1.0), std::f32::consts::FRAC_PI_2);
            assert_bits(atan2(p, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
            assert_eq!(exp(p, 0.0), 1.0);
            assert_eq!(exp2(p, -3.0), 0.125);
            assert_eq!(exp2(p, 10.0), 1024.0);
        }
    }

    #[cfg(feature = "strict-math")]
    #[test]
    fn strict_math_is_the_libm_crate() {
        let x = f32::from_bits(0x400c_dd30);
        for p in PLATFORMS {
            assert_bits(sin(p, x), libm::sinf(x));
            assert_bits(pow(p, x, 0.3), libm::powf(x, 0.3));
            assert_bits(exp2(p, -x), libm::exp2f(-x));
        }
    }
}
