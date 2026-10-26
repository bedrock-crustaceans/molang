//! The C library function each platform calls for a transcendental, `sqrt` or the remainder, and the
//! NaN it returns for an argument outside its domain.
//!
//! Every platform's functions are modelled by the bodies in [`transcendental`] for now: each `f64`
//! function rounded to `f32`. A platform-specific body replaces its arm of the `match`.

use super::transcendental;
use crate::numeric::platform::Libm;
use crate::numeric::{LinuxRelease, Platform};

/// The quiet NaN with the sign bit clear.
const POSITIVE_NAN: f32 = f32::from_bits(0x7fc0_0000);

/// The quiet NaN with the sign bit set.
const NEGATIVE_NAN: f32 = f32::from_bits(0xffc0_0000);

/// The NaN of `sinf`, `cosf`, `powf`, `fmodf` and `sqrtf` outside their domain.
#[inline]
const fn domain_nan(p: Platform) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt => NEGATIVE_NAN,
        Libm::Bionic => POSITIVE_NAN,
    }
}

/// The NaN of `logf` of a negative number: the 1.26.52 server binds glibc's compatibility
/// wrapper, which returns `log(−∞)`'s positive NaN.
#[inline]
const fn ln_nan(p: Platform) -> f32 {
    match p.libm() {
        Libm::Glibc(LinuxRelease::V1_26_52) | Libm::Bionic => POSITIVE_NAN,
        Libm::Glibc(LinuxRelease::V1_26_36) | Libm::Ucrt => NEGATIVE_NAN,
    }
}

/// The NaN of `asinf` / `acosf` beyond ±1.
#[inline]
const fn inverse_trig_nan(p: Platform) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Bionic => POSITIVE_NAN,
        Libm::Ucrt => NEGATIVE_NAN,
    }
}

/// `sinf`.
#[inline]
pub(crate) fn sin(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::sin(x, domain_nan(p)),
    }
}

/// `cosf`.
#[inline]
pub(crate) fn cos(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::cos(x, domain_nan(p)),
    }
}

/// `asinf`.
#[inline]
pub(crate) fn asin(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::asin(x, inverse_trig_nan(p)),
    }
}

/// `acosf`.
#[inline]
pub(crate) fn acos(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::acos(x, inverse_trig_nan(p)),
    }
}

/// `atanf`.
#[inline]
pub(crate) fn atan(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::atan(x),
    }
}

/// `atan2f(y, x)`.
#[inline]
pub(crate) fn atan2(p: Platform, y: f32, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::atan2(y, x),
    }
}

/// `expf`.
#[inline]
pub(crate) fn exp(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::exp(x),
    }
}

/// `exp2f` on the servers; the Android client calls `powf(2, x)` instead.
#[inline]
pub(crate) fn exp2(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt => transcendental::exp2(x),
        Libm::Bionic => pow(p, 2.0, x),
    }
}

/// `logf`.
#[inline]
pub(crate) fn ln(p: Platform, x: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::ln(x, ln_nan(p)),
    }
}

/// `powf(a, b)`.
#[inline]
pub(crate) fn pow(p: Platform, a: f32, b: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::pow(a, b, domain_nan(p)),
    }
}

/// The square root: the processor's for `x ≥ 0`, else `sqrtf` on the servers; Android's processor
/// instruction gives its default NaN.
#[inline]
pub(crate) fn sqrt(p: Platform, x: f32) -> f32 {
    transcendental::sqrt(x, domain_nan(p))
}

/// `fmodf(a, b)`.
#[inline]
pub(crate) fn rem(p: Platform, a: f32, b: f32) -> f32 {
    match p.libm() {
        Libm::Glibc(_) | Libm::Ucrt | Libm::Bionic => transcendental::rem(a, b, domain_nan(p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;

    fn bits(x: f32) -> u32 {
        x.to_bits()
    }

    #[test]
    fn each_platform_chooses_the_nan_of_each_invalid_argument() {
        #[rustfmt::skip]
        let rows: [(Platform, [u32; 8]); 4] = [
            //               sin(∞)       cos(−∞)      ln(−1)       pow(−8, ⅓)   sqrt(−1)     asin(2)      acos(−2)     fmod(1, 0)
            (LINUX,         [0xffc0_0000, 0xffc0_0000, 0x7fc0_0000, 0xffc0_0000, 0xffc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0xffc0_0000]),
            (LINUX_1_26_36, [0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0xffc0_0000]),
            (WINDOWS,       [0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000]),
            (ANDROID,       [0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000]),
        ];
        for (p, expected) in rows {
            let got = [
                sin(p, INF),
                cos(p, -INF),
                ln(p, -1.0),
                pow(p, -8.0, 1.0 / 3.0),
                sqrt(p, -1.0),
                asin(p, 2.0),
                acos(p, -2.0),
                rem(p, 1.0, 0.0),
            ];
            assert_eq!(got.map(bits), expected, "{p:?}");
        }
    }

    #[test]
    fn a_nan_argument_comes_back_quiet_on_every_platform() {
        let negative_signalling = f32::from_bits(0xff80_0003);
        for p in PLATFORMS {
            for got in [
                sin(p, negative_signalling),
                cos(p, negative_signalling),
                asin(p, negative_signalling),
                acos(p, negative_signalling),
                atan(p, negative_signalling),
                exp(p, negative_signalling),
                exp2(p, negative_signalling),
                ln(p, negative_signalling),
                sqrt(p, negative_signalling),
                pow(p, negative_signalling, 2.0),
                pow(p, 2.0, negative_signalling),
                atan2(p, negative_signalling, 1.0),
                rem(p, negative_signalling, 1.0),
            ] {
                assert_eq!(bits(got), 0xffc0_0003, "{p:?}");
            }
            assert_eq!(bits(pow(p, negative_signalling, 3.0)), 0x7fc0_0003);
        }
    }

    #[test]
    fn android_takes_exp2_from_pow() {
        for x in [-10.0_f32, -1.0, -0.5, 0.0, 0.3, 1.0, 7.25, 128.0, -150.0] {
            assert_bits(exp2(ANDROID, x), pow(ANDROID, 2.0, x));
            assert_bits(exp2(LINUX, x), transcendental::exp2(x));
        }
    }
}
