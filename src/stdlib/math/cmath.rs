//! The C library each platform calls for a transcendental, `sqrt` or the remainder: [`Libm`] picks
//! the body and the NaN of an invalid argument.

use super::{glibc, transcendental};
use crate::numeric::platform::Fpu;
use crate::numeric::{LinuxRelease, Platform, fpu};

/// The quiet NaN with the sign bit set.
const NEGATIVE_NAN: f32 = f32::from_bits(0xffc0_0000);

/// The bodies that compute `sin` … `pow`, and the easings' sine table through `sin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Libm {
    /// The `f64` functions of [`transcendental`], rounded once, with `0xffc00000` for every
    /// invalid operation, `asin` and `acos` of `|x| > 1` included.
    RoundedF64,
    /// glibc's single-precision functions as its FMA build on x86-64 computes them, with their own
    /// NaNs: `0xffc00000` for `sin`/`cos` of ±∞ and an invalid `pow`, `0x7fc00000` for `asin`/
    /// `acos` of `|x| > 1`, and for `ln` of a negative the one of the bound `logf` symbol.
    Glibc(LogfSymbol),
    /// The same algorithms with an arm64 processor's NaNs: `0x7fc00000` for every invalid
    /// operation, and of two NaN arguments the first signalling one.
    GlibcArm64,
}

/// The `logf` symbol a build linked against glibc binds; the other functions give the same values
/// through either version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LogfSymbol {
    /// `logf@GLIBC_2.2.5`, the compatibility wrapper: `ln` of a negative is `0x7fc00000`.
    Compat,
    /// `logf@GLIBC_2.27`: `ln` of a negative is `0xffc00000`.
    Current,
}

/// The library `p` calls. `logf` of a negative number gives the positive NaN on 1.26.52 and the
/// negative one on 1.26.36; bionic shares glibc's algorithms for most functions, so the Android
/// client takes the ports with its own NaNs as the closest match.
#[inline]
pub(crate) const fn libm(p: Platform) -> Libm {
    match p {
        Platform::BdsLinux(LinuxRelease::V1_26_52) => Libm::Glibc(LogfSymbol::Compat),
        Platform::BdsLinux(LinuxRelease::V1_26_36) => Libm::Glibc(LogfSymbol::Current),
        Platform::BdsWindows => Libm::RoundedF64,
        Platform::AndroidArm64 => Libm::GlibcArm64,
    }
}

/// `r`, or when it is NaN the arm64 NaN of the arguments `a` and `b`.
#[inline]
fn on_arm64(r: f32, a: f32, b: f32) -> f32 {
    if r.is_nan() {
        fpu::nan_of(Fpu::Arm64, a, b)
    } else {
        r
    }
}

impl Libm {
    pub(crate) fn sin(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::sin(x, NEGATIVE_NAN),
            Self::Glibc(_) => glibc::sinf(x),
            Self::GlibcArm64 => on_arm64(glibc::sinf(x), x, x),
        }
    }

    pub(crate) fn cos(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::cos(x, NEGATIVE_NAN),
            Self::Glibc(_) => glibc::cosf(x),
            Self::GlibcArm64 => on_arm64(glibc::cosf(x), x, x),
        }
    }

    pub(crate) fn asin(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::asin(x, NEGATIVE_NAN),
            Self::Glibc(_) | Self::GlibcArm64 => glibc::asinf(x),
        }
    }

    pub(crate) fn acos(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::acos(x, NEGATIVE_NAN),
            Self::Glibc(_) | Self::GlibcArm64 => glibc::acosf(x),
        }
    }

    pub(crate) fn atan(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::atan(x),
            Self::Glibc(_) | Self::GlibcArm64 => glibc::atanf(x),
        }
    }

    pub(crate) fn atan2(self, y: f32, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::atan2(y, x),
            Self::Glibc(_) => glibc::atan2f(y, x),
            Self::GlibcArm64 => on_arm64(glibc::atan2f(y, x), y, x),
        }
    }

    pub(crate) fn exp(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::exp(x),
            Self::Glibc(_) | Self::GlibcArm64 => glibc::expf(x),
        }
    }

    pub(crate) fn exp2(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::exp2(x),
            Self::Glibc(_) | Self::GlibcArm64 => glibc::exp2f(x),
        }
    }

    pub(crate) fn ln(self, x: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::ln(x, NEGATIVE_NAN),
            Self::Glibc(LogfSymbol::Compat) => glibc::logf_compat(x),
            Self::Glibc(LogfSymbol::Current) => glibc::logf(x),
            Self::GlibcArm64 => on_arm64(glibc::logf(x), x, x),
        }
    }

    pub(crate) fn pow(self, a: f32, b: f32) -> f32 {
        match self {
            Self::RoundedF64 => transcendental::pow(a, b, NEGATIVE_NAN),
            Self::Glibc(_) => glibc::powf(a, b),
            Self::GlibcArm64 => on_arm64(glibc::powf(a, b), a, b),
        }
    }
}

/// `sinf`.
#[inline]
pub(crate) fn sin(p: Platform, x: f32) -> f32 {
    libm(p).sin(x)
}

/// `cosf`.
#[inline]
pub(crate) fn cos(p: Platform, x: f32) -> f32 {
    libm(p).cos(x)
}

/// `asinf`.
#[inline]
pub(crate) fn asin(p: Platform, x: f32) -> f32 {
    libm(p).asin(x)
}

/// `acosf`.
#[inline]
pub(crate) fn acos(p: Platform, x: f32) -> f32 {
    libm(p).acos(x)
}

/// `atanf`.
#[inline]
pub(crate) fn atan(p: Platform, x: f32) -> f32 {
    libm(p).atan(x)
}

/// `atan2f(y, x)`.
#[inline]
pub(crate) fn atan2(p: Platform, y: f32, x: f32) -> f32 {
    libm(p).atan2(y, x)
}

/// `expf`.
#[inline]
pub(crate) fn exp(p: Platform, x: f32) -> f32 {
    libm(p).exp(x)
}

/// `exp2f` on the servers; the Android client calls `powf(2, x)` instead.
#[inline]
pub(crate) fn exp2(p: Platform, x: f32) -> f32 {
    match p {
        Platform::AndroidArm64 => pow(p, 2.0, x),
        _ => libm(p).exp2(x),
    }
}

/// `logf`.
#[inline]
pub(crate) fn ln(p: Platform, x: f32) -> f32 {
    libm(p).ln(x)
}

/// `powf(a, b)`.
#[inline]
pub(crate) fn pow(p: Platform, a: f32, b: f32) -> f32 {
    libm(p).pow(a, b)
}

/// The square root: the processor's for `x ≥ 0`; a negative `x` gives the platform's default NaN.
#[inline]
pub(crate) fn sqrt(p: Platform, x: f32) -> f32 {
    transcendental::sqrt(x, fpu::default_nan(p))
}

/// `fmodf(a, b)`; an invalid one gives the platform's default NaN.
#[inline]
pub(crate) fn rem(p: Platform, a: f32, b: f32) -> f32 {
    transcendental::rem(a, b, fpu::default_nan(p))
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
    fn the_servers_and_android_take_the_glibc_ports() {
        let x = f32::from_bits(0x400c_dd30);
        for p in [LINUX, LINUX_1_26_36, ANDROID] {
            assert_bits(sin(p, x), glibc::sinf(x));
            assert_bits(ln(p, 3.0), glibc::logf(3.0));
            assert_bits(atan2(p, x, 0.5), glibc::atan2f(x, 0.5));
        }
        assert_bits(sin(WINDOWS, x), transcendental::sin(x, NAN));
        assert_bits(ln(WINDOWS, 3.0), transcendental::ln(3.0, NAN));
    }

    #[test]
    fn the_linux_releases_differ_only_in_the_nan_of_ln() {
        let arguments = [
            -INF,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            1e-30,
            0.3,
            0.5,
            1.0,
            2.0,
            13.0,
            1e30,
            INF,
            NAN,
            f32::from_bits(0x3288_85a3),
            f32::from_bits(0x3982_6222),
            f32::from_bits(0x3d8d_6b23),
            f32::from_bits(0xbd8d_6b23),
        ];
        for x in arguments {
            for y in arguments {
                assert_eq!(bits(atan2(LINUX, y, x)), bits(atan2(LINUX_1_26_36, y, x)));
                assert_eq!(bits(pow(LINUX, y, x)), bits(pow(LINUX_1_26_36, y, x)));
            }
            for f in [sin, cos, asin, acos, atan, exp, exp2] {
                assert_eq!(bits(f(LINUX, x)), bits(f(LINUX_1_26_36, x)), "{x}");
            }
            if x >= 0.0 || x.is_nan() {
                assert_eq!(bits(ln(LINUX, x)), bits(ln(LINUX_1_26_36, x)), "{x}");
            }
        }
        assert_ne!(bits(ln(LINUX, -1.0)), bits(ln(LINUX_1_26_36, -1.0)));
    }

    #[test]
    fn android_takes_exp2_from_pow() {
        for x in [-10.0_f32, -1.0, -0.5, 0.0, 0.3, 1.0, 7.25, 128.0, -150.0] {
            assert_bits(exp2(ANDROID, x), pow(ANDROID, 2.0, x));
            assert_bits(exp2(LINUX, x), glibc::exp2f(x));
            assert_bits(exp2(WINDOWS, x), transcendental::exp2(x));
        }
    }

    #[test]
    fn each_library_chooses_its_nans() {
        for (libm, sin_inf, ln_negative, asin_beyond) in [
            (Libm::RoundedF64, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000),
            (
                Libm::Glibc(LogfSymbol::Compat),
                0xffc0_0000,
                0x7fc0_0000,
                0x7fc0_0000,
            ),
            (
                Libm::Glibc(LogfSymbol::Current),
                0xffc0_0000,
                0xffc0_0000,
                0x7fc0_0000,
            ),
            (Libm::GlibcArm64, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000),
        ] {
            assert_eq!(bits(libm.sin(INF)), sin_inf, "{libm:?}");
            assert_eq!(bits(libm.cos(-INF)), sin_inf, "{libm:?}");
            assert_eq!(bits(libm.ln(-1.0)), ln_negative, "{libm:?}");
            assert_eq!(bits(libm.asin(1.001)), asin_beyond, "{libm:?}");
            assert_eq!(bits(libm.acos(-INF)), asin_beyond, "{libm:?}");
            assert_eq!(bits(libm.asin(SIGNALLING)), 0x7fc0_0001, "{libm:?}");
            assert_eq!(bits(libm.acos(-NAN)), 0xffc0_0000, "{libm:?}");
            assert_eq!(bits(libm.pow(-8.0, 0.5)), sin_inf, "{libm:?}");
            assert_eq!(bits(libm.atan(SIGNALLING)), 0x7fc0_0001, "{libm:?}");
            assert_eq!(bits(libm.atan2(1.0, 0.0)), 0x3fc9_0fdb, "{libm:?}");
            assert_eq!(libm.exp(0.0), 1.0, "{libm:?}");
            assert_eq!(libm.exp2(-3.0), 0.125, "{libm:?}");
        }
    }
}
