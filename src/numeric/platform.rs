//! [`Platform`]: whose float behaviour an expression reproduces.

/// Whose float behaviour an expression reproduces: a dedicated server or a client, and where it
/// matters the release.
///
/// The host chooses it with
/// [`CompileOptions::with_platform`](crate::compile::CompileOptions::with_platform);
/// the compiled expression records it, and constant folding and evaluation both follow it, so a
/// result is the same on every host. The default is [`Platform::BdsLinux`] at
/// [`LinuxRelease::LATEST`].
///
/// Every platform rounds each multiply, add and divide on its own, in formula order (there is no
/// fused multiply-add), makes every comparison with a NaN false, returns the second operand from
/// `math.min` / `math.max` when either is NaN or both are zeros, lets a NaN divisor pass the
/// division guard and a NaN pass the `math.asin` / `math.acos` clamp, and gives `math.sign(NaN)` =
/// +1. They differ in:
///
/// | | `BdsLinux`, `BdsWindows` | `AndroidArm64` |
/// | --- | --- | --- |
/// | NaN operands of `+ − · /` | the left one, quietened | the first signalling one quietened, else the first quiet one |
/// | an invalid `+ − · /` (`∞ − ∞`, `0·∞`, `0/0`) | `0xffc00000` | `0x7fc00000` |
/// | float → int (`loop` counts, array indices, die counts, the sine-table index) out of range or NaN | `i32::MIN` | saturated; NaN → 0 |
///
/// and in the NaN of an invalid standard function:
///
/// | | `BdsLinux` | `BdsWindows` | `AndroidArm64` |
/// | --- | --- | --- | --- |
/// | `math.sin`, `math.cos` of ±∞; `math.pow`, `math.mod` invalid | `0xffc00000` | `0xffc00000` | `0x7fc00000` |
/// | `math.sqrt` of a negative number | `0xffc00000` | `0xffc00000` | `0x7fc00000` |
/// | `math.ln` of a negative number | per [`LinuxRelease`] | `0xffc00000` | `0x7fc00000` |
/// | `math.asin`, `math.acos` beyond the tolerance | `0x7fc00000` | `0xffc00000` | `0x7fc00000` |
///
/// The transcendental functions and the easings' sine table are glibc's single-precision functions
/// on `BdsLinux`, as glibc 2.41 computes them on a CPU with FMA.
///
/// `AndroidArm64` takes the same functions for `sin`, `cos`, `exp`, `exp2`, `ln` and `pow`, whose
/// algorithms bionic shares with glibc. bionic builds them with contracted multiply-adds and the
/// arm64 round-to-integer instructions, neither of which is modelled, so they can miss its last
/// bit. Its `asin`, `acos`, `atan` and `atan2` come from FreeBSD's code, which the `libm` crate
/// carries; that it matches the device is inferred, not checked. Android parity of the
/// transcendental functions is a best effort.
///
/// `BdsWindows` evaluates them in `f64` and rounds once, the model of its C library's
/// double-precision kernels; its results depend on the server's CPU, and its single-precision
/// `asinf` and `acosf` kernels are not modelled, so either can miss the last bit.
///
/// On `BdsWindows`, ten easings (`in_out_quad`, `in_out_cubic`, `in_out_quart`, `in_out_quint`,
/// `in_out_expo`, `in_circ`, `out_circ`, `in_out_circ`, `in_out_bounce`, `in_out_back`) add the
/// start first and so return the start's NaN of two, and `math.floor`, `math.ceil`, `math.round`,
/// `math.trunc` and `math.abs` compute their post-op as `O + S·x`: values are the same, only the
/// choice between two NaNs differs.
///
/// ```
/// # #[cfg(feature = "stdlib")]
/// # {
/// use molang::compile::{CompileOptions, compile};
/// use molang::numeric::Platform;
/// use molang::version::MolangVersion;
///
/// let sign_of_nan = |platform| {
///     let options = CompileOptions::server(MolangVersion::LATEST).with_platform(platform);
///     let expr = compile("math.copy_sign(1, math.ln(-1))", &options).into_result().unwrap().0;
///     expr.as_constant()
/// };
/// assert_eq!(sign_of_nan(Platform::default()), Some(1.0));
/// assert_eq!(sign_of_nan(Platform::BdsWindows), Some(-1.0));
/// # }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Platform {
    /// The dedicated server on Linux x86-64: SSE arithmetic, `i32::MIN` for a float → int
    /// conversion out of range, and the release's C library functions.
    BdsLinux(LinuxRelease),
    /// The dedicated server on Windows x86-64: the arithmetic of [`Platform::BdsLinux`], with the
    /// NaN differences listed above.
    BdsWindows,
    /// The Android client on arm64: the formulas of the servers with the arm64 NaN rules and a
    /// saturating float → int conversion.
    AndroidArm64,
}

impl Default for Platform {
    /// [`Platform::BdsLinux`] at [`LinuxRelease::LATEST`].
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A release of the Linux dedicated server, where its float behaviour depends on it.
///
/// The behaviour changed between 1.26.36 and 1.26.52; the releases in between were not analysed.
/// A host on one of them that needs `math.ln(-1)`'s sign exact must check which variant it matches.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum LinuxRelease {
    /// 1.26.36 and earlier: `math.ln` of a negative number gives `0xffc00000`.
    V1_26_36,
    /// 1.26.52 and later: `math.ln` of a negative number gives `0x7fc00000`.
    V1_26_52,
}

impl LinuxRelease {
    /// The newest release analysed: [`LinuxRelease::V1_26_52`].
    pub const LATEST: Self = Self::V1_26_52;
}

impl Default for LinuxRelease {
    /// [`LinuxRelease::LATEST`].
    fn default() -> Self {
        Self::LATEST
    }
}

/// Whose rules choose the NaN of an arithmetic operation and convert a float to an int.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Fpu {
    /// SSE: the left NaN operand, `0xffc00000` for an invalid operation, `i32::MIN` out of range.
    X86_64,
    /// arm64: the first signalling NaN operand, else the first quiet one, `0x7fc00000` for an
    /// invalid operation, saturating conversions.
    Arm64,
}

impl Platform {
    /// [`Platform::default`], usable in a `const fn`.
    pub(crate) const DEFAULT: Self = Self::BdsLinux(LinuxRelease::LATEST);

    /// The processor rules of this platform.
    #[inline]
    pub(crate) const fn fpu(self) -> Fpu {
        match self {
            Self::BdsLinux(_) | Self::BdsWindows => Fpu::X86_64,
            Self::AndroidArm64 => Fpu::Arm64,
        }
    }

    /// Whether some easings and the rounding post-ops take their operands in the other order:
    /// the same values, another NaN of two.
    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) const fn reorders_operands(self) -> bool {
        matches!(self, Self::BdsWindows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_newest_linux_server() {
        assert_eq!(
            Platform::default(),
            Platform::BdsLinux(LinuxRelease::V1_26_52)
        );
        assert_eq!(LinuxRelease::LATEST, LinuxRelease::V1_26_52);
        assert_eq!(LinuxRelease::default(), LinuxRelease::LATEST);
        assert!(LinuxRelease::V1_26_36 < LinuxRelease::V1_26_52);
    }

    #[test]
    fn the_servers_share_the_x86_64_rules() {
        assert_eq!(Platform::default().fpu(), Fpu::X86_64);
        assert_eq!(
            Platform::BdsLinux(LinuxRelease::V1_26_36).fpu(),
            Fpu::X86_64
        );
        assert_eq!(Platform::BdsWindows.fpu(), Fpu::X86_64);
        assert_eq!(Platform::AndroidArm64.fpu(), Fpu::Arm64);
    }
}
