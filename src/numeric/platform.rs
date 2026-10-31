//! [`Platform`]: whose float behaviour an expression reproduces.

/// Whose float behaviour an expression reproduces: the host's own, or with the `platform-quirks`
/// feature the processor rules of a dedicated server or a client.
///
/// The host chooses it with
/// [`CompileOptions::with_platform`](crate::compile::CompileOptions::with_platform);
/// the compiled expression records it, and constant folding and evaluation both follow it. The
/// default is [`Platform::Native`].
///
/// Every platform keeps the engine's rules: each multiply, add and divide is rounded on its own,
/// in formula order (there is no fused multiply-add), every comparison with a NaN is false,
/// `math.min` / `math.max` return the second operand when either is NaN or both are zeros, a NaN
/// divisor passes the division guard and a NaN passes the `math.asin` / `math.acos` clamp, and
/// `math.sign(NaN)` = +1.
///
/// [`Platform::Native`] leaves the rest to the host and to Rust: the NaN of an operation is
/// whichever the host's processor returns, and a float → int conversion (array indices, die
/// counts, the sine-table index) is Rust's `as i32`, saturating with NaN → 0. Its results are host
/// dependent: `math.copy_sign(1, math.ln(-1))` folds to −1 on x86-64 and +1 on arm64, a folded
/// constant carries the NaN of the machine that compiled it, and Rust and LLVM do not guarantee a
/// NaN's sign or payload even on one host. `platform-quirks` gives host-independent bits.
///
/// The `platform-quirks` feature adds the emulated platforms, which differ in:
///
/// | | `BdsLinux`, `BdsWindows` | `AndroidArm64` |
/// | --- | --- | --- |
/// | NaN operands of `+ − · /` and of a transcendental function | the left one, quietened | the first signalling one quietened, else the first quiet one |
/// | an invalid operation (`∞ − ∞`, `0·∞`, `0/0`, `math.sqrt(-1)`, `math.ln(-1)`, `math.sin` of ±∞, an invalid `math.pow` or `math.mod`) | `0xffc00000` | `0x7fc00000` |
/// | float → int out of range or NaN | `i32::MIN` | saturated; NaN → 0 |
///
/// On `BdsWindows`, ten easings (`in_out_quad`, `in_out_cubic`, `in_out_quart`, `in_out_quint`,
/// `in_out_expo`, `in_circ`, `out_circ`, `in_out_circ`, `in_out_bounce`, `in_out_back`) add the
/// start first and so return the start's NaN of two, and `math.floor`, `math.ceil`, `math.round`,
/// `math.trunc` and `math.abs` compute their post-op as `O + S·x`: values are the same, only the
/// choice between two NaNs differs.
///
/// On every emulated platform `math.asin` and `math.acos` beyond the tolerance give
/// `0x7fc00000`, and `math.pow` of a NaN to an odd integer gives that NaN with its sign bit
/// cleared. Both are measured on BDS Linux only; that BDS Windows and Android give the same is
/// assumed. The transcendental functions themselves are not part of the platform: they come from
/// the host's C library, or with the `strict-math` feature from the `libm` crate.
///
/// ```
/// # #[cfg(all(feature = "stdlib", feature = "platform-quirks"))]
/// # {
/// use molang::compile::{CompileOptions, compile};
/// use molang::numeric::Platform;
/// use molang::version::MolangVersion;
///
/// let sign_of_nan = |platform| -> Result<Option<f32>, Box<dyn std::error::Error>> {
///     let options = CompileOptions::server(MolangVersion::LATEST).with_platform(platform);
///     let (expr, _) = compile("math.copy_sign(1, math.ln(-1))", &options).into_result()?;
///     Ok(expr.as_constant())
/// };
/// assert_eq!(sign_of_nan(Platform::BdsLinux)?, Some(-1.0));
/// assert_eq!(sign_of_nan(Platform::AndroidArm64)?, Some(1.0));
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Platform {
    /// The host's processor and Rust's float → int conversion.
    #[default]
    Native,
    /// The dedicated server on Linux x86-64: SSE arithmetic and `i32::MIN` for a float → int
    /// conversion out of range.
    #[cfg(feature = "platform-quirks")]
    #[cfg_attr(docsrs, doc(cfg(feature = "platform-quirks")))]
    BdsLinux,
    /// The dedicated server on Windows x86-64: the arithmetic of [`Platform::BdsLinux`], with the
    /// operand order listed above.
    #[cfg(feature = "platform-quirks")]
    #[cfg_attr(docsrs, doc(cfg(feature = "platform-quirks")))]
    BdsWindows,
    /// The Android client on arm64: the formulas of the servers with the arm64 NaN rules and a
    /// saturating float → int conversion.
    #[cfg(feature = "platform-quirks")]
    #[cfg_attr(docsrs, doc(cfg(feature = "platform-quirks")))]
    AndroidArm64,
}

/// Whose rules choose the NaN of an arithmetic operation and convert a float to an int.
#[cfg(feature = "platform-quirks")]
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
    pub(crate) const DEFAULT: Self = Self::Native;

    /// The emulated processor rules of this platform; `None` for the host's own.
    #[cfg(feature = "platform-quirks")]
    #[inline]
    pub(crate) const fn fpu(self) -> Option<Fpu> {
        match self {
            Self::Native => None,
            Self::BdsLinux | Self::BdsWindows => Some(Fpu::X86_64),
            Self::AndroidArm64 => Some(Fpu::Arm64),
        }
    }

    /// Whether some easings and the rounding post-ops take their operands in the other order:
    /// the same values, another NaN of two.
    #[cfg(feature = "stdlib")]
    #[inline]
    pub(crate) const fn reorders_operands(self) -> bool {
        match self {
            Self::Native => false,
            #[cfg(feature = "platform-quirks")]
            Self::BdsLinux | Self::AndroidArm64 => false,
            #[cfg(feature = "platform-quirks")]
            Self::BdsWindows => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_host() {
        assert_eq!(Platform::default(), Platform::Native);
        assert_eq!(Platform::DEFAULT, Platform::Native);
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn the_servers_share_the_x86_64_rules() {
        assert_eq!(Platform::Native.fpu(), None);
        assert_eq!(Platform::BdsLinux.fpu(), Some(Fpu::X86_64));
        assert_eq!(Platform::BdsWindows.fpu(), Some(Fpu::X86_64));
        assert_eq!(Platform::AndroidArm64.fpu(), Some(Fpu::Arm64));
    }
}
