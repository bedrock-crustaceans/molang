//! [`Platform`]: whose float behaviour an expression reproduces.

/// Whose float behaviour an expression reproduces: the processor rules of a dedicated server or a
/// client.
///
/// The host chooses it with
/// [`CompileOptions::with_platform`](crate::compile::CompileOptions::with_platform);
/// the compiled expression records it, and constant folding and evaluation both follow it. The
/// default is [`Platform::BdsLinux`].
///
/// Every platform rounds each multiply, add and divide on its own, in formula order (there is no
/// fused multiply-add), makes every comparison with a NaN false, returns the second operand from
/// `math.min` / `math.max` when either is NaN or both are zeros, lets a NaN divisor pass the
/// division guard and a NaN pass the `math.asin` / `math.acos` clamp, and gives `math.sign(NaN)` =
/// +1. They differ in:
///
/// | | `BdsLinux`, `BdsWindows` | `AndroidArm64` |
/// | --- | --- | --- |
/// | NaN operands of `+ − · /` and of a transcendental function | the left one, quietened | the first signalling one quietened, else the first quiet one |
/// | an invalid operation (`∞ − ∞`, `0·∞`, `0/0`, `math.sqrt(-1)`, `math.ln(-1)`, `math.sin` of ±∞, an invalid `math.pow` or `math.mod`) | `0xffc00000` | `0x7fc00000` |
/// | float → int (`loop` counts, array indices, die counts, the sine-table index) out of range or NaN | `i32::MIN` | saturated; NaN → 0 |
///
/// On `BdsWindows`, ten easings (`in_out_quad`, `in_out_cubic`, `in_out_quart`, `in_out_quint`,
/// `in_out_expo`, `in_circ`, `out_circ`, `in_out_circ`, `in_out_bounce`, `in_out_back`) add the
/// start first and so return the start's NaN of two, and `math.floor`, `math.ceil`, `math.round`,
/// `math.trunc` and `math.abs` compute their post-op as `O + S·x`: values are the same, only the
/// choice between two NaNs differs.
///
/// On every platform `math.asin` and `math.acos` beyond the tolerance give `0x7fc00000`, and
/// `math.pow` of a NaN to an odd integer gives that NaN with its sign bit cleared. Both are measured
/// on BDS Linux only; that BDS Windows and Android give the same is assumed. The transcendental functions themselves are not part of the platform: they come from the
/// host's C library, or with the `strict-math` feature from the `libm` crate.
///
/// ```
/// # #[cfg(feature = "stdlib")]
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
/// assert_eq!(sign_of_nan(Platform::default())?, Some(-1.0));
/// assert_eq!(sign_of_nan(Platform::AndroidArm64)?, Some(1.0));
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Platform {
    /// The dedicated server on Linux x86-64: SSE arithmetic and `i32::MIN` for a float → int
    /// conversion out of range.
    BdsLinux,
    /// The dedicated server on Windows x86-64: the arithmetic of [`Platform::BdsLinux`], with the
    /// operand order listed above.
    BdsWindows,
    /// The Android client on arm64: the formulas of the servers with the arm64 NaN rules and a
    /// saturating float → int conversion.
    AndroidArm64,
}

impl Default for Platform {
    /// [`Platform::BdsLinux`].
    fn default() -> Self {
        Self::DEFAULT
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
    pub(crate) const DEFAULT: Self = Self::BdsLinux;

    /// The processor rules of this platform.
    #[inline]
    pub(crate) const fn fpu(self) -> Fpu {
        match self {
            Self::BdsLinux | Self::BdsWindows => Fpu::X86_64,
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
    fn the_default_is_the_linux_server() {
        assert_eq!(Platform::default(), Platform::BdsLinux);
    }

    #[test]
    fn the_servers_share_the_x86_64_rules() {
        assert_eq!(Platform::BdsLinux.fpu(), Fpu::X86_64);
        assert_eq!(Platform::BdsWindows.fpu(), Fpu::X86_64);
        assert_eq!(Platform::AndroidArm64.fpu(), Fpu::Arm64);
    }
}
