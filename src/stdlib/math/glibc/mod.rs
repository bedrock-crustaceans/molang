//! glibc's single-precision functions, ported bit for bit as glibc 2.41 computes them on an x86-64
//! CPU with FMA.
//!
//! On such a CPU glibc runs a build of `sinf`, `cosf`, `expf`, `exp2f`, `logf`, `log2f` and `powf`
//! compiled for FMA, in which the compiler fused most `a·b + c`: every `mul_add` here stands where
//! that build has a fused instruction, and every other operation is rounded on its own. `asinf`,
//! `acosf`, `atanf` and `atan2f` have a single build and fuse only where their source calls `fma`.
//!
//! NaN results follow x86-64: a NaN argument comes back quiet (of two, the one glibc adds first),
//! an invalid operation gives `0xffc00000` and a compatibility wrapper's error `0x7fc00000`. The
//! functions are named after the symbols they reproduce; a `_compat` suffix marks the
//! `GLIBC_2.2.5` wrapper where its results differ from the current symbol's.

mod exp;
mod log;
mod sincos;

pub use exp::{exp2f, expf};
#[allow(unused_imports, reason = "only the oracle test takes `log2`")]
pub use log::log2f;
pub use log::{logf, logf_compat};
pub use sincos::{cosf, sinf};

const QUIET_BIT: u32 = 0x0040_0000;

/// The NaN of an invalid operation on x86-64, `(x − x) / (x − x)` for a non-NaN `x`.
const DEFAULT_NAN: f32 = f32::from_bits(0xffc0_0000);

/// The NaN a compatibility wrapper returns for a domain error.
const WRAPPER_NAN: f32 = f32::from_bits(0x7fc0_0000);

#[inline]
fn quiet(x: f32) -> f32 {
    f32::from_bits(x.to_bits() | QUIET_BIT)
}

/// `a + b` with a NaN operand: the first NaN, quietened.
#[cold]
fn nan_sum(a: f32, b: f32) -> f32 {
    quiet(if a.is_nan() { a } else { b })
}

/// glibc's `__math_invalidf(x)`, `(x − x) / (x − x)`.
#[cold]
fn invalid(x: f32) -> f32 {
    if x.is_nan() { quiet(x) } else { DEFAULT_NAN }
}

/// glibc's `xflowf`: `y` squared with the sign of the result, rounding as an overflow (`y` large)
/// or underflow (`y` small) does.
#[cold]
fn xflow(negative: bool, y: f32) -> f32 {
    (if negative { -y } else { y }) * y
}

/// glibc's `__math_oflowf`: `(2⁹⁷)²`.
fn overflow(negative: bool) -> f32 {
    xflow(negative, f32::from_bits(0x7000_0000))
}

/// glibc's `__math_uflowf`: `(2⁻⁹⁵)²`.
fn underflow(negative: bool) -> f32 {
    xflow(negative, f32::from_bits(0x1000_0000))
}

/// glibc's `__math_may_uflowf`: the smallest subnormal, `(1.25·2⁻⁷⁵)²` rounded.
fn may_underflow(negative: bool) -> f32 {
    xflow(negative, hex("0x1.4p-75") as f32)
}

/// The `f64` the C hexadecimal floating literal `literal` denotes, such as
/// `-0x1.921fb54442d18p+0`. The literal must be exact in `f64`, and normal or zero.
const fn hex(literal: &str) -> f64 {
    let b = literal.as_bytes();
    let negative = b[0] == b'-';
    let mut i = if negative { 1 } else { 0 };
    assert!(b[i] == b'0' && b[i + 1] == b'x');
    let lead = b[i + 2];
    assert!(lead == b'0' || lead == b'1');
    i += 3;
    let mut fraction: u64 = 0;
    let mut digits = 0;
    if b[i] == b'.' {
        i += 1;
        while b[i] != b'p' {
            let d = b[i];
            let value = match d {
                b'0'..=b'9' => d - b'0',
                b'a'..=b'f' => d - b'a' + 10,
                b'A'..=b'F' => d - b'A' + 10,
                _ => panic!("not a hexadecimal digit"),
            };
            fraction = fraction << 4 | value as u64;
            digits += 1;
            i += 1;
        }
    }
    assert!(digits <= 13 && b[i] == b'p');
    fraction <<= 4 * (13 - digits);
    i += 1;
    let exponent_negative = b[i] == b'-';
    if b[i] == b'-' || b[i] == b'+' {
        i += 1;
    }
    let mut exponent: i64 = 0;
    while i < b.len() {
        exponent = exponent * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if exponent_negative {
        exponent = -exponent;
    }
    let sign = (negative as u64) << 63;
    if lead == b'0' {
        assert!(fraction == 0);
        return f64::from_bits(sign);
    }
    let biased = exponent + 1023;
    assert!(biased > 0 && biased < 0x7ff);
    f64::from_bits(sign | (biased as u64) << 52 | fraction)
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Checks `f` against `(argument bits, result bits)` rows.
    pub(crate) fn check_rows(name: &str, f: fn(f32) -> f32, rows: &[(u32, u32)]) {
        for &(x, expected) in rows {
            let got = f(f32::from_bits(x)).to_bits();
            assert_eq!(
                got, expected,
                "{name}({x:#010x}) = {got:#010x}, expected {expected:#010x}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_literals_denote_their_values() {
        assert_eq!(hex("0x1p+0"), 1.0);
        assert_eq!(hex("0x0p+0").to_bits(), 0);
        assert_eq!(hex("-0x0p+0").to_bits(), 0x8000_0000_0000_0000);
        assert_eq!(hex("0x1.8p+52"), 6_755_399_441_055_744.0);
        assert_eq!(hex("-0x1.921fb54442d18p+0"), -std::f64::consts::FRAC_PI_2);
        assert_eq!(hex("0x1.921fb54442d18p+1"), std::f64::consts::PI);
        assert_eq!(hex("0x1.62e42fefa39efp-1"), std::f64::consts::LN_2);
        assert_eq!(hex("0x1.ffffffd0c621cp-2").to_bits(), 0x3fdf_ffff_fd0c_621c);
        assert_eq!(hex("0x1.0fdaa22168c23p-7").to_bits(), 0x3f80_fdaa_2216_8c23);
        assert_eq!((hex("0x1.4p-75") as f32).to_bits(), 0x1a20_0000);
    }

    #[test]
    fn the_error_helpers_round_as_glibc_does() {
        assert_eq!(overflow(false), f32::INFINITY);
        assert_eq!(overflow(true), f32::NEG_INFINITY);
        assert_eq!(underflow(false).to_bits(), 0);
        assert_eq!(underflow(true).to_bits(), 0x8000_0000);
        assert_eq!(may_underflow(false).to_bits(), 1);
        assert_eq!(may_underflow(true).to_bits(), 0x8000_0001);
        assert_eq!(invalid(f32::INFINITY).to_bits(), 0xffc0_0000);
        assert_eq!(invalid(f32::from_bits(0xff80_0001)).to_bits(), 0xffc0_0001);
        assert_eq!(
            nan_sum(1.0, f32::from_bits(0x7f80_0002)).to_bits(),
            0x7fc0_0002
        );
        assert_eq!(
            nan_sum(f32::from_bits(0xffa0_0000), f32::from_bits(0x7f80_0002)).to_bits(),
            0xffe0_0000
        );
    }
}
