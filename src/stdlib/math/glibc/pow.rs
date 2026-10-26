//! `powf`: `exp2(y·log2 x)`, both halves in `f64`.
//!
//! The `GLIBC_2.2.5` wrapper returns the same values.

use super::exp::{POLY, SHIFT_SCALED, scale_poly};
use super::log::{LOG2_TABLE, split};
use super::{hex, invalid, may_underflow, nan_sum, overflow, quiet, underflow};

/// The quintic of `log1p(r)/ln 2`, highest degree first, without a constant term.
const POLY_LOG2: [f64; 5] = [
    hex("0x1.27616c9496e0bp-2"),
    hex("-0x1.71969a075c67ap-2"),
    hex("0x1.ec70a6ca7baddp-2"),
    hex("-0x1.7154748bef6c8p-1"),
    hex("0x1.71547652ab82bp0"),
];

/// Added to `k`'s bits, it negates `2^(k/32)`: the sign bit sits 47 + 16 bits up.
const SIGN_BIAS: u64 = 1 << (5 + 11);

/// `log2 x` of the bits `ix` of a positive `x`, normalised.
fn log2_inline(ix: u32) -> f64 {
    let (i, k, z) = split(ix);
    let (invc, logc) = LOG2_TABLE[i];
    let r = z.mul_add(invc, -1.0);
    let y0 = logc + f64::from(k);
    let r2 = r * r;
    let y = POLY_LOG2[0].mul_add(r, POLY_LOG2[1]);
    let p = POLY_LOG2[2].mul_add(r, POLY_LOG2[3]);
    let r4 = r2 * r2;
    let q = POLY_LOG2[4].mul_add(r, y0);
    let q = p.mul_add(r2, q);
    y.mul_add(r4, q)
}

/// `2^xd` for `xd` in `[−150, 128]`, negated by a `sign_bias` of [`SIGN_BIAS`].
fn exp2_inline(xd: f64, sign_bias: u64) -> f64 {
    let kd = xd + SHIFT_SCALED;
    let ki = kd.to_bits();
    let kd = kd - SHIFT_SCALED;
    let r = xd - kd;
    // The bias is a multiple of the table size, so the row stays `k`'s.
    scale_poly(ki.wrapping_add(sign_bias), r, &POLY)
}

/// 0 if `iy`, a non-zero finite value's bits, is not an integer, 1 if odd, 2 if even.
fn checkint(iy: u32) -> u32 {
    let e = (iy >> 23) & 0xff;
    if e < 0x7f {
        return 0;
    }
    if e > 0x7f + 23 {
        return 2;
    }
    if iy & ((1 << (0x7f + 23 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1 << (0x7f + 23 - e)) != 0 {
        return 1;
    }
    2
}

/// Whether the bits `ix` are of ±0, ±∞ or a NaN.
fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 2 * 0x7f80_0000 - 1
}

fn is_signalling(x: f32) -> bool {
    (x.to_bits() ^ 0x0040_0000) & 0x7fff_ffff > 0x7fc0_0000
}

/// `powf`. A signalling NaN is quietened even where a quiet one gives 1 (`pow(sNaN, 0)`); a finite
/// negative `x` with a non-integer `y` gives `0xffc00000`.
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut sign_bias = 0;
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        if zeroinfnan(iy) {
            return special_y(x, y);
        }
        if zeroinfnan(ix) {
            return special_x(x, y);
        }
        if ix & 0x8000_0000 != 0 {
            match checkint(iy) {
                0 => return invalid(x),
                1 => sign_bias = SIGN_BIAS,
                _ => {}
            }
            ix &= 0x7fff_ffff;
        }
        if ix < 0x0080_0000 {
            ix = (x * hex("0x1p23") as f32).to_bits() & 0x7fff_ffff;
            ix = ix.wrapping_sub(23 << 23);
        }
    }
    let ylogx = f64::from(y) * log2_inline(ix);
    if (ylogx.to_bits() >> 47 & 0xffff) >= 126.0_f64.to_bits() >> 47 {
        let negative = sign_bias != 0;
        // In round-to-nearest the source's second overflow bound falls through.
        if ylogx > hex("0x1.fffffffd1d571p+6") {
            return overflow(negative);
        }
        if ylogx <= -150.0 {
            return underflow(negative);
        }
        if ylogx < -149.0 {
            return may_underflow(negative);
        }
    }
    exp2_inline(ylogx, sign_bias) as f32
}

/// `y` is ±0, ±∞ or NaN.
#[cold]
fn special_y(x: f32, y: f32) -> f32 {
    let (ix, iy) = (x.to_bits(), y.to_bits());
    if iy.wrapping_mul(2) == 0 {
        return if is_signalling(x) { quiet(x) } else { 1.0 };
    }
    if ix == 0x3f80_0000 {
        return if is_signalling(y) { quiet(y) } else { 1.0 };
    }
    if ix.wrapping_mul(2) > 2 * 0x7f80_0000 || iy.wrapping_mul(2) > 2 * 0x7f80_0000 {
        return nan_sum(x, y);
    }
    if ix.wrapping_mul(2) == 2 * 0x3f80_0000 {
        return 1.0;
    }
    if (ix.wrapping_mul(2) < 2 * 0x3f80_0000) == (iy & 0x8000_0000 == 0) {
        return 0.0;
    }
    y * y
}

/// `x` is ±0, ±∞ or NaN and `y` is not.
#[cold]
fn special_x(x: f32, y: f32) -> f32 {
    let (ix, iy) = (x.to_bits(), y.to_bits());
    let odd = ix & 0x8000_0000 != 0 && checkint(iy) == 1;
    if x.is_nan() {
        // `x·x`, then negated, then `1/x²`: the NaN passes through with only the sign changing.
        let x2 = quiet(x);
        return if odd { -x2 } else { x2 };
    }
    let x2 = if odd { -(x * x) } else { x * x };
    if ix.wrapping_mul(2) == 0 && iy & 0x8000_0000 != 0 {
        return if odd {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    if iy & 0x8000_0000 != 0 { 1.0 / x2 } else { x2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::math::glibc::DEFAULT_NAN;

    #[test]
    fn powf_matches_glibc() {
        for &(x, y, expected) in POWF {
            let got = powf(f32::from_bits(x), f32::from_bits(y)).to_bits();
            assert_eq!(got, expected, "powf({x:#010x}, {y:#010x}) = {got:#010x}");
        }
    }

    #[test]
    fn invalid_and_nan_cases() {
        assert_eq!(powf(-8.0, 1.0 / 3.0).to_bits(), DEFAULT_NAN.to_bits());
        let snan = f32::from_bits(0x7f80_0001);
        assert_eq!(powf(snan, 0.0).to_bits(), 0x7fc0_0001);
        assert_eq!(powf(f32::NAN, 0.0), 1.0);
        assert_eq!(powf(1.0, f32::NAN), 1.0);
        assert_eq!(
            powf(f32::from_bits(0xffc0_0005), 3.0).to_bits(),
            0x7fc0_0005
        );
        assert_eq!(
            powf(f32::from_bits(0xffc0_0005), 2.0).to_bits(),
            0xffc0_0005
        );
    }

    /// Reference results of glibc's `powf`, as `(x, y, powf(x, y))`.
    const POWF: &[(u32, u32, u32)] = &[
        // The build without FMA rounds these the other way.
        (0x4032_678b, 0xc1e6_0c30, 0x2a31_427e),
        (0x4108_7801, 0xc1f5_c837, 0x0ffe_9a53),
        (0x410e_9d3f, 0xc208_2bc9, 0x09bc_f57e),
        (0x0000_0000, 0x0000_0000, 0x3f80_0000),
        (0x0000_0000, 0x7f80_0000, 0x0000_0000),
        (0x0000_0000, 0xff80_0000, 0x7f80_0000),
        (0x0000_0000, 0xffc0_0000, 0xffc0_0000),
        (0x0000_0000, 0x3f80_0000, 0x0000_0000),
        (0x0000_0000, 0x4040_0000, 0x0000_0000),
        (0x0000_0000, 0xbfc0_0000, 0x7f80_0000),
        (0x8000_0000, 0x0000_0000, 0x3f80_0000),
        (0x8000_0000, 0x7f80_0000, 0x0000_0000),
        (0x8000_0000, 0xff80_0000, 0x7f80_0000),
        (0x8000_0000, 0xffc0_0000, 0xffc0_0000),
        (0x8000_0000, 0x3f80_0000, 0x8000_0000),
        (0x8000_0000, 0x4040_0000, 0x8000_0000),
        (0x8000_0000, 0xbfc0_0000, 0x7f80_0000),
        (0x7f80_0000, 0x0000_0000, 0x3f80_0000),
        (0x7f80_0000, 0x7f80_0000, 0x7f80_0000),
        (0x7f80_0000, 0xff80_0000, 0x0000_0000),
        (0x7f80_0000, 0xffc0_0000, 0xffc0_0000),
        (0x7f80_0000, 0x3f80_0000, 0x7f80_0000),
        (0x7f80_0000, 0x4040_0000, 0x7f80_0000),
        (0x7f80_0000, 0xbfc0_0000, 0x0000_0000),
        (0xff80_0000, 0x0000_0000, 0x3f80_0000),
        (0xff80_0000, 0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0xff80_0000, 0x0000_0000),
        (0xff80_0000, 0xffc0_0000, 0xffc0_0000),
        (0xff80_0000, 0x3f80_0000, 0xff80_0000),
        (0xff80_0000, 0x4040_0000, 0xff80_0000),
        (0xff80_0000, 0xbfc0_0000, 0x0000_0000),
        (0x7fc0_0000, 0x0000_0000, 0x3f80_0000),
        (0x7fc0_0000, 0x7f80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0xff80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0xffc0_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0x3f80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0x4040_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0xbfc0_0000, 0x7fc0_0000),
        (0xffc0_0005, 0x0000_0000, 0x3f80_0000),
        (0xffc0_0005, 0x7f80_0000, 0xffc0_0005),
        (0xffc0_0005, 0xff80_0000, 0xffc0_0005),
        (0xffc0_0005, 0xffc0_0000, 0xffc0_0005),
        (0xffc0_0005, 0x3f80_0000, 0x7fc0_0005),
        (0xffc0_0005, 0x4040_0000, 0x7fc0_0005),
        (0xffc0_0005, 0xbfc0_0000, 0xffc0_0005),
        (0x7f80_0001, 0x0000_0000, 0x7fc0_0001),
        (0x7f80_0001, 0x7f80_0000, 0x7fc0_0001),
        (0x7f80_0001, 0xff80_0000, 0x7fc0_0001),
        (0x7f80_0001, 0xffc0_0000, 0x7fc0_0001),
        (0x7f80_0001, 0x3f80_0000, 0x7fc0_0001),
        (0x7f80_0001, 0x4040_0000, 0x7fc0_0001),
        (0x7f80_0001, 0xbfc0_0000, 0x7fc0_0001),
        (0x3f80_0000, 0x0000_0000, 0x3f80_0000),
        (0x3f80_0000, 0x7f80_0000, 0x3f80_0000),
        (0x3f80_0000, 0xff80_0000, 0x3f80_0000),
        (0x3f80_0000, 0xffc0_0000, 0x3f80_0000),
        (0x3f80_0000, 0x3f80_0000, 0x3f80_0000),
        (0x3f80_0000, 0x4040_0000, 0x3f80_0000),
        (0x3f80_0000, 0xbfc0_0000, 0x3f80_0000),
        (0xbf80_0000, 0x0000_0000, 0x3f80_0000),
        (0xbf80_0000, 0x7f80_0000, 0x3f80_0000),
        (0xbf80_0000, 0xff80_0000, 0x3f80_0000),
        (0xbf80_0000, 0xffc0_0000, 0xffc0_0000),
        (0xbf80_0000, 0x3f80_0000, 0xbf80_0000),
        (0xbf80_0000, 0x4040_0000, 0xbf80_0000),
        (0xbf80_0000, 0xbfc0_0000, 0xffc0_0000),
        (0x4000_0000, 0x0000_0000, 0x3f80_0000),
        (0x4000_0000, 0x7f80_0000, 0x7f80_0000),
        (0x4000_0000, 0xff80_0000, 0x0000_0000),
        (0x4000_0000, 0xffc0_0000, 0xffc0_0000),
        (0x4000_0000, 0x3f80_0000, 0x4000_0000),
        (0x4000_0000, 0x4040_0000, 0x4100_0000),
        (0x4000_0000, 0xbfc0_0000, 0x3eb5_04f3),
        (0xc000_0000, 0x0000_0000, 0x3f80_0000),
        (0xc000_0000, 0x7f80_0000, 0x7f80_0000),
        (0xc000_0000, 0xff80_0000, 0x0000_0000),
        (0xc000_0000, 0xffc0_0000, 0xffc0_0000),
        (0xc000_0000, 0x3f80_0000, 0xc000_0000),
        (0xc000_0000, 0x4040_0000, 0xc100_0000),
        (0xc000_0000, 0xbfc0_0000, 0xffc0_0000),
        (0x4000_0000, 0x4300_0000, 0x7f80_0000),
        (0x4000_0000, 0xc316_0000, 0x0000_0000),
        (0x4000_0000, 0xc315_8000, 0x0000_0001),
        (0xc000_0000, 0x42fe_0000, 0xff00_0000),
        (0xc000_0000, 0x4040_0000, 0xc100_0000),
        (0x3f80_0001, 0x4f00_0000, 0x7f80_0000),
        (0x0000_0001, 0x3f00_0000, 0x1a35_04f3),
        (0x40c0_632d, 0x41f6_3328, 0x6747_b225),
        (0x3fe1_d5a7, 0xc0f1_1670, 0x3c63_5851),
        (0x40e7_136d, 0x41b7_1e78, 0x601c_2d22),
        (0x40e6_1003, 0x4186_d8a2, 0x577a_a7b6),
        (0x410d_464c, 0x418f_b936, 0x5baf_5180),
        (0x4094_3e80, 0xc204_5345, 0x1ae3_7f58),
        (0x409c_8ef3, 0xc19c_9544, 0x290f_ba43),
        (0x3fb0_2453, 0xc210_48f0, 0x3727_3e23),
        (0x411b_3d29, 0xc1ec_b9d1, 0x0efe_7796),
        (0x409f_9d05, 0xbe1c_1600, 0x3f48_61ae),
        (0x40a9_1f4c, 0x4218_7f7e, 0x6d3e_4c1f),
        (0x4090_fd5a, 0xc109_ac10, 0x3617_91f5),
        (0x401b_6db0, 0xc12b_b62a, 0x3899_7b67),
        (0x3ffe_970a, 0x4188_2c00, 0x47ec_86b2),
        (0x4038_fbaf, 0xc1d2_5e1f, 0x2b54_f11e),
        (0x405d_2154, 0x4203_1de6, 0x5cc6_9e85),
        (0x411a_c94f, 0xc1b1_e1cb, 0x1b12_e935),
        (0x4116_ac2e, 0x4149_e220, 0x53e2_3e0b),
        (0x3f71_53f6, 0x41bb_41b4, 0x3e80_9d84),
        (0x4087_89dc, 0x404c_1d00, 0x42c7_b89b),
        (0x405b_b0c0, 0x417d_f220, 0x4d97_4073),
        (0x4107_25fc, 0x4112_8540, 0x4d92_12c7),
        (0x4090_b725, 0x4198_970e, 0x5438_2ece),
        (0x40a4_6c50, 0xc187_dab8, 0x2b6f_16f9),
        (0x40fa_e5fa, 0x41cb_aac0, 0x6546_e283),
    ];
}
