//! `asinf` and `acosf`, correctly rounded: a fast polynomial whose result is kept when an error
//! bound cannot change its rounding, else an accurate one.
//!
//! Both are the `GLIBC_2.2.5` wrappers, the only symbols: `|x| > 1` gives `0x7fc00000`.

use super::{WRAPPER_NAN, hex, quiet};

/// The degree-11 polynomial of the accurate paths, in `z`.
fn poly12(z: f64, c: &[f64; 12]) -> f64 {
    let z2 = z * z;
    let z4 = z2 * z2;
    let mut c0 = c[0] + z * c[1];
    let c2 = c[2] + z * c[3];
    let mut c4 = c[4] + z * c[5];
    let c6 = c[6] + z * c[7];
    let mut c8 = c[8] + z * c[9];
    let c10 = c[10] + z * c[11];
    c0 += c2 * z2;
    c4 += c6 * z2;
    c8 += z2 * c10;
    c0 += z4 * (c4 + z4 * c8);
    c0
}

/// The odd degree-31 polynomial of the fast paths, `z·p(z²)`.
fn poly16(z: f64, b: &[f64; 16]) -> f64 {
    let z2 = z * z;
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let z16 = z8 * z8;
    z * ((((b[0] + z2 * b[1]) + z4 * (b[2] + z2 * b[3]))
        + z8 * ((b[4] + z2 * b[5]) + z4 * (b[6] + z2 * b[7])))
        + z16
            * (((b[8] + z2 * b[9]) + z4 * (b[10] + z2 * b[11]))
                + z8 * ((b[12] + z2 * b[13]) + z4 * (b[14] + z2 * b[15]))))
}

/// The accurate path's polynomial for `|x|` below 0.5.
const C_SMALL: [f64; 12] = [
    hex("0x1.555555555529cp-3"),
    hex("0x1.333333337e0ddp-4"),
    hex("0x1.6db6db3b4465ep-5"),
    hex("0x1.f1c72e13ac306p-6"),
    hex("0x1.6e89cebe06bc4p-6"),
    hex("0x1.1c6dcf5289094p-6"),
    hex("0x1.c6dbbcc7c6315p-7"),
    hex("0x1.8f8dc2615e996p-7"),
    hex("0x1.a5833b7bf15e8p-8"),
    hex("0x1.43f44ace1665cp-6"),
    hex("-0x1.0fb17df881c73p-6"),
    hex("0x1.07520c026b2d6p-5"),
];

/// The accurate path's polynomial in `1 − |x|`, for `|x|` from 0.5.
const C_LARGE: [f64; 12] = [
    hex("0x1.6a09e667f3bcbp+0"),
    hex("0x1.e2b7dddff2db9p-4"),
    hex("0x1.b27247ab42dbcp-6"),
    hex("0x1.02995cc4e0744p-7"),
    hex("0x1.5ffb0276ec8eap-9"),
    hex("0x1.033885a928decp-10"),
    hex("0x1.911f2be23f8c7p-12"),
    hex("0x1.4c3c55d2437fdp-13"),
    hex("0x1.af477e1d7b461p-15"),
    hex("0x1.abd6bdff67dcbp-15"),
    hex("-0x1.1717e86d0fa28p-16"),
    hex("0x1.6ff526de46023p-16"),
];

const ASIN_FAST: [f64; 16] = [
    hex("0x1.0000000000005p+0"),
    hex("0x1.55557aeca105dp-3"),
    hex("0x1.3314ec3db7d12p-4"),
    hex("0x1.775738a5a6f92p-5"),
    hex("0x1.5d5f7ce1c8538p-8"),
    hex("0x1.605c6d58740fp-2"),
    hex("-0x1.5728b732d73c6p+1"),
    hex("0x1.f152170f151ebp+3"),
    hex("-0x1.f962ea3ca992ep+5"),
    hex("0x1.71971e17375ap+7"),
    hex("-0x1.860512b4ba23p+8"),
    hex("0x1.26a3b8d4bdb14p+9"),
    hex("-0x1.36f2ea5698b51p+9"),
    hex("0x1.b3d722aebfa2ep+8"),
    hex("-0x1.6cf89703b1289p+7"),
    hex("0x1.1518af6a65e2dp+5"),
];

const ACOS_FAST: [f64; 16] = [
    hex("0x1.fffffffd9ccb8p-1"),
    hex("0x1.5555c94838007p-3"),
    hex("0x1.32ded4b7c20fap-4"),
    hex("0x1.8566df703309ep-5"),
    hex("-0x1.980c959bec9a3p-6"),
    hex("0x1.56fbb04998344p-1"),
    hex("-0x1.403d8e4c49f52p+2"),
    hex("0x1.b06c3e9f311eap+4"),
    hex("-0x1.9ea97c4e2c21fp+6"),
    hex("0x1.200b8261cc61bp+8"),
    hex("-0x1.2274c2799a5c7p+9"),
    hex("0x1.a558a59cc19d3p+9"),
    hex("-0x1.aca4b6a529ffp+9"),
    hex("0x1.228744703f813p+9"),
    hex("-0x1.d7dbb0b322228p+7"),
    hex("0x1.5c2018c0c0105p+5"),
];

const PI2: f64 = hex("0x1.921fb54442d18p+0");

/// The result for `|x| > 1` or NaN.
#[cold]
fn beyond_one(x: f32) -> f32 {
    if x.is_nan() { quiet(x) } else { WRAPPER_NAN }
}

/// `asinf`.
pub fn asinf(x: f32) -> f32 {
    let xs = f64::from(x);
    let ax = x.to_bits() << 1;
    if ax > 0x7f << 24 {
        return beyond_one(x);
    }
    if ax < 0x7ec2_9000 {
        if ax < 115 << 24 {
            return x.mul_add(hex("0x1p-25") as f32, x);
        }
        let r = poly16(xs, &ASIN_FAST);
        let ub = r as f32;
        let lb = (r - xs * hex("0x1.efa8ebp-31")) as f32;
        if ub == lb {
            return ub;
        }
    }
    let r = if ax < 0x7e << 24 {
        let z2 = xs * xs;
        xs + (xs * z2) * poly12(z2, &C_SMALL)
    } else {
        if ax == 0x7e55_688a {
            return (hex("0x1.75b8a2p-1") as f32).copysign(x) + (hex("0x1p-26") as f32).copysign(x);
        }
        if ax == 0x7e10_7434 {
            return (hex("0x1.1f4b64p-1") as f32).copysign(x) + (hex("0x1p-26") as f32).copysign(x);
        }
        let z = 1.0 - xs.abs();
        let s = z.sqrt();
        (PI2 - s * poly12(z, &C_LARGE)).copysign(xs)
    };
    r as f32
}

/// `acosf`.
pub fn acosf(x: f32) -> f32 {
    let xs = f64::from(x);
    let t = x.to_bits();
    let ax = t << 1;
    if ax >= 0x7f << 24 {
        return match t {
            0x3f80_0000 => 0.0,
            0xbf80_0000 => hex("0x1.921fb6p+1") as f32 + hex("-0x1p-24") as f32,
            _ => beyond_one(x),
        };
    }
    if ax < 0x7ec2_a1dc {
        if ax <= 0x4000_0000 {
            return PI2 as f32;
        }
        let r = poly16(xs, &ACOS_FAST);
        let ub = (hex("0x1.921fb54574191p+0") - r) as f32;
        let lb = (hex("0x1.921fb543118ap+0") - r) as f32;
        if ub == lb {
            return ub;
        }
    }
    let r = if ax < 0x7e << 24 {
        if t == 0x3288_85a3 {
            return hex("0x1.921fb6p+0") as f32 + hex("0x1p-25") as f32;
        }
        if t == 0x3982_6222 {
            return hex("0x1.920f6ap+0") as f32 + hex("0x1p-25") as f32;
        }
        let x2 = xs * xs;
        (PI2 - xs) - (xs * x2) * poly12(x2, &C_SMALL)
    } else {
        let z = 1.0 - xs.abs();
        let s = z.sqrt().copysign(xs);
        let o = if t >> 31 == 0 {
            0.0
        } else {
            hex("0x1.921fb54442d18p+1")
        };
        o + s * poly12(z, &C_LARGE)
    };
    r as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::math::glibc::test_support::check_rows;

    #[test]
    fn asinf_matches_glibc() {
        check_rows("asinf", asinf, ASINF);
    }

    #[test]
    fn acosf_matches_glibc() {
        check_rows("acosf", acosf, ACOSF);
    }

    /// Recorded from glibc 2.41 on x86-64.
    const ASINF: &[(u32, u32)] = &[
        (0x0000_0000, 0x0000_0000),
        (0x8000_0000, 0x8000_0000),
        (0x7f80_0000, 0x7fc0_0000),
        (0xff80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0x0000_0001),
        (0x807f_ffff, 0x807f_ffff),
        (0x0080_0000, 0x0080_0000),
        (0x7f7f_ffff, 0x7fc0_0000),
        (0xff7f_ffff, 0x7fc0_0000),
        (0x3f80_0000, 0x3fc9_0fdb),
        (0xbf80_0000, 0xbfc9_0fdb),
        (0x3f61_4800, 0x3f89_b630),
        (0x3f61_47ff, 0x3f89_b62f),
        (0x3980_0000, 0x3980_0000),
        (0x397f_ffff, 0x397f_ffff),
        (0x3f00_0000, 0x3f06_0a92),
        (0x3eff_ffff, 0x3f06_0a91),
        (0x3f2a_b445, 0x3f3a_dc51),
        (0xbf08_3a1a, 0xbf0f_a5b2),
        (0x3f80_0001, 0x7fc0_0000),
        (0xbf5e_aa78, 0xbf87_02d4),
        (0xbf3c_ba46, 0xbf54_3594),
        (0x3f14_d866, 0x3f1e_d7e0),
        (0xbf60_5dec, 0xbf88_c177),
        (0x3ef2_f070, 0x3efd_2012),
        (0xbed2_eea4, 0xbed9_6822),
        (0xbead_a290, 0xbeb1_25fa),
        (0xbe21_34f8, 0xbe21_e15d),
        (0xbe17_5998, 0xbe17_e812),
        (0xbf30_5f0e, 0xbf42_91fb),
        (0x3f21_1a4c, 0x3f2e_3fdb),
        (0xbe95_dd94, 0xbe98_17af),
    ];

    /// Recorded from glibc 2.41 on x86-64.
    const ACOSF: &[(u32, u32)] = &[
        (0x0000_0000, 0x3fc9_0fdb),
        (0x8000_0000, 0x3fc9_0fdb),
        (0x7f80_0000, 0x7fc0_0000),
        (0xff80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0x3fc9_0fdb),
        (0x807f_ffff, 0x3fc9_0fdb),
        (0x0080_0000, 0x3fc9_0fdb),
        (0x7f7f_ffff, 0x7fc0_0000),
        (0xff7f_ffff, 0x7fc0_0000),
        (0x3f80_0000, 0x0000_0000),
        (0xbf80_0000, 0x4049_0fdb),
        (0x3f61_50ee, 0x3efd_410c),
        (0x3f61_50ed, 0x3efd_4110),
        (0x2000_0000, 0x3fc9_0fdb),
        (0x2000_0001, 0x3fc9_0fdb),
        (0x3f00_0000, 0x3f86_0a92),
        (0x3288_85a3, 0x3fc9_0fdb),
        (0x3982_6222, 0x3fc9_07b5),
        (0x3f80_0001, 0x7fc0_0000),
        (0xbf5e_aa78, 0x4028_0957),
        (0xbf3c_ba46, 0x4019_9552),
        (0x3f14_d866, 0x3f73_47d5),
        (0xbf60_5dec, 0x4028_e8a9),
        (0x3ef2_f070, 0x3f89_c7d6),
        (0xbed2_eea4, 0x3fff_69e3),
        (0xbead_a290, 0x3ff5_5959),
        (0xbe21_34f8, 0x3fdd_4c06),
        (0xbe17_5998, 0x3fdc_0cdd),
        (0xbf30_5f0e, 0x4015_2c6c),
        (0x3f21_1a4c, 0x3f63_dfdb),
        (0xbe95_dd94, 0x3fef_15c6),
    ];
}
