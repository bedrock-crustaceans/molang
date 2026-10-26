//! `atanf` and `atan2f`, correctly rounded: a rational approximation in `f64`, and for `atan2f`
//! a double-double fallback near rounding boundaries.

use super::{hex, nan_sum, quiet};

const PI2: f64 = hex("0x1.921fb54442d18p+0");

/// `cn(z²)/cd(z²)` of the given numerator and denominator coefficients.
fn rational(z2: f64, cn: &[f64; 7], cd: &[f64; 7]) -> (f64, f64) {
    let z4 = z2 * z2;
    let z8 = z4 * z4;
    let mut cn0 = cn[0] + z2 * cn[1];
    let cn2 = cn[2] + z2 * cn[3];
    let mut cn4 = cn[4] + z2 * cn[5];
    cn0 += z4 * cn2;
    cn4 += z4 * cn[6];
    cn0 += z8 * cn4;
    let mut cd0 = cd[0] + z2 * cd[1];
    let cd2 = cd[2] + z2 * cd[3];
    let mut cd4 = cd[4] + z2 * cd[5];
    cd0 += z4 * cd2;
    cd4 += z4 * cd[6];
    cd0 += z8 * cd4;
    (cn0, cd0)
}

const ATAN_CN: [f64; 7] = [
    hex("0x1.51eccde075d67p-2"),
    hex("0x1.a76bb5637f2f2p-1"),
    hex("0x1.81e0eed20de88p-1"),
    hex("0x1.376c8ca67d11dp-2"),
    hex("0x1.aec7b69202ac6p-5"),
    hex("0x1.9561899acc73ep-9"),
    hex("0x1.bf9fa5b67e6p-16"),
];

const ATAN_CD: [f64; 7] = [
    hex("0x1.51eccde075d66p-2"),
    hex("0x1.dfbdd7b392d28p-1"),
    hex("0x1p+0"),
    hex("0x1.fd22bf0e89b54p-2"),
    hex("0x1.d91ff8b576282p-4"),
    hex("0x1.653ea99fc9bbp-7"),
    hex("0x1.1e7fcc202340ap-12"),
];

/// `atanf`.
pub fn atanf(x: f32) -> f32 {
    let t = x.to_bits();
    let e = (t >> 23) & 0xff;
    let gt = e >= 127;
    let ta = t & 0x7fff_ffff;
    if ta >= 0x4c70_0518 {
        if ta > 0x7f80_0000 {
            return quiet(x);
        }
        return PI2.copysign(f64::from(x)) as f32;
    }
    if e < 127 - 13 {
        if e < 127 - 25 {
            if t << 1 == 0 {
                return x;
            }
            return (-x).mul_add(x.abs(), x);
        }
        return (hex("-0x1.5555555555555p-2") as f32 * x).mul_add(x * x, x);
    }
    let z = if gt { 1.0 / f64::from(x) } else { f64::from(x) };
    let (cn0, cd0) = rational(z * z, &ATAN_CN, &ATAN_CD);
    let r = cn0 * z / cd0;
    if !gt {
        return r as f32;
    }
    let r = hex("0x1.0fdaa22168c23p-7").copysign(z) - r + hex("0x1.9p0").copysign(z);
    r as f32
}

const ATAN2_CN: [f64; 7] = [
    hex("0x1p+0"),
    hex("0x1.40e0698f94c35p+1"),
    hex("0x1.248c5da347f0dp+1"),
    hex("0x1.d873386572976p-1"),
    hex("0x1.46fa40b20f1dp-3"),
    hex("0x1.33f5e041eed0fp-7"),
    hex("0x1.546bbf28667c5p-14"),
];

const ATAN2_CD: [f64; 7] = [
    hex("0x1p+0"),
    hex("0x1.6b8b143a3f6dap+1"),
    hex("0x1.8421201d18ed5p+1"),
    hex("0x1.8221d086914ebp+0"),
    hex("0x1.670657e3a07bap-2"),
    hex("0x1.0f4951fd1e72dp-5"),
    hex("0x1.b3874b8798286p-11"),
];

const PI: f64 = hex("0x1.921fb54442d18p+1");
const PI2_LOW: f64 = hex("0x1.1a62633145c07p-54");

/// The angle added per octant, indexed by `y`'s sign·4 + `x`'s sign·2 + `|y| > |x|`, and its low
/// part.
const OFF: [f64; 8] = [0.0, PI2, PI, PI2, -0.0, -PI2, -PI, -PI2];
const OFF_LOW: [f64; 8] = [
    0.0,
    PI2_LOW,
    2.0 * PI2_LOW,
    PI2_LOW,
    -0.0,
    -PI2_LOW,
    -2.0 * PI2_LOW,
    -PI2_LOW,
];

const SGN: [f64; 2] = [1.0, -1.0];

/// The double-double atan series in `z²`, highest power last.
const SERIES: [(f64, f64); 32] = [
    (hex("0x1p+0"), hex("-0x1.8c1dac5492248p-87")),
    (hex("-0x1.5555555555555p-2"), hex("-0x1.55553bf3a2abep-56")),
    (hex("0x1.999999999999ap-3"), hex("-0x1.99deed1ec9071p-57")),
    (hex("-0x1.2492492492492p-3"), hex("-0x1.fd99c8d18269ap-58")),
    (hex("0x1.c71c71c71c717p-4"), hex("-0x1.651eee4c4d9dp-61")),
    (hex("-0x1.745d1745d1649p-4"), hex("-0x1.632683d6c44a6p-58")),
    (hex("0x1.3b13b13b11c63p-4"), hex("0x1.bf69c1f8af41dp-58")),
    (hex("-0x1.11111110e6338p-4"), hex("0x1.3c3e431e8bb68p-61")),
    (hex("0x1.e1e1e1dc45c4ap-5"), hex("-0x1.be2db05c77bbfp-59")),
    (hex("-0x1.af286b8164b4fp-5"), hex("0x1.a4673491f0942p-61")),
    (hex("0x1.86185e9ad4846p-5"), hex("0x1.e12e32d79fceep-59")),
    (hex("-0x1.642c6d5161faep-5"), hex("0x1.3ce76c1ca03fp-59")),
    (hex("0x1.47ad6f277e5bfp-5"), hex("-0x1.abd8d85bdb714p-60")),
    (hex("-0x1.2f64a2ee8896dp-5"), hex("0x1.ef87d4b615323p-61")),
    (hex("0x1.1a6a2b31741b5p-5"), hex("0x1.a5d9d973547eep-62")),
    (hex("-0x1.07fbdad65e0a6p-5"), hex("-0x1.65ac07f5d35f4p-61")),
    (hex("0x1.ee9932a9a5f8bp-6"), hex("0x1.f8b9623f6f55ap-61")),
    (hex("-0x1.ce8b5b9584dc6p-6"), hex("0x1.fe5af96e8ea2dp-61")),
    (hex("0x1.ac9cb288087b7p-6"), hex("-0x1.450cdfceaf5cap-60")),
    (hex("-0x1.84b025351f3e6p-6"), hex("0x1.579561b0d73dap-61")),
    (hex("0x1.52f5b8ecdd52bp-6"), hex("0x1.036bd2c6fba47p-60")),
    (hex("-0x1.163a8c44909dcp-6"), hex("0x1.18f735ffb9f16p-60")),
    (hex("0x1.a400dce3eea6fp-7"), hex("-0x1.c90569c0c1b5cp-61")),
    (hex("-0x1.1caa78ae6db3ap-7"), hex("-0x1.4c60f8161ea09p-61")),
    (hex("0x1.52672453c0731p-8"), hex("0x1.834efb598c338p-62")),
    (hex("-0x1.5850c5be137cfp-9"), hex("-0x1.445fc150ca7f5p-63")),
    (hex("0x1.23eb98d22e1cap-10"), hex("-0x1.388fbaf1d783p-64")),
    (hex("-0x1.8f4e974a40741p-12"), hex("0x1.271198a97da34p-66")),
    (hex("0x1.a5cf2e9cf76e5p-14"), hex("-0x1.887eb4a63b665p-68")),
    (hex("-0x1.420c270719e32p-16"), hex("0x1.efd595b27888bp-71")),
    (hex("0x1.3ba2d69b51677p-19"), hex("-0x1.4fb06829cdfc7p-73")),
    (hex("-0x1.29b7e6f676385p-23"), hex("-0x1.a783b6de718fbp-77")),
];

/// The double-double product `(xh + xl)·(ch + cl)`.
fn muldd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let high_low = ch * xl;
    let low_high = cl * xh;
    let product = ch * xh;
    let error = ch.mul_add(xh, -product) + (low_high + high_low);
    let h = product + error;
    (h, (product - h) + error)
}

/// [`SERIES`] at the double-double `xh + xl`.
fn polydd(xh: f64, xl: f64) -> (f64, f64) {
    let (mut ch, mut cl) = SERIES[SERIES.len() - 1];
    for &(c, c_low) in SERIES[..SERIES.len() - 1].iter().rev() {
        (ch, cl) = muldd(xh, xl, ch, cl);
        let th = ch + c;
        let tl = (c - th) + ch;
        ch = th;
        cl += tl + c_low;
    }
    (ch, cl)
}

/// `atan2f` for `|y/x| < 2⁻²⁴`: `z − z³/3` for `z = y/x`, nudged off a rounding boundary.
fn atan2f_tiny(y: f32, x: f32) -> f32 {
    let dy = f64::from(y);
    let dx = f64::from(x);
    let z = dy / dx;
    let e = (-z).mul_add(dx, dy);
    let c = hex("-0x1.5555555555555p-2");
    let zz = z * z;
    let cz = c * z;
    let e = e / dx + cz * zz;
    let mut t = z.to_bits();
    if t.trailing_zeros() >= 28 {
        if z * e > 0.0 {
            t = t.wrapping_add(1);
        } else {
            t = t.wrapping_sub(1);
        }
    }
    f64::from_bits(t) as f32
}

/// `atan2f(y, x)`. Of two NaNs `y` wins.
pub fn atan2f(y: f32, x: f32) -> f32 {
    let ux = x.to_bits();
    let uy = y.to_bits();
    let ax = ux & 0x7fff_ffff;
    let ay = uy & 0x7fff_ffff;
    let y_sign = (uy >> 31) as usize;
    let x_negative = ux >> 31 != 0;
    if ay >= 0xff << 23 || ax >= 0xff << 23 {
        if ay > 0xff << 23 || ax > 0xff << 23 {
            return nan_sum(y, x);
        }
        let yinf = ay == 0xff << 23;
        let xinf = ax == 0xff << 23;
        if yinf && xinf {
            let angle = if x_negative {
                hex("0x1.2d97c7f3321d2p+1")
            } else {
                hex("0x1.921fb54442d18p-1")
            };
            return (angle * SGN[y_sign]) as f32;
        }
        if xinf {
            let angle = if x_negative { PI } else { 0.0 };
            return (angle * SGN[y_sign]) as f32;
        }
        if yinf {
            return (PI2 * SGN[y_sign]) as f32;
        }
    }
    if ay == 0 {
        if ax == 0 {
            let i = y_sign * 4 + usize::from(x_negative) * 2;
            return if x_negative {
                (OFF[i] + OFF_LOW[i]) as f32
            } else {
                OFF[i] as f32
            };
        }
        if !x_negative {
            return (0.0 * SGN[y_sign]) as f32;
        }
    }
    let gt = usize::from(ay > ax);
    let i = y_sign * 4 + usize::from(x_negative) * 2 + gt;
    let m = [0.0, 1.0];
    let zx = f64::from(x);
    let zy = f64::from(y);
    let z = (m[gt] * zx + m[1 - gt] * zy) / (m[gt] * zy + m[1 - gt] * zx);
    let d = ax as i32 - ay as i32;
    let r = if d < 27 << 23 && d > -(27 << 23) {
        let (cn0, cd0) = rational(z * z, &ATAN2_CN, &ATAN2_CD);
        cn0 / cd0
    } else {
        1.0
    };
    let z = z * SGN[gt];
    let r = z * r + OFF[i];
    if (r.to_bits().wrapping_add(8) & 0xfff_ffff) > 16 {
        return r as f32;
    }
    if ay < ax && (ax - ay) >> 23 >= 25 {
        return atan2f_tiny(y, x);
    }
    let (zh, zl) = if gt == 0 {
        let zh = zy / zx;
        (zh, zh.mul_add(-zx, zy) / zx)
    } else {
        let zh = zx / zy;
        (zh, zh.mul_add(-zy, zx) / zy)
    };
    let (z2h, z2l) = muldd(zh, zl, zh, zl);
    let (ph, pl) = polydd(z2h, z2l);
    let zh = zh * SGN[gt];
    let zl = zl * SGN[gt];
    let (ph, pl) = muldd(zh, zl, ph, pl);
    let sh = ph + OFF[i];
    let sl = ((OFF[i] - sh) + ph) + pl + OFF_LOW[i];
    let th = f64::from(sh as f32);
    let dh = sh - th;
    let mut tm = dh + sl;
    if th + th * hex("0x1p-60") == th - th * hex("0x1p-60") {
        let tth = (th.to_bits() & 0x7ff << 52).wrapping_sub(24 << 52);
        if tm.abs() > f64::from_bits(tth) {
            tm *= 1.25;
        } else {
            tm *= 0.75;
        }
    }
    (th + tm) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::math::glibc::test_support::check_rows;

    #[test]
    fn atanf_matches_glibc() {
        check_rows("atanf", atanf, ATANF);
    }

    #[test]
    fn atan2f_matches_glibc() {
        for &(y, x, expected) in ATAN2F {
            let got = atan2f(f32::from_bits(y), f32::from_bits(x)).to_bits();
            assert_eq!(got, expected, "atan2f({y:#010x}, {x:#010x}) = {got:#010x}");
        }
    }

    /// Recorded from glibc 2.41 on x86-64.
    const ATANF: &[(u32, u32)] = &[
        (0x0000_0000, 0x0000_0000),
        (0x8000_0000, 0x8000_0000),
        (0x7f80_0000, 0x3fc9_0fdb),
        (0xff80_0000, 0xbfc9_0fdb),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0x0000_0001),
        (0x807f_ffff, 0x807f_ffff),
        (0x0080_0000, 0x0080_0000),
        (0x7f7f_ffff, 0x3fc9_0fdb),
        (0xff7f_ffff, 0xbfc9_0fdb),
        (0x3f80_0000, 0x3f49_0fdb),
        (0xbf80_0000, 0xbf49_0fdb),
        (0x4c70_0518, 0x3fc9_0fdb),
        (0x4c70_0517, 0x3fc9_0fda),
        (0x3900_0000, 0x3900_0000),
        (0x38ff_ffff, 0x38ff_ffff),
        (0x3300_0000, 0x3300_0000),
        (0x32ff_ffff, 0x32ff_ffff),
        (0x3d8d_6b23, 0x3d8d_31c3),
        (0x4bff_ffff, 0x3fc9_0fda),
        (0xc22d_f52e, 0xbfc6_1e84),
        (0xc213_7187, 0xbfc5_971c),
        (0x41e8_9220, 0x3fc4_a924),
        (0xc22f_4960, 0xbfc6_243a),
        (0x41bd_cbd8, 0x3fc3_ab7c),
        (0xc1a4_ca70, 0xbfc2_da55),
        (0xc187_a700, 0xbfc1_859f),
        (0xc0fb_e2c0, 0xbfb8_e322),
        (0xc0ec_7c00, 0xbfb7_d89b),
        (0xc209_ca43, 0xbfc5_58e1),
        (0x41fb_b918, 0x3fc4_fece),
        (0xc16a_2a38, 0xbfc0_545c),
    ];

    /// `(y, x, atan2f(y, x))`, recorded from glibc 2.41 on x86-64.
    const ATAN2F: &[(u32, u32, u32)] = &[
        (0x0000_0000, 0x0000_0000, 0x0000_0000),
        (0x0000_0000, 0x7f80_0000, 0x0000_0000),
        (0x0000_0000, 0xff80_0000, 0x4049_0fdb),
        (0x0000_0000, 0xffc0_0000, 0xffc0_0000),
        (0x0000_0000, 0x3f80_0000, 0x0000_0000),
        (0x0000_0000, 0x4040_0000, 0x0000_0000),
        (0x0000_0000, 0xbfc0_0000, 0x4049_0fdb),
        (0x8000_0000, 0x0000_0000, 0x8000_0000),
        (0x8000_0000, 0x7f80_0000, 0x8000_0000),
        (0x8000_0000, 0xff80_0000, 0xc049_0fdb),
        (0x8000_0000, 0xffc0_0000, 0xffc0_0000),
        (0x8000_0000, 0x3f80_0000, 0x8000_0000),
        (0x8000_0000, 0x4040_0000, 0x8000_0000),
        (0x8000_0000, 0xbfc0_0000, 0xc049_0fdb),
        (0x7f80_0000, 0x0000_0000, 0x3fc9_0fdb),
        (0x7f80_0000, 0x7f80_0000, 0x3f49_0fdb),
        (0x7f80_0000, 0xff80_0000, 0x4016_cbe4),
        (0x7f80_0000, 0xffc0_0000, 0xffc0_0000),
        (0x7f80_0000, 0x3f80_0000, 0x3fc9_0fdb),
        (0x7f80_0000, 0x4040_0000, 0x3fc9_0fdb),
        (0x7f80_0000, 0xbfc0_0000, 0x3fc9_0fdb),
        (0xff80_0000, 0x0000_0000, 0xbfc9_0fdb),
        (0xff80_0000, 0x7f80_0000, 0xbf49_0fdb),
        (0xff80_0000, 0xff80_0000, 0xc016_cbe4),
        (0xff80_0000, 0xffc0_0000, 0xffc0_0000),
        (0xff80_0000, 0x3f80_0000, 0xbfc9_0fdb),
        (0xff80_0000, 0x4040_0000, 0xbfc9_0fdb),
        (0xff80_0000, 0xbfc0_0000, 0xbfc9_0fdb),
        (0x7fc0_0000, 0x0000_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0x7f80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0xff80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0xffc0_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0x3f80_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0x4040_0000, 0x7fc0_0000),
        (0x7fc0_0000, 0xbfc0_0000, 0x7fc0_0000),
        (0xffc0_0005, 0x0000_0000, 0xffc0_0005),
        (0xffc0_0005, 0x7f80_0000, 0xffc0_0005),
        (0xffc0_0005, 0xff80_0000, 0xffc0_0005),
        (0xffc0_0005, 0xffc0_0000, 0xffc0_0005),
        (0xffc0_0005, 0x3f80_0000, 0xffc0_0005),
        (0xffc0_0005, 0x4040_0000, 0xffc0_0005),
        (0xffc0_0005, 0xbfc0_0000, 0xffc0_0005),
        (0x7f80_0001, 0x0000_0000, 0x7fc0_0001),
        (0x7f80_0001, 0x7f80_0000, 0x7fc0_0001),
        (0x7f80_0001, 0xff80_0000, 0x7fc0_0001),
        (0x7f80_0001, 0xffc0_0000, 0x7fc0_0001),
        (0x7f80_0001, 0x3f80_0000, 0x7fc0_0001),
        (0x7f80_0001, 0x4040_0000, 0x7fc0_0001),
        (0x7f80_0001, 0xbfc0_0000, 0x7fc0_0001),
        (0x3f80_0000, 0x0000_0000, 0x3fc9_0fdb),
        (0x3f80_0000, 0x7f80_0000, 0x0000_0000),
        (0x3f80_0000, 0xff80_0000, 0x4049_0fdb),
        (0x3f80_0000, 0xffc0_0000, 0xffc0_0000),
        (0x3f80_0000, 0x3f80_0000, 0x3f49_0fdb),
        (0x3f80_0000, 0x4040_0000, 0x3ea4_bc7d),
        (0x3f80_0000, 0xbfc0_0000, 0x4023_6e05),
        (0xbf80_0000, 0x0000_0000, 0xbfc9_0fdb),
        (0xbf80_0000, 0x7f80_0000, 0x8000_0000),
        (0xbf80_0000, 0xff80_0000, 0xc049_0fdb),
        (0xbf80_0000, 0xffc0_0000, 0xffc0_0000),
        (0xbf80_0000, 0x3f80_0000, 0xbf49_0fdb),
        (0xbf80_0000, 0x4040_0000, 0xbea4_bc7d),
        (0xbf80_0000, 0xbfc0_0000, 0xc023_6e05),
        (0x4000_0000, 0x0000_0000, 0x3fc9_0fdb),
        (0x4000_0000, 0x7f80_0000, 0x0000_0000),
        (0x4000_0000, 0xff80_0000, 0x4049_0fdb),
        (0x4000_0000, 0xffc0_0000, 0xffc0_0000),
        (0x4000_0000, 0x3f80_0000, 0x3f8d_b70d),
        (0x4000_0000, 0x4040_0000, 0x3f16_8757),
        (0x4000_0000, 0xbfc0_0000, 0x400d_b70d),
        (0xc000_0000, 0x0000_0000, 0xbfc9_0fdb),
        (0xc000_0000, 0x7f80_0000, 0x8000_0000),
        (0xc000_0000, 0xff80_0000, 0xc049_0fdb),
        (0xc000_0000, 0xffc0_0000, 0xffc0_0000),
        (0xc000_0000, 0x3f80_0000, 0xbf8d_b70d),
        (0xc000_0000, 0x4040_0000, 0xbf16_8757),
        (0xc000_0000, 0xbfc0_0000, 0xc00d_b70d),
        (0x0000_0001, 0x7f7f_ffff, 0x0000_0000),
        (0x3f80_0000, 0x4c00_0000, 0x3300_0000),
        (0x8000_0000, 0xbf80_0000, 0xc049_0fdb),
        (0x0000_0000, 0x8000_0000, 0x4049_0fdb),
        (0xc089_cd64, 0xbfde_0738, 0xbffa_135d),
        (0xc105_3318, 0xc059_762c, 0xbffa_a9dc),
        (0xc0e0_6398, 0x4041_db68, 0xbf94_de7e),
        (0xc0e5_c902, 0xbff4_3218, 0xbfea_4d07),
        (0x4042_9ff0, 0xc0a9_4996, 0x4027_ac27),
        (0x4112_1e88, 0x3fde_3650, 0x3fb1_0445),
        (0x4090_99ec, 0xc112_8811, 0x402b_ba02),
        (0xc03b_9064, 0x3fe1_9c30, 0xbf83_c128),
        (0x40ac_cf62, 0xc0ef_d404, 0x4021_19d8),
        (0x4015_48c0, 0xbf78_db80, 0x3ffb_9abb),
    ];
}
