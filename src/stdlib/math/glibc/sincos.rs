//! `sinf` and `cosf`: a polynomial in `f64` after a reduction to `[−π/4, π/4]`.

use super::{hex, invalid};

/// The sine of each quadrant, `2/π·2²⁴`, `π/2` and the polynomials of one half of the table.
struct Poly {
    sign: [f64; 4],
    hpi_inv: f64,
    hpi: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    s1: f64,
    s2: f64,
    s3: f64,
}

/// The second entry's cosine polynomial is negated, for quadrants 2 and 3.
const TABLE: [Poly; 2] = [
    Poly {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: hex("0x1.45F306DC9C883p+23"),
        hpi: hex("0x1.921FB54442D18p0"),
        c0: hex("0x1p0"),
        c1: hex("-0x1.ffffffd0c621cp-2"),
        c2: hex("0x1.55553e1068f19p-5"),
        c3: hex("-0x1.6c087e89a359dp-10"),
        c4: hex("0x1.99343027bf8c3p-16"),
        s1: hex("-0x1.555545995a603p-3"),
        s2: hex("0x1.1107605230bc4p-7"),
        s3: hex("-0x1.994eb3774cf24p-13"),
    },
    Poly {
        sign: [1.0, -1.0, -1.0, 1.0],
        hpi_inv: hex("0x1.45F306DC9C883p+23"),
        hpi: hex("0x1.921FB54442D18p0"),
        c0: hex("-0x1p0"),
        c1: hex("0x1.ffffffd0c621cp-2"),
        c2: hex("-0x1.55553e1068f19p-5"),
        c3: hex("0x1.6c087e89a359dp-10"),
        c4: hex("-0x1.99343027bf8c3p-16"),
        s1: hex("-0x1.555545995a603p-3"),
        s2: hex("0x1.1107605230bc4p-7"),
        s3: hex("-0x1.994eb3774cf24p-13"),
    },
];

/// 4/π to 192 bits, eight new bits per entry.
const INV_PIO4: [u32; 24] = [
    0xa2,
    0xa2f9,
    0xa2_f983,
    0xa2f9_836e,
    0xf983_6e4e,
    0x836e_4e44,
    0x6e4e_4415,
    0x4e44_1529,
    0x4415_29fc,
    0x1529_fc27,
    0x29fc_2757,
    0xfc27_57d1,
    0x2757_d1f5,
    0x57d1_f534,
    0xd1f5_34dd,
    0xf534_ddc0,
    0x34dd_c0db,
    0xddc0_db62,
    0xc0db_6295,
    0xdb62_9599,
    0x6295_993c,
    0x9599_3c43,
    0x993c_4390,
    0x3c43_9041,
];

/// 2π·2⁻⁶⁴.
const PI63: f64 = hex("0x1.921FB54442D18p-62");

const PIO4: f32 = hex("0x1.921FB6p-1") as f32;

/// The top 12 bits of `x`'s representation without the sign.
fn abstop12(x: f32) -> u32 {
    (x.to_bits() >> 20) & 0x7ff
}

/// `x` reduced to `[−π/4, π/4]` and its quadrant, for `|x| < 120`.
fn reduce_fast(x: f64, p: &Poly) -> (f64, i32) {
    let r = x * p.hpi_inv;
    let n = ((r as i32) + 0x80_0000) >> 24;
    (f64::from(n).mul_add(-p.hpi, x), n)
}

/// The finite `|x| ≥ 120` of bits `xi` reduced to `[−π/4, π/4]` and its quadrant, from the
/// exact 2.62 fixed-point product with 4/π.
fn reduce_large(xi: u32) -> (f64, i32) {
    let arr = &INV_PIO4[((xi >> 26) & 15) as usize..];
    let shift = (xi >> 23) & 7;
    let xi = ((xi & 0xff_ffff) | 0x80_0000) << shift;
    let res0 = u64::from(xi.wrapping_mul(arr[0]));
    let res1 = u64::from(xi) * u64::from(arr[4]);
    let res2 = u64::from(xi) * u64::from(arr[8]);
    let res0 = ((res2 >> 32) | (res0 << 32)).wrapping_add(res1);
    let n = res0.wrapping_add(1 << 61) >> 62;
    let res0 = res0.wrapping_sub(n << 62);
    ((res0 as i64) as f64 * PI63, n as i32)
}

/// The sine polynomial of `x` (`x2` its square) for an even quadrant `n`, the cosine one for an
/// odd.
fn poly(x: f64, x2: f64, p: &Poly, n: i32) -> f32 {
    if n & 1 == 0 {
        let x3 = x * x2;
        let s1 = x2.mul_add(p.s3, p.s2);
        let x7 = x3 * x2;
        let s = x3.mul_add(p.s1, x);
        x7.mul_add(s1, s) as f32
    } else {
        let x4 = x2 * x2;
        let c2 = x2.mul_add(p.c4, p.c3);
        let c1 = x2.mul_add(p.c1, p.c0);
        let x6 = x4 * x2;
        let c = x4.mul_add(p.c2, c1);
        x6.mul_add(c2, c) as f32
    }
}

/// The polynomial half and sign for quadrant `q`.
fn half(q: i32) -> (&'static Poly, f64) {
    (
        &TABLE[usize::from(q & 2 != 0)],
        TABLE[0].sign[(q & 3) as usize],
    )
}

/// `sinf`; ±∞ gives `0xffc00000`.
pub(crate) fn sinf(y: f32) -> f32 {
    let x = f64::from(y);
    let top = abstop12(y);
    if top < abstop12(PIO4) {
        if top < abstop12(hex("0x1p-12") as f32) {
            return y;
        }
        poly(x, x * x, &TABLE[0], 0)
    } else if top < abstop12(120.0) {
        let (r, n) = reduce_fast(x, &TABLE[0]);
        let (p, s) = half(n);
        poly(r * s, r * r, p, n)
    } else if top < abstop12(f32::INFINITY) {
        let xi = y.to_bits();
        let (r, n) = reduce_large(xi);
        let (p, s) = half(n.wrapping_add((xi >> 31) as i32));
        poly(r * s, r * r, p, n)
    } else {
        invalid(y)
    }
}

/// `cosf`; ±∞ gives `0xffc00000`.
pub(crate) fn cosf(y: f32) -> f32 {
    let x = f64::from(y);
    let top = abstop12(y);
    if top < abstop12(PIO4) {
        if top < abstop12(hex("0x1p-12") as f32) {
            return 1.0;
        }
        poly(x, x * x, &TABLE[0], 1)
    } else if top < abstop12(120.0) {
        let (r, n) = reduce_fast(x, &TABLE[0]);
        let (p, s) = half(n);
        poly(r * s, r * r, p, n ^ 1)
    } else if top < abstop12(f32::INFINITY) {
        let xi = y.to_bits();
        let (r, n) = reduce_large(xi);
        let (p, s) = half(n.wrapping_add((xi >> 31) as i32));
        poly(r * s, r * r, p, n ^ 1)
    } else {
        invalid(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::math::glibc::test_support::check_rows;

    #[test]
    fn sinf_matches_glibc() {
        check_rows("sinf", sinf, SINF);
    }

    #[test]
    fn cosf_matches_glibc() {
        check_rows("cosf", cosf, COSF);
    }

    /// Reference results of glibc's `sinf`.
    const SINF: &[(u32, u32)] = &[
        // The build without FMA rounds these the other way.
        (0x4255_b0a9, 0xbc7d_08a9),
        (0x42a3_5c07, 0xbada_a3b4),
        (0x42a3_5d44, 0x3a44_b889),
        (0x0000_0000, 0x0000_0000),
        (0x8000_0000, 0x8000_0000),
        (0x7f80_0000, 0xffc0_0000),
        (0xff80_0000, 0xffc0_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0x0000_0001),
        (0x807f_ffff, 0x807f_ffff),
        (0x0080_0000, 0x0080_0000),
        (0x7f7f_ffff, 0xbf05_99b3),
        (0xff7f_ffff, 0x3f05_99b3),
        (0x3f80_0000, 0x3f57_6aa4),
        (0xbf80_0000, 0xbf57_6aa4),
        (0x3e68_5696, 0x3e66_5992),
        (0x3fe6_1aad, 0x3f79_7051),
        (0x3fc9_0fdb, 0x3f80_0000),
        (0xc149_0fdb, 0xb4bb_bd2e),
        (0x3f49_0fdb, 0x3f35_04f3),
        (0x3980_0000, 0x3980_0000),
        (0x42f0_0000, 0x3f14_a2ef),
        (0x42ef_ffff, 0x3f14_a287),
        (0x4b00_0001, 0xbf06_7728),
        (0x38c9_0fdb, 0x38c9_0fdb),
        (0x3fc9_0fdb, 0x3f80_0000),
        (0xc340_d4dd, 0x3f6e_12da),
        (0x4185_7250, 0xbf53_9533),
        (0x4315_0fe4, 0xbf7c_9527),
        (0x4335_c1e8, 0xbee0_f8af),
        (0xc2f3_c565, 0xbf18_2da7),
        (0x41fc_aab8, 0x3e2a_a692),
        (0xc064_c280, 0x3ed6_baee),
        (0x4148_7340, 0xbd1c_90de),
        (0xc338_963c, 0xbf31_aded),
        (0xc317_ed06, 0xbf67_777b),
        (0xc30e_911a, 0x3f6e_25d3),
        (0xc21b_1e78, 0xbf61_dbd5),
        (0x3f3e_7900, 0x3f2d_60ff),
        (0x4180_80b8, 0xbeb1_e990),
        (0x4327_4258, 0xbf2f_62e4),
        (0x40e0_7c20, 0x3f2b_17ee),
        (0x4343_4990, 0x3ef9_532f),
        (0xc23d_5428, 0x3e53_c10d),
        (0x42e7_7318, 0x3efb_cae9),
        (0x4230_91b0, 0x3e23_1e0d),
        (0xc343_4fc8, 0xbf02_0e87),
        (0x4265_5538, 0x3f34_ddcc),
        (0x4239_8dfc, 0x3f2b_b6f0),
        (0xc345_9ae1, 0xbe9e_cad1),
        (0x4303_d1f8, 0xbe01_651f),
        (0x4190_f428, 0xbf2a_c886),
        (0x41ff_d698, 0x3f08_d1a4),
    ];

    /// Reference results of glibc's `cosf`.
    const COSF: &[(u32, u32)] = &[
        // The build without FMA rounds these the other way.
        (0x418a_3adb, 0xb7b4_f770),
        (0x418a_3adc, 0xb7a4_f770),
        (0x41bc_76d9, 0xbb80_3f6b),
        (0x0000_0000, 0x3f80_0000),
        (0x8000_0000, 0x3f80_0000),
        (0x7f80_0000, 0xffc0_0000),
        (0xff80_0000, 0xffc0_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0x3f80_0000),
        (0x807f_ffff, 0x3f80_0000),
        (0x0080_0000, 0x3f80_0000),
        (0x7f7f_ffff, 0x3f5a_5f96),
        (0xff7f_ffff, 0x3f5a_5f96),
        (0x3f80_0000, 0x3f0a_5140),
        (0xbf80_0000, 0x3f0a_5140),
        (0x3e68_5696, 0x3f79_7051),
        (0x3fe6_1aad, 0xbe66_598f),
        (0x3fc9_0fdb, 0xb33b_bd2e),
        (0xc149_0fdb, 0x3f80_0000),
        (0x3f49_0fdb, 0x3f35_04f3),
        (0x3980_0000, 0x3f80_0000),
        (0x42f0_0000, 0x3f50_6e2a),
        (0x42ef_ffff, 0x3f50_6e74),
        (0x4b00_0001, 0xbf59_d77f),
        (0xc340_d4dd, 0xbebc_38fe),
        (0x4185_7250, 0xbf10_1d11),
        (0x4315_0fe4, 0xbe26_c217),
        (0x4335_c1e8, 0x3f65_f685),
        (0xc2f3_c565, 0xbf4d_dbcc),
        (0x41fc_aab8, 0x3f7c_6b8a),
        (0xc064_c280, 0xbf68_65d6),
        (0x4148_7340, 0x3f7f_d01b),
        (0xc338_963c, 0xbf38_4c7c),
        (0xc317_ed06, 0x3eda_b742),
        (0xc30e_911a, 0xbebb_d8e5),
        (0xc21b_1e78, 0x3ef1_07a2),
        (0xc233_c30c, 0x3f13_33f1),
        (0x42fd_df24, 0x3e96_a939),
        (0xc28d_bdde, 0xbe3c_5c64),
        (0xc2d7_d183, 0x3eea_7ef7),
        (0x4341_e2ac, 0x3f20_787c),
        (0x40ea_3920, 0x3f02_6938),
        (0xc318_4126, 0x3de6_7046),
        (0xc2e6_3162, 0xbed4_a7bb),
        (0xbf9d_3e80, 0x3eab_de2f),
        (0xc2f0_7e91, 0x3f25_b977),
        (0xc101_44c0, 0xbe64_c36e),
        (0xc328_54f6, 0x3e81_fa4b),
        (0xc332_3380, 0xbf25_26eb),
        (0x4285_929c, 0xbf2f_f918),
        (0x4099_97e0, 0x3db2_c4b1),
    ];
}
