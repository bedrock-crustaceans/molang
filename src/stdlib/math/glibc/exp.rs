//! `expf` and `exp2f`: `2^(k/32)` from a table times a cubic in the remainder, in `f64`.
//!
//! The `GLIBC_2.2.5` wrappers return the same values: their overflow is +∞ and their underflow
//! +0, as here.

use super::{hex, may_underflow, overflow, quiet, underflow};

/// `log2` of the table size.
pub(super) const TABLE_BITS: u32 = 5;

/// `2^(i/32)`'s bits minus `i << 47`, so that adding `k << 47` scales by `2^(k div 32)`.
pub(super) const TABLE: [u64; 32] = [
    0x3ff0_0000_0000_0000,
    0x3fef_d9b0_d315_8574,
    0x3fef_b558_6cf9_890f,
    0x3fef_9301_d012_5b51,
    0x3fef_72b8_3c7d_517b,
    0x3fef_5487_3168_b9aa,
    0x3fef_387a_6e75_6238,
    0x3fef_1e9d_f51f_dee1,
    0x3fef_06fe_0a31_b715,
    0x3fee_f1a7_373a_a9cb,
    0x3fee_dea6_4c12_3422,
    0x3fee_ce08_6061_892d,
    0x3fee_bfda_d536_2a27,
    0x3fee_b42b_569d_4f82,
    0x3fee_ab07_dd48_5429,
    0x3fee_a47e_b03a_5585,
    0x3fee_a09e_667f_3bcd,
    0x3fee_9f75_e8ec_5f74,
    0x3fee_a114_73eb_0187,
    0x3fee_a589_994c_ce13,
    0x3fee_ace5_422a_a0db,
    0x3fee_b737_b0cd_c5e5,
    0x3fee_c491_82a3_f090,
    0x3fee_d503_b23e_255d,
    0x3fee_e89f_995a_d3ad,
    0x3fee_ff76_f2fb_5e47,
    0x3fef_199b_dd85_529c,
    0x3fef_3720_dcef_9069,
    0x3fef_5818_dcfb_a487,
    0x3fef_7c97_337b_9b5f,
    0x3fef_a4af_a2a4_90da,
    0x3fef_d076_5b6e_4540,
];

/// Adding it rounds an `f64` below 2⁴⁶ to a multiple of 1/32 and leaves `k` in the low bits.
pub(super) const SHIFT_SCALED: f64 = hex("0x1.8p+52") / 32.0;

/// The cubic of `2^r` for `r` in `[−1/64, 1/64]`, highest degree first; the constant term is 1.
pub(super) const POLY: [f64; 3] = [
    hex("0x1.c6af84b912394p-5"),
    hex("0x1.ebfce50fac4f3p-3"),
    hex("0x1.62e42ff0c52d6p-1"),
];

const SHIFT: f64 = hex("0x1.8p+52");

/// 32/ln 2.
const INV_LN2_SCALED: f64 = hex("0x1.71547652b82fep+0") * 32.0;

/// [`POLY`] for `r` scaled by 32.
const POLY_SCALED: [f64; 3] = [
    hex("0x1.c6af84b912394p-5") / 32.0 / 32.0 / 32.0,
    hex("0x1.ebfce50fac4f3p-3") / 32.0 / 32.0,
    hex("0x1.62e42ff0c52d6p-1") / 32.0,
];

fn top12(x: f32) -> u32 {
    x.to_bits() >> 20
}

/// `2^(k/32)·(c₀r³ + c₁r² + c₂r + 1)` for `k`'s bits `ki` (sign bias included) and remainder `r`.
pub(super) fn scale_poly(ki: u64, r: f64, poly: &[f64; 3]) -> f64 {
    let t = TABLE[(ki % (1 << TABLE_BITS)) as usize].wrapping_add(ki << (52 - TABLE_BITS));
    let s = f64::from_bits(t);
    let z = poly[0].mul_add(r, poly[1]);
    let r2 = r * r;
    let y = poly[2].mul_add(r, 1.0);
    z.mul_add(r2, y) * s
}

/// `expf`.
pub(crate) fn expf(x: f32) -> f32 {
    let xd = f64::from(x);
    let abstop = top12(x) & 0x7ff;
    if abstop >= top12(88.0) {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        if abstop >= top12(f32::INFINITY) {
            return if x.is_nan() { quiet(x) } else { x };
        }
        if x > hex("0x1.62e42ep6") as f32 {
            return overflow(false);
        }
        if x < hex("-0x1.9fe368p6") as f32 {
            return underflow(false);
        }
        if x < hex("-0x1.9d1d9ep6") as f32 {
            return may_underflow(false);
        }
    }
    // The FMA build never rounds `INV_LN2_SCALED·xd` on its own.
    let kd = INV_LN2_SCALED.mul_add(xd, SHIFT);
    let ki = kd.to_bits();
    let kd = kd - SHIFT;
    let r = INV_LN2_SCALED.mul_add(xd, -kd);
    scale_poly(ki, r, &POLY_SCALED) as f32
}

/// `exp2f`.
pub(crate) fn exp2f(x: f32) -> f32 {
    let xd = f64::from(x);
    let abstop = top12(x) & 0x7ff;
    if abstop >= top12(128.0) {
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        if abstop >= top12(f32::INFINITY) {
            return if x.is_nan() { quiet(x) } else { x };
        }
        if x > 0.0 {
            return overflow(false);
        }
        if x <= -150.0 {
            return underflow(false);
        }
        if x < -149.0 {
            return may_underflow(false);
        }
    }
    let kd = xd + SHIFT_SCALED;
    let ki = kd.to_bits();
    let kd = kd - SHIFT_SCALED;
    let r = xd - kd;
    scale_poly(ki, r, &POLY) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::math::glibc::test_support::check_rows;

    #[test]
    fn expf_matches_glibc() {
        check_rows("expf", expf, EXPF);
    }

    #[test]
    fn exp2f_matches_glibc() {
        check_rows("exp2f", exp2f, EXP2F);
    }

    /// Reference results of glibc's `expf`.
    const EXPF: &[(u32, u32)] = &[
        // The build without FMA rounds these the other way.
        (0x4202_422f, 0x56fc_9f1c),
        (0xc27c_65d9, 0x11fa_2993),
        (0x0000_0000, 0x3f80_0000),
        (0x8000_0000, 0x3f80_0000),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0x0000_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0x3f80_0000),
        (0x807f_ffff, 0x3f80_0000),
        (0x0080_0000, 0x3f80_0000),
        (0x7f7f_ffff, 0x7f80_0000),
        (0xff7f_ffff, 0x0000_0000),
        (0x3f80_0000, 0x402d_f854),
        (0xbf80_0000, 0x3ebc_5ab2),
        (0x42b1_7217, 0x7f7f_ff84),
        (0x42b1_7218, 0x7f80_0000),
        (0xc2cf_f1b4, 0x0000_0001),
        (0xc2cf_f1b5, 0x0000_0000),
        (0xc2ce_8ecf, 0x0000_0001),
        (0xc2ce_8ed0, 0x0000_0001),
        (0x42b0_0000, 0x7ef8_82b7),
        (0x3380_0000, 0x3f80_0001),
        (0xc2cb_0beb, 0x0000_0006),
        (0x3db8_b800, 0x3f8c_14e2),
        (0x4280_9718, 0x6dd8_803a),
        (0x42a0_4e02, 0x7946_d528),
        (0xc286_3a9e, 0x0f10_6f02),
        (0x40ea_2c70, 0x44bc_6254),
        (0xc11b_bcb0, 0x3878_8e8f),
        (0xbff6_4100, 0x3e15_8c20),
        (0xc2c3_0c9c, 0x0000_013c),
        (0xc2a3_5e3c, 0x048e_8026),
        (0xc29a_4a30, 0x07d0_6ed1),
        (0xc1d6_7728, 0x2c20_39fd),
        (0xc279_c058, 0x1272_6652),
        (0xc0b5_6f90, 0x3b61_fd31),
        (0xc284_2900, 0x0fcb_2d85),
        (0xc21f_215e, 0x22c2_cd36),
        (0x42b0_e76e, 0x7f43_436e),
        (0xc234_f042, 0x1ed5_d0fc),
        (0xc294_e274, 0x09c2_584d),
        (0xc20e_9a54, 0x25bd_9c52),
        (0xc237_e9d6, 0x1e4b_452a),
        (0x41f7_db40, 0x55cf_996f),
        (0xc22e_ca39, 0x1ff8_a157),
        (0x40df_5390, 0x4486_394f),
        (0xc232_05a0, 0x1f5d_a6a5),
        (0xc186_e8ac, 0x334b_cac8),
        (0x41d1_39e0, 0x5254_7aaa),
    ];

    /// Reference results of glibc's `exp2f`.
    const EXP2F: &[(u32, u32)] = &[
        (0x0000_0000, 0x3f80_0000),
        (0x8000_0000, 0x3f80_0000),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0x0000_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0x3f80_0000),
        (0x807f_ffff, 0x3f80_0000),
        (0x0080_0000, 0x3f80_0000),
        (0x7f7f_ffff, 0x7f80_0000),
        (0xff7f_ffff, 0x0000_0000),
        (0x3f80_0000, 0x4000_0000),
        (0xbf80_0000, 0x3f00_0000),
        (0x4300_0000, 0x7f80_0000),
        (0x42ff_ffff, 0x7f7f_ffa7),
        (0xc316_0000, 0x0000_0000),
        (0xc315_ffff, 0x0000_0001),
        (0xc315_0000, 0x0000_0001),
        (0xc314_ffff, 0x0000_0001),
        (0x3380_0000, 0x3f80_0000),
        (0xbf80_0000, 0x3f00_0000),
        (0xc304_c53a, 0x0001_2c29),
        (0xc2e4_6bbc, 0x065d_41df),
        (0x428c_ccb0, 0x62a8_df0d),
        (0xc305_b35d, 0x0000_9d84),
        (0x425d_b6fc, 0x5b2c_4a42),
        (0xc289_5a82, 0x1d20_24e4),
        (0xc269_e9ce, 0x2237_c26f),
        (0xc204_28f8, 0x2ef8_ff89),
        (0xc1fd_8a00, 0x2f9e_6bb4),
        (0xc2d6_e7f8, 0x09bb_0161),
        (0x429a_34c4, 0x6609_7a64),
        (0xc24f_ea5a, 0x2581_e3b7),
        (0x4292_62dc, 0x6412_548f),
        (0xc227_14d6, 0x2a96_1633),
        (0x42cd_65be, 0x72cf_c064),
        (0x4271_370c, 0x5d9d_ff54),
        (0xc125_4e00, 0x3a4b_708d),
        (0xc23e_2dc0, 0x27af_7fd9),
        (0x42dd_b8fc, 0x76e8_88b2),
        (0x42fd_5938, 0x7ecc_424a),
        (0x427c_cc60, 0x5f12_fdb9),
        (0x42b0_3b1a, 0x6b8a_a993),
        (0xc2cf_647a, 0x0b9d_ff54),
        (0xc270_ab42, 0x2163_fa95),
        (0xc285_5334, 0x1e21_bc54),
        (0x42a3_e502, 0x6876_d068),
        (0x4106_0e30, 0x43a6_64fd),
    ];
}
