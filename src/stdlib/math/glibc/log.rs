//! `logf` and `log2f`: `x = 2^k·z` with `z` near a table point `c`, then `log1p(z/c − 1)` as a
//! polynomial in `f64`.

use super::{WRAPPER_NAN, hex, invalid};

/// The start of the range `[OFF, 2·OFF)` the reduced `z` lies in.
pub(super) const OFF: u32 = 0x3f33_0000;

/// `1/c` and `ln c` for the 16 subintervals of `[OFF, 2·OFF)`.
const LN_TABLE: [(f64, f64); 16] = [
    (hex("0x1.661ec79f8f3bep+0"), hex("-0x1.57bf7808caadep-2")),
    (hex("0x1.571ed4aaf883dp+0"), hex("-0x1.2bef0a7c06ddbp-2")),
    (hex("0x1.49539f0f010bp+0"), hex("-0x1.01eae7f513a67p-2")),
    (hex("0x1.3c995b0b80385p+0"), hex("-0x1.b31d8a68224e9p-3")),
    (hex("0x1.30d190c8864a5p+0"), hex("-0x1.6574f0ac07758p-3")),
    (hex("0x1.25e227b0b8eap+0"), hex("-0x1.1aa2bc79c81p-3")),
    (hex("0x1.1bb4a4a1a343fp+0"), hex("-0x1.a4e76ce8c0e5ep-4")),
    (hex("0x1.12358f08ae5bap+0"), hex("-0x1.1973c5a611cccp-4")),
    (hex("0x1.0953f419900a7p+0"), hex("-0x1.252f438e10c1ep-5")),
    (hex("0x1p+0"), hex("0x0p+0")),
    (hex("0x1.e608cfd9a47acp-1"), hex("0x1.aa5aa5df25984p-5")),
    (hex("0x1.ca4b31f026aap-1"), hex("0x1.c5e53aa362eb4p-4")),
    (hex("0x1.b2036576afce6p-1"), hex("0x1.526e57720db08p-3")),
    (hex("0x1.9c2d163a1aa2dp-1"), hex("0x1.bc2860d22477p-3")),
    (hex("0x1.886e6037841edp-1"), hex("0x1.1058bc8a07ee1p-2")),
    (hex("0x1.767dcf5534862p-1"), hex("0x1.4043057b6ee09p-2")),
];

const LN2: f64 = hex("0x1.62e42fefa39efp-1");

/// The cubic of `log1p(r)` above `r`, highest degree first.
const LN_POLY: [f64; 3] = [
    hex("-0x1.00ea348b88334p-2"),
    hex("0x1.5575b0be00b6ap-2"),
    hex("-0x1.ffffef20a4123p-2"),
];

/// `1/c` and `log2 c`, as [`LN_TABLE`].
pub(super) const LOG2_TABLE: [(f64, f64); 16] = [
    (hex("0x1.661ec79f8f3bep+0"), hex("-0x1.efec65b963019p-2")),
    (hex("0x1.571ed4aaf883dp+0"), hex("-0x1.b0b6832d4fca4p-2")),
    (hex("0x1.49539f0f010bp+0"), hex("-0x1.7418b0a1fb77bp-2")),
    (hex("0x1.3c995b0b80385p+0"), hex("-0x1.39de91a6dcf7bp-2")),
    (hex("0x1.30d190c8864a5p+0"), hex("-0x1.01d9bf3f2b631p-2")),
    (hex("0x1.25e227b0b8eap+0"), hex("-0x1.97c1d1b3b7afp-3")),
    (hex("0x1.1bb4a4a1a343fp+0"), hex("-0x1.2f9e393af3c9fp-3")),
    (hex("0x1.12358f08ae5bap+0"), hex("-0x1.960cbbf788d5cp-4")),
    (hex("0x1.0953f419900a7p+0"), hex("-0x1.a6f9db6475fcep-5")),
    (hex("0x1p+0"), hex("0x0p+0")),
    (hex("0x1.e608cfd9a47acp-1"), hex("0x1.338ca9f24f53dp-4")),
    (hex("0x1.ca4b31f026aap-1"), hex("0x1.476a9543891bap-3")),
    (hex("0x1.b2036576afce6p-1"), hex("0x1.e840b4ac4e4d2p-3")),
    (hex("0x1.9c2d163a1aa2dp-1"), hex("0x1.40645f0c6651cp-2")),
    (hex("0x1.886e6037841edp-1"), hex("0x1.88e9c2c1b9ff8p-2")),
    (hex("0x1.767dcf5534862p-1"), hex("0x1.ce0a44eb17bccp-2")),
];

/// The quartic of `log1p(r)/ln 2`, highest degree first, without a constant term.
const LOG2_POLY: [f64; 4] = [
    hex("-0x1.712b6f70a7e4dp-2"),
    hex("0x1.ecabf496832ep-2"),
    hex("-0x1.715479ffae3dep-1"),
    hex("0x1.715475f35c8b8p0"),
];

/// The bits of a positive normal `x`, or the result for zero, a negative, ∞ or NaN; a subnormal
/// comes back scaled by 2²³ with 23 taken off its exponent field.
fn normalised(x: f32) -> Result<u32, f32> {
    let ix = x.to_bits();
    if ix.wrapping_sub(0x0080_0000) < 0x7f80_0000 - 0x0080_0000 {
        return Ok(ix);
    }
    if ix.wrapping_mul(2) == 0 {
        return Err(f32::NEG_INFINITY);
    }
    if ix == 0x7f80_0000 {
        return Err(x);
    }
    if ix & 0x8000_0000 != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
        return Err(invalid(x));
    }
    Ok((x * hex("0x1p23") as f32).to_bits().wrapping_sub(23 << 23))
}

/// The table row `i`, the exponent `k` and the reduced `z` of the normalised bits `ix`.
pub(super) fn split(ix: u32) -> (usize, i32, f64) {
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> 19) % 16) as usize;
    let k = (tmp as i32) >> 23;
    let iz = ix.wrapping_sub(tmp & 0xff80_0000);
    (i, k, f64::from(f32::from_bits(iz)))
}

/// `logf@GLIBC_2.27`: a negative `x` gives `0xffc00000`.
pub fn logf(x: f32) -> f32 {
    if x.to_bits() == 0x3f80_0000 {
        return 0.0;
    }
    let ix = match normalised(x) {
        Ok(ix) => ix,
        Err(special) => return special,
    };
    let (i, k, z) = split(ix);
    let (invc, logc) = LN_TABLE[i];
    let r = z.mul_add(invc, -1.0);
    let y0 = f64::from(k).mul_add(LN2, logc);
    let r2 = r * r;
    let y = LN_POLY[1].mul_add(r, LN_POLY[2]);
    let y = LN_POLY[0].mul_add(r2, y);
    y.mul_add(r2, y0 + r) as f32
}

/// `logf@GLIBC_2.2.5`, the compatibility wrapper: a negative `x` gives `0x7fc00000`.
pub fn logf_compat(x: f32) -> f32 {
    if x < 0.0 { WRAPPER_NAN } else { logf(x) }
}

/// `log2f@GLIBC_2.27`: a negative `x` gives `0xffc00000`.
pub fn log2f(x: f32) -> f32 {
    if x.to_bits() == 0x3f80_0000 {
        return 0.0;
    }
    let ix = match normalised(x) {
        Ok(ix) => ix,
        Err(special) => return special,
    };
    let (i, k, z) = split(ix);
    let (invc, logc) = LOG2_TABLE[i];
    let r = z.mul_add(invc, -1.0);
    let y0 = logc + f64::from(k);
    let r2 = r * r;
    let y = LOG2_POLY[1].mul_add(r, LOG2_POLY[2]);
    let y = LOG2_POLY[0].mul_add(r2, y);
    let p = LOG2_POLY[3].mul_add(r, y0);
    y.mul_add(r2, p) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::math::glibc::DEFAULT_NAN;
    use crate::stdlib::math::glibc::test_support::check_rows;

    #[test]
    fn logf_matches_glibc() {
        check_rows("logf", logf, LOGF);
    }

    #[test]
    fn log2f_matches_glibc() {
        check_rows("log2f", log2f, LOG2F);
    }

    #[test]
    fn only_a_negative_argument_differs_in_the_compatibility_wrapper() {
        for x in [
            -1.0,
            -f32::MIN_POSITIVE,
            -1.0e-45,
            f32::NEG_INFINITY,
            -f32::MAX,
        ] {
            assert_eq!(logf(x).to_bits(), DEFAULT_NAN.to_bits(), "logf({x})");
            assert_eq!(logf_compat(x).to_bits(), 0x7fc0_0000, "logf_compat({x})");
        }
        for (x, _) in LOGF {
            let x = f32::from_bits(*x);
            if x >= 0.0 || x.is_nan() {
                assert_eq!(logf_compat(x).to_bits(), logf(x).to_bits(), "{x}");
            }
        }
    }

    /// Recorded from glibc 2.41 on x86-64 with FMA.
    const LOGF: &[(u32, u32)] = &[
        (0x0000_0000, 0xff80_0000),
        (0x8000_0000, 0xff80_0000),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0xffc0_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0xc2ce_8ed0),
        (0x807f_ffff, 0xffc0_0000),
        (0x0080_0000, 0xc2ae_ac50),
        (0x7f7f_ffff, 0x42b1_7218),
        (0xff7f_ffff, 0xffc0_0000),
        (0x3f80_0000, 0x0000_0000),
        (0xbf80_0000, 0xffc0_0000),
        (0x3f33_0000, 0xbeb7_3077),
        (0x3f32_ffff, 0xbeb7_307a),
        (0x3f7f_ffff, 0xb380_0000),
        (0x3f80_0001, 0x33ff_ffff),
        (0x0000_0003, 0xc2cc_5c53),
        (0x4f80_0000, 0x41b1_7218),
        (0x3fe5_646a, 0x3f15_59e8),
        (0x4258_ae4a, 0x407f_7f13),
        (0x42ae_87f2, 0x408f_01af),
        (0x42be_e0f4, 0x4091_df29),
        (0x419c_3a9b, 0x403e_334c),
        (0x4267_9557, 0x4081_e06d),
        (0x4244_6cf6, 0x4079_3720),
        (0x4254_8734, 0x407e_4207),
        (0x4076_9c3a, 0x3fac_a99c),
        (0x4140_4be8, 0x401f_21fe),
        (0x4165_bb99, 0x402a_845a),
        (0x4221_3862, 0x406c_9314),
        (0x4142_9fa8, 0x401f_e715),
        (0x42c2_321e, 0x4092_6c4c),
        (0x42a1_c4d4, 0x408c_93a2),
        (0x3f88_f61c, 0x3d8a_95d8),
        (0x42b3_ced8, 0x408f_f5b2),
        (0x3f7d_fac0, 0xbc01_d355),
        (0x4292_db84, 0x4089_7b6a),
        (0x3f7b_3434, 0xbc9a_ee34),
        (0x3edc_3ed0, 0xbf57_f505),
        (0x42b2_d367, 0x408f_c8d4),
        (0x4212_ab26, 0x4066_8537),
        (0x4224_8a7f, 0x406d_e11f),
        (0x3f81_a15a, 0x3c4f_5baa),
        (0x40d2_c16b, 0x3ff1_4678),
        (0x4288_7c5c, 0x4087_236b),
    ];

    /// Recorded from glibc 2.41 on x86-64 with FMA.
    const LOG2F: &[(u32, u32)] = &[
        (0x0000_0000, 0xff80_0000),
        (0x8000_0000, 0xff80_0000),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0xffc0_0000),
        (0x7fc0_0000, 0x7fc0_0000),
        (0xffc0_0000, 0xffc0_0000),
        (0x7f80_0001, 0x7fc0_0001),
        (0xffc1_2345, 0xffc1_2345),
        (0x0000_0001, 0xc315_0000),
        (0x807f_ffff, 0xffc0_0000),
        (0x0080_0000, 0xc2fc_0000),
        (0x7f7f_ffff, 0x4300_0000),
        (0xff7f_ffff, 0xffc0_0000),
        (0x3f80_0000, 0x0000_0000),
        (0xbf80_0000, 0xffc0_0000),
        (0x3f33_0000, 0xbf04_24a6),
        (0x3f32_ffff, 0xbf04_24a8),
        (0x3f7f_ffff, 0xb3b8_aa3b),
        (0x3f80_0001, 0x3438_aa3a),
        (0x0000_0003, 0xc313_6a40),
        (0x4f80_0000, 0x4200_0000),
        (0x40d0_5692, 0x402c_fa67),
        (0x4152_39e5, 0x406d_cfa4),
        (0x429e_2488, 0x40c9_c33e),
        (0x40c5_b4fd, 0x4028_2460),
        (0x4293_72f6, 0x40c6_87c4),
        (0x41eb_3590, 0x409c_16f2),
        (0x4204_2c80, 0x40a1_7b3b),
        (0x4228_83a8, 0x40ac_b1fd),
        (0x422a_7080, 0x40ad_383e),
        (0x4178_d6f4, 0x407d_6172),
        (0x42a2_ee46, 0x40cb_23c2),
        (0x420d_7572, 0x40a4_9d99),
        (0x41db_700a, 0x4098_e2a6),
        (0x3f99_c21c, 0x3e87_6f28),
        (0x40ba_4942, 0x4022_a5eb),
        (0x418c_afeb, 0x4084_5cf4),
        (0x42bb_8978, 0x40d1_a20c),
        (0x42a1_87ae, 0x40ca_bdb7),
        (0x42ab_a257, 0x40cd_8acb),
        (0x4202_bf3f, 0x40a0_faf6),
        (0x3ee3_37f8, 0xbf96_0627),
        (0x4152_cccc, 0x406e_1013),
        (0x4246_1248, 0x40b4_27f4),
        (0x3f7e_c130, 0xbbe6_892a),
        (0x4109_d353, 0x4046_d428),
        (0x4291_c061, 0x40c5_fee9),
        (0x41ed_4acd, 0x409c_7f26),
    ];
}
