//! `asinf`, `acosf` and `atanf`. glibc rounds each correctly, so the `f64` result rounded once
//! gives the same bits, except at the few arguments listed here.
//!
//! `asinf` and `acosf` are the `GLIBC_2.2.5` wrappers, the only symbols: `|x| > 1` gives
//! `0x7fc00000`.

use super::WRAPPER_NAN;
use crate::stdlib::math::transcendental;

/// `acosf` where the rounded `f64` result is off, as `(argument bits, result bits)`.
const ACOSF_FIXES: [(u32, u32); 2] = [(0x3288_85a3, 0x3fc9_0fdb), (0x3982_6222, 0x3fc9_07b5)];

/// `atanf` where the rounded `f64` result is off, as `(argument bits, result bits)`.
const ATANF_FIXES: [(u32, u32); 2] = [(0x3d8d_6b23, 0x3d8d_31c3), (0xbd8d_6b23, 0xbd8d_31c3)];

#[inline]
fn fixed(fixes: &[(u32, u32)], x: f32) -> Option<f32> {
    let bits = x.to_bits();
    fixes
        .iter()
        .find(|&&(argument, _)| argument == bits)
        .map(|&(_, result)| f32::from_bits(result))
}

/// `asinf`.
pub(crate) fn asinf(x: f32) -> f32 {
    transcendental::asin(x, WRAPPER_NAN)
}

/// `acosf`.
pub(crate) fn acosf(x: f32) -> f32 {
    fixed(&ACOSF_FIXES, x).unwrap_or_else(|| transcendental::acos(x, WRAPPER_NAN))
}

/// `atanf`.
pub(crate) fn atanf(x: f32) -> f32 {
    fixed(&ATANF_FIXES, x).unwrap_or_else(|| transcendental::atan(x))
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

    #[test]
    fn atanf_matches_glibc() {
        check_rows("atanf", atanf, ATANF);
    }

    fn rounded_acosf(x: f32) -> f32 {
        transcendental::acos(x, WRAPPER_NAN)
    }

    #[test]
    fn each_fix_is_one_ulp_from_the_rounded_result_and_its_neighbours_need_none() {
        for (fixes, f, rounded) in [
            (
                &ACOSF_FIXES,
                acosf as fn(f32) -> f32,
                rounded_acosf as fn(f32) -> f32,
            ),
            (&ATANF_FIXES, atanf, transcendental::atan),
        ] {
            for &(argument, result) in fixes {
                let x = f32::from_bits(argument);
                assert_eq!(f(x).to_bits(), result, "{argument:#010x}");
                assert_eq!(rounded(x).to_bits().abs_diff(result), 1, "{argument:#010x}");
                for neighbour in [argument - 2, argument - 1, argument + 1, argument + 2] {
                    let x = f32::from_bits(neighbour);
                    assert_eq!(f(x).to_bits(), rounded(x).to_bits(), "{neighbour:#010x}");
                }
            }
        }
    }

    #[test]
    fn the_neighbours_of_the_fixes_match_glibc() {
        check_rows("acosf", acosf, ACOSF_NEIGHBOURS);
        check_rows("atanf", atanf, ATANF_NEIGHBOURS);
    }

    /// glibc's `acosf` around its fixes.
    const ACOSF_NEIGHBOURS: &[(u32, u32)] = &[
        (0x3288_85a1, 0x3fc9_0fdb),
        (0x3288_85a2, 0x3fc9_0fdb),
        (0x3288_85a4, 0x3fc9_0fda),
        (0x3288_85a5, 0x3fc9_0fda),
        (0x3982_6220, 0x3fc9_07b5),
        (0x3982_6221, 0x3fc9_07b5),
        (0x3982_6223, 0x3fc9_07b4),
        (0x3982_6224, 0x3fc9_07b4),
    ];

    /// glibc's `atanf` around its fixes.
    const ATANF_NEIGHBOURS: &[(u32, u32)] = &[
        (0x3d8d_6b21, 0x3d8d_31c1),
        (0x3d8d_6b22, 0x3d8d_31c2),
        (0x3d8d_6b24, 0x3d8d_31c3),
        (0x3d8d_6b25, 0x3d8d_31c4),
        (0xbd8d_6b21, 0xbd8d_31c1),
        (0xbd8d_6b22, 0xbd8d_31c2),
        (0xbd8d_6b24, 0xbd8d_31c3),
        (0xbd8d_6b25, 0xbd8d_31c4),
    ];

    /// Reference results of glibc's `asinf`.
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

    /// Reference results of glibc's `acosf`.
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

    /// Reference results of glibc's `atanf`.
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
}
