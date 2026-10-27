//! bionic's `asinf`, `acosf`, `atanf` and `atan2f`, which it builds from FreeBSD's code: the
//! `libm` crate's port of that code, with FreeBSD's results where the port departs from it.
//!
//! The port adds `0x1p-120` where FreeBSD adds the low part of `π/2`, so FreeBSD rounds `acos(0)`,
//! `acos(-1)` and `atan(±∞)` one ulp higher. Neither is checked against a device.

/// `π/2` rounded, FreeBSD's `pio2_hi + pio2_lo`.
const PI_2: f32 = f32::from_bits(0x3fc9_0fdb);
/// `π` rounded.
const PI: f32 = f32::from_bits(0x4049_0fdb);
/// `π/4` rounded.
const PI_4: f32 = f32::from_bits(0x3f49_0fdb);
/// `π` less [`PI`].
const PI_LOW: f32 = f32::from_bits(0xb3bb_bd2e);

pub(crate) fn asinf(x: f32) -> f32 {
    libm::asinf(x)
}

pub(crate) fn acosf(x: f32) -> f32 {
    let ix = x.to_bits() & 0x7fff_ffff;
    if ix <= 0x3280_0000 {
        PI_2
    } else if x == -1.0 {
        PI
    } else {
        libm::acosf(x)
    }
}

pub(crate) fn atanf(x: f32) -> f32 {
    let ix = x.to_bits() & 0x7fff_ffff;
    if (0x4c80_0000..=0x7f80_0000).contains(&ix) {
        PI_2.copysign(x)
    } else {
        libm::atanf(x)
    }
}

/// `atan2f(y, x)` through [`atanf`].
pub(crate) fn atan2f(y: f32, x: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x == 1.0 {
        return atanf(y);
    }
    let (y_negative, x_negative) = (y.is_sign_negative(), x.is_sign_negative());
    let iy = y.to_bits() & 0x7fff_ffff;
    let ix = x.to_bits() & 0x7fff_ffff;
    let signed = |z: f32| if y_negative { -z } else { z };
    if iy == 0 {
        return if x_negative { signed(PI) } else { y };
    }
    if ix == 0 {
        return signed(PI_2);
    }
    if ix == 0x7f80_0000 {
        return signed(match (iy == 0x7f80_0000, x_negative) {
            (true, false) => PI_4,
            (true, true) => 3.0 * PI_4,
            (false, false) => 0.0,
            (false, true) => PI,
        });
    }
    if iy == 0x7f80_0000 {
        return signed(PI_2);
    }
    // The exponent difference; both are finite and non-zero, so the bits fit an `i32`.
    let k = (iy.cast_signed() - ix.cast_signed()) >> 23;
    if k > 26 {
        return signed(PI_2 + 0.5 * PI_LOW);
    }
    let z = if k < -26 && x_negative {
        0.0
    } else {
        atanf((y / x).abs())
    };
    match (y_negative, x_negative) {
        (false, false) => z,
        (true, false) => -z,
        (false, true) => PI - (z - PI_LOW),
        (true, true) => (z - PI_LOW) - PI,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f32) -> u32 {
        x.to_bits()
    }

    #[test]
    fn freebsd_rounds_the_special_points_up_where_the_port_rounds_down() {
        for (got, port, expected) in [
            (acosf(0.0), libm::acosf(0.0), 0x3fc9_0fdb),
            (acosf(-0.0), libm::acosf(-0.0), 0x3fc9_0fdb),
            (acosf(1e-9), libm::acosf(1e-9), 0x3fc9_0fdb),
            (acosf(-1.0), libm::acosf(-1.0), 0x4049_0fdb),
            (
                atanf(f32::INFINITY),
                libm::atanf(f32::INFINITY),
                0x3fc9_0fdb,
            ),
            (
                atanf(f32::NEG_INFINITY),
                libm::atanf(f32::NEG_INFINITY),
                0xbfc9_0fdb,
            ),
            (atanf(1e20), libm::atanf(1e20), 0x3fc9_0fdb),
            (
                atan2f(f32::INFINITY, 1.0),
                libm::atan2f(f32::INFINITY, 1.0),
                0x3fc9_0fdb,
            ),
        ] {
            assert_eq!(bits(got), expected);
            assert_eq!(bits(port), expected - 1);
        }
    }

    #[test]
    fn elsewhere_the_port_is_kept() {
        for x in [
            0x3280_0001_u32,
            0x3f00_0000,
            0x3f61_50ee,
            0xbf5e_aa78,
            0x3f7f_ffff,
            0xbf7f_ffff,
            0x4c7f_ffff,
            0x3f80_0000,
            0x3f80_0001,
            0x7fc0_0000,
        ] {
            let x = f32::from_bits(x);
            for (got, port) in [
                (asinf(x), libm::asinf(x)),
                (acosf(x), libm::acosf(x)),
                (atanf(x), libm::atanf(x)),
            ] {
                assert!(
                    bits(got) == bits(port) || got.is_nan() && port.is_nan(),
                    "{x}"
                );
            }
        }
    }

    #[test]
    fn atan2f_agrees_with_the_port_off_its_special_points() {
        let values = [
            -1e30_f32,
            -1e10,
            -7.5,
            -3.0,
            -1.0,
            -0.25,
            -1e-10,
            -0.0,
            0.0,
            1e-30,
            1e-10,
            0.25,
            0.5,
            2.0,
            3.0,
            7.5,
            1e10,
            1e30,
            -f32::INFINITY,
            f32::INFINITY,
        ];
        for y in values {
            for x in values {
                if x == 1.0 || (y / x).abs() >= 67_108_864.0 {
                    continue;
                }
                assert_eq!(
                    bits(atan2f(y, x)),
                    bits(libm::atan2f(y, x)),
                    "atan2({y}, {x})"
                );
            }
        }
        assert_eq!(bits(atan2f(0.0, -1.0)), bits(PI));
        assert_eq!(bits(atan2f(-0.0, -1.0)), bits(-PI));
        assert_eq!(bits(atan2f(1.0, 0.0)), 0x3fc9_0fdb);
        assert_eq!(bits(atan2f(1e30, 1e-10)), 0x3fc9_0fdb);
        assert_eq!(
            bits(atan2f(f32::INFINITY, f32::NEG_INFINITY)),
            bits(3.0 * PI_4)
        );
        assert_eq!(bits(atan2f(2.0, 1.0)), bits(atanf(2.0)));
        assert!(atan2f(f32::NAN, 1.0).is_nan());
    }
}
