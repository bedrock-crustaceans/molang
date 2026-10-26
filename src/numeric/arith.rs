//! The engine's primitive float operations, as a [`Platform`] performs them.
//!
//! Each operation is rounded on its own; the platform chooses the NaN of a NaN result and the
//! integer of an out-of-range conversion.

use super::{Platform, fpu};

/// `a·b + c`, rounded twice: the NaN of the product, then of `c`.
#[inline]
pub fn mul_add(platform: Platform, a: f32, b: f32, c: f32) -> f32 {
    fpu::mul_add(platform, a, b, c)
}

/// `c − a·b`, rounded twice: the NaN of `c`, then of the product.
#[inline]
pub fn mul_sub(platform: Platform, a: f32, b: f32, c: f32) -> f32 {
    fpu::sub(platform, c, fpu::mul(platform, a, b))
}

/// `a + b`.
#[inline]
pub fn add(platform: Platform, a: f32, b: f32) -> f32 {
    fpu::add(platform, a, b)
}

/// `a − b`.
#[inline]
pub fn sub(platform: Platform, a: f32, b: f32) -> f32 {
    fpu::sub(platform, a, b)
}

/// `a·b`.
#[inline]
pub fn mul(platform: Platform, a: f32, b: f32) -> f32 {
    fpu::mul(platform, a, b)
}

/// `a / b`.
#[inline]
pub fn div(platform: Platform, a: f32, b: f32) -> f32 {
    fpu::div(platform, a, b)
}

/// `(a > b) ? a : b` on every platform: `b` when either is NaN or both are zeros.
#[inline]
pub fn max(a: f32, b: f32) -> f32 {
    fpu::max(a, b)
}

/// `(a < b) ? a : b` on every platform: `b` when either is NaN or both are zeros.
#[inline]
pub fn min(a: f32, b: f32) -> f32 {
    fpu::min(a, b)
}

/// Truncating float-to-int. Out of range or NaN, [`Platform::BdsLinux`] and
/// [`Platform::BdsWindows`] give `i32::MIN`; [`Platform::AndroidArm64`] saturates and maps NaN to
/// 0.
#[inline]
pub fn to_int(platform: Platform, v: f32) -> i32 {
    fpu::to_int(platform, v)
}

#[cfg(test)]
mod tests {
    use crate::numeric::test_support::*;

    #[test]
    fn the_wrappers_are_the_primitives() {
        let x = third();
        for p in PLATFORMS {
            assert_bits(super::mul_add(p, x, 3.0, -1.0), 0.0);
            assert_bits(super::mul_sub(p, x, 3.0, 1.0), 0.0);
            assert_bits(super::add(p, 0.1, 0.2), 0.1 + 0.2);
            assert_bits(super::sub(p, 0.1, 0.2), 0.1 - 0.2);
            assert_bits(super::mul(p, 0.1, 0.2), 0.1 * 0.2);
            assert_bits(super::div(p, 0.1, 0.2), 0.1 / 0.2);
        }
        assert_bits(super::max(NAN, 4.0), 4.0);
        assert!(super::max(4.0, NAN).is_nan());
        assert!(super::min(4.0, NAN).is_nan());
        assert_eq!(super::to_int(LINUX, NAN), i32::MIN);
        assert_eq!(super::to_int(ANDROID, NAN), 0);
        assert_eq!(
            super::mul_sub(LINUX, NAN, 1.0, -NAN).to_bits(),
            (-NAN).to_bits()
        );
    }
}
