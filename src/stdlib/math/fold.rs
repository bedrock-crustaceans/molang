//! The constant fold of `math.mod`.

use crate::numeric::Platform;

/// An all-constant `math.mod` is folded to the plain truncated remainder, without the post-op a
/// run-time `math.mod` applies (which turns a −0 remainder into +0): `math.mod(-4, 2)` folds to −0.
pub(crate) fn modulo(platform: Platform, a: f32, b: f32) -> f32 {
    super::cmath::rem(platform, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;

    #[test]
    fn the_fold_keeps_the_sign_of_a_zero_remainder() {
        for p in PLATFORMS {
            assert_eq!(modulo(p, -4.0, 2.0).to_bits(), (-0.0_f32).to_bits());
            assert_eq!(modulo(p, 4.0, 2.0).to_bits(), 0.0_f32.to_bits());
            assert_eq!(modulo(p, 7.0, 3.0), 1.0);
        }
    }
}
