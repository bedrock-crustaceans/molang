//! The readings of a result: how a field of the game turns an evaluated [`Value`] into what it
//! needs.

use std::ops::RangeInclusive;

use super::Value;
use crate::numeric::Platform;
use crate::numeric::arith::to_int;
use crate::numeric::truthy;
use crate::vm::host::Host;

/// What an int entity property adds before it truncates: 2^-23.
const PROPERTY_BIAS: f32 = f32::EPSILON;

/// The readings of a result. Each starts from [`Value::to_f32`] (a string is the low bits of its
/// hash, every handle and resource 0.0), and a field that wants a float takes `to_f32` as is. The
/// readings that truncate depend on the [`Platform`]: out of range or NaN saturates (NaN to 0),
/// except on the emulated servers (`platform-quirks`), which give `i32::MIN`.
///
/// | Field | Reading |
/// | --- | --- |
/// | block permutation conditions, block and item event trigger conditions, `bone_visibility` | [`rounds_to_nonzero`](Self::rounds_to_nonzero) |
/// | item and block descriptor `tags`, bool entity-property defaults | [`is_nonzero`](Self::is_nonzero) |
/// | ambient-sound conditions | [`exceeds_epsilon`](Self::exceeds_epsilon) |
/// | anger-sound conditions | [`is_positive`](Self::is_positive) |
/// | experience reward entries (summed by the caller) | [`round_to_int`](Self::round_to_int) |
/// | int entity-property defaults | [`to_property_int`](Self::to_property_int) |
/// | float entity-property defaults | [`to_property_float`](Self::to_property_float) |
/// | animation values, render-controller colours and the other float fields | [`to_f32`](Self::to_f32) |
impl<H: Host> Value<H> {
    /// `x != 0`: NaN is true, `-0.0` false. The reading of item and block descriptor `tags` and of
    /// bool entity-property defaults.
    #[inline]
    pub fn is_nonzero(&self) -> bool {
        truthy(self.to_f32())
    }

    /// `x > 0`: NaN is false. The reading of anger-sound conditions.
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.to_f32() > 0.0
    }

    /// `x > 2^-23`: NaN is false. The reading of ambient-sound conditions.
    #[inline]
    pub fn exceeds_epsilon(&self) -> bool {
        self.to_f32() > f32::EPSILON
    }

    /// `x` truncated to an int as `platform` converts: `2.9` is 2, `-2.9` is −2.
    #[inline]
    pub fn to_int(&self, platform: Platform) -> i32 {
        to_int(platform, self.to_f32())
    }

    /// `x` rounded half away from zero, then truncated as `platform` converts. The reading of an
    /// experience reward entry, which the caller sums.
    #[inline]
    pub fn round_to_int(&self, platform: Platform) -> i32 {
        to_int(platform, self.to_f32().round())
    }

    /// `(int)round(x) != 0`: `0.4` is false and `0.5` true, and a NaN or an `x` beyond the int
    /// range follows `platform`'s conversion (NaN is true on a server and false on Android). The
    /// reading of block permutation conditions, block and item event trigger conditions and
    /// `bone_visibility`.
    #[inline]
    pub fn rounds_to_nonzero(&self, platform: Platform) -> bool {
        self.round_to_int(platform) != 0
    }

    /// `clamp(trunc(x + 2^-23), min, max)` with `platform`'s truncation: the default of an int
    /// entity property. NaN gives `min` on a server and 0 clamped on Android, and `1e10` in
    /// `0..=10` gives 0 on a server and 10 on Android. `None` when the value is not a number
    /// ([`Value::Float`]); the caller logs it and chooses the fallback.
    ///
    /// # Panics
    ///
    /// In a debug build, when the range is empty.
    pub fn to_property_int(&self, range: RangeInclusive<i32>, platform: Platform) -> Option<i32> {
        debug_assert!(!range.is_empty(), "an empty property range");
        let Self::Float(x) = self else { return None };
        Some(
            to_int(platform, *x + PROPERTY_BIAS)
                .max(*range.start())
                .min(*range.end()),
        )
    }

    /// `x > max ? max : (x > min ? x : min)`: the default of a float entity property. NaN gives
    /// `min` on every platform. `None` when the value is not a number ([`Value::Float`]).
    ///
    /// # Panics
    ///
    /// In a debug build, when the range is empty or has a NaN bound.
    pub fn to_property_float(&self, range: RangeInclusive<f32>) -> Option<f32> {
        debug_assert!(!range.is_empty(), "an empty property range");
        let Self::Float(x) = *self else { return None };
        let (min, max) = range.into_inner();
        Some(if x > max {
            max
        } else if x > min {
            x
        } else {
            min
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::HashedStr;
    use crate::vm::host::NoHost;

    type V = Value<NoHost>;

    use crate::numeric::test_support::PLATFORMS;

    #[cfg(feature = "platform-quirks")]
    const SERVERS: [Platform; 2] = [Platform::BdsLinux, Platform::BdsWindows];
    #[cfg(feature = "platform-quirks")]
    const ANDROID: Platform = Platform::AndroidArm64;
    const NAN: f32 = f32::NAN;
    const INF: f32 = f32::INFINITY;
    const TWO_31: f32 = 2_147_483_648.0;

    fn f(x: f32) -> V {
        V::Float(x)
    }

    #[test]
    fn the_float_readings_ignore_the_platform() {
        // (x, is_nonzero, is_positive, exceeds_epsilon)
        let cases: [(f32, bool, bool, bool); 10] = [
            (0.0, false, false, false),
            (-0.0, false, false, false),
            (0.4, true, true, true),
            (-0.5, true, false, false),
            (1.0e-8, true, true, false),
            (1.192_092_9e-7, true, true, false),
            (2.0e-7, true, true, true),
            (INF, true, true, true),
            (-INF, true, false, false),
            (NAN, true, false, false),
        ];
        for (x, nonzero, positive, epsilon) in cases {
            assert_eq!(f(x).is_nonzero(), nonzero, "{x}");
            assert_eq!(f(x).is_positive(), positive, "{x}");
            assert_eq!(f(x).exceeds_epsilon(), epsilon, "{x}");
        }
    }

    #[test]
    fn rounding_is_half_away_from_zero_before_the_truncation() {
        for p in PLATFORMS {
            assert_eq!(f(0.4).round_to_int(p), 0);
            assert_eq!(f(0.5).round_to_int(p), 1);
            assert_eq!(f(-0.5).round_to_int(p), -1);
            assert_eq!(f(-0.4).round_to_int(p), 0);
            assert_eq!(f(2.5).round_to_int(p), 3);
            assert_eq!(f(-2.5).round_to_int(p), -3);
            assert_eq!(f(0.0).round_to_int(p), 0);
            assert_eq!(f(-0.0).round_to_int(p), 0);
            assert_eq!(f(2.9).to_int(p), 2);
            assert_eq!(f(-2.9).to_int(p), -2);
            assert!(!f(0.4).rounds_to_nonzero(p));
            assert!(f(0.5).rounds_to_nonzero(p));
            assert!(f(-0.5).rounds_to_nonzero(p));
            assert!(!f(-0.0).rounds_to_nonzero(p));
        }
    }

    #[test]
    fn natively_the_integer_readings_saturate_out_of_range() {
        let p = Platform::Native;
        assert_eq!(f(NAN).to_int(p), 0);
        assert_eq!(f(NAN).round_to_int(p), 0);
        assert!(!f(NAN).rounds_to_nonzero(p));
        for x in [INF, TWO_31, 1.0e10] {
            assert_eq!(f(x).to_int(p), i32::MAX, "{x}");
            assert_eq!(f(x).round_to_int(p), i32::MAX, "{x}");
            assert!(f(x).rounds_to_nonzero(p), "{x}");
        }
        for x in [-INF, -TWO_31, -1.0e10] {
            assert_eq!(f(x).to_int(p), i32::MIN, "{x}");
            assert!(f(x).rounds_to_nonzero(p), "{x}");
        }
        assert_eq!(f(1.0e10).to_property_int(0..=10, p), Some(10));
        assert_eq!(f(INF).to_property_int(0..=10, p), Some(10));
        assert_eq!(f(NAN).to_property_int(0..=10, p), Some(0));
        assert_eq!(f(NAN).to_property_int(3..=10, p), Some(3));
        assert_eq!(f(-1.0e10).to_property_int(-3..=10, p), Some(-3));
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn the_integer_readings_follow_the_platform_out_of_range() {
        for p in SERVERS {
            for x in [NAN, INF, -INF, TWO_31, 1.0e10, -1.0e10] {
                assert_eq!(f(x).to_int(p), i32::MIN, "{p:?} {x}");
                assert_eq!(f(x).round_to_int(p), i32::MIN, "{p:?} {x}");
                assert!(f(x).rounds_to_nonzero(p), "{p:?} {x}");
            }
        }
        assert_eq!(f(NAN).to_int(ANDROID), 0);
        assert_eq!(f(NAN).round_to_int(ANDROID), 0);
        assert!(!f(NAN).rounds_to_nonzero(ANDROID));
        for x in [INF, TWO_31, 1.0e10] {
            assert_eq!(f(x).to_int(ANDROID), i32::MAX, "{x}");
            assert_eq!(f(x).round_to_int(ANDROID), i32::MAX, "{x}");
            assert!(f(x).rounds_to_nonzero(ANDROID), "{x}");
        }
        for x in [-INF, -TWO_31, -1.0e10] {
            assert_eq!(f(x).to_int(ANDROID), i32::MIN, "{x}");
            assert!(f(x).rounds_to_nonzero(ANDROID), "{x}");
        }
    }

    #[test]
    fn a_property_default_biases_truncates_and_clamps() {
        for p in PLATFORMS {
            assert_eq!(f(3.0).to_property_int(0..=10, p), Some(3));
            assert_eq!(f(2.9).to_property_int(0..=10, p), Some(2));
            assert_eq!(f(-0.5).to_property_int(0..=10, p), Some(0));
            assert_eq!(f(-5.0).to_property_int(-3..=10, p), Some(-3));
            assert_eq!(f(11.0).to_property_int(0..=10, p), Some(10));
            assert_eq!(f(-0.0).to_property_int(0..=10, p), Some(0));
            assert_eq!(f(-INF).to_property_int(-4..=10, p), Some(-4));
            // The bias keeps a value just below an integer from truncating down.
            assert_eq!(f(2.999_999_8).to_property_int(0..=10, p), Some(3));
        }
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn a_property_default_out_of_range_differs_by_platform() {
        for p in SERVERS {
            for x in [1.0e10, TWO_31, NAN, INF] {
                assert_eq!(f(x).to_property_int(0..=10, p), Some(0), "{p:?} {x}");
                assert_eq!(f(x).to_property_int(-2..=10, p), Some(-2), "{p:?} {x}");
            }
            assert_eq!(f(-1.0e10).to_property_int(0..=10, p), Some(0));
        }
        assert_eq!(f(1.0e10).to_property_int(0..=10, ANDROID), Some(10));
        assert_eq!(f(TWO_31).to_property_int(0..=10, ANDROID), Some(10));
        assert_eq!(f(INF).to_property_int(0..=10, ANDROID), Some(10));
        assert_eq!(f(NAN).to_property_int(0..=10, ANDROID), Some(0));
        assert_eq!(f(NAN).to_property_int(3..=10, ANDROID), Some(3));
        assert_eq!(f(NAN).to_property_int(-3..=10, ANDROID), Some(0));
        assert_eq!(f(-1.0e10).to_property_int(-3..=10, ANDROID), Some(-3));
    }

    #[test]
    fn other_kinds_read_as_arithmetic_does() {
        let p = Platform::default();
        let hash = V::Hash(HashedStr::from_u64(0x3fc0_0000)); // 1.5 in the low 32 bits
        assert!(hash.is_nonzero() && hash.is_positive() && hash.exceeds_epsilon());
        assert_eq!(hash.to_int(p), 1);
        assert_eq!(hash.round_to_int(p), 2);
        assert!(hash.rounds_to_nonzero(p));
        assert_eq!(hash.to_property_int(0..=10, p), None);
        assert_eq!(hash.to_property_float(0.0..=10.0), None);

        let empty = V::Hash(HashedStr::from_u64(0));
        let actor = V::Actor(());
        for zero in [&empty, &actor] {
            assert!(!zero.is_nonzero() && !zero.is_positive() && !zero.exceeds_epsilon());
            assert!(!zero.rounds_to_nonzero(p));
            assert_eq!(zero.to_int(p), 0);
            assert_eq!(zero.to_property_int(2..=10, p), None);
        }
    }

    #[test]
    fn a_float_property_default_picks_by_comparison_so_nan_gives_min() {
        let r = 1.0..=4.0;
        for (x, want) in [
            (2.5, 2.5),
            (0.0, 1.0),
            (9.0, 4.0),
            (f32::INFINITY, 4.0),
            (f32::NEG_INFINITY, 1.0),
            (f32::NAN, 1.0),
            (1.0, 1.0),
            (4.0, 4.0),
        ] {
            assert_eq!(f(x).to_property_float(r.clone()), Some(want), "{x}");
        }
        assert_eq!(V::Actor(()).to_property_float(r), None);
    }
}
