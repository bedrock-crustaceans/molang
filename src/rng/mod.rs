//! The random sources of the `math.random*` / `math.die_roll*` operators: anything that is a
//! [`RandomSource`].
//!
//! The source is a value the embedding owns. Each draw is one [`RandomSource::sample`]; how the
//! generator's word becomes that sample belongs to the source, and differs by context:
//!
//! | Context | Generator | Mapping |
//! |---|---|---|
//! | Entities, items, blocks, commands, client animation, scripting, particles' `math.random` | one xorshift128 for the whole process (`vm::ProcessRng`) | [`unit_31`] |
//! | Feature rules and the scatter features they place | an MT19937 per chunk and per rule | [`unit_32`] |
//! | Helper-placed features (saplings, `/place feature`) | an MT19937 per thread | [`unit_32`] |
//! | Particles' `random_1..4` variables (not a Molang operator) | an MT19937 per thread | [`unit_32`] |
//! | The UI player preview | none | the constant 0 ([`FixedRng::ZERO`]) |
//!
//! The crate has no generator of its own beyond the process one: a host brings MT19937 (the
//! `rand_mt` crate, say) and maps its words with [`unit_32`]. A closure `FnMut() -> f32` is a
//! source.

#[cfg(feature = "vm")]
pub(crate) mod xorshift;

/// What the random operators draw from: one sample in `[0, 1]` per draw.
///
/// How the generator's output becomes the sample is the source's business (see the
/// [module](self) table). The operators clamp a sample outside `[0, 1]`, NaN to 0. A closure is a
/// source, as are `&mut dyn RandomSource` and `Box<dyn RandomSource>`.
///
/// ```
/// use molang::rng::{FixedRng, RandomSource, unit_32};
///
/// assert_eq!(FixedRng::HALF.sample(), 0.5);
///
/// // A generator of the host's, mapped by a closure.
/// let mut words = [0x8000_0000_u32, u32::MAX].into_iter();
/// let mut closure = move || unit_32(words.next().unwrap_or(0));
/// assert_eq!((closure.sample(), closure.sample()), (0.5, 1.0));
///
/// /// A source of its own: every sample is a quarter.
/// struct Quarter;
/// impl RandomSource for Quarter {
///     fn sample(&mut self) -> f32 {
///         0.25
///     }
/// }
/// let dynamic: &mut dyn RandomSource = &mut Quarter;
/// assert_eq!(dynamic.sample(), 0.25);
/// ```
pub trait RandomSource {
    /// Draws one sample.
    fn sample(&mut self) -> f32;
}

// Forwarding is for trait objects only: a generic `&mut S` or `Box<S>` would overlap the closure
// impl, since both are `FnMut` when `S` is.
impl RandomSource for &mut (dyn RandomSource + '_) {
    #[inline]
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

impl RandomSource for &mut (dyn RandomSource + Send + '_) {
    #[inline]
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

impl RandomSource for Box<dyn RandomSource + '_> {
    #[inline]
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

impl RandomSource for Box<dyn RandomSource + Send + '_> {
    #[inline]
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

impl<F: FnMut() -> f32> RandomSource for F {
    #[inline]
    fn sample(&mut self) -> f32 {
        self()
    }
}

/// The 31-bit mapping: `(word & 0x7fff_ffff) · 2^-31` in `f32`, rounded to nearest.
///
/// The mapping of entities, items, blocks, commands, client animation, scripting and the
/// `math.random` of particles, over the one xorshift128 of the process (`vm::ProcessRng`). The top
/// bit is ignored, and the words from `0x7fff_ffc0` round to exactly 1.0, so the range is
/// `[0, 1]`.
///
/// ```
/// use molang::rng::unit_31;
///
/// assert_eq!(unit_31(1 << 30), 0.5);
/// assert_eq!(unit_31(0x8000_0000 | 1 << 30), 0.5);
/// assert_eq!(unit_31(0x7fff_ffff), 1.0);
/// ```
#[must_use]
#[inline]
pub const fn unit_31(word: u32) -> f32 {
    (word & 0x7fff_ffff) as f32 * (1.0 / 2_147_483_648.0)
}

/// The 32-bit mapping: `word · 2^-32` in `f64`, then rounded to `f32`.
///
/// The mapping of world generation, over an MT19937: feature rules and scatter features (one per
/// chunk and per rule) and helper-placed features (one per thread). Particles map their
/// `random_1..4` variables with it too, which no Molang operator draws. The words from
/// `0xffff_ff80` round to exactly 1.0, `u32::MAX` included, so the range is `[0, 1]`.
///
/// ```
/// use molang::rng::unit_32;
///
/// assert_eq!(unit_32(0x8000_0000), 0.5);
/// assert_eq!(unit_32(u32::MAX), 1.0);
/// ```
#[must_use]
#[inline]
pub const fn unit_32(word: u32) -> f32 {
    (word as f64 * (1.0 / 4_294_967_296.0)) as f32
}

/// A source whose every sample is the same word under the 31-bit mapping ([`unit_31`]).
///
/// ```
/// use molang::rng::{FixedRng, RandomSource};
///
/// assert_eq!(FixedRng::from_sample(0.25).unwrap().sample(), 0.25);
/// assert_eq!(FixedRng::from_sample(1e-10), None, "not k · 2^-31");
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedRng(pub u32);

impl FixedRng {
    /// Always the sample 0: every random operator returns its lower bound.
    pub const ZERO: Self = Self(0);
    /// Always the sample 0.5.
    pub const HALF: Self = Self(1 << 30);
    /// Always the sample 1: every random operator returns its upper bound.
    pub const ONE: Self = Self(0x7fff_ffff);

    /// The source whose sample is `sample`; `None` when no word maps to it: NaN, -0, outside
    /// `[0, 1]`, or not a multiple of 2^-31.
    pub const fn from_sample(sample: f32) -> Option<Self> {
        if sample.is_nan() || sample.is_sign_negative() || sample > 1.0 {
            return None;
        }
        if sample == 1.0 {
            return Some(Self::ONE);
        }
        let scaled = sample * 2_147_483_648.0;
        let word = scaled as u32;
        if word as f32 == scaled {
            Some(Self(word))
        } else {
            None
        }
    }
}

impl RandomSource for FixedRng {
    #[inline]
    fn sample(&mut self) -> f32 {
        unit_31(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct words spread over the whole range.
    fn words(n: u32) -> impl Iterator<Item = u32> {
        (0..n).map(|i| i.wrapping_mul(0x9e37_79b9) ^ (i >> 3))
    }

    #[test]
    fn bits31_masks_the_top_bit_and_scales_by_two_to_the_minus_31() {
        for w in words(64) {
            let expected = (w & 0x7fff_ffff) as f32 * 2.0_f32.powi(-31);
            assert_eq!(unit_31(w).to_bits(), expected.to_bits());
            assert_eq!(unit_31(w).to_bits(), unit_31(w ^ 0x8000_0000).to_bits());
        }
    }

    #[test]
    fn bits31_maps_the_extreme_words_to_the_ends_of_the_unit_interval() {
        assert_eq!(unit_31(1).to_bits(), 0x3000_0000);
        // Spacing below 2^31 is 128: 2^31 − 65 rounds down to 1 − 2^-24, the tie at 2^31 − 64
        // rounds to even, which is 2^31.
        assert_eq!(unit_31(0x7fff_ffbf).to_bits(), 0x3f7f_ffff);
        assert_eq!(unit_31(0x7fff_ffc0), 1.0);
    }

    #[test]
    fn the_mappings_pin_their_ends() {
        assert_eq!(unit_31(0).to_bits(), 0);
        assert_eq!(unit_31(0x7fff_ffff).to_bits(), 0x3f80_0000);
        assert_eq!(unit_31(0x8000_0000).to_bits(), 0);
        assert_eq!(unit_31(u32::MAX).to_bits(), 0x3f80_0000);
        assert_eq!(unit_32(0).to_bits(), 0);
        assert_eq!(unit_32(u32::MAX).to_bits(), 0x3f80_0000);
        assert_eq!(unit_32(0xffff_ff7f).to_bits(), 0x3f7f_ffff);
        assert_eq!(unit_32(0xffff_ff80).to_bits(), 0x3f80_0000);
        assert_eq!(unit_32(1).to_bits(), 0x2f80_0000);
    }

    #[test]
    fn the_32_bit_mapping_keeps_the_top_bit_and_rounds_once() {
        assert_eq!(unit_32(0x8000_0000), 0.5);
        // 2^25 + 3 rounds up under the 32-bit mapping; its half, 2^24 + 1, under the 31-bit
        // mapping is a tie and rounds to even.
        assert_eq!(unit_32((1 << 25) + 3).to_bits(), 0x3c00_0001);
        assert_eq!(unit_31((1 << 24) + 1).to_bits(), 0x3c00_0000);
    }

    #[test]
    fn a_closure_is_a_source_drawn_once_per_sample() {
        let mut calls = 0;
        let mut closure = || {
            calls += 1;
            unit_31(calls << 29)
        };
        assert_eq!(
            [closure.sample(), closure.sample(), closure.sample()],
            [0.25, 0.5, 0.75]
        );
        assert_eq!(calls, 3);
    }

    #[test]
    fn a_dynamic_or_boxed_source_forwards_to_the_source() {
        let mut fixed = FixedRng::HALF;
        let mut dynamic: &mut dyn RandomSource = &mut fixed;
        assert_eq!(RandomSource::sample(&mut dynamic), 0.5);
        let mut sendable: &mut (dyn RandomSource + Send) = &mut fixed;
        assert_eq!(RandomSource::sample(&mut sendable), 0.5);
        let mut boxed: Box<dyn RandomSource> = Box::new(FixedRng::ONE);
        assert_eq!(boxed.sample(), 1.0);
        let mut boxed: Box<dyn RandomSource + Send> = Box::new(|| 0.125);
        assert_eq!(boxed.sample(), 0.125);
        let mut count = 0.0;
        let mut closure = || {
            count += 1.0;
            count
        };
        let borrowed = &mut closure;
        assert_eq!((borrowed.sample(), borrowed.sample()), (1.0, 2.0));
    }

    #[test]
    fn the_sources_are_send_and_sync() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<FixedRng>();
    }

    #[test]
    fn fixed_sources() {
        for (mut source, expected) in [
            (FixedRng::ZERO, 0.0),
            (FixedRng::HALF, 0.5),
            (FixedRng::ONE, 1.0),
        ] {
            assert_eq!(source.sample(), expected);
            assert_eq!(FixedRng::from_sample(expected), Some(source));
        }
        let mut fixed = FixedRng(0x1234_5678);
        assert_eq!(fixed.sample(), unit_31(0x1234_5678));
        assert_eq!(fixed.sample(), unit_31(0x1234_5678));
        assert_eq!(fixed, FixedRng(0x1234_5678), "drawing leaves it as it was");
    }

    #[test]
    fn from_sample_takes_exactly_the_samples_a_word_maps_to() {
        for value in [
            0.25,
            0.3,
            0.6,
            1.0 / 3.0,
            0.999_999_94,
            2.0_f32.powi(-8),
            2.0_f32.powi(-31),
            3.0 * 2.0_f32.powi(-31),
        ] {
            let mut fixed = FixedRng::from_sample(value).unwrap_or_else(|| panic!("{value}"));
            assert_eq!(fixed.sample().to_bits(), value.to_bits(), "{value}");
        }
        for value in [
            -0.0,
            -1.0,
            1.000_000_1,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e-10,
            2.0_f32.powi(-32),
            1.5 * 2.0_f32.powi(-31),
        ] {
            assert_eq!(FixedRng::from_sample(value), None, "{value}");
        }
        assert_eq!(FixedRng::from_sample(0.0), Some(FixedRng::ZERO));
    }

    #[test]
    fn every_word_round_trips_through_from_sample() {
        for word in words(10_000) {
            let value = FixedRng(word).sample();
            assert_eq!(
                FixedRng::from_sample(value).map(|mut f| f.sample().to_bits()),
                Some(value.to_bits())
            );
        }
    }
}
