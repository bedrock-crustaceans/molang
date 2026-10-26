//! The random sources of the `math.random*` / `math.die_roll*` operators: anything that is a
//! [`RandomSource`].
//!
//! The source is a value the embedding owns, so evaluation is reproducible and
//! thread-independent. Each draw is one [`RandomSource::sample`], whose mapping from the
//! generator's output belongs to the source:
//!
//! | Source | Sample from one 32-bit output `u` | Used by |
//! |---|---|---|
//! | [`Xorshift128`], [`Bits31`], [`FixedRng`] | `(u & 0x7fff_ffff) · 2^-31`, in `f32` | entities, items, blocks, commands, particles' `math.random` |
//! | [`MersenneTwister`] | `u · 2^-32` in `f64`, then rounded to `f32` | world generation: feature rules and scatter features |
//!
//! Both mappings can give exactly 1. `rand_core` is re-exported here, so a host needs no
//! dependency of its own.

use rand_core::{Infallible, TryRng, utils};

mod mersenne_twister;
mod xorshift;

pub use mersenne_twister::MersenneTwister;
pub use xorshift::Xorshift128;

/// The `rand_core` 0.10 crate, whose `Rng` [`Bits31`] adapts and the generators here implement;
/// a new major version of it is a breaking change here.
pub use rand_core;

/// What the random operators draw from: one sample in `[0, 1]` per draw.
///
/// How the generator's output becomes the sample is the source's business (see the
/// [module](self) table). The operators clamp a sample outside `[0, 1]`, NaN to 0.
///
/// ```
/// use molang::rng::{Bits31, FixedRng, MersenneTwister, RandomSource, Xorshift128};
///
/// assert_eq!(FixedRng::HALF.sample(), 0.5);
/// assert_eq!(Xorshift128::new().sample(), Bits31(Xorshift128::new()).sample());
/// assert_ne!(MersenneTwister::new(5489).sample(), Bits31(MersenneTwister::new(5489)).sample());
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

impl<S: RandomSource + ?Sized> RandomSource for &mut S {
    #[inline]
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

impl<S: RandomSource + ?Sized> RandomSource for Box<S> {
    #[inline]
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

/// The 31-bit mapping: `(word & 0x7fff_ffff) · 2^-31`, rounded to nearest, which can be exactly
/// 1.0.
#[inline]
const fn bits31(word: u32) -> f32 {
    (word & 0x7fff_ffff) as f32 * (1.0 / 2_147_483_648.0)
}

/// Draws one `next_u32` of `rng` under the 31-bit mapping.
#[inline]
pub fn sample<R: rand_core::Rng + ?Sized>(rng: &mut R) -> f32 {
    bits31(rng.next_u32())
}

/// Any `rand_core` generator as a [`RandomSource`] with the 31-bit mapping: each sample is one
/// `next_u32` `u` mapped to `(u & 0x7fff_ffff) · 2^-31`, rounded to nearest.
///
/// ```
/// use molang::rng::{Bits31, FixedRng, RandomSource};
///
/// assert_eq!(Bits31(FixedRng(0x8000_0000 | 1 << 30)).sample(), 0.5, "the top bit is ignored");
/// assert_eq!(Bits31(FixedRng(u32::MAX)).sample(), 1.0);
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bits31<R>(pub R);

impl<R: rand_core::Rng> RandomSource for Bits31<R> {
    #[inline]
    fn sample(&mut self) -> f32 {
        bits31(self.0.next_u32())
    }
}

/// A source whose every `next_u32` is the same word, so every sample is that word under the
/// 31-bit mapping.
///
/// `next_u64` is the word in both halves; `fill_bytes` repeats its little-endian bytes.
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
        bits31(self.0)
    }
}

impl TryRng for FixedRng {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.0)
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        utils::next_u64_via_u32(self)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        utils::fill_bytes_via_next_word(dst, || self.try_next_u32())
    }
}

#[cfg(test)]
mod tests {
    use rand_core::Rng;

    use super::*;

    /// Plays back its words in order and counts the calls of each method.
    #[derive(Default)]
    struct Counted {
        words: Vec<u32>,
        u32_calls: usize,
        other_calls: usize,
    }

    impl TryRng for Counted {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Infallible> {
            self.u32_calls += 1;
            Ok(self.words[self.u32_calls - 1])
        }

        fn try_next_u64(&mut self) -> Result<u64, Infallible> {
            self.other_calls += 1;
            Ok(0)
        }

        fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Infallible> {
            self.other_calls += 1;
            Ok(())
        }
    }

    #[test]
    fn a_bits31_sample_is_one_next_u32_and_nothing_else() {
        let mut source = Bits31(Counted {
            words: vec![0, 1 << 30, 0x7fff_ffff, 0x8000_0000, u32::MAX, 0x1234_5678],
            ..Counted::default()
        });
        let samples: Vec<f32> = (0..6).map(|_| source.sample()).collect();
        assert_eq!(
            samples,
            [
                0.0,
                0.5,
                1.0,
                0.0,
                1.0,
                0x1234_5678 as f32 / 2_147_483_648.0
            ]
        );
        assert_eq!((source.0.u32_calls, source.0.other_calls), (6, 0));
    }

    #[test]
    fn bits31_masks_the_top_bit_and_scales_by_two_to_the_minus_31() {
        let mut words = Xorshift128::new();
        for _ in 0..64 {
            let w = words.next_u32();
            let expected = (w & 0x7fff_ffff) as f32 * 2.0_f32.powi(-31);
            assert_eq!(bits31(w).to_bits(), expected.to_bits());
            assert_eq!(bits31(w).to_bits(), bits31(w ^ 0x8000_0000).to_bits());
        }
    }

    #[test]
    fn bits31_maps_the_extreme_words_to_the_ends_of_the_unit_interval() {
        assert_eq!(bits31(0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(bits31(1).to_bits(), 0x3000_0000);
        // Spacing below 2^31 is 128: 2^31 − 65 rounds down to 1 − 2^-24, the tie at 2^31 − 64
        // rounds to even, which is 2^31.
        assert_eq!(bits31(0x7fff_ffbf).to_bits(), 0x3f7f_ffff);
        assert_eq!(bits31(0x7fff_ffc0), 1.0);
        assert_eq!(bits31(0x7fff_ffff), 1.0);
        assert_eq!(bits31(0x8000_0000).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn the_adapter_over_a_generator_samples_as_the_generator_itself() {
        let mut adapted = Bits31(Xorshift128::new());
        let mut own = Xorshift128::new();
        for _ in 0..16 {
            assert_eq!(adapted.sample().to_bits(), own.sample().to_bits());
        }
        let mut borrowed = Xorshift128::new();
        let mut dynamic: Bits31<&mut dyn Rng> = Bits31(&mut borrowed);
        assert_eq!(
            dynamic.sample().to_bits(),
            Xorshift128::new().sample().to_bits()
        );
    }

    #[test]
    fn a_dynamic_borrowed_or_boxed_source_is_a_source() {
        let mut rng = Xorshift128::new();
        let first = RandomSource::sample(&mut &mut rng);
        let dynamic: &mut dyn RandomSource = &mut rng;
        let second = dynamic.sample();
        let mut boxed: Box<dyn RandomSource> = Box::new(rng);
        let third = boxed.sample();
        let mut reference = Xorshift128::new();
        assert_eq!(
            (first, second, third),
            (reference.sample(), reference.sample(), reference.sample())
        );
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
        assert_eq!(
            (fixed.next_u32(), fixed.next_u32()),
            (0x1234_5678, 0x1234_5678)
        );
        assert_eq!(fixed.next_u64(), 0x1234_5678_1234_5678);
        let mut bytes = [0_u8; 6];
        fixed.fill_bytes(&mut bytes);
        assert_eq!(bytes, [0x78, 0x56, 0x34, 0x12, 0x78, 0x56]);
        assert_eq!(fixed.sample(), Bits31(fixed).sample());
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
        let mut rng = Xorshift128::new();
        for _ in 0..10_000 {
            let value = FixedRng(rng.next_u32()).sample();
            assert_eq!(
                FixedRng::from_sample(value).map(|mut f| f.sample().to_bits()),
                Some(value.to_bits())
            );
        }
    }
}
