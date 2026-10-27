//! The xorshift128 behind `vm::ProcessRng`, kept here because it must be `const` for a `static`.

use super::{RandomSource, unit_31};

/// Marsaglia's xorshift128 from the standard seeds; a sample is one step under [`unit_31`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xorshift128 {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl Xorshift128 {
    /// A generator from the standard seeds.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            x: 123_456_789,
            y: 362_436_069,
            z: 521_288_629,
            w: 88_675_123,
        }
    }

    /// Advances the state and returns the new `w`.
    #[inline]
    pub const fn next_u32(&mut self) -> u32 {
        let t = self.x ^ (self.x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w = self.w ^ (self.w >> 19) ^ t ^ (t >> 8);
        self.w
    }
}

impl Default for Xorshift128 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSource for Xorshift128 {
    #[inline]
    fn sample(&mut self) -> f32 {
        unit_31(self.next_u32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_outputs_from_the_standard_seeds() {
        let mut words = Xorshift128::new();
        assert_eq!(
            [words.next_u32(), words.next_u32(), words.next_u32()],
            [3_701_687_786, 458_299_110, 2_500_872_618]
        );
        let mut samples = Xorshift128::new();
        assert_eq!(
            [samples.sample(), samples.sample(), samples.sample()].map(f32::to_bits),
            [0x3f39_468c, 0x3e5a_88b7, 0x3e28_824d]
        );
        assert_eq!(words, samples);
        assert_eq!(Xorshift128::default(), Xorshift128::new());
    }
}
