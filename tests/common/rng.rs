//! A host's own xorshift128, from the `rand_xorshift` crate, as a random source of its own per
//! test.

use molang::rng::{RandomSource, unit_31};
use rand_core::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

/// The standard seeds `[x, y, z, w]` as `rand_xorshift` takes them.
const STANDARD_SEED: [u32; 4] = [123_456_789, 362_436_069, 521_288_629, 88_675_123];

/// xorshift128 from the standard seeds under the 31-bit mapping: the sequence of
/// `vm::ProcessRng` after a reset, in a value of its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xorshift128(XorShiftRng);

impl Xorshift128 {
    pub fn new() -> Self {
        let [x, y, z, w] = STANDARD_SEED.map(u32::to_le_bytes);
        Self(XorShiftRng::from_seed(
            [x, y, z, w].concat().try_into().expect("16 bytes"),
        ))
    }

    /// A generator of its own from `seed`, `rand_core`'s default expansion of it.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self(XorShiftRng::seed_from_u64(seed))
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
}

impl Default for Xorshift128 {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSource for Xorshift128 {
    fn sample(&mut self) -> f32 {
        unit_31(self.next_u32())
    }
}
