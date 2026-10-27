//! [`MersenneTwister`], the generator of world generation.

use std::fmt;

use rand_core::{Infallible, TryRng, utils};

use super::{RandomSource, unit_32};

/// Words of state.
const N: usize = 624;
/// The offset of the word mixed into each twisted word.
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The 32-bit Mersenne Twister, MT19937: `init_genrand` seeding and `genrand_int32` output.
///
/// A [`RandomSource::sample`] is one output `u` as `u · 2^-32` in `f64`, then rounded to `f32`:
/// every output from `0xffff_ff80` up rounds to exactly 1.0. Unlike the 31-bit mapping, it uses
/// all 32 bits of the output, on a 2^-32 grid. `next_u32` is one output; `next_u64` is two, the
/// first in the low half; `fill_bytes` takes one per 4 bytes, little-endian.
///
/// The constructors `for_*` seed the generators of world generation; the per-chunk decoration
/// generator is what a feature rule's `math.random` draws from. A level seed is taken as its low
/// 32 bits.
///
/// ```
/// use molang::rng::{MersenneTwister, RandomSource, rand_core::Rng};
///
/// let mut mt = MersenneTwister::new(5489);
/// assert_eq!(mt.clone().next_u32(), 3_499_211_612);
/// assert_eq!(mt.sample(), (3_499_211_612.0_f64 / 4_294_967_296.0) as f32);
/// assert_eq!(mt.next_int(), 581_869_302 >> 1);
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct MersenneTwister {
    seed: u32,
    state: [u32; N],
    /// The next word to temper; `N` when the state must be twisted first.
    index: usize,
}

impl MersenneTwister {
    /// The generator seeded with `seed` (`init_genrand`): `mt[0] = seed`, then
    /// `mt[i] = 1812433253 · (mt[i-1] ^ (mt[i-1] >> 30)) + i`, wrapping, for `i` up to 623.
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        let mut state = [0; N];
        state[0] = seed;
        let mut i = 1;
        while i < N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253_u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
            i += 1;
        }
        Self {
            seed,
            state,
            index: N,
        }
    }

    /// The seed this generator was created with, whatever it has drawn since.
    #[must_use]
    pub const fn seed(&self) -> u32 {
        self.seed
    }

    /// The generator of a chunk's feature-rule pass: `MersenneTwister::new((cx·a + cz·b) ^ S)`,
    /// wrapping, with `S` the low 32 bits of the level seed, `(cx, cz)` the chunk coordinates, and
    /// `a`, `b` the first two [`next_int`](Self::next_int)s of `MersenneTwister::new(S)`, each with
    /// its low bit set.
    ///
    /// It drives the `math.random` of every rule's expressions in the pass, sharing one state
    /// across the rules; [`for_feature_rule`](Self::for_feature_rule) of its
    /// [`seed`](Self::seed) builds each rule's own.
    #[must_use]
    pub const fn for_chunk_decoration(level_seed: u32, chunk_x: i32, chunk_z: i32) -> Self {
        Self::new(chunk_seed(level_seed, chunk_x, chunk_z))
    }

    /// The generator of a structure start in a chunk: seeded as
    /// [`for_chunk_decoration`](Self::for_chunk_decoration), then one
    /// [`next_int`](Self::next_int) discarded.
    #[must_use]
    pub const fn for_structure_start(level_seed: u32, chunk_x: i32, chunk_z: i32) -> Self {
        let mut mt = Self::for_chunk_decoration(level_seed, chunk_x, chunk_z);
        mt.next_int();
        mt
    }

    /// The generator of one feature rule: `MersenneTwister::new(((s << 6) + (s >> 2) + h +
    /// 0x9e3779b9) ^ s)`, wrapping, with `s` the [`seed`](Self::seed) of the pass's generator (not
    /// its state) and `h` the low 32 bits of the rule's identifier hash. The host supplies the
    /// hash; the crate does not define it.
    ///
    /// It draws the rule's chance, iterations and coordinate distributions. When the iterations
    /// are more than 0, the placed feature gets a second generator equal to a fresh one of these.
    ///
    /// ```
    /// use molang::rng::MersenneTwister;
    ///
    /// let chunk = MersenneTwister::for_chunk_decoration(12_345, 3, -7);
    /// let rule = MersenneTwister::for_feature_rule(chunk.seed(), 0xf7a7_6dfd);
    /// assert_eq!(rule.seed(), 3_348_516_332);
    /// ```
    #[must_use]
    pub const fn for_feature_rule(chunk_seed: u32, identifier_hash: u32) -> Self {
        let s = chunk_seed;
        let mixed = (s << 6)
            .wrapping_add(s >> 2)
            .wrapping_add(identifier_hash)
            .wrapping_add(0x9e37_79b9);
        Self::new(mixed ^ s)
    }

    /// One output shifted right by 1: in `[0, 2^31)`.
    pub const fn next_int(&mut self) -> u32 {
        self.next_word() >> 1
    }

    /// One output modulo `bound`, or 0 with no draw when `bound` is 0.
    pub const fn next_int_below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            0
        } else {
            self.next_word() % bound
        }
    }

    /// `genrand_int32`: the next tempered word.
    const fn next_word(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// Regenerates all `N` words.
    const fn twist(&mut self) {
        let mut i = 0;
        while i < N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mag = if y & 1 == 0 { 0 } else { MATRIX_A };
            self.state[i] = self.state[(i + M) % N] ^ (y >> 1) ^ mag;
            i += 1;
        }
        self.index = 0;
    }
}

/// `(cx·a + cz·b) ^ S` with `a`, `b` the first two `next_int`s of `MT(S)`, low bit set.
const fn chunk_seed(level_seed: u32, chunk_x: i32, chunk_z: i32) -> u32 {
    let mut level = MersenneTwister::new(level_seed);
    let a = level.next_int() | 1;
    let b = level.next_int() | 1;
    (chunk_x.cast_unsigned().wrapping_mul(a)).wrapping_add(chunk_z.cast_unsigned().wrapping_mul(b))
        ^ level_seed
}

impl fmt::Debug for MersenneTwister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MersenneTwister")
            .field("seed", &self.seed)
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl RandomSource for MersenneTwister {
    #[inline]
    fn sample(&mut self) -> f32 {
        unit_32(self.next_word())
    }
}

impl TryRng for MersenneTwister {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.next_word())
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
    use crate::rng::{Bits31, FixedRng, unit_32};

    /// MT19937 written as the plain recurrence over an ever-growing sequence:
    /// `x[k+624] = x[k+397] ^ twist(x[k], x[k+1])`, output `temper(x[k+624])`.
    fn reference_outputs(seed: u32, count: usize) -> Vec<u32> {
        let mut x: Vec<u32> = vec![seed];
        for i in 1..624_u32 {
            let prev = x[x.len() - 1];
            x.push(
                1_812_433_253_u32
                    .wrapping_mul(prev ^ (prev >> 30))
                    .wrapping_add(i),
            );
        }
        (0..count)
            .map(|k| {
                let y = (x[k] & 0x8000_0000) | (x[k + 1] & 0x7fff_ffff);
                let next = x[k + 397] ^ (y >> 1) ^ ((y & 1) * 0x9908_b0df);
                x.push(next);
                let mut t = next;
                t ^= t >> 11;
                t ^= (t << 7) & 0x9d2c_5680;
                t ^= (t << 15) & 0xefc6_0000;
                t ^ (t >> 18)
            })
            .collect()
    }

    fn outputs(seed: u32, count: usize) -> Vec<u32> {
        let mut mt = MersenneTwister::new(seed);
        (0..count).map(|_| mt.next_u32()).collect()
    }

    #[test]
    fn the_published_outputs_of_seed_5489() {
        assert_eq!(
            outputs(5489, 20),
            [
                3_499_211_612,
                581_869_302,
                3_890_346_734,
                3_586_334_585,
                545_404_204,
                4_161_255_391,
                3_922_919_429,
                949_333_985,
                2_715_962_298,
                1_323_567_403,
                418_932_835,
                2_350_294_565,
                1_196_140_740,
                809_094_426,
                2_348_838_239,
                4_264_392_720,
                4_112_460_519,
                4_279_768_804,
                4_144_164_697,
                4_156_218_106
            ]
        );
        // The 10000th output, one of the published reference outputs.
        assert_eq!(outputs(5489, 10_000)[9_999], 4_123_659_995);
    }

    #[test]
    fn the_first_outputs_of_other_seeds() {
        assert_eq!(
            outputs(0, 5),
            [
                2_357_136_044,
                2_546_248_239,
                3_071_714_933,
                3_626_093_760,
                2_588_848_963
            ]
        );
        assert_eq!(
            outputs(1, 5),
            [
                1_791_095_845,
                4_282_876_139,
                3_093_770_124,
                4_005_303_368,
                491_263
            ]
        );
        assert_eq!(
            outputs(u32::MAX, 5),
            [
                419_326_371,
                479_346_978,
                3_918_654_476,
                2_416_749_639,
                3_388_880_820
            ]
        );
        assert_eq!(
            outputs(12_345, 5),
            [
                3_992_670_690,
                3_823_185_381,
                1_358_822_685,
                561_383_553,
                789_925_284
            ]
        );
    }

    #[test]
    fn outputs_match_the_plain_recurrence_across_several_twists() {
        for seed in [0, 1, 5489, 12_345, 0x8000_0000, u32::MAX] {
            assert_eq!(
                outputs(seed, 2_000),
                reference_outputs(seed, 2_000),
                "{seed}"
            );
        }
    }

    #[test]
    fn a_sample_is_the_output_through_f64_times_two_to_the_minus_32() {
        let mut words = MersenneTwister::new(5489);
        let mut samples = MersenneTwister::new(5489);
        for _ in 0..1_000 {
            let expected = (f64::from(words.next_u32()) / 4_294_967_296.0) as f32;
            assert_eq!(samples.sample().to_bits(), expected.to_bits());
        }
        assert_eq!(words, samples);
        let mut mt = MersenneTwister::new(5489);
        let bits: Vec<u32> = (0..4).map(|_| mt.sample().to_bits()).collect();
        assert_eq!(bits, [0x3f50_91bb, 0x3e0a_ba7c, 0x3f67_e1fb, 0x3f55_c31f]);
    }

    #[test]
    fn the_32_bit_mapping_reaches_one_and_keeps_the_top_bit() {
        assert_eq!(unit_32(0xffff_ff80), 1.0);
        assert_eq!(unit_32(u32::MAX), 1.0);
        assert_eq!(unit_32(0xffff_ff7f).to_bits(), 0x3f7f_ffff);
        assert_eq!(unit_32(0x8000_0000), 0.5);
        assert_eq!(unit_32(1).to_bits(), 0x2f80_0000);
        assert_eq!(unit_32(0).to_bits(), 0.0_f32.to_bits());
        // 2^25 + 3 rounds up under the 32-bit mapping; g >> 1 = 2^24 + 1 under the 31-bit mapping
        // is a tie and rounds to even.
        assert_eq!(unit_32((1 << 25) + 3).to_bits(), 0x3c00_0001);
        assert_eq!(
            Bits31(FixedRng((1 << 24) + 1)).sample().to_bits(),
            0x3c00_0000
        );
    }

    #[test]
    fn next_int_is_the_output_shifted_right_by_one() {
        let mut ints = MersenneTwister::new(5489);
        let words = outputs(5489, 8);
        for word in words {
            assert_eq!(ints.next_int(), word >> 1);
        }
    }

    #[test]
    fn next_int_below_is_the_unsigned_remainder_and_draws_nothing_for_zero() {
        let mut mt = MersenneTwister::new(5489);
        assert_eq!(mt.next_int_below(0), 0);
        assert_eq!(mt.next_int_below(16), 3_499_211_612 % 16);
        assert_eq!(mt.next_int_below(u32::MAX), 581_869_302);
        assert_eq!(mt.next_int_below(1), 0);
        assert_eq!(mt.next_u32(), 3_586_334_585, "zero bound drew nothing");
    }

    #[test]
    fn the_chunk_decoration_seed_by_hand() {
        let level_seed = 12_345;
        let mut level = MersenneTwister::new(level_seed);
        let a = (level.next_u32() >> 1) | 1;
        let b = (level.next_u32() >> 1) | 1;
        assert_eq!((a, b), (1_996_335_345, 1_911_592_691));
        let seed =
            (3_u32.wrapping_mul(a)).wrapping_add((-7_i32 as u32).wrapping_mul(b)) ^ level_seed;
        assert_eq!(seed, 1_197_795_863);
        let chunk = MersenneTwister::for_chunk_decoration(level_seed, 3, -7);
        assert_eq!(chunk, MersenneTwister::new(seed));
        assert_eq!(chunk.seed(), seed);
        assert_eq!(
            outputs(seed, 4),
            [2_024_310_572, 244_950_694, 2_533_344_077, 2_787_966_974]
        );
    }

    #[test]
    fn the_structure_start_discards_one_draw() {
        let mut start = MersenneTwister::for_structure_start(12_345, 3, -7);
        assert_eq!(start.seed(), 1_197_795_863);
        assert_eq!(
            (0..3).map(|_| start.next_u32()).collect::<Vec<_>>(),
            [244_950_694, 2_533_344_077, 2_787_966_974]
        );
    }

    #[test]
    fn the_feature_rule_seed_mixes_the_chunk_seed_not_its_state() {
        let mut chunk = MersenneTwister::for_chunk_decoration(12_345, 3, -7);
        let hash = 0xf7a7_6dfd;
        let s = chunk.seed();
        let by_hand = ((s << 6)
            .wrapping_add(s >> 2)
            .wrapping_add(hash)
            .wrapping_add(0x9e37_79b9))
            ^ s;
        assert_eq!(by_hand, 3_348_516_332);
        let fresh = MersenneTwister::for_feature_rule(chunk.seed(), hash);
        assert_eq!(fresh, MersenneTwister::new(by_hand));
        chunk.next_u32();
        assert_eq!(
            MersenneTwister::for_feature_rule(chunk.seed(), hash),
            fresh,
            "drawing changes nothing"
        );
        assert_eq!(fresh.clone().next_u32(), 760_173_523);
    }

    #[test]
    fn the_constructors_run_at_compile_time() {
        const CHUNK: MersenneTwister = MersenneTwister::for_chunk_decoration(12_345, 3, -7);
        const START: MersenneTwister = MersenneTwister::for_structure_start(12_345, 3, -7);
        const RULE: MersenneTwister = MersenneTwister::for_feature_rule(CHUNK.seed(), 0xf7a7_6dfd);
        assert_eq!(CHUNK.seed(), 1_197_795_863);
        assert_eq!(START.clone().next_u32(), 244_950_694);
        assert_eq!(RULE.seed(), 3_348_516_332);
    }

    #[test]
    fn wide_draws_and_bytes_are_whole_outputs_little_endian() {
        let words = outputs(5489, 4);
        let mut mt = MersenneTwister::new(5489);
        assert_eq!(
            mt.next_u64(),
            u64::from(words[1]) << 32 | u64::from(words[0])
        );
        let mut bytes = [0_u8; 6];
        mt.fill_bytes(&mut bytes);
        assert_eq!(bytes[..4], words[2].to_le_bytes());
        assert_eq!(bytes[4..], words[3].to_le_bytes()[..2]);
        assert_eq!(mt, {
            let mut after = MersenneTwister::new(5489);
            for _ in 0..4 {
                after.next_u32();
            }
            after
        });
    }

    #[test]
    fn debug_shows_the_seed_and_position() {
        let mut mt = MersenneTwister::new(7);
        mt.next_u32();
        assert_eq!(
            format!("{mt:?}"),
            "MersenneTwister { seed: 7, index: 1, .. }"
        );
    }
}
