//! [`ProcessRng`]: one random sequence shared by every evaluation.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::rng::RandomSource;
use crate::rng::xorshift::Xorshift128;

static PROCESS_XORSHIFT: Mutex<Xorshift128> = Mutex::new(Xorshift128::new());

/// A handle to one xorshift128 for the whole process, from the standard seeds
/// (123456789, 362436069, 521288629, 88675123) and never reseeded; each sample is one step under
/// [`unit_31`](crate::rng::unit_31).
///
/// It is the default source of [`HostEnv`](super::HostEnv) and [`NoHostEnv`](super::NoHostEnv),
/// the one sequence that entities, items, blocks, commands, client animation and scripting all
/// draw from. Concurrent evaluations interleave their draws; each sample takes the
/// lock once, so the state is never torn. A host wanting a sequence of its own per environment
/// passes another source.
///
/// ```
/// use molang::rng::{RandomSource, unit_31};
/// use molang::vm::ProcessRng;
///
/// ProcessRng::reset();
/// assert_eq!(ProcessRng.sample(), unit_31(3_701_687_786));
/// assert_eq!(ProcessRng.sample(), unit_31(458_299_110));
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProcessRng;

impl ProcessRng {
    /// Puts the process-wide generator back to the standard seeds.
    pub fn reset() {
        *locked() = Xorshift128::new();
    }
}

fn locked() -> MutexGuard<'static, Xorshift128> {
    PROCESS_XORSHIFT
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

impl RandomSource for ProcessRng {
    fn sample(&mut self) -> f32 {
        locked().sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator is process-wide, so these tests must not run concurrently.
    static LOCK: Mutex<()> = Mutex::new(());

    fn exclusive() -> MutexGuard<'static, ()> {
        let guard = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        ProcessRng::reset();
        guard
    }

    fn reference(n: usize) -> Vec<f32> {
        let mut rng = Xorshift128::new();
        (0..n).map(|_| rng.sample()).collect()
    }

    #[test]
    fn the_first_samples_are_those_of_a_freshly_seeded_generator() {
        let _guard = exclusive();
        let mut rng = ProcessRng;
        let got: Vec<u32> = (0..5).map(|_| rng.sample().to_bits()).collect();
        let want: Vec<u32> = reference(5).into_iter().map(f32::to_bits).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn a_random_source_sample_is_one_step_of_the_one_generator() {
        let _guard = exclusive();
        let mut reference = Xorshift128::new();
        assert_eq!(ProcessRng.sample().to_bits(), reference.sample().to_bits());
        assert_eq!(ProcessRng.sample().to_bits(), reference.sample().to_bits());
    }

    #[test]
    fn reset_restarts_the_sequence() {
        let _guard = exclusive();
        let first = ProcessRng.sample();
        for _ in 0..3 {
            ProcessRng.sample();
        }
        assert_ne!(ProcessRng.sample().to_bits(), first.to_bits());
        ProcessRng::reset();
        assert_eq!(ProcessRng.sample().to_bits(), first.to_bits());
        assert_eq!(first.to_bits(), reference(1)[0].to_bits());
    }

    #[test]
    fn every_value_shares_the_one_generator() {
        let _guard = exclusive();
        let (mut a, mut b) = (ProcessRng, ProcessRng);
        let got = [a.sample(), b.sample(), a.sample(), b.sample(), a.sample()];
        let want = reference(5);
        for (got, want) in got.iter().zip(&want) {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn samples_stay_in_the_unit_interval() {
        let _guard = exclusive();
        for _ in 0..10_000 {
            let sample = ProcessRng.sample();
            assert!((0.0..=1.0).contains(&sample), "{sample}");
        }
    }

    #[test]
    fn concurrent_draws_hand_out_each_sample_exactly_once() {
        let _guard = exclusive();
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..1000)
                        .map(|_| ProcessRng.sample().to_bits())
                        .collect::<Vec<u32>>()
                })
            })
            .collect();
        let mut drawn: Vec<u32> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect();
        drawn.sort_unstable();
        let mut want: Vec<u32> = reference(4000).into_iter().map(f32::to_bits).collect();
        want.sort_unstable();
        assert_eq!(drawn, want);
    }

    #[test]
    fn a_poisoned_lock_is_recovered() {
        let _guard = exclusive();
        let poisoner = std::thread::spawn(|| {
            let _held = PROCESS_XORSHIFT.lock().unwrap();
            panic!("poisons the generator's lock on purpose");
        });
        assert!(poisoner.join().is_err());
        assert!(PROCESS_XORSHIFT.is_poisoned());
        ProcessRng::reset();
        assert_eq!(ProcessRng.sample().to_bits(), reference(1)[0].to_bits());
        assert_eq!(ProcessRng.sample().to_bits(), reference(2)[1].to_bits());
    }

    #[test]
    fn the_handle_is_a_zero_sized_copy_with_equality() {
        let a = ProcessRng;
        let b = a;
        assert_eq!(a, b);
        assert_eq!(size_of::<ProcessRng>(), 0);
        assert_eq!(format!("{a:?}"), "ProcessRng");
    }
}
