//! Scalar semantics of the operators, comparisons and post-ops, as pure `f32` functions; the
//! `math.*` functions are in `stdlib::math`.
//!
//! The constant folder and the VM both call these. A constant operand (a literal, a call with
//! constant arguments, a product of constants) folds to its value before its parent applies a
//! negation, a term or a factor to it; a run-time operand takes them into its post-op `S·x + O`.
//! So a folded result can differ from the run-time form of the same expression in:
//!
//! - the sign of a zero: `-((0) * (1))` folds to −0 and `-(v.a * v.b)` runs to +0;
//!   `((-1) * (0)) + 0` folds to +0 and `(v.a * v.b) + 0` runs to −0;
//! - the sign or the choice of a NaN: a folded negation flips its sign, a run-time one leaves it;
//!   with a constant term left of a NaN operand, the fold takes the term's NaN where the run-time
//!   form can take the operand's: `math.ln(-1) + 3*math.sqrt(-1)` and
//!   `math.ln(-1) + v.k*math.sqrt(-1)`;
//! - `math.sign`: with a negative argument the post-op gives `−(S + O)`, the folded value
//!   `−S + O`;
//! - a division the guard stops: at run time it is 0 without its post-op;
//! - a literal divisor: the run-time form multiplies by its reciprocal: `(1/3)/3` folds to
//!   `0x3de38e39`, `v.k/3` runs to `0x3de38e3a`; below the guard's threshold the factor is 0, so a
//!   NaN or infinite dividend gives NaN where the fold gives 0: `math.sqrt(-1)/0` and `v.k/0`;
//! - a NaN divisor: the fold gives 0, the run-time form NaN: `1/math.sqrt(-1)` and `1/v.k`;
//! - `math.mod`: a zero divisor folds to NaN and runs to 0, and a −0 remainder folds to −0 and
//!   runs to +0: `math.mod(3, 0)` and `math.mod(3, v.k)`, `math.mod(-4, 2)` and
//!   `math.mod(v.k, 2)`;
//! - the rounding of sums: the run-time form flattens nested sums, adds the constant terms
//!   together first and multiplies the terms of an inner sum by its factor: `1 * 1 + (1/3) + 2`
//!   folds to `0x40555556`, `v.a * v.b + (1/3) + 2` runs to `0x40555555`;
//! - the rounding of products: the run-time form multiplies chained constant factors together
//!   first: `(0.1*3)*(1/3)` folds to `0x3dccccce`, `(v.k*3)*(1/3)` runs to `0x3dcccccd`; and
//!   `3.4e38*3.4e38*1e-45+1` folds to +∞ where `v.k*3.4e38*1e-45+1` does not overflow;
//! - an infinite factor: a constant factor folds into its operand's post-op, so an infinite one
//!   makes the offset `∞·O`, a NaN for an operand without an offset: `v.x * math.exp(1000)` and
//!   `(v.x < 2) * math.exp(1000)` are NaN where the same infinity read from a variable multiplies.
//!
//! The float behaviour is a [`Platform`], which the host passes in the compile options and the
//! compiled expression records. Every platform rounds each multiply, add and divide on its own, in
//! formula order; the platforms differ in the NaN an operation returns and in float → int
//! conversion ([`Platform`] lists the differences). The transcendental `math.*` functions come from
//! the host's C library, or with the `strict-math` feature from the `libm` crate, whose bits are the
//! same on every host.
//!
//! The arithmetic and the standard functions choose the sign and payload of every NaN they return,
//! so those are the same on every host too. On 32-bit x86 a value returned through the x87
//! register may have a signalling NaN quietened; 32-bit x86 without SSE2 does not compile.
//!
//! Every instruction function takes the node's [`PostOp`] as its last argument and returns the
//! final value; pass [`PostOp::IDENTITY`] for the plain form.

#[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
compile_error!("molang needs SSE2 on 32-bit x86: x87 rounds f32 operations in extended precision");

pub mod arith;
pub(crate) mod fpu;
mod instr;
pub(crate) mod platform;
mod post_op;

pub use instr::{
    add, array_index, div, div_guard, eq, fold_const_div, fold_const_divisor, ge, gt, le, lt, mul,
    ne, negate, not, truthy,
};
pub use platform::Platform;
pub use post_op::PostOp;

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) const ID: PostOp = PostOp::IDENTITY;
    pub(crate) const NAN: f32 = f32::NAN;
    pub(crate) const INF: f32 = f32::INFINITY;

    /// The Linux server, the default platform.
    pub(crate) const LINUX: Platform = Platform::BdsLinux;
    pub(crate) const WINDOWS: Platform = Platform::BdsWindows;
    pub(crate) const ANDROID: Platform = Platform::AndroidArm64;
    /// Every platform, for a test that runs once with each.
    pub(crate) const PLATFORMS: [Platform; 3] = [LINUX, WINDOWS, ANDROID];

    /// `servers` on [`Platform::BdsLinux`] and [`Platform::BdsWindows`], `android` on
    /// [`Platform::AndroidArm64`].
    pub(crate) fn servers_or_android<T>(p: Platform, servers: T, android: T) -> T {
        match p.fpu() {
            platform::Fpu::X86_64 => servers,
            platform::Fpu::Arm64 => android,
        }
    }

    /// A post-op that is not the identity and has exact arithmetic on small integers.
    pub(crate) const AFFINE: PostOp = PostOp::new(2.0, 1.0);

    /// `|actual − expected| <= tolerance`; a non-finite expectation must match exactly.
    pub(crate) fn within(actual: f32, expected: f32, tolerance: f32) -> bool {
        if expected.is_nan() {
            actual.is_nan()
        } else if expected.is_infinite() {
            actual == expected
        } else {
            (actual - expected).abs() <= tolerance
        }
    }

    #[track_caller]
    pub(crate) fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            within(actual, expected, tolerance),
            "{actual:e} is not {expected:e} ± {tolerance:e}"
        );
    }

    /// The literal `1 / 3` as the optimiser folds it.
    pub(crate) fn third() -> f32 {
        fold_const_div(LINUX, 1.0, 3.0)
    }

    /// A grid over and beyond `[0, 1]`, a fixed pseudo-random scatter, and the values around the
    /// easing thresholds, centres and phases, so that a constant one step off or a branch taken at
    /// a threshold shows.
    #[allow(clippy::manual_midpoint)]
    pub(crate) fn sweep_ts() -> Vec<f32> {
        let mut ts: Vec<f32> = (-128..=768).map(|k| k as f32 / 512.0).collect();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..30_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ts.push((state >> 40) as f32 / 16_777_216.0 * 2.0 - 0.5);
        }
        // Bounce thresholds and centres, their mirrored and shifted forms, and the elastic shift.
        let specials = [
            0x3eba_2e8c_u32,
            0x3f3a_2e8c,
            0x3f68_ba2f,
            0x3f0b_a2e9,
            0x3f51_745d,
            0x3f74_5d17,
            0x3ee8_ba2e,
            0x3e3a_2e8c,
            0x3d3a_2e90,
            0x3fc5_d174,
            0x3fe8_ba2e,
            0x3ffa_2e8c,
            0x3f00_0000,
            0x3e80_0000,
            0x3f40_0000,
            0x3d99_999a,
            0x3e99_999a,
            0x3eaa_aaab,
        ];
        for bits in specials {
            for delta in -4_i32..=4 {
                let s = f32::from_bits(bits.wrapping_add_signed(delta));
                ts.extend([
                    s,
                    -s,
                    1.0 - s,
                    s * 2.0,
                    s * 0.5,
                    (1.0 + s) * 0.5,
                    (1.0 - s) * 0.5,
                    2.0 - s,
                    s + 0.5,
                ]);
            }
        }
        ts
    }

    /// The `(start, end)` pairs of [`sweep_hash`].
    pub(crate) const SWEEP_RANGES: [(f32, f32); 4] =
        [(0.0, 1.0), (1.0, 5.0), (-3.0, 7.5), (0.3, 0.1)];

    /// The ulps a transcendental result of the host's C library may miss the `libm` crate's by.
    pub(crate) const HOST_ULPS: u32 = 2;

    /// Whether `a` and `b` are both NaN, or within `calls` × [`HOST_ULPS`] ulps of the larger of
    /// `scale`, `|a|` and `|b|`: a formula that subtracts a transcendental result from a constant
    /// keeps its absolute error, not its relative one, and each call adds its own.
    pub(crate) fn within_host_ulps_of(a: f32, b: f32, scale: f32, calls: u32) -> bool {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() && b.is_nan();
        }
        if a.is_infinite() || b.is_infinite() {
            return a == b;
        }
        let scale = scale.max(a.abs()).max(b.abs());
        let ulp = f64::from(f32::from_bits(scale.to_bits() & 0x7f80_0000).max(f32::MIN_POSITIVE))
            * f64::from(f32::EPSILON);
        (f64::from(a) - f64::from(b)).abs() <= f64::from(HOST_ULPS * calls) * ulp
    }

    /// FNV-1a over the bits of `f` at every [`sweep_ts`] point, for `(start, end)` pairs whose
    /// range is not a power of two (so fused and separate rounding differ); all NaNs hash alike.
    pub(crate) fn sweep_hash(f: impl Fn(f32, f32, f32) -> f32) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for (start, end) in SWEEP_RANGES {
            for t in sweep_ts() {
                let value = f(start, end, t);
                let bits = if value.is_nan() {
                    0x7fc0_0000
                } else {
                    value.to_bits()
                };
                for byte in bits.to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        hash
    }

    #[track_caller]
    pub(crate) fn assert_bits(actual: f32, expected: f32) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual:e} ({:#010x}) != {expected:e} ({:#010x})",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;

    #[test]
    fn within_holds_a_non_finite_expectation_exactly() {
        assert!(within(1.05, 1.0, 0.1) && !within(1.2, 1.0, 0.1));
        assert!(within(NAN, NAN, 0.0) && !within(0.0, NAN, 1.0));
        assert!(within(INF, INF, 0.0) && !within(f32::MAX, INF, f32::MAX));
    }
}
