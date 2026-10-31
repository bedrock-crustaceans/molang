//! The affine post-op that the optimiser folds into a value-producing node.

use super::{Platform, fpu};

/// The `raw·scale + offset` applied to a node's result, into which the optimiser folds `x·c`,
/// `x + c` and `−x`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PostOp {
    /// The multiplier `S`.
    pub scale: f32,
    /// The addend `O`.
    pub offset: f32,
}

impl Default for PostOp {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl PostOp {
    /// `(1, 0)`: the plain form.
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        offset: 0.0,
    };

    /// A post-op with the given scale and offset.
    #[inline]
    pub const fn new(scale: f32, offset: f32) -> Self {
        Self { scale, offset }
    }

    /// Whether this is the plain form (an offset of `−0` counts).
    #[inline]
    pub fn is_identity(self) -> bool {
        self.scale == 1.0 && self.offset == 0.0
    }

    /// `raw·S + O`, rounded twice: the NaN of `raw`, then of `S`, then of `O`.
    ///
    /// The identity returns `raw` untouched, so `math.ceil(-0.5)` stays `−0`.
    #[inline]
    pub fn apply(self, platform: Platform, raw: f32) -> f32 {
        if self.is_identity() {
            raw
        } else {
            fpu::mul_add(platform, raw, self.scale, self.offset)
        }
    }

    /// The "true" result of a comparison or logical node: `S + O`.
    #[inline]
    pub fn truthy_value(self, platform: Platform) -> f32 {
        fpu::add(platform, self.scale, self.offset)
    }

    /// The "false" result of a comparison or logical node: `O`.
    #[inline]
    pub fn falsy_value(self) -> f32 {
        self.offset
    }

    /// The result of a comparison or logical node: one of two precomputed constants, never a
    /// multiply.
    #[inline]
    pub fn select(self, platform: Platform, condition: bool) -> f32 {
        if condition {
            self.truthy_value(platform)
        } else {
            self.falsy_value()
        }
    }

    /// Folds `Mul(child, c)` into the child's post-op, with `self` the post-op of the
    /// multiplication.
    ///
    /// The scale is `(S·c)·child.S`; the offset is `((c·child.O)·S) + O`.
    #[inline]
    #[must_use]
    pub fn fold_scaled(self, platform: Platform, c: f32, child: Self) -> Self {
        let k = fpu::mul(platform, self.scale, c);
        Self {
            scale: fpu::mul(platform, k, child.scale),
            offset: fpu::mul_add(
                platform,
                fpu::mul(platform, c, child.offset),
                self.scale,
                self.offset,
            ),
        }
    }

    /// Folds `Negate(child)` into the child's post-op, with `self` the post-op of the negation:
    /// `(−(S·child.S), O − child.O·S)`.
    #[inline]
    #[must_use]
    pub fn fold_negated(self, platform: Platform, child: Self) -> Self {
        Self {
            scale: -fpu::mul(platform, self.scale, child.scale),
            offset: fpu::sub(
                platform,
                self.offset,
                fpu::mul(platform, child.offset, self.scale),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::test_support::*;
    use crate::stdlib::math;

    const P: Platform = NATIVE;

    #[test]
    fn identity_post_op_returns_the_raw_value() {
        assert_eq!(ID.apply(P, -0.0).to_bits(), (-0.0_f32).to_bits());
        // Applying (1, 0) would turn −0 into +0.
        assert_eq!(math::ceil(P, -0.5, ID).to_bits(), (-0.0_f32).to_bits());
        assert!(ID.is_identity());
        assert!(PostOp::new(1.0, -0.0).is_identity());
        assert!(!PostOp::new(1.0, 1.0).is_identity());
        assert_eq!(PostOp::default(), ID);
    }

    #[test]
    fn post_op_constructors_and_constants() {
        assert_eq!(
            PostOp::IDENTITY,
            PostOp {
                scale: 1.0,
                offset: 0.0
            }
        );
        assert_eq!(
            PostOp::new(2.5, -3.0),
            PostOp {
                scale: 2.5,
                offset: -3.0
            }
        );
        assert_eq!(PostOp::default(), PostOp::IDENTITY);
        let post = PostOp::new(2.0, 3.0);
        let copy = post;
        assert_eq!(post, copy);
        assert_ne!(PostOp::new(2.0, 3.0), PostOp::new(2.0, 3.5));
        assert_ne!(PostOp::new(2.0, 3.0), PostOp::new(2.5, 3.0));
    }

    #[test]
    fn is_identity_is_exactly_scale_one_and_offset_zero() {
        assert!(PostOp::new(1.0, 0.0).is_identity());
        assert!(PostOp::new(1.0, -0.0).is_identity());
        for post in [
            PostOp::new(-1.0, 0.0),
            PostOp::new(0.0, 0.0),
            PostOp::new(1.000_000_1, 0.0),
            PostOp::new(0.999_999_94, 0.0),
            PostOp::new(1.0, f32::MIN_POSITIVE),
            PostOp::new(1.0, 1.0e-45),
            PostOp::new(NAN, 0.0),
            PostOp::new(1.0, NAN),
            PostOp::new(INF, 0.0),
            PostOp::new(1.0, INF),
        ] {
            assert!(!post.is_identity(), "{post:?}");
        }
    }

    #[test]
    fn apply_returns_the_raw_bits_for_the_identity() {
        for raw in [0.0_f32, -0.0, 1.5, -2.5, INF, -INF, f32::MAX, 1.0e-45] {
            assert_bits(ID.apply(P, raw), raw);
        }
        assert!(ID.apply(P, NAN).is_nan());
        assert_eq!(ID.apply(P, NAN).to_bits(), NAN.to_bits());
        assert_bits(PostOp::new(1.0, -0.0).apply(P, -0.0), -0.0);
    }

    #[test]
    fn apply_computes_raw_times_scale_plus_offset() {
        assert_eq!(PostOp::new(2.0, 1.0).apply(P, 3.0), 7.0);
        assert_eq!(PostOp::new(-2.0, 0.5).apply(P, 3.0), -5.5);
        assert_eq!(PostOp::new(0.0, 4.0).apply(P, 123.0), 4.0);
        assert_eq!(PostOp::new(1.0, 2.0).apply(P, 3.0), 5.0);
        assert_eq!(PostOp::new(2.0, 0.0).apply(P, 3.0), 6.0);
    }

    #[test]
    fn a_real_post_op_normalises_negative_zero() {
        // (−0)·2 + (+0) = +0 but (−0)·2 + (−0) = −0.
        assert_bits(PostOp::new(2.0, 0.0).apply(P, -0.0), 0.0);
        assert_bits(PostOp::new(2.0, -0.0).apply(P, -0.0), -0.0);
        assert_bits(PostOp::new(-1.0, 0.0).apply(P, 0.0), 0.0);
    }

    #[test]
    fn apply_with_non_finite_operands() {
        assert!(PostOp::new(2.0, 1.0).apply(P, NAN).is_nan());
        assert_eq!(PostOp::new(2.0, 1.0).apply(P, INF), INF);
        assert_eq!(PostOp::new(-2.0, 1.0).apply(P, INF), -INF);
        assert!(PostOp::new(0.0, 1.0).apply(P, INF).is_nan());
        assert!(PostOp::new(2.0, -INF).apply(P, INF).is_nan());
        assert_eq!(PostOp::new(2.0, INF).apply(P, 1.0), INF);
        assert!(PostOp::new(NAN, 0.0).apply(P, 1.0).is_nan());
    }

    #[test]
    fn the_post_op_rounds_twice_on_every_platform() {
        // v.x = 1/3; v.x * 3 - 1
        let post = PostOp::new(3.0, -1.0);
        for p in PLATFORMS {
            assert_eq!(post.apply(p, third()).to_bits(), 0.0_f32.to_bits());
        }
    }

    #[test]
    fn a_nan_raw_scale_or_offset_gives_nan_on_every_platform() {
        for p in PLATFORMS {
            assert!(PostOp::new(2.0, 1.0).apply(p, NAN).is_nan());
            assert!(PostOp::new(NAN, 1.0).apply(p, 2.0).is_nan());
            assert!(PostOp::new(2.0, NAN).apply(p, 2.0).is_nan());
            assert!(PostOp::new(0.0, 1.0).apply(p, INF).is_nan());
            assert!(PostOp::new(INF, -INF).truthy_value(p).is_nan());
        }
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn the_post_op_chooses_the_nan_of_raw_then_scale_then_offset() {
        let (q1, q2, q3) = (
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc0_0002),
            f32::from_bits(0x7f80_0003),
        );
        for p in EMULATED {
            assert_eq!(PostOp::new(q2, 1.0).apply(p, q1).to_bits(), 0x7fc0_0001);
            assert_eq!(PostOp::new(2.0, q1).apply(p, q2).to_bits(), 0xffc0_0002);
        }
        // A signalling offset wins on Android only.
        assert_eq!(PostOp::new(2.0, q3).apply(LINUX, q2).to_bits(), 0xffc0_0002);
        assert_eq!(
            PostOp::new(2.0, q3).apply(ANDROID, q2).to_bits(),
            0x7fc0_0003
        );
        assert_eq!(
            PostOp::new(0.0, 1.0).apply(LINUX, INF).to_bits(),
            0xffc0_0000
        );
        assert_eq!(
            PostOp::new(0.0, 1.0).apply(ANDROID, INF).to_bits(),
            0x7fc0_0000
        );
    }

    #[test]
    fn truthy_and_falsy_values_are_scale_plus_offset_and_offset() {
        assert_eq!(ID.truthy_value(P), 1.0);
        assert_eq!(ID.falsy_value(), 0.0);
        let post = PostOp::new(2.0, 1.1);
        assert_eq!(post.truthy_value(P), 2.0 + 1.1);
        assert_eq!(post.falsy_value(), 1.1);
        let post = PostOp::new(-2.0, 0.5);
        assert_eq!(post.truthy_value(P), -1.5);
        assert_eq!(post.falsy_value(), 0.5);
        assert_bits(PostOp::new(0.0, -0.0).falsy_value(), -0.0);
        assert!(PostOp::new(NAN, 1.0).truthy_value(P).is_nan());
        assert_eq!(PostOp::new(NAN, 1.0).falsy_value(), 1.0);
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn the_truthy_value_of_a_nan_post_op_is_the_platforms() {
        let signalling = f32::from_bits(0x7f80_0001);
        let quiet = f32::from_bits(0xffc0_0002);
        for (p, expected) in [(LINUX, 0xffc0_0002), (ANDROID, 0x7fc0_0001)] {
            assert_eq!(
                PostOp::new(quiet, signalling).truthy_value(p).to_bits(),
                expected
            );
            assert_eq!(
                PostOp::new(quiet, signalling).select(p, true).to_bits(),
                expected
            );
        }
        for (p, expected) in [(LINUX, 0xffc0_0000), (ANDROID, 0x7fc0_0000)] {
            assert_eq!(PostOp::new(INF, -INF).truthy_value(p).to_bits(), expected);
        }
    }

    #[test]
    fn comparison_results_are_precomputed_constants() {
        assert_eq!(ID.select(P, true), 1.0);
        assert_eq!(ID.select(P, false), 0.0);
        let post = PostOp::new(2.0, 1.1);
        assert_eq!(post.select(P, true), 2.0 + 1.1);
        assert_eq!(post.select(P, false), 1.1);
        assert_eq!(post.truthy_value(P), 3.1);
        assert_eq!(post.falsy_value(), 1.1);
    }

    #[test]
    fn select_picks_the_precomputed_constant_on_every_platform() {
        for post in [
            ID,
            PostOp::new(-3.0, 7.0),
            PostOp::new(0.0, -0.0),
            PostOp::new(1.0, -0.0),
        ] {
            for p in PLATFORMS {
                assert_bits(post.select(p, true), post.truthy_value(p));
                assert_bits(post.select(p, false), post.falsy_value());
            }
        }
        // (1, −0) is the identity for `apply`, but `select` still returns O = −0.
        assert_bits(PostOp::new(1.0, -0.0).select(P, false), -0.0);
    }

    #[test]
    fn optimiser_folds_compose_post_ops() {
        // (x·2 + 1)·3 under a node (·5 + 7): k = 5·3, scale = 15·2, offset = 15·1 + 7.
        assert_eq!(
            PostOp::new(5.0, 7.0).fold_scaled(P, 3.0, PostOp::new(2.0, 1.0)),
            PostOp::new(30.0, 22.0)
        );
        assert_eq!(ID.fold_scaled(P, 3.0, ID), PostOp::new(3.0, 0.0));
        // −(x·2 + 1) under a node (·5 + 7): scale = −(5·2), offset = 7 − 1·5.
        assert_eq!(
            PostOp::new(5.0, 7.0).fold_negated(P, PostOp::new(2.0, 1.0)),
            PostOp::new(-10.0, 2.0)
        );
        assert_eq!(ID.fold_negated(P, ID), PostOp::new(-1.0, 0.0));
        // The offset groupings differ once the node's own scale is not 1.
        let (parent, c, child) = (PostOp::new(0.1, 0.25), 0.3_f32, PostOp::new(2.0, 0.7));
        for p in PLATFORMS {
            assert_eq!(
                parent.fold_scaled(p, c, child).offset,
                (c * child.offset) * parent.scale + parent.offset
            );
        }
        assert_ne!(
            (c * child.offset) * parent.scale,
            (parent.scale * c) * child.offset
        );
        assert_eq!(
            parent.fold_scaled(P, c, child).scale,
            (parent.scale * c) * child.scale
        );
        // The offset is rounded twice.
        let (parent, child) = (PostOp::new(3.0, -1.0), PostOp::new(1.0, third()));
        for p in PLATFORMS {
            assert_eq!(parent.fold_scaled(p, 1.0, child).offset, 0.0);
        }
    }

    #[test]
    fn fold_scaled_multiplies_the_scales_in_the_documented_order() {
        let (parent, c, child) = (PostOp::new(0.1, 0.0), 0.3_f32, PostOp::new(0.7, 0.0));
        let expected = (0.1_f32 * 0.3) * 0.7;
        assert_ne!(
            expected,
            0.1 * (0.3 * 0.7),
            "the grouping must matter for this check"
        );
        assert_bits(parent.fold_scaled(P, c, child).scale, expected);
    }

    #[test]
    fn fold_scaled_with_nan_and_infinite_constants() {
        assert!(ID.fold_scaled(P, NAN, ID).scale.is_nan());
        assert_eq!(ID.fold_scaled(P, INF, ID).scale, INF);
        assert!(
            ID.fold_scaled(P, 0.0, PostOp::new(1.0, INF))
                .offset
                .is_nan()
        );
        assert_eq!(ID.fold_scaled(P, 2.0, PostOp::new(1.0, INF)).offset, INF);
    }

    #[test]
    fn fold_scaled_rounds_the_offset_twice() {
        let parent = PostOp::new(1.0, -1.0);
        let (c, child) = (third(), PostOp::new(1.0, 3.0));
        for p in PLATFORMS {
            assert_eq!(parent.fold_scaled(p, c, child).offset, 0.0);
        }
    }

    #[test]
    fn fold_scaled_multiplies_the_constant_into_the_child_offset_first() {
        let (parent, c, child) = (PostOp::new(0.1, 0.1), 0.3_f32, PostOp::new(1.0, 0.7));
        assert_bits(
            parent.fold_scaled(P, c, child).offset,
            f32::from_bits(0x3df7_ceda),
        );
        assert_ne!(
            (c * child.offset) * parent.scale + parent.offset,
            (parent.scale * c) * child.offset + parent.offset
        );
    }

    #[test]
    fn fold_negated_follows_its_formula() {
        // −(x·2 + 1) under (·5 + 7): scale −10, offset 7 − 1·5 = 2.
        assert_eq!(
            PostOp::new(5.0, 7.0).fold_negated(P, PostOp::new(2.0, 1.0)),
            PostOp::new(-10.0, 2.0)
        );
        let once = ID.fold_negated(P, PostOp::new(3.0, 4.0));
        assert_eq!(once, PostOp::new(-3.0, -4.0));
        let twice = ID.fold_negated(P, once);
        assert_eq!(twice, PostOp::new(3.0, 4.0));
    }

    #[test]
    fn fold_negated_rounds_the_offset_twice() {
        // 3·(1/3) rounds to 1 before the subtraction.
        let (parent, child) = (PostOp::new(3.0, 1.0), PostOp::new(1.0, third()));
        for p in PLATFORMS {
            assert_bits(parent.fold_negated(p, child).offset, 0.0);
            assert_bits(parent.fold_negated(p, child).scale, -3.0);
        }
    }

    #[cfg(feature = "platform-quirks")]
    #[test]
    fn fold_negated_chooses_the_nan_of_the_scale_product() {
        let (q1, q2) = (f32::from_bits(0x7fc0_0001), f32::from_bits(0xffc0_0002));
        assert_eq!(
            PostOp::new(q1, 0.0)
                .fold_negated(LINUX, PostOp::new(q2, 0.0))
                .scale
                .to_bits(),
            0xffc0_0001
        );
        let generated = |p| {
            PostOp::new(INF, 0.0)
                .fold_negated(p, PostOp::new(0.0, 0.0))
                .scale
                .to_bits()
        };
        assert_eq!(generated(LINUX), 0x7fc0_0000);
        assert_eq!(generated(WINDOWS), 0x7fc0_0000);
        assert_eq!(generated(ANDROID), 0xffc0_0000);
    }

    #[test]
    fn fold_negated_scale_is_the_negated_product_of_the_scales() {
        let folded = PostOp::new(0.1, 0.0).fold_negated(P, PostOp::new(0.3, 0.0));
        assert_bits(folded.scale, -(0.1_f32 * 0.3));
        // A zero scale gives −0.
        let folded = PostOp::new(0.0, 0.0).fold_negated(P, PostOp::new(2.0, 0.0));
        assert_bits(folded.scale, -0.0);
    }
}
