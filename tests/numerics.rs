//! Numeric behaviour through the compiler and the VM: the math library, the platforms, the division
//! guard per version, NaN and zero rules, and the random operators.

#![cfg(all(feature = "vm", feature = "stdlib"))]

mod common;

mod math_functions {
    // Expected values are written with nine significant digits.
    #![allow(clippy::excessive_precision)]

    use crate::common::P;
    use crate::common::{Samples, assert_bits, compile_support::server_at};
    use molang::catalog::{QueryAdmission, QuerySetMask};
    use molang::compile::{CompileFailure, compile};
    use molang::numeric::PostOp;
    use molang::ops::OpSet;
    use molang::stdlib::{MathFn, math};
    use molang::vm::{NoHostEnv, Value, VariableName};

    const NAN: f32 = f32::NAN;

    /// Compiles (which must succeed) and evaluates with `v.<name>` preset and a random source that
    /// plays back `samples` (0 once they run out). Returns the value and the number of draws.
    fn run(source: &str, vars: &[(&str, f32)], samples: &[f32]) -> (f32, usize) {
        let compiled = compile(source, &server_at(13));
        assert_eq!(
            compiled.failure(),
            None,
            "{source:?}: {:?}",
            compiled.diagnostics()
        );
        let expr = compiled.expr().cloned().expect("an expression");
        let mut env = NoHostEnv::new();
        for (name, value) in vars {
            env.variables_mut()
                .set(VariableName::new(name), Value::Float(*value));
        }
        let mut rng = Samples::new(samples);
        let value = {
            let mut cx = env.cx();
            cx.rng = &mut rng;
            expr.eval_f32(&mut cx)
        };
        assert!(env.sink().is_empty(), "{source:?}: {:?}", env.sink());
        (value, rng.draws)
    }

    fn eval(source: &str) -> f32 {
        run(source, &[], &[]).0
    }

    /// The constant `source` folds to.
    #[track_caller]
    fn folded_constant(source: &str) -> f32 {
        let compiled = compile(source, &server_at(13));
        let constant = compiled.expr().and_then(|expr| expr.as_constant());
        constant.unwrap_or_else(|| panic!("{source:?} does not fold"))
    }

    fn eval_vars(source: &str, vars: &[(&str, f32)]) -> f32 {
        run(source, vars, &[]).0
    }

    /// A float that is not exactly representable in a short literal: `1/3` in f32.
    fn third() -> f32 {
        1.0 / 3.0
    }

    /// Post-ops folded at compile time choose the NaN the same operations choose at run time: `∞ −
    /// ∞`, `∞·0`, two different NaNs, and the bounds of a literal `math.random`.
    #[test]
    fn folded_post_ops_choose_the_nans_of_their_run_time_forms() {
        let (sqrt, asin) = (eval("math.sqrt(-1)"), eval("math.asin(2)"));
        assert_ne!(sqrt.to_bits(), asin.to_bits());
        let vars = [
            ("x", 2.0),
            ("p", f32::INFINITY),
            ("q", f32::NEG_INFINITY),
            ("z", 0.0),
            ("s", sqrt),
            ("a", asin),
        ];
        for (folded, run_time) in [
            ("(v.x + 1e39) + (v.x - 1e39)", "(v.x + v.p) + (v.x + v.q)"),
            ("(v.x - 1e39) + (v.x + 1e39)", "(v.x + v.q) + (v.x + v.p)"),
            ("(v.x * 1e39) * 0", "(v.x * v.p) * v.z"),
            ("1e39 / 1e39", "v.p / v.p"),
            (
                "(v.x + math.sqrt(-1)) + (v.x + math.asin(2))",
                "(v.x + v.s) + (v.x + v.a)",
            ),
            (
                "(v.x + math.asin(2)) + (v.x + math.sqrt(-1))",
                "(v.x + v.a) + (v.x + v.s)",
            ),
            ("v.x + 1e39 + v.x - 1e39", "v.x + v.p + v.x + v.q"),
            ("math.random(1e39, 1e39)", "math.random(v.p, v.p)"),
        ] {
            let (constant, run) = (eval_vars(folded, &vars), eval_vars(run_time, &vars));
            assert!(constant.is_nan(), "{folded} = {constant}");
            assert_eq!(
                constant.to_bits(),
                run.to_bits(),
                "{folded} against {run_time}"
            );
        }
    }

    /// Math functions are operations, not queries: they compile with no query set and at version
    /// −1.
    #[test]
    fn math_functions_ignore_query_versions_and_sets() {
        for function in MathFn::all() {
            let meta = function.meta();
            let arguments = vec!["1"; usize::from(meta.min_args)].join(", ");
            let source = if *function == MathFn::Pi {
                function.token().to_owned()
            } else {
                format!("{}({arguments})", function.token())
            };
            for version in [-1_i16, 1, 13] {
                let options =
                    server_at(version).with_admission(QueryAdmission::Sets(QuerySetMask::empty()));
                let compiled = compile(&source, &options);
                assert_eq!(
                    compiled.failure(),
                    None,
                    "{source} at v{version}: {:?}",
                    compiled.diagnostics()
                );
            }
        }
        // A query, by contrast, does not resolve without a set.
        let options = server_at(13).with_admission(QueryAdmission::Sets(QuerySetMask::empty()));
        assert_eq!(
            compile("query.is_baby", &options).failure(),
            Some(CompileFailure::Rejected)
        );
    }

    /// The post-op `raw·S + O` applies to every value-producing function, folded or run-time.
    #[test]
    fn every_math_result_takes_the_post_op() {
        let cases: &[(&str, f32)] = &[
            ("math.abs(v.a)", 1.5),
            ("math.ceil(v.a)", 2.0),
            ("math.floor(v.a)", 1.0),
            ("math.round(v.a)", 2.0),
            ("math.trunc(v.a)", 1.0),
            ("math.sqrt(v.b)", 2.0),
            ("math.max(v.a, v.b)", 4.0),
            ("math.min(v.a, v.b)", 1.5),
            ("math.clamp(v.b, 0, v.a)", 1.5),
            ("math.lerp(0, v.b, 0.5)", 2.0),
            ("math.mod(v.b, v.a)", 1.0),
            ("math.pow(v.b, 0.5)", 2.0),
            ("math.copy_sign(v.a, -1)", -1.5),
            ("math.hermite_blend(0.5)", 0.5),
            ("math.inverse_lerp(0, v.b, 1)", 0.25),
            ("math.min_angle(v.a * 100 + 220)", 10.0),
            ("math.ease_in_quad(0, v.b, 0.5)", 1.0),
            ("math.pi", math::PI),
        ];
        for (call, raw) in cases {
            let vars = [("a", 1.5), ("b", 4.0)];
            assert_eq!(eval_vars(call, &vars), *raw, "{call}");
            for (s, o) in [(2.0_f32, 1.0_f32), (-3.0, 0.5), (1.0, -1.0)] {
                let source = format!("({call}) * {s} + {o}");
                assert_eq!(eval_vars(&source, &vars), raw * s + o, "{source}");
            }
        }
    }

    #[test]
    fn abs_clears_the_sign_bit() {
        assert_eq!(eval("math.abs(-2)"), 2.0);
        assert_bits(eval_vars("math.abs(v.x)", &[("x", -0.0)]), 0.0, "abs(-0)");
        assert_eq!(
            eval_vars("math.abs(v.x)", &[("x", f32::NEG_INFINITY)]),
            f32::INFINITY
        );
        assert!(eval_vars("math.abs(v.x)", &[("x", NAN)]).is_nan());
    }

    /// Arguments up to ±1.0005 clamp to ±1; 1.0006 is NaN; results are in degrees.
    #[test]
    fn inverse_trig_clamp_window() {
        assert_eq!(eval("math.acos(1.0005)"), 0.0);
        assert_eq!(eval_vars("math.acos(v.x)", &[("x", 1.0005)]), 0.0);
        // The `libm` crate's `acos(-1)` is one ulp below π.
        let half_turn = if cfg!(feature = "strict-math") {
            f32::from_bits(0x4333_ffff)
        } else {
            180.0
        };
        assert_eq!(eval_vars("math.acos(v.x)", &[("x", -1.0001)]), half_turn);
        assert_eq!(eval_vars("math.acos(v.x)", &[("x", -1.0005)]), half_turn);
        assert_eq!(eval_vars("math.asin(v.x)", &[("x", 1.0005)]), 90.0);
        assert_eq!(eval_vars("math.asin(v.x)", &[("x", -1.0004)]), -90.0);
        for x in [1.0006_f32, -1.0006, 2.0] {
            assert!(
                eval_vars("math.acos(v.x)", &[("x", x)]).is_nan(),
                "acos({x})"
            );
            assert!(
                eval_vars("math.asin(v.x)", &[("x", x)]).is_nan(),
                "asin({x})"
            );
        }
        assert!(eval("math.acos(1.0006)").is_nan());
        let x = 0.3_f32;
        assert_bits(
            eval_vars("math.acos(v.x)", &[("x", x)]),
            crate::common::library::acos(x) * math::RAD_TO_DEG,
            "acos(0.3)",
        );
        assert_bits(
            eval_vars("math.asin(v.x)", &[("x", x)]),
            crate::common::library::asin(x) * math::RAD_TO_DEG,
            "asin(0.3)",
        );
    }

    /// A NaN argument passes the clamp.
    #[test]
    fn inverse_trig_of_nan() {
        let nan = [("x", NAN)];
        assert!(eval_vars("math.acos(v.x)", &nan).is_nan());
        assert!(eval_vars("math.asin(v.x)", &nan).is_nan());
    }

    /// `atan` and `atan2(y, x)` in degrees, y first; `atan2(0, 0)` = 0.
    #[test]
    fn atan_and_atan2() {
        let x = 0.3_f32;
        assert_bits(
            eval_vars("math.atan(v.x)", &[("x", x)]),
            crate::common::library::atan(x) * math::RAD_TO_DEG,
            "atan(0.3)",
        );
        assert_eq!(eval("math.atan(1)"), 45.0);
        assert_eq!(eval("math.atan2(1, 0)"), 90.0);
        assert_eq!(eval("math.atan2(0, 1)"), 0.0);
        assert_eq!(eval("math.atan2(0, -1)"), 180.0);
        assert_eq!(
            eval_vars("math.atan2(v.y, v.x)", &[("y", 0.0), ("x", 0.0)]),
            0.0
        );
        assert_eq!(eval("math.atan2(0, 0)"), 0.0);
        let (y, x) = (-1.0_f32, -7.0_f32);
        assert_bits(
            eval_vars("math.atan2(v.y, v.x)", &[("y", y), ("x", x)]),
            crate::common::library::atan2(y, x) * math::RAD_TO_DEG,
            "atan2(-1, -7)",
        );
    }

    /// Ceil toward +∞ (keeping −0), floor toward −∞, round half away from zero, trunc toward zero.
    #[test]
    fn rounding() {
        for (source, x, expected) in [
            ("math.ceil(v.x)", -0.5_f32, -0.0_f32),
            ("math.ceil(v.x)", 1.1, 2.0),
            ("math.ceil(v.x)", -1.1, -1.0),
            ("math.floor(v.x)", -0.5, -1.0),
            ("math.floor(v.x)", 1.9, 1.0),
            ("math.floor(v.x)", -1.1, -2.0),
            ("math.round(v.x)", 0.5, 1.0),
            ("math.round(v.x)", -0.5, -1.0),
            ("math.round(v.x)", -1.5, -2.0),
            ("math.round(v.x)", -2.5, -3.0),
            ("math.round(v.x)", 2.5, 3.0),
            ("math.round(v.x)", 2.49, 2.0),
            ("math.trunc(v.x)", 1.7, 1.0),
            ("math.trunc(v.x)", -1.7, -1.0),
            ("math.trunc(v.x)", -0.5, -0.0),
            ("math.trunc(v.x)", 1.0e10, 1.0e10),
        ] {
            assert_bits(
                eval_vars(source, &[("x", x)]),
                expected,
                &format!("{source} with {x}"),
            );
        }
        assert_bits(eval("math.ceil(-0.5)"), -0.0, "folded ceil(-0.5)");
        assert_eq!(eval("math.round(-2.5)"), -3.0);
        assert_eq!(eval("math.trunc(-2.9)"), -2.0);
    }

    /// Clamp tests the upper bound first; a NaN value gives the lower bound; a NaN lower bound is
    /// returned.
    #[test]
    fn clamp_order_and_nan() {
        assert_eq!(eval("math.clamp(3, 2, 1)"), 1.0);
        assert_eq!(eval("math.clamp(-1, -2, -3)"), -3.0);
        let vars = [("v", 3.0), ("lo", 2.0), ("hi", 1.0)];
        assert_eq!(eval_vars("math.clamp(v.v, v.lo, v.hi)", &vars), 1.0);
        assert_eq!(
            eval_vars(
                "math.clamp(v.v, v.lo, v.hi)",
                &[("v", 1.5), ("lo", 2.0), ("hi", 1.0)]
            ),
            1.0
        );
        assert_eq!(
            eval_vars(
                "math.clamp(v.v, v.lo, v.hi)",
                &[("v", 0.5), ("lo", 2.0), ("hi", 1.0)]
            ),
            2.0
        );
        assert_eq!(eval_vars("math.clamp(v.v, 1, 2)", &[("v", NAN)]), 1.0);
        assert_eq!(eval_vars("math.clamp(v.v, 1, 5)", &[("v", NAN)]), 1.0);
        // A NaN upper bound never matches `v > hi`.
        assert_eq!(eval_vars("math.clamp(9, 1, v.hi)", &[("hi", NAN)]), 9.0);
        assert!(eval_vars("math.clamp(4, v.lo, 5)", &[("lo", NAN)]).is_nan());
        assert!(eval_vars("math.clamp(v.v, v.lo, 5)", &[("v", 4.0), ("lo", NAN)]).is_nan());
    }

    /// `copy_sign(a, b)`: the magnitude of `a`, the sign bit of `b` (−0 included).
    #[test]
    fn copy_sign_is_a_bit_select() {
        assert_eq!(eval("math.copy_sign(-1.1, 3.1)"), 1.1);
        assert_eq!(eval("math.copy_sign(2, -0.5)"), -2.0);
        assert_eq!(eval_vars("math.copy_sign(2, v.b)", &[("b", -0.0)]), -2.0);
        assert_eq!(eval_vars("math.copy_sign(2, v.b)", &[("b", 0.0)]), 2.0);
        assert_bits(
            eval_vars("math.copy_sign(v.a, -1)", &[("a", 0.0)]),
            -0.0,
            "copy_sign(0, -1)",
        );
        let nan = eval_vars("math.copy_sign(v.a, -1)", &[("a", NAN)]);
        assert!(nan.is_nan() && nan.is_sign_negative());
    }

    /// Not quantised like the easings: the bits equal the sine / cosine of `x·0.017453292`.
    #[test]
    fn sin_and_cos_are_plain_in_degrees() {
        let deg = f32::from_bits(0x3c8e_fa35);
        assert_eq!(deg, 0.017_453_292);
        for x in [
            0.0_f32,
            1.0,
            30.0,
            45.0,
            90.0 / 1.3,
            138.461_54,
            180.0,
            -77.7,
            1000.0,
            12_345.6,
        ] {
            let radians = x * deg;
            assert_bits(
                eval_vars("math.sin(v.x)", &[("x", x)]),
                crate::common::library::sin(radians),
                &format!("sin({x})"),
            );
            assert_bits(
                eval_vars("math.cos(v.x)", &[("x", x)]),
                crate::common::library::cos(radians),
                &format!("cos({x})"),
            );
        }
        assert_eq!(eval("math.sin(180)"), -8.742_278e-8);
        assert_eq!(eval_vars("math.sin(v.x)", &[("x", 180.0)]), -8.742_278e-8);
        // The plain sine is not the quantised one: sin(90/1.3) differs from the table sine.
        let x = 90.0_f32 / 1.3;
        let table_index = ((x * deg * f32::from_bits(0x4622_f983)) as i32 & 0xffff) as f32;
        assert_ne!(
            eval_vars("math.sin(v.x)", &[("x", x)]),
            crate::common::library::sin(table_index / f32::from_bits(0x4622_f983))
        );
    }

    /// `die_roll(n, a, b)`: `trunc(n)` rolls (none below 1), bounds floored and sorted, one draw
    /// per roll; the sum need not be integral. Each roll adds `r·hi + (1 − r)·lo` to the sum.
    #[test]
    fn die_roll() {
        assert_eq!(
            run("math.die_roll(3, 1, 6)", &[], &[0.0, 0.0, 0.0]),
            (3.0, 3)
        );
        assert_eq!(
            run("math.die_roll(3, 1, 6)", &[], &[1.0, 1.0, 1.0]),
            (18.0, 3)
        );
        assert_eq!(
            run("math.die_roll(3, 6, 1)", &[], &[0.5, 0.5, 0.5]),
            (10.5, 3),
            "inverted bounds, not integral"
        );
        // trunc(2.9) = 2 rolls; floor(1.9) = 1, floor(6.9) = 6.
        assert_eq!(
            run(
                "math.die_roll(v.n, v.a, v.b)",
                &[("n", 2.9), ("a", 1.9), ("b", 6.9)],
                &[1.0, 1.0]
            ),
            (12.0, 2)
        );
        for n in [0.9_f32, 0.0, -2.0] {
            assert_eq!(
                run("math.die_roll(v.n, 1, 6)", &[("n", n)], &[0.5]),
                (0.0, 0),
                "n = {n}"
            );
        }
        assert_eq!(
            run("math.die_roll(v.n, 1, 6) * 2 + 1.5", &[("n", 0.0)], &[]),
            (1.5, 0)
        );
        let r = third();
        let (lo, hi) = (1.0_f32, 7.0_f32);
        let vars = [("a", 1.1), ("b", 7.3)];
        let rolled = (0.0 + (r * hi + (1.0 - r) * lo)) + (r * hi + (1.0 - r) * lo);
        assert_bits(
            run("math.die_roll(2, v.a, v.b)", &vars, &[r, r]).0,
            rolled,
            "die_roll",
        );
    }

    /// `die_roll_integer`: each roll is an integer in [lo, hi] (floored, clamped), summed, each
    /// `floor((1 − r)·lo + r·top)`.
    #[test]
    fn die_roll_integer() {
        assert_eq!(
            run("math.die_roll_integer(3, 1, 6)", &[], &[0.0, 0.0, 0.0]),
            (3.0, 3)
        );
        assert_eq!(
            run("math.die_roll_integer(3, 1, 6)", &[], &[1.0, 1.0, 1.0]),
            (18.0, 3)
        );
        assert_eq!(
            run("math.die_roll_integer(3, 6, 1)", &[], &[0.5, 0.5, 0.5]),
            (9.0, 3)
        );
        assert_eq!(
            run("math.die_roll_integer(2, 1.9, 6.9)", &[], &[0.0, 1.0]),
            (7.0, 2)
        );
        assert_eq!(
            run("math.die_roll_integer(0.5, 1, 6)", &[], &[0.5]),
            (0.0, 0)
        );
        for sample in [0.0_f32, 0.25, 0.5, 0.999_999_9, 1.0] {
            let (value, _) = run("math.die_roll_integer(1, 2, 4)", &[], &[sample]);
            assert!(
                (2.0..=4.0).contains(&value) && value.fract() == 0.0,
                "{sample} → {value}"
            );
        }
        // `floor(r·span + lo)` would give 0 at (−3, 2) with the sample 0.5.
        let vars = [("a", -3.0), ("b", 2.0)];
        assert_eq!(
            run("math.die_roll_integer(1, v.a, v.b)", &vars, &[0.5]).0,
            -1.0
        );
        let (lo, hi, r) = (-3.0_f32, 2.0_f32, 0.5_f32);
        let top = (-f32::EPSILON * hi + 1.0) + hi;
        assert_eq!(((1.0 - r) * lo + top * r).floor(), -1.0);
        let span = hi.mul_add(-f32::EPSILON, (hi + 1.0) - lo);
        assert_eq!(r.mul_add(span, lo).floor(), 0.0);
    }

    #[test]
    fn exp_ln_pow_sqrt() {
        for x in [0.0_f32, 1.0, -1.0, 0.3, 10.0, 88.0] {
            assert_bits(
                eval_vars("math.exp(v.x)", &[("x", x)]),
                crate::common::library::exp(x),
                &format!("exp({x})"),
            );
        }
        for x in [1.0_f32, 2.0, 0.5, 1.0e-3, 1.0e30] {
            assert_bits(
                eval_vars("math.ln(v.x)", &[("x", x)]),
                crate::common::library::ln(x),
                &format!("ln({x})"),
            );
        }
        assert_eq!(eval("math.ln(0)"), f32::NEG_INFINITY);
        assert!(eval("math.ln(-1)").is_nan());
        for (a, b) in [
            (2.0_f32, 10.0_f32),
            (2.0, 0.5),
            (-2.0, 3.0),
            (0.0, 0.0),
            (10.0, -2.0),
        ] {
            assert_bits(
                eval_vars("math.pow(v.a, v.b)", &[("a", a), ("b", b)]),
                crate::common::library::pow(a, b),
                &format!("pow({a}, {b})"),
            );
        }
        assert_eq!(eval("math.sqrt(16)"), 4.0);
        assert!(eval("math.sqrt(-1)").is_nan());
        assert!(eval_vars("math.sqrt(v.x)", &[("x", -1.0)]).is_nan());
        assert_bits(
            eval_vars("math.sqrt(v.x)", &[("x", 2.0)]),
            2.0_f32.sqrt(),
            "sqrt(2)",
        );
        assert_bits(
            eval_vars("math.sqrt(v.x)", &[("x", -0.0)]),
            -0.0,
            "sqrt(-0)",
        );
    }

    /// `3t² − 2t³`, `t` not clamped, as the terms `(3t)·t − ((t + t)·t)·t`.
    #[test]
    fn hermite_blend() {
        assert_eq!(eval("math.hermite_blend(0.5)"), 0.5);
        assert_eq!(eval_vars("math.hermite_blend(v.t)", &[("t", 2.0)]), -4.0);
        assert_eq!(eval_vars("math.hermite_blend(v.t)", &[("t", -1.0)]), 5.0);
        let t = f32::from_bits(0x3fde_6363);
        assert_bits(
            eval_vars("math.hermite_blend(v.t)", &[("t", t)]),
            (3.0 * t) * t - ((t + t) * t) * t,
            "",
        );
    }

    /// `lerp(a, b, t)` = `a + t·(b − a)` with `t` not clamped, unfused.
    #[test]
    fn lerp() {
        assert_eq!(eval("math.lerp(0, 10, 0.5)"), 5.0);
        assert_eq!(eval_vars("math.lerp(0, 10, v.t)", &[("t", 2.0)]), 20.0);
        assert_eq!(eval_vars("math.lerp(0, 10, v.t)", &[("t", -0.5)]), -5.0);
        assert_eq!(eval("math.lerp(10, 0, 0.25)"), 7.5);
        let la = 1.1_f32;
        let lb = la * 3.3;
        let t = third();
        let vars = [("a", la), ("b", lb), ("t", t)];
        let unfused = la + t * (lb - la);
        let fused = t.mul_add(lb - la, la);
        assert_ne!(unfused, fused);
        assert_bits(
            eval_vars("math.lerp(v.a, v.b, v.t)", &vars),
            unfused,
            "lerp",
        );
    }

    /// `lerprotate` interpolates over the wrapped difference and is not
    /// re-wrapped; `min_angle` wraps to [−180, 180).
    #[test]
    fn angles() {
        assert_eq!(eval("math.lerprotate(350, 10, 0.5)"), 360.0);
        assert_eq!(
            eval_vars("math.lerprotate(v.a, 10, 0.5)", &[("a", 350.0)]),
            360.0
        );
        assert_eq!(eval("math.lerprotate(10, 350, 0.5)"), 0.0);
        assert_eq!(eval("math.lerprotate(0, 90, 2)"), 180.0);
        for (x, wrapped) in [
            (180.0_f32, -180.0_f32),
            (370.0, 10.0),
            (-370.0, -10.0),
            (-180.0, -180.0),
            (540.0, -180.0),
            (179.5, 179.5),
            (0.0, 0.0),
        ] {
            assert_eq!(
                eval_vars("math.min_angle(v.x)", &[("x", x)]),
                wrapped,
                "min_angle({x})"
            );
            assert_eq!(
                eval(&format!("math.min_angle({x})")),
                wrapped,
                "folded min_angle({x})"
            );
        }
    }

    /// `min_angle` shifts its argument by +180 before the remainder, and that sum is rounded: 2^-14
    /// above 1000 is a tie at 1180 (grid 2^-13) that rounds away, so the wrapped angle is exactly
    /// -80.
    #[test]
    fn min_angle_rounds_the_argument_shifted_by_plus_180() {
        let step = 2.0_f32.powi(-14);
        assert_bits(
            eval_vars("math.min_angle(v.x)", &[("x", 1000.0 + step)]),
            -80.0,
            "min_angle(1000 + 2^-14)",
        );
        assert_bits(
            eval_vars("math.min_angle(v.x)", &[("x", -1000.0 - step)]),
            79.999_94,
            "min_angle(-1000 - 2^-14)",
        );
    }

    /// A NaN gives the second operand when both are variables and is dropped against a constant.
    #[test]
    fn min_max_with_nan() {
        let nan = [("n", NAN), ("f", 4.0)];
        for function in ["max", "min"] {
            let first = format!("math.{function}(v.n, v.f)");
            let second = format!("math.{function}(v.f, v.n)");
            assert_eq!(eval_vars(&first, &nan), 4.0, "{first}");
            assert!(eval_vars(&second, &nan).is_nan(), "{second}");
            for constant in [
                format!("math.{function}(4, v.n)"),
                format!("math.{function}(v.n, 4)"),
            ] {
                assert_eq!(eval_vars(&constant, &nan), 4.0, "{constant}");
            }
            assert!(eval_vars(&format!("math.{function}(v.n, v.n)"), &nan).is_nan());
        }
        assert_eq!(eval("math.max(0, 1)"), 1.0);
        assert_eq!(eval("math.max(-1, -2)"), -1.0);
        assert_eq!(eval("math.min(0, 1)"), 0.0);
        assert_eq!(eval("math.min(-1, -2)"), -2.0);
    }

    /// `mod` is the truncated remainder (sign of the dividend); a run-time zero
    /// divisor gives the post-op offset; a literal zero divisor has no test and gives NaN.
    #[test]
    fn mod_rules() {
        assert_eq!(
            eval_vars("math.mod(v.a, 3)", &[("a", -5.1)]),
            -5.1_f32 % 3.0
        );
        assert!((eval_vars("math.mod(v.a, 3)", &[("a", -5.1)]) + 2.1).abs() < 1e-6);
        assert_eq!(
            eval_vars("math.mod(v.a, v.b)", &[("a", 7.5), ("b", -2.0)]),
            1.5
        );
        let zero = [("x", 1.0), ("y", 0.0)];
        assert_eq!(eval_vars("math.mod(v.x, v.y)", &zero), 0.0);
        assert_eq!(eval_vars("math.mod(v.x, v.y) + 0.125", &zero), 0.125);
        assert_eq!(eval_vars("math.mod(v.x, v.y) * 3 + 0.125", &zero), 0.125);
        assert_eq!(
            eval_vars("math.mod(v.x, v.y * 2)", &zero),
            0.0,
            "an expression divisor"
        );
        assert!(eval_vars("math.mod(v.x, 0)", &zero).is_nan());
        assert!(eval("math.mod(1, 0)").is_nan());
        assert!(eval("math.mod(5, 0)").is_nan());
    }

    /// `math.pi` is the f32 π and takes no parentheses: `math.pi()` is rejected (#8).
    #[test]
    fn pi() {
        assert_bits(eval("math.pi"), f32::from_bits(0x4049_0fdb), "math.pi");
        assert_eq!(eval("math.pi"), 3.141_592_74);
        assert_eq!(eval("math.PI * 2"), 6.283_185_5);
        assert_eq!(eval("Math.Pi / 2"), std::f32::consts::FRAC_PI_2);
        let compiled = compile("math.pi()", &server_at(13));
        assert_eq!(compiled.failure(), Some(CompileFailure::Rejected));
        assert!(
            compiled
                .diagnostics()
                .iter()
                .any(|d| d.message().starts_with(
                    "found multiple operations without a combining operation between them"
                )),
            "{:?}",
            compiled.diagnostics()
        );
    }

    /// Sorted bounds, a closed interval; literal bounds fold into the post-op (`S = hi − lo`, `O =
    /// lo`). Run-time bounds interpolate as `hi·r + (1 − r)·lo`.
    #[test]
    fn random() {
        for (sample, plain, scaled) in [
            (0.0_f32, 2.0_f32, 10.0_f32),
            (0.5, 4.5, 17.5),
            (1.0, 7.0, 25.0),
        ] {
            assert_eq!(run("math.random(2.0, 7.0)", &[], &[sample]), (plain, 1));
            assert_eq!(
                run("math.random(7.0, 2.0)", &[], &[sample]),
                (plain, 1),
                "inverted"
            );
            assert_eq!(
                run("3.0 * math.random(2.0, 7.0) + 4.0", &[], &[sample]),
                (scaled, 1)
            );
            let vars = [("a", 2.0), ("b", 7.0)];
            assert_eq!(run("math.random(v.a, v.b)", &vars, &[sample]).0, plain);
            assert_eq!(
                run("math.random(v.b, v.a)", &vars, &[sample]).0,
                plain,
                "inverted"
            );
        }
        // Literal bounds: the folded post-op (S', O') = ((hi − lo)·S, lo·S + O), applied to r.
        let r = third();
        let folded = PostOp::new(15.0, 10.0);
        assert_bits(
            run("math.random(7, 2) * 3 + 4", &[], &[r]).0,
            folded.apply(P, r),
            "folded random",
        );
        assert_bits(
            run("math.random(2, 7)", &[], &[r]).0,
            PostOp::new(5.0, 2.0).apply(P, r),
            "folded random",
        );
        let r = third();
        assert_bits(
            run("math.random(7, 2) * 3 + 4", &[], &[r]).0,
            r * 15.0 + 10.0,
            "folded",
        );
        let (lo, hi) = (third(), f32::from_bits(0x3f91_bc0d));
        let r = f32::from_bits(0x3ecf_a978);
        let vars = [("a", lo), ("b", hi)];
        assert_bits(
            run("math.random(v.a, v.b)", &vars, &[r]).0,
            hi * r + (1.0 - r) * lo,
            "run-time",
        );
        // An infinite bound survives.
        let vars = [("a", -180.0), ("b", f32::NEG_INFINITY)];
        assert_eq!(
            run("math.random(v.a, v.b)", &vars, &[0.25]).0,
            f32::NEG_INFINITY
        );
    }

    /// A NaN bound is kept, in either position.
    #[test]
    fn random_with_a_nan_bound() {
        let nan = [("n", NAN)];
        for sample in [0.0_f32, 0.5, 1.0] {
            assert!(run("math.random(v.n, 4)", &nan, &[sample]).0.is_nan());
            assert!(run("math.random(4, v.n)", &nan, &[sample]).0.is_nan());
        }
    }

    /// An infinite bound gives 0, with literal and with run-time bounds.
    #[test]
    fn random_integer_with_an_infinite_bound() {
        let vars = [("a", 0.0), ("b", f32::INFINITY)];
        for sample in [0.0_f32, 0.5, 1.0] {
            for source in [
                "math.random_integer(0, 1e39)",
                "math.random_integer(1e39, 0)",
            ] {
                let (value, _) = run(source, &[], &[sample]);
                assert_eq!(value, 0.0, "{source} at {sample}");
            }
            let (value, _) = run("math.random_integer(v.a, v.b)", &vars, &[sample]);
            assert_eq!(value, 0.0, "run-time bounds at {sample}");
        }
    }

    /// Sorted, unrounded bounds, an integer draw clamped to [lo, hi].
    #[test]
    fn random_integer() {
        for (sample, plain, scaled) in [
            (0.0_f32, 2.0_f32, 10.0_f32),
            (0.5, 4.0, 16.0),
            (1.0, 7.0, 25.0),
        ] {
            assert_eq!(run("math.random_integer(2, 7)", &[], &[sample]), (plain, 1));
            assert_eq!(
                run("math.random_integer(7, 2)", &[], &[sample]).0,
                plain,
                "inverted"
            );
            assert_eq!(
                run("3 * math.random_integer(2, 7) + 4", &[], &[sample]).0,
                scaled
            );
            let vars = [("a", 7.0), ("b", 2.0)];
            assert_eq!(
                run("math.random_integer(v.a, v.b)", &vars, &[sample]).0,
                plain,
                "inverted variables"
            );
        }
        // The bounds are not rounded: at r = 0 the floor falls below lo, and the clamp returns lo.
        assert_eq!(run("math.random_integer(0.1, 1000001)", &[], &[0.0]).0, 0.1);
        assert_eq!(
            run("math.random_integer(v.a, 1000001)", &[("a", 0.1)], &[0.0]).0,
            0.1
        );
        for sample in [0.0_f32, 0.3, 0.999_999_9, 1.0] {
            let value = run(
                "math.random_integer(v.a, v.b)",
                &[("a", -2.0), ("b", 3.0)],
                &[sample],
            )
            .0;
            assert!(
                (-2.0..=3.0).contains(&value) && value.fract() == 0.0,
                "{sample} → {value}"
            );
        }
        let vars = [("a", -3.0), ("b", 2.0)];
        assert_eq!(run("math.random_integer(v.a, v.b)", &vars, &[0.5]).0, -1.0);
    }

    /// `sign(x)` is `x < 0 ? −1 : 1`, so both zeros and NaN give 1.
    #[test]
    fn sign() {
        assert_eq!(eval("math.sign(0)"), 1.0);
        assert_eq!(eval_vars("math.sign(v.x)", &[("x", 0.0)]), 1.0);
        assert_eq!(eval_vars("math.sign(v.x)", &[("x", -0.0)]), 1.0);
        assert_eq!(eval_vars("math.sign(-v.x)", &[("x", 0.0)]), 1.0);
        assert_eq!(eval_vars("math.sign(v.x)", &[("x", -1.0e-30)]), -1.0);
        assert_eq!(eval_vars("math.sign(v.x)", &[("x", 7.0)]), 1.0);
        assert_eq!(
            eval_vars("math.sign(v.x)", &[("x", f32::NEG_INFINITY)]),
            -1.0
        );
        assert_eq!(eval_vars("math.sign(v.x)", &[("x", NAN)]), 1.0);
    }

    /// With a post-op, the negative branch of `sign` is `−(S + O)`, not `O − S`.
    #[test]
    fn sign_post_op_quirk() {
        let neg = [("x", -1.0)];
        assert_eq!(eval_vars("math.sign(v.x) + 1", &neg), -2.0);
        assert_eq!(eval_vars("math.sign(v.x) * -2 + 1", &neg), 1.0);
        assert_eq!(eval_vars("math.sign(v.x) * 3 + 0.5", &neg), -3.5);
        let pos = [("x", 1.0)];
        assert_eq!(eval_vars("math.sign(v.x) + 1", &pos), 2.0);
        assert_eq!(eval_vars("math.sign(v.x) * -2 + 1", &pos), -1.0);
    }

    /// A constant argument folds to its value first, so the factors, terms and negations around
    /// the call are constant arithmetic: a negative argument with a term gives `−S + O`, where the
    /// run-time post-op gives `−(S + O)`.
    #[test]
    fn sign_of_a_constant_folds_to_its_value_first() {
        for (folded, constant, run, x, run_time) in [
            (
                "math.sign(-1) * 2 + 1",
                -1.0,
                "math.sign(v.x) * 2 + 1",
                -1.0,
                -3.0,
            ),
            ("math.sign(-1) + 1", 0.0, "math.sign(v.x) + 1", -1.0, -2.0),
            (
                "math.sign(-1) * -2 + 1",
                3.0,
                "math.sign(v.x) * -2 + 1",
                -1.0,
                1.0,
            ),
            (
                "math.sign(-1) * 2 * 3 + 1",
                -5.0,
                "math.sign(v.x) * 2 * 3 + 1",
                -1.0,
                -7.0,
            ),
            (
                "1 - math.sign(-1) * 2",
                3.0,
                "1 - math.sign(v.x) * 2",
                -1.0,
                1.0,
            ),
            (
                "-math.sign(-1) * 2 + 1",
                3.0,
                "-math.sign(v.x) * 2 + 1",
                -1.0,
                1.0,
            ),
            (
                "math.sign(1) * -2 + 1",
                -1.0,
                "math.sign(v.x) * -2 + 1",
                1.0,
                -1.0,
            ),
        ] {
            assert_eq!(folded_constant(folded), constant, "{folded}");
            assert_eq!(eval_vars(run, &[("x", x)]), run_time, "{run}");
        }
        // sign(NaN) is the positive branch.
        assert_eq!(folded_constant("math.sign(math.sqrt(-1)) * 2 + 1"), 3.0);
        assert_eq!(eval_vars("math.sign(v.x) * 2 + 1", &[("x", NAN)]), 3.0);
    }

    /// `inverse_lerp(a, b, v)` = `(v − a)/(b − a)`, unclamped and with no zero guard.
    #[test]
    fn inverse_lerp() {
        assert_eq!(eval("math.inverse_lerp(1, 5, 3)"), 0.5);
        assert_eq!(
            eval_vars("math.inverse_lerp(1, 5, v.v)", &[("v", 9.0)]),
            2.0
        );
        assert_eq!(
            eval_vars("math.inverse_lerp(1, 5, v.v)", &[("v", -3.0)]),
            -1.0
        );
        assert_eq!(
            eval_vars("math.inverse_lerp(5, 5, v.v)", &[("v", 4.0)]),
            f32::NEG_INFINITY
        );
        assert_eq!(
            eval_vars("math.inverse_lerp(5, 5, v.v)", &[("v", 6.0)]),
            f32::INFINITY
        );
        assert!(eval_vars("math.inverse_lerp(5, 5, v.v)", &[("v", 5.0)]).is_nan());
    }

    type Curve = fn(f64) -> f64;

    fn powi(t: f64, n: i32) -> f64 {
        t.powi(n)
    }

    fn in_out(t: f64, first: impl Fn(f64) -> f64, second: impl Fn(f64) -> f64) -> f64 {
        let t2 = 2.0 * t;
        if t2 < 1.0 { first(t2) } else { second(t2) }
    }

    fn out_bounce(t: f64) -> f64 {
        let (n1, d1) = (7.5625, 2.75);
        if t < 1.0 / d1 {
            n1 * t * t
        } else if t < 2.0 / d1 {
            let u = t - 1.5 / d1;
            n1 * u * u + 0.75
        } else if t < 2.5 / d1 {
            let u = t - 2.25 / d1;
            n1 * u * u + 0.9375
        } else {
            let u = t - 2.625 / d1;
            n1 * u * u + 0.984_375
        }
    }

    const C1: f64 = 1.701_58;
    const C3: f64 = C1 + 1.0;
    const C2: f64 = C1 * 1.525;
    const ELASTIC: f64 = 2.0 * std::f64::consts::PI / 3.0;

    /// Every curve as a formula (f(t) of `start + (end − start)·f(t)`).
    fn curves() -> Vec<(&'static str, Curve, f64)> {
        use std::f64::consts::PI;
        // (function, f, tolerance as a fraction of the span)
        vec![
            ("ease_in_quad", |t| powi(t, 2), 1e-6),
            ("ease_out_quad", |t| 1.0 - powi(1.0 - t, 2), 1e-6),
            (
                "ease_in_out_quad",
                |t| in_out(t, |u| 0.5 * powi(u, 2), |u| 1.0 - 0.5 * powi(2.0 - u, 2)),
                1e-6,
            ),
            ("ease_in_cubic", |t| powi(t, 3), 1e-6),
            ("ease_out_cubic", |t| 1.0 - powi(1.0 - t, 3), 1e-6),
            (
                "ease_in_out_cubic",
                |t| in_out(t, |u| 0.5 * powi(u, 3), |u| 1.0 - 0.5 * powi(2.0 - u, 3)),
                1e-6,
            ),
            ("ease_in_quart", |t| powi(t, 4), 1e-6),
            ("ease_out_quart", |t| 1.0 - powi(1.0 - t, 4), 1e-6),
            (
                "ease_in_out_quart",
                |t| in_out(t, |u| 0.5 * powi(u, 4), |u| 1.0 - 0.5 * powi(2.0 - u, 4)),
                1e-6,
            ),
            ("ease_in_quint", |t| powi(t, 5), 1e-6),
            ("ease_out_quint", |t| 1.0 - powi(1.0 - t, 5), 1e-6),
            (
                "ease_in_out_quint",
                |t| in_out(t, |u| 0.5 * powi(u, 5), |u| 1.0 - 0.5 * powi(2.0 - u, 5)),
                1e-6,
            ),
            // The sine curves are quantised to the 65,536-entry table: within one entry of the
            // plain sine.
            ("ease_in_sine", |t| 1.0 - (t * PI / 2.0).cos(), 2e-4),
            ("ease_out_sine", |t| (t * PI / 2.0).sin(), 2e-4),
            ("ease_in_out_sine", |t| -((PI * t).cos() - 1.0) / 2.0, 2e-4),
            ("ease_in_expo", |t| (10.0 * t - 10.0).exp2(), 1e-6),
            ("ease_out_expo", |t| 1.0 - (-10.0 * t).exp2(), 1e-6),
            (
                "ease_in_out_expo",
                |t| {
                    if t < 0.5 {
                        (20.0 * t - 10.0).exp2() / 2.0
                    } else {
                        (2.0 - (-20.0 * t + 10.0).exp2()) / 2.0
                    }
                },
                1e-6,
            ),
            ("ease_in_circ", |t| 1.0 - (1.0 - t * t).sqrt(), 1e-6),
            (
                "ease_out_circ",
                |t| (1.0 - (t - 1.0) * (t - 1.0)).sqrt(),
                1e-6,
            ),
            (
                "ease_in_out_circ",
                |t| {
                    if t < 0.5 {
                        (1.0 - (1.0 - powi(2.0 * t, 2)).sqrt()) / 2.0
                    } else {
                        ((1.0 - powi(-2.0 * t + 2.0, 2)).sqrt() + 1.0) / 2.0
                    }
                },
                1e-6,
            ),
            ("ease_in_back", |t| t * t * (C3 * t - C1), 1e-6),
            (
                "ease_out_back",
                |t| 1.0 + C3 * powi(t - 1.0, 3) + C1 * powi(t - 1.0, 2),
                1e-6,
            ),
            (
                "ease_in_out_back",
                |t| {
                    if t < 0.5 {
                        powi(2.0 * t, 2) * ((C2 + 1.0) * 2.0 * t - C2) / 2.0
                    } else {
                        (powi(2.0 * t - 2.0, 2) * ((C2 + 1.0) * (2.0 * t - 2.0) + C2) + 2.0) / 2.0
                    }
                },
                1e-6,
            ),
            // The elastic curves read the table too, and their angle is (10t − 10.75)·2π/3.
            (
                "ease_in_elastic",
                |t| -(10.0 * t - 10.0).exp2() * ((10.0 * t - 10.75) * ELASTIC).sin(),
                2e-4,
            ),
            (
                "ease_out_elastic",
                |t| (-10.0 * t).exp2() * ((10.0 * t - 0.75) * ELASTIC).sin() + 1.0,
                2e-4,
            ),
            (
                "ease_in_out_elastic",
                |t| {
                    let s = ((20.0 * t - 10.75) * ELASTIC).sin();
                    if 2.0 * t < 1.0 {
                        -0.5 * (20.0 * t - 10.0).exp2() * s
                    } else {
                        1.0 + 0.5 * (10.0 - 20.0 * t).exp2() * s
                    }
                },
                2e-4,
            ),
            ("ease_out_bounce", out_bounce, 1e-6),
            ("ease_in_bounce", |t| 1.0 - out_bounce(1.0 - t), 1e-6),
            (
                "ease_in_out_bounce",
                |t| {
                    if t < 0.5 {
                        (1.0 - out_bounce(1.0 - 2.0 * t)) / 2.0
                    } else {
                        (1.0 + out_bounce(2.0 * t - 1.0)) / 2.0
                    }
                },
                1e-6,
            ),
        ]
    }

    fn check_curve(function: &str, f: Curve, tolerance: f64, ts: &[f32]) {
        for &(start, end) in &[(1.0_f32, 5.0_f32), (-2.0, 0.5), (3.0, -1.0)] {
            let span = f64::from(end) - f64::from(start);
            for &t in ts {
                // The endpoints of the elastic curves are special cases (`elastic_endpoints`).
                if function.contains("elastic") && (t == 0.0 || t == 1.0) {
                    continue;
                }
                let source = format!("math.{function}(v.s, v.e, v.t)");
                let actual = f64::from(eval_vars(&source, &[("s", start), ("e", end), ("t", t)]));
                let expected = f64::from(start) + span * f(f64::from(t));
                assert!(
                    (actual - expected).abs() <= tolerance * span.abs().max(1.0) + 1e-6,
                    "{function}({start}, {end}, {t}) = {actual}, the formula gives {expected}"
                );
            }
        }
        // Literal operands fold to the same value as run-time ones.
        let folded = eval(&format!("math.{function}(1, 5, 0.3)"));
        let runtime = eval_vars(&format!("math.{function}(1, 5, v.t)"), &[("t", 0.3)]);
        assert_bits(folded, runtime, &format!("folded {function}"));
    }

    const UNIT: [f32; 13] = [
        0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.75, 0.9, 0.97, 1.0,
    ];

    #[test]
    fn easing_curves_follow_their_formulas() {
        for (function, f, tolerance) in curves() {
            check_curve(function, f, tolerance, &UNIT);
        }
    }

    /// The polynomial, expo and back curves keep their formula outside [0, 1].
    #[test]
    fn easings_do_not_clamp_t() {
        let outside = [-1.0_f32, -0.5, 1.25, 2.0];
        for (function, f, tolerance) in curves() {
            let unclamped = ["quad", "cubic", "quart", "quint", "expo", "back"]
                .iter()
                .any(|kind| function.ends_with(kind));
            if unclamped {
                check_curve(function, f, tolerance.max(1e-5), &outside);
            }
        }
        assert_eq!(eval("math.ease_in_quad(0, 1, 2)"), 4.0);
        assert_eq!(
            eval_vars("math.ease_in_quad(0, 1, v.t)", &[("t", -2.0)]),
            4.0
        );
        assert_eq!(
            eval_vars("math.ease_in_cubic(0, 1, v.t)", &[("t", -2.0)]),
            -8.0
        );
        assert_eq!(
            eval_vars("math.ease_out_quad(1, 5, v.t)", &[("t", 2.0)]),
            1.0
        );
    }

    /// The sine and elastic curves read a 65,536-entry sine table at the truncated index of an
    /// angle rounded in three steps, so for some `t` a single rounding would land one entry apart.
    #[test]
    fn the_table_easings_round_their_angle_in_steps() {
        let pinned: [(&str, f32, f32, f32, u32); 6] = [
            ("math.ease_in_elastic", 0.0, 1.0, 0.2235, 0xbb80_080a),
            ("math.ease_in_elastic", 0.0, 1.0, 0.4485, 0x3c3c_cdfe),
            ("math.ease_out_elastic", 0.0, 1.0, 0.0375, 0x3ee8_d482),
            ("math.ease_out_elastic", 0.0, 1.0, 0.1875, 0x3f98_ad80),
            ("math.ease_in_out_elastic", 0.0, 1.0, 0.0125, 0x2e5f_42b2),
            ("math.ease_in_out_elastic", 0.0, 1.0, 0.0875, 0xb429_1280),
        ];
        for (function, start, end, t, bits) in pinned.as_slice() {
            let source = format!("{function}(v.s, v.e, v.t)");
            let vars = [("s", *start), ("e", *end), ("t", *t)];
            assert_eq!(
                eval_vars(&source, &vars).to_bits(),
                *bits,
                "{function}({start}, {end}, {t})"
            );
            let folded = format!("{function}({start}, {end}, {t})");
            assert_eq!(eval(&folded).to_bits(), *bits, "folded {folded}");
        }
    }

    /// The start as a variable holding −0.0 and as the literal `-0.0` give the same result; the
    /// other operands cannot be literals, since a decimal literal does not reach the smallest
    /// subnormal.
    #[test]
    fn in_out_bounce_second_half_with_a_negative_zero_start_evaluated() {
        let tiny = f32::from_bits(1); // 1.4e-45
        assert_eq!(eval("math.copy_sign(1, -0.0)"), -1.0);
        let vars = [("s", -0.0_f32), ("e", -tiny), ("t", 0.5)];
        for source in [
            "math.ease_in_out_bounce(v.s, v.e, v.t)",
            "math.ease_in_out_bounce(-0.0, v.e, v.t)",
            "math.ease_in_out_bounce(-0.0, v.e, 0.5)",
        ] {
            assert_eq!(
                eval_vars(source, &vars).to_bits(),
                0.0_f32.to_bits(),
                "{source}"
            );
        }
    }

    #[test]
    fn expo_has_no_endpoint_special_cases() {
        assert_eq!(
            eval_vars("math.ease_in_expo(1, 5, v.t)", &[("t", 0.0)]),
            1.003_906_25
        );
        assert_eq!(eval("math.ease_in_expo(1, 5, 0)"), 1.003_906_25);
        // e − (e − s)·2^−10
        assert_eq!(
            eval_vars("math.ease_out_expo(1, 5, v.t)", &[("t", 1.0)]),
            5.0 - 4.0 / 1024.0
        );
        assert_eq!(
            eval_vars("math.ease_in_out_expo(1, 5, v.t)", &[("t", 0.0)]),
            1.0 + 4.0 / 2048.0
        );
        assert_eq!(
            eval_vars("math.ease_in_out_expo(1, 5, v.t)", &[("t", 1.0)]),
            5.0 - 4.0 / 2048.0
        );
    }

    /// The elastic curves special-case `t == 0` (start) and `t == 1` (end), that end as
    /// `start + (end − start)`.
    #[test]
    fn elastic_endpoints() {
        let (start, end) = (0.3_f32, 0.1_f32);
        assert_ne!(start + (end - start), end);
        for function in ["ease_in_elastic", "ease_out_elastic", "ease_in_out_elastic"] {
            let source = format!("math.{function}(v.s, v.e, v.t)");
            assert_eq!(
                eval_vars(&source, &[("s", start), ("e", end), ("t", 0.0)]),
                start,
                "{function} at 0"
            );
            assert_eq!(
                eval_vars(&source, &[("s", 1.0), ("e", 5.0), ("t", 1.0)]),
                5.0,
                "{function} at 1"
            );
            assert_eq!(
                eval_vars(&source, &[("s", start), ("e", end), ("t", 1.0)]),
                start + (end - start),
                "{function}"
            );
        }
        let value = eval_vars("math.ease_in_out_elastic(1, 5, v.t)", &[("t", 0.3)]);
        assert!((value - 0.937_503_457).abs() <= 1e-6, "{value}");
    }

    /// At `t = 0.5` the in-out elastic takes its second branch, `start + (−c·0.5 + c)`: for
    /// the smallest subnormal range `c` the half `c/2` ties to even (0), so the result is `c`,
    /// where the first branch (`c·0.5 + start`) would give 0.
    #[test]
    fn in_out_elastic_midpoint_on_a_subnormal_range_keeps_the_rounding_of_the_second_branch() {
        let tiny = f32::from_bits(1);
        let three = f32::from_bits(3);
        let source = "math.ease_in_out_elastic(v.s, v.e, v.t)";
        let at = |end: f32| eval_vars(source, &[("s", 0.0), ("e", end), ("t", 0.5)]);
        assert_eq!(at(tiny).to_bits(), tiny.to_bits());
        assert_eq!(at(-tiny).to_bits(), (-tiny).to_bits());
        assert_eq!(at(three).to_bits(), tiny.to_bits());
        assert_eq!(at(4.0), 2.0);
    }

    /// The sine and elastic easings read a 65,536-entry table at `trunc(rad·10430.378) mod 65536`
    /// (cosine `+ 16384`); the entry is the sine of `i / 10430.378`. `math.sin` itself is not
    /// quantised.
    #[test]
    fn sine_table() {
        let scale = f32::from_bits(0x4622_f983);
        assert_eq!(scale, 10_430.378);
        let half_pi = f32::from_bits(0x3fc9_0fdb);
        for step_count in [1_u8, 7, 30, 50, 81, 99] {
            let t = f32::from(step_count) / 100.0;
            let radians = t * half_pi;
            let index = ((radians * scale) as i32 & 0xffff) as f32;
            // ease_out_sine(0, 1, t) is the table sine of t·π/2.
            assert_bits(
                eval_vars("math.ease_out_sine(0, 1, v.t)", &[("t", t)]),
                crate::common::library::sin(index / scale),
                "entry",
            );
        }
        // The cosine: ease_in_sine(0, 1, t) is 1 minus the entry trunc(t·π/2·10430.378 + 16384)
        // mod 65536.
        let t = 0.3_f32;
        let index = (((t * half_pi) * scale + 16_384.0) as i32 & 0xffff) as f32;
        assert_bits(
            eval_vars("math.ease_in_sine(0, 1, v.t)", &[("t", t)]),
            1.0 - crate::common::library::sin(index / scale),
            "cosine",
        );
    }

    /// Every random-family sample is clamped to [0, 1] (NaN to 0) before use. A generator's samples
    /// lie in [0, 1], so raw samples outside it reach the functions the evaluator calls only when
    /// given directly.
    #[test]
    fn samples_are_clamped() {
        const ID: PostOp = PostOp::IDENTITY;
        for (sample, expected) in [(1.0_f32, 7.0_f32), (0.0, 2.0), (0.5, 4.5)] {
            assert_eq!(
                run("math.random(2, 7)", &[], &[sample]).0,
                expected,
                "random, {sample}"
            );
            assert_eq!(
                run("math.random(v.a, 7)", &[("a", 2.0)], &[sample]).0,
                expected,
                "run-time random, {sample}"
            );
        }
        for (sample, expected) in [
            (2.0_f32, 7.0_f32),
            (-1.0, 2.0),
            (NAN, 2.0),
            (f32::INFINITY, 7.0),
            (0.5, 4.5),
        ] {
            assert_eq!(
                math::random_folded(P, sample, math::random_const_bounds(P, 2.0, 7.0, ID)),
                expected,
                "random, {sample}"
            );
            assert_eq!(
                math::random(P, 2.0, 7.0, sample, ID),
                expected,
                "run-time random, {sample}"
            );
        }
        assert_eq!(math::random_integer_const_bounds(P, 2.0, 7.0, 5.0, ID), 7.0);
        assert_eq!(
            math::random_integer_const_bounds(P, 2.0, 7.0, -5.0, ID),
            2.0
        );
        let rolled = |mut roll: math::DieRoll, samples: [f32; 2]| {
            for sample in samples {
                roll.roll(sample);
            }
            roll.finish(ID)
        };
        assert_eq!(
            rolled(math::DieRoll::new(P, 2.0, 1.0, 6.0), [9.0, -9.0]),
            7.0
        );
        assert_eq!(
            rolled(math::DieRoll::new_integer(P, 2.0, 1.0, 6.0), [9.0, NAN]),
            7.0
        );
        assert_eq!(run("math.random_integer(2, 7)", &[], &[1.0]).0, 7.0);
        assert_eq!(
            run("math.die_roll_integer(2, 1, 6)", &[], &[1.0, 0.0]).0,
            7.0
        );
    }

    /// Draw count and order: a random operator draws once, after its bounds (and their
    /// own draws) are evaluated; `die_roll*` draws `k` times after both bounds.
    #[test]
    fn draw_count_and_order() {
        // The inner (bound) draw is first: 0.5 → lo = 0.5, then the outer draw 0.0 → 0.5.
        assert_eq!(
            run("math.random(math.random(0, 1), 10)", &[], &[0.5, 0.0]),
            (0.5, 2)
        );
        // Both bounds' draws, then three rolls of [lo, hi] = [floor(0.2·10), floor(0.9·10)] = [2,
        // 9].
        assert_eq!(
            run(
                "math.die_roll(3, math.random(0, 10), math.random(0, 10))",
                &[],
                &[0.2, 0.9, 0.0, 0.0, 1.0]
            ),
            (13.0, 5)
        );
        assert_eq!(
            run(
                "math.die_roll_integer(2, math.random(0, 10), 6)",
                &[],
                &[0.1, 0.0, 1.0]
            ),
            (7.0, 3)
        );
        // One draw per operator, none for a skipped branch.
        assert_eq!(
            run("v.c = 0; return v.c ? math.random(0, 1) : 2;", &[], &[0.5]),
            (2.0, 0)
        );
        assert_eq!(
            run(
                "math.random_integer(0, 3) + math.random(0, 1)",
                &[],
                &[1.0, 0.5]
            ),
            (3.5, 2)
        );
    }

    /// At every version; the rejected expression evaluates to 0.
    #[test]
    fn wrong_arity_is_rejected_before_evaluation() {
        for version in [1_i16, 2, 13] {
            for source in [
                "math.max(3)",
                "math.min(3)",
                "math.mod(7)",
                "math.pow(2)",
                "math.clamp(1, 2)",
                "math.abs(1, 2)",
                "math.sin()",
            ] {
                let compiled = compile(source, &server_at(version));
                assert_eq!(
                    compiled.failure(),
                    Some(CompileFailure::Rejected),
                    "{source} at v{version}"
                );
                let mut env = NoHostEnv::new();
                assert_eq!(
                    compiled
                        .expr()
                        .cloned()
                        .map_or(0.0, |e| e.eval_f32(&mut env.cx())),
                    0.0
                );
            }
            let compiled = compile("math.max(3)", &server_at(version));
            assert!(
                compiled
                    .diagnostics()
                    .iter()
                    .any(|d| d.message().starts_with("Unexpected number of parameters to Max 'math.max' function - expected 2, found 1.")),
                "{:?}",
                compiled.diagnostics()
            );
        }
    }

    /// With assignments and random draws disallowed, `math.random` and
    /// `math.random_integer` are rejected, `math.die_roll*` never are.
    #[test]
    fn random_is_a_side_effect_die_roll_is_not() {
        let strict = server_at(13).with_allowed_ops(OpSet::all().without_assignments_or_random());
        let lenient = server_at(13).with_allowed_ops(OpSet::all().without_assignments());
        for source in [
            "math.random(0, 1)",
            "math.random_integer(0, 1)",
            "1 + math.random(v.a, 2)",
        ] {
            assert_eq!(
                compile(source, &strict).failure(),
                Some(CompileFailure::Rejected),
                "{source}"
            );
            assert_eq!(compile(source, &lenient).failure(), None, "{source}");
        }
        for source in ["math.die_roll(1, 0, 1)", "math.die_roll_integer(2, 1, 6)"] {
            assert_eq!(compile(source, &strict).failure(), None, "{source}");
            assert_eq!(compile(source, &lenient).failure(), None, "{source}");
        }
    }
}

mod platforms {
    //! The platforms end to end: source text compiled for each platform and run. Where a test
    //! names no platform, every platform gives the same bits.

    // Expected values are written with nine significant digits.
    #![allow(clippy::excessive_precision)]

    use crate::common::{
        PLATFORMS, assert_bits, compile_support::server_at, on, servers_or_android,
    };
    use molang::compile::compile;
    use molang::numeric::{Platform, PostOp};
    use molang::stdlib::math;
    use molang::vm::{EvalLimits, NoHostEnv, Value, VariableName};

    const NAN: f32 = f32::NAN;
    const ID: PostOp = PostOp::IDENTITY;

    /// Compiles at `version` for `platform` (which must succeed) and evaluates with `v.<name>`
    /// preset: the value and the run-time messages.
    fn run_on(
        platform: Platform,
        version: i16,
        source: &str,
        vars: &[(&str, f32)],
    ) -> (f32, Vec<String>) {
        let compiled = compile(source, &on(platform, server_at(version)));
        assert_eq!(
            compiled.failure(),
            None,
            "{source:?}: {:?}",
            compiled.diagnostics()
        );
        let expr = compiled.expr().cloned().expect("an expression");
        let mut env = NoHostEnv::new();
        for (name, value) in vars {
            env.variables_mut()
                .set(VariableName::new(name), Value::Float(*value));
        }
        let value = expr.eval_f32(&mut env.cx());
        (value, env.sink_mut().take())
    }

    /// [`run_on`] on every platform, which must give the same bits and messages.
    #[track_caller]
    fn run_at(version: i16, source: &str, vars: &[(&str, f32)]) -> (f32, Vec<String>) {
        let (value, messages) = run_on(PLATFORMS[0], version, source, vars);
        for platform in PLATFORMS {
            let (other, other_messages) = run_on(platform, version, source, vars);
            assert_eq!(
                (other.to_bits(), &other_messages),
                (value.to_bits(), &messages),
                "{source:?} on {platform:?}"
            );
        }
        (value, messages)
    }

    /// The value on `platform`, which must log nothing.
    #[track_caller]
    fn eval_on(platform: Platform, source: &str, vars: &[(&str, f32)]) -> f32 {
        let (value, messages) = run_on(platform, 13, source, vars);
        assert!(messages.is_empty(), "{source:?}: {messages:?}");
        value
    }

    /// The value every platform gives, which must log nothing.
    #[track_caller]
    fn eval_vars(source: &str, vars: &[(&str, f32)]) -> f32 {
        let (value, messages) = run_at(13, source, vars);
        assert!(messages.is_empty(), "{source:?}: {messages:?}");
        value
    }

    #[track_caller]
    fn eval(source: &str) -> f32 {
        eval_vars(source, &[])
    }

    /// The constant `source` folds to on `platform`.
    #[track_caller]
    fn folded_on(platform: Platform, source: &str) -> f32 {
        let compiled = compile(source, &on(platform, server_at(13)));
        let constant = compiled.expr().and_then(|expr| expr.as_constant());
        constant.unwrap_or_else(|| panic!("{source:?} does not fold"))
    }

    /// The constant `source` folds to on every platform.
    #[track_caller]
    fn folded_constant(source: &str) -> f32 {
        let constant = folded_on(PLATFORMS[0], source);
        for platform in PLATFORMS {
            assert_bits(folded_on(platform, source), constant, source);
        }
        constant
    }

    /// Rounded at every step (no wider intermediate); boolean results are exactly 1.0 / 0.0.
    #[test]
    fn binary32_arithmetic_and_boolean_results() {
        // 2^24 + 1 is not an f32: in f32 the sum is 2^24 and the product exactly 1; a wider
        // intermediate would give 1 + 2^-24, which rounds to 1.0000001.
        let vars = [("a", 16_777_216.0), ("b", 1.0), ("c", 1.0 / 16_777_216.0)];
        assert_eq!(eval_vars("(v.a + v.b) * v.c", &vars), 1.0);
        let vars = [("a", 1.0), ("b", 3.0)];
        assert_bits(eval_vars("v.a / v.b", &vars), 1.0_f32 / 3.0, "1/3");
        assert_bits(
            eval_vars("v.a / v.b * v.b", &vars),
            (1.0_f32 / 3.0) * 3.0,
            "(1/3)·3",
        );
        assert_eq!(eval("7 / 2"), 3.5);
        assert_eq!(eval_vars("v.a / 2", &[("a", 7.0)]), 3.5);
        for (source, expected) in [
            ("v.a < v.b", 1.0_f32),
            ("v.a > v.b", 0.0),
            ("v.a == v.a", 1.0),
            ("v.a != v.a", 0.0),
            ("v.a && v.b", 1.0),
            ("v.a || v.b", 1.0),
            ("!v.a", 0.0),
            ("!!v.b", 1.0),
            ("2 < 3", 1.0),
            ("!5", 0.0),
        ] {
            assert_bits(
                eval_vars(source, &[("a", 2.0), ("b", 3.0)]),
                expected,
                source,
            );
        }
        assert_bits(eval("0.1"), 0.1_f32, "0.1");
        assert_bits(
            eval("1.17549435e-38"),
            f32::MIN_POSITIVE,
            "f32::MIN_POSITIVE",
        );
    }

    /// The post-op is `raw·S` then `+ O`, rounded twice.
    #[test]
    fn post_op_rounding() {
        for source in [
            "v.x = 1 / 3; return v.x * 3 - 1;",
            "v.x = 1; v.y = 3; v.q = v.x / v.y; return v.q * 3 - 1;",
            "v.x = 1 / 3; return (v.x + 0) * 3 - 1;",
        ] {
            assert_bits(eval(source), 0.0, source);
        }
        let x = 1.0_f32 / 3.0;
        assert_bits(eval_vars("v.x * 3 - 1", &[("x", x)]), x * 3.0 - 1.0, "");
    }

    /// `a / b * S + O` divides first, `(a/b)·S + O`.
    #[test]
    fn division_post_op() {
        let (a, b) = (1000.0_f32, 13.0_f32);
        let vars = [("a", a), ("b", b)];
        assert_ne!((a / b) * 3.0, (a * 3.0) / b);
        assert_bits(eval_vars("v.a / v.b * 3", &vars), (a / b) * 3.0, "");
        assert_bits(
            eval_vars("v.a / v.b * 3 + 0.5", &vars),
            (a / b) * 3.0 + 0.5,
            "with offset",
        );
    }

    /// `a * b * S + O` is `(a·b)·S + O`.
    #[test]
    fn multiplication_post_op() {
        let (k, j) = (1.4_f32, 0.75_f32);
        let vars = [("k", k), ("j", j)];
        assert_ne!((k * j) * 7.0, k * (j * 7.0));
        assert_ne!((k * j) * 7.0, j * (k * 7.0));
        assert_bits(eval_vars("v.k * v.j * 7", &vars), (k * j) * 7.0, "");
        assert_bits(
            eval_vars("v.k * v.j * 7 + 0.25", &vars),
            (k * j) * 7.0 + 0.25,
            "with offset",
        );
    }

    /// A constant operand folds to its value before its parent adds a term to it or multiplies it:
    /// the product and the term are two roundings, as in a run-time operand's post-op.
    #[test]
    fn a_constant_operand_folds_to_its_value_before_a_term_or_factor() {
        let third = 1.0_f32 / 3.0;
        let bits = f32::from_bits;
        for (folded, constant, run, vars, run_time) in [
            (
                "(-1/3) * 3 + 1",
                0.0,
                "v.d * 3 + 1",
                &[("d", -third)][..],
                0.0,
            ),
            (
                "(-1/3) * 3 + 1",
                0.0,
                "v.d * v.e + 1",
                &[("d", -third), ("e", 3.0)],
                0.0,
            ),
            ("(1/3) * 3 - 1", 0.0, "v.d * 3 - 1", &[("d", third)], 0.0),
            ("1 - (1/3) * 3", 0.0, "1 - v.d * 3", &[("d", third)], 0.0),
            (
                "math.sqrt(0.1) * 3 - 1",
                bits(0xbd52_3180),
                "math.sqrt(v.d) * 3 - 1",
                &[("d", 0.1)],
                bits(0xbd52_3180),
            ),
            (
                "math.inverse_lerp(0.5, 2, 0) * 3 + 1",
                0.0,
                "math.inverse_lerp(v.d, 2, v.e) * 3 + 1",
                &[("d", 0.5), ("e", 0.0)],
                0.0,
            ),
            (
                "math.acos(0) + (1/3)",
                crate::common::library::pinned(
                    0x42b4_aaaa,
                    0x42b4_aaab,
                    math::acos(LINUX, 0.0, ID) + third,
                ),
                "math.acos(v.x) + (1/3)",
                &[("x", 0.0)],
                crate::common::library::pinned(
                    0x42b4_aaaa,
                    0x42b4_aaab,
                    math::acos(LINUX, 0.0, ID) + third,
                ),
            ),
            (
                "math.acos(1/3) * (1/3)",
                crate::common::library::pinned(
                    0x41bc_13a6,
                    0x41bc_13a6,
                    math::acos(LINUX, third, ID) * third,
                ),
                "math.acos(v.x) * (1/3)",
                &[("x", third)],
                crate::common::library::pinned(
                    0x41bc_13a6,
                    0x41bc_13a6,
                    math::acos(LINUX, third, ID) * third,
                ),
            ),
            (
                "math.atan2(1, -3) * (1/3)",
                crate::common::library::pinned(
                    0x4257_6b8a,
                    0x4257_6b8a,
                    math::atan2(LINUX, 1.0, -3.0, ID) * third,
                ),
                "math.atan2(v.x, -3) * (1/3)",
                &[("x", 1.0)],
                crate::common::library::pinned(
                    0x4257_6b8a,
                    0x4257_6b8a,
                    math::atan2(LINUX, 1.0, -3.0, ID) * third,
                ),
            ),
            // The run-time form adds its constant terms together first.
            (
                "1 * 1 + (1/3) + 2",
                bits(0x4055_5556),
                "v.a * v.b + (1/3) + 2",
                &[("a", 1.0), ("b", 1.0)],
                bits(0x4055_5555),
            ),
        ] {
            assert_bits(folded_constant(folded), constant, folded);
            assert_bits(eval_vars(run, vars), run_time, run);
        }
        // Without a term the product is the plain one, and a further factor multiplies its value.
        assert_bits(folded_constant("(-1) * 0"), -0.0, "(-1) * 0");
        assert_bits(
            folded_constant("(-1/3) * 3 * 3 + 1"),
            -2.0,
            "(-1/3) * 3 * 3 + 1",
        );
    }

    /// A folded constant operand keeps its sign of zero and of NaN: a negation flips the sign bit,
    /// a zero term adds and a factor multiplies, where a run-time operand computes `S·x + O`.
    #[test]
    fn a_folded_constant_operand_keeps_its_signs_of_zero_and_nan() {
        for (folded, constant, run, vars, run_time) in [
            (
                "-((0) * (1))",
                -0.0,
                "-(v.a * v.b)",
                &[("a", 0.0), ("b", 1.0)][..],
                0.0,
            ),
            (
                "((-1) * (0)) + 0",
                0.0,
                "(v.a * v.b) + 0",
                &[("a", -1.0), ("b", 0.0)],
                -0.0,
            ),
            (
                "math.abs(0) * (-1)",
                -0.0,
                "math.abs(v.x) * (-1)",
                &[("x", 0.0)],
                0.0,
            ),
            // A constant divisor of 0 folds the division to 0 and is a factor 0 at run time.
            (
                "math.sign(-1) / 0",
                0.0,
                "math.sign(v.x) / 0",
                &[("x", -1.0)],
                -0.0,
            ),
        ] {
            assert_bits(folded_constant(folded), constant, folded);
            assert_bits(eval_vars(run, vars), run_time, run);
        }
        // The fold flips the sign of the invalid NaN, the run keeps it.
        for platform in PLATFORMS {
            let nan = invalid(platform);
            let negated = f32::from_bits(nan.to_bits() ^ 0x8000_0000);
            let source = "-((math.sqrt(-1)) * (1))";
            assert_bits(folded_on(platform, source), negated, source);
            let vars = [("a", -1.0), ("b", 1.0)];
            let source = "-(math.sqrt(v.a) * v.b)";
            assert_bits(eval_on(platform, source, &vars), nan, source);
        }
        // An assigned constant −0 is stored as +0.
        assert_bits(eval("t.x = -0; return t.x;"), 0.0, "t.x = -0");
        assert_bits(
            eval_vars("t.x = v.z; return t.x;", &[("z", -0.0)]),
            -0.0,
            "t.x = v.z",
        );
    }

    /// An infinite constant factor folds into its operand's post-op and makes the offset `∞·O`, a
    /// NaN for an operand without an offset, where the same infinity read from a variable
    /// multiplies.
    #[test]
    fn an_infinite_constant_factor_folds_into_a_nan_offset() {
        let vars = [("o", 1.0), ("inf", f32::INFINITY)];
        for (folded, run) in [
            ("(v.o < 2) * math.exp(1000) + 1", "(v.o < 2) * v.inf + 1"),
            ("(v.o < 2) * 1e39 + 1", "(v.o < 2) * v.inf + 1"),
            ("(v.o && 1) * math.exp(1000) + 1", "(v.o && 1) * v.inf + 1"),
            ("v.o * math.exp(1000) + 1", "v.o * v.inf + 1"),
            ("v.o * math.exp(1000)", "v.o * v.inf"),
        ] {
            for platform in PLATFORMS {
                let nan = invalid(platform);
                assert_bits(eval_on(platform, folded, &vars), nan, folded);
            }
            assert_bits(eval_vars(run, &vars), f32::INFINITY, run);
        }
        // An operand with an offset of 1 gets the offset ∞.
        let source = "(v.x + 1) * math.exp(1000)";
        assert_bits(eval_vars(source, &[("x", 1.0)]), f32::INFINITY, source);
    }

    /// One example of each of these fold / run-time differences from the `numeric` module docs: a
    /// literal divisor, chained constant factors, two NaNs with a constant term on the left, a NaN
    /// divisor and `math.mod`.
    #[test]
    fn literal_divisors_constant_factors_nan_terms_and_mod_differ_as_listed() {
        let bits = f32::from_bits;
        let sqrt_nan = bits(0xffc0_0000);
        let big = folded_constant("3.4e38");
        for (folded, constant, run, vars, run_time) in [
            (
                "(1/3)/3",
                0x3de3_8e39,
                "v.k/3",
                &[("k", 1.0 / 3.0)][..],
                0x3de3_8e3a,
            ),
            (
                "100/3.4e38",
                0x02c8_2a85,
                "v.k/3.4e38",
                &[("k", 100.0)],
                0x02c8_2a88,
            ),
            (
                "math.sqrt(-1)/0",
                0,
                "v.k/0",
                &[("k", sqrt_nan)],
                sqrt_nan.to_bits(),
            ),
            (
                "(0.1*3)*(1/3)",
                0x3dcc_ccce,
                "(v.k*3)*(1/3)",
                &[("k", 0.1)],
                0x3dcc_cccd,
            ),
            (
                "3.4e38*3.4e38*1e-45+1",
                0x7f80_0000,
                "v.k*3.4e38*1e-45+1",
                &[("k", big)],
                0x74ff_934a,
            ),
            (
                "1/math.sqrt(-1)",
                0,
                "1/v.k",
                &[("k", sqrt_nan)],
                0xffc0_0000,
            ),
            (
                "math.mod(-4, 2)",
                0x8000_0000,
                "math.mod(v.k, 2)",
                &[("k", -4.0)],
                0,
            ),
        ] {
            assert_bits(folded_constant(folded), bits(constant), folded);
            assert_bits(eval_vars(run, vars), bits(run_time), run);
        }
        for platform in PLATFORMS {
            // Two NaNs with a constant term on the left: the fold keeps the term's, the run the
            // other.
            let source = "math.ln(-1) + 3*math.sqrt(-1)";
            let ln = folded_on(platform, "math.ln(-1)");
            assert_bits(folded_on(platform, source), ln, source);
            let source = "math.ln(-1) + v.k*math.sqrt(-1)";
            let run = eval_on(platform, source, &[("k", 3.0)]);
            assert_bits(run, invalid(platform), source);
            assert_bits(
                folded_on(platform, "math.mod(3, 0)"),
                invalid(platform),
                "mod",
            );
            assert_bits(
                eval_on(platform, "math.mod(3, v.k)", &[("k", 0.0)]),
                0.0,
                "mod",
            );
        }
    }

    /// `/` groups before `*`; `0.1 + 0.2` is 0.30000001.
    #[test]
    fn grouping_and_f32_sums() {
        assert_eq!(eval("7 * 3 / 9"), 2.333_333_49);
        assert_eq!(eval("v.x = 1; return v.x * 3 / 9;"), 0.333_333_343);
        assert_eq!(eval("v.a = 7; return v.a * 3 / 9;"), 2.333_333_49);
        assert_eq!(eval("2 * 3 / 4 * 5"), 7.5);
        assert_eq!(eval("0.1 + 0.2"), 0.300_000_01);
        assert_eq!(eval_vars("v.a + 0.2", &[("a", 0.1)]), 0.300_000_01);
        assert_eq!(
            eval_vars("v.a + v.b", &[("a", 0.1), ("b", 0.2)]),
            0.300_000_01
        );
    }

    /// −0 is false, NaN true, and a value that is not a number (a struct) false.
    #[test]
    fn truthiness() {
        assert_eq!(eval("!-0.0"), 1.0);
        assert_eq!(eval("-0.5 ? 5 : 6"), 5.0);
        assert_eq!(eval("0.0000001 ? 5 : 6"), 5.0);
        assert_eq!(eval_vars("v.x ? 5 : 6", &[("x", -0.0)]), 6.0);
        let nan = [("n", NAN)];
        assert_eq!(eval_vars("v.n ? 5 : 6", &nan), 5.0);
        assert_eq!(eval_vars("!v.n", &nan), 0.0);
        assert_eq!(eval_vars("v.n && 1", &nan), 1.0);
        assert_eq!(eval_vars("0 || v.n", &nan), 1.0);
        assert_eq!(eval("v.s.a = 1; return v.s ? 5 : 6;"), 6.0);
        assert_eq!(eval("v.s.a = 1; return !v.s;"), 1.0);
    }

    /// `==` / `!=` are exact, and NaN is unequal to everything, itself included.
    #[test]
    fn equality_is_exact() {
        let close = [("a", 1.0), ("b", 1.000_000_1)];
        assert_eq!(eval_vars("v.a == v.b", &close), 0.0);
        assert_eq!(eval_vars("v.a != v.b", &close), 1.0);
        assert_eq!(eval_vars("v.a == v.a", &close), 1.0);
        assert_eq!(eval_vars("v.z == 0", &[("z", -0.0)]), 1.0);
        let nan = [("n", NAN), ("o", 1.0)];
        for (source, expected) in [
            ("v.n == v.n", 0.0),
            ("v.n != v.n", 1.0),
            ("v.n == v.o", 0.0),
            ("v.o == v.n", 0.0),
            ("v.n != 1", 1.0),
            ("1 != v.n", 1.0),
        ] {
            assert_eq!(eval_vars(source, &nan), expected, "{source}");
        }
    }

    /// Every ordered comparison with a NaN is false.
    #[test]
    fn nan_comparisons() {
        let vars = [("n", NAN), ("o", 4.0)];
        for source in [
            "v.n < v.o",
            "v.n <= v.o",
            "v.n > v.o",
            "v.n >= v.o",
            "v.o < v.n",
            "v.o <= v.n",
            "v.o > v.n",
            "v.o >= v.n",
            "v.n < 4",
            "4 > v.n",
            "v.n <= v.n",
            "v.n >= v.n",
        ] {
            assert_eq!(eval_vars(source, &vars), 0.0, "{source}");
        }
    }

    /// A comparison or logical result under a post-op is one of the precomputed
    /// constants `F = O`, `T = S + O`.
    #[test]
    fn comparison_results_are_precomputed_constants() {
        let vars = [("a", 1.0), ("b", 2.0)];
        assert_bits(
            eval_vars("(v.a < v.b) * 0.1 + 0.2", &vars),
            0.1_f32 + 0.2,
            "T",
        );
        assert_bits(eval_vars("(v.a > v.b) * 0.1 + 0.2", &vars), 0.2, "F");
        assert_bits(eval_vars("(v.a == v.a) * -2 - 1", &vars), -3.0, "T");
        assert_bits(eval_vars("(v.a && v.b) * 3 + 1", &vars), 4.0, "T");
        assert_bits(eval_vars("(!v.a) * 3 + 1", &vars), 1.0, "F");
        assert_bits(eval_vars("(v.a > v.b) * 3", &vars), 0.0, "F without offset");
    }

    /// `min`, `max` and the lower bound of `clamp` give the second operand when a NaN is involved
    /// (both operands variables; a constant operand counts as the second), and a NaN bound of
    /// `random` survives in either position.
    #[test]
    fn nan_selects() {
        let vars = [("n", NAN), ("f", 4.0)];
        assert_eq!(eval_vars("math.max(v.n, v.f)", &vars), 4.0);
        assert!(eval_vars("math.max(v.f, v.n)", &vars).is_nan());
        assert_eq!(eval_vars("math.min(v.n, v.f)", &vars), 4.0);
        assert!(eval_vars("math.min(v.f, v.n)", &vars).is_nan());
        assert!(eval_vars("math.clamp(v.f, v.n, 5)", &vars).is_nan());
        assert_eq!(eval_vars("math.clamp(v.n, 1, 5)", &vars), 1.0);
        assert!(eval_vars("math.random(v.n, v.f)", &vars).is_nan());
        assert!(eval_vars("math.random(v.f, v.n)", &vars).is_nan());
        assert_eq!(eval_vars("math.max(4, v.n)", &vars), 4.0);
        assert_eq!(eval_vars("math.min(4, v.n)", &vars), 4.0);
    }

    /// Signs of zero, the division guard, the inverse-trigonometry tolerance, rounding, `math.mod`,
    /// loop counts and a missing read.
    #[test]
    fn shared_behaviour() {
        assert_eq!(eval("math.sign(0)"), 1.0);
        assert_eq!(eval_vars("math.sign(v.z)", &[("z", -0.0)]), 1.0);
        assert_eq!(eval_vars("5 / v.g", &[("g", 0.000_000_1)]), 0.0);
        assert_eq!(eval_vars("5 / v.g", &[("g", 0.0)]), 0.0);
        for (version, expected) in [(13_i16, -5.0_f32), (7, -5.0), (6, 5.0)] {
            assert_eq!(
                run_at(version, "5 / v.h", &[("h", -1.0)]).0,
                expected,
                "v{version}"
            );
        }
        assert_eq!(eval_vars("math.asin(v.w)", &[("w", 1.0004)]), 90.0);
        for platform in PLATFORMS {
            assert!(eval_on(platform, "math.asin(v.w)", &[("w", 1.001)]).is_nan());
        }
        assert_eq!(eval_vars("math.round(v.m)", &[("m", -2.5)]), -3.0);
        assert_eq!(eval("math.min_angle(180)"), -180.0);
        assert_eq!(eval_vars("math.min_angle(v.d)", &[("d", 180.0)]), -180.0);
        for platform in PLATFORMS {
            assert!(eval_on(platform, "math.mod(1, 0)", &[]).is_nan());
            assert!(eval_on(platform, "math.mod(v.one, 0)", &[("one", 1.0)]).is_nan());
        }
        assert_eq!(
            eval_vars("math.mod(v.one, v.zero)", &[("one", 1.0), ("zero", 0.0)]),
            0.0
        );
        assert_eq!(
            eval("t.i = 0; loop(2.5, {t.i = t.i + 1;}); return t.i;"),
            3.0
        );
        assert_eq!(
            eval("t.i = 0; loop(-1, {t.i = t.i + 1;}); return t.i;"),
            0.0
        );
        let (value, messages) = run_at(13, "v.q = 1; v.r = v.missing; v.s = 2; return 3;", &[]);
        assert_eq!(value, 0.0);
        assert_eq!(messages.len(), 1, "{messages:?}");
        assert!(messages[0].contains("variable.missing"), "{messages:?}");
        assert_eq!(eval("7 * 3 / 9"), 7.0 * 0.333_333_34);
        // A constant NaN divisor folds to 0; a run-time one passes the guard.
        assert_eq!(eval("1 / math.sqrt(-1)"), 0.0);
        assert_eq!(eval_vars("v.o / math.sqrt(-1)", &[("o", 1.0)]), 0.0);
        let nan = [("o", 1.0), ("n", NAN)];
        assert!(eval_vars("v.o / v.n", &nan).is_nan());
    }

    #[test]
    fn sign_of_nan() {
        assert_eq!(eval_vars("math.sign(v.n)", &[("n", NAN)]), 1.0);
    }

    /// The NaN of an invalid operation on numbers.
    fn invalid(platform: Platform) -> f32 {
        f32::from_bits(servers_or_android(platform, 0xffc0_0000, 0x7fc0_0000))
    }

    const LINUX: Platform = Platform::BdsLinux;

    /// `value` on each platform, in the order of [`PLATFORMS`].
    #[track_caller]
    fn each(f: impl Fn(Platform) -> f32) -> [u32; 3] {
        PLATFORMS.map(|platform| f(platform).to_bits())
    }

    /// The NaN of an invalid operation, folded and at run time.
    #[test]
    fn the_nan_of_an_invalid_operation_is_the_platforms() {
        const POS: u32 = 0x7fc0_0000;
        const NEG: u32 = 0xffc0_0000;
        assert_eq!(PLATFORMS[0], LINUX);
        for (call, x, expected) in [
            ("math.ln", -1.0, [NEG, NEG, POS]),
            ("math.sqrt", -1.0, [NEG, NEG, POS]),
            ("math.asin", 1.001, [POS, POS, POS]),
            ("math.acos", -1.001, [POS, POS, POS]),
            ("math.sin", f32::INFINITY, [NEG, NEG, POS]),
            ("math.cos", f32::INFINITY, [NEG, NEG, POS]),
        ] {
            let run = format!("{call}(v.x)");
            assert_eq!(each(|p| eval_on(p, &run, &[("x", x)])), expected, "{run}");
        }
        for (source, expected) in [
            ("math.ln(-1)", [NEG, NEG, POS]),
            ("math.sqrt(-1)", [NEG, NEG, POS]),
            ("math.pow(-8, 1/3)", [NEG, NEG, POS]),
            ("math.mod(3, 0)", [NEG, NEG, POS]),
        ] {
            assert_eq!(each(|p| folded_on(p, source)), expected, "{source}");
        }
        // `∞ − ∞` and `0·∞` at run time.
        let vars = [("i", f32::INFINITY), ("j", f32::NEG_INFINITY), ("z", 0.0)];
        for source in ["v.i + v.j", "v.z * v.i"] {
            assert_eq!(
                each(|p| eval_on(p, source, &vars)),
                [NEG, NEG, POS],
                "{source}"
            );
        }
    }

    /// Two NaN operands: the servers keep the left one, Android a signalling one first.
    #[test]
    fn the_nan_an_operation_keeps_is_the_platforms() {
        let quiet = f32::from_bits(0x7fc0_0001);
        let signalling = f32::from_bits(0xff80_0002);
        let vars = [("q", quiet), ("s", signalling)];
        assert_eq!(
            each(|p| eval_on(p, "v.q * v.s", &vars)),
            [0x7fc0_0001, 0x7fc0_0001, 0xffc0_0002]
        );
        assert_eq!(each(|p| eval_on(p, "v.s * v.q", &vars)), [0xffc0_0002; 3]);
    }

    /// A count of 2^31 or more is no rolls on the servers and 2^31 − 1 on Android, which the step
    /// budget ends.
    #[test]
    fn a_die_count_beyond_the_int_range_is_the_platforms() {
        for platform in PLATFORMS {
            let compiled = compile("math.die_roll(v.n, 1, 2)", &on(platform, server_at(13)));
            let expr = compiled.expr().cloned().expect("an expression");
            let mut env = NoHostEnv::new();
            env.variables_mut()
                .set(VariableName::new("n"), Value::Float(3e9));
            let mut cx = env.cx();
            cx.limits = EvalLimits::NONE.with_total_steps(Some(1_000));
            assert_eq!(expr.eval_f32(&mut cx), 0.0, "{platform:?}");
            let stopped = env.sink_mut().take().len() == 1;
            assert_eq!(stopped, platform == Platform::AndroidArm64, "{platform:?}");
        }
    }

    /// On Windows ten easings add the start first, so of a NaN start and a NaN range they keep the
    /// start's NaN.
    #[test]
    fn the_windows_easings_keep_the_starts_nan() {
        let start = f32::from_bits(0x7fc0_0001);
        let end = f32::from_bits(0x7fc0_0002);
        let vars = [("s", start), ("e", end), ("t", 0.25)];
        for (function, windows_differs) in [
            ("ease_in_out_quad", true),
            ("ease_in_circ", true),
            ("ease_in_quad", false),
        ] {
            let source = format!("math.{function}(v.s, v.e, v.t)");
            let linux = eval_on(LINUX, &source, &vars);
            let windows = eval_on(Platform::BdsWindows, &source, &vars);
            assert_eq!(
                linux.to_bits() != windows.to_bits(),
                windows_differs,
                "{function}: {:08x} {:08x}",
                linux.to_bits(),
                windows.to_bits()
            );
        }
    }

    /// The operand consumed last is evaluated first: a division evaluates its divisor before its
    /// numerator, a comparison and a function's arguments left to right. The first missing read
    /// aborts, so its message names the operand evaluated first.
    #[test]
    fn operand_evaluation_order() {
        for (source, first) in [
            ("v.num / v.den", "den"),
            ("v.l < v.r", "l"),
            ("v.l >= v.r", "l"),
            ("v.l * v.r", "l"),
            ("math.max(v.l, v.r)", "l"),
            ("math.pow(v.l, v.r)", "l"),
            ("math.mod(v.l, v.r)", "l"),
            ("math.atan2(v.l, v.r)", "l"),
            ("math.random(v.l, v.r)", "l"),
            ("math.clamp(v.l, v.r, v.s)", "l"),
            ("math.lerp(v.l, v.r, v.s)", "l"),
            ("math.die_roll(v.l, v.r, v.s)", "l"),
        ] {
            let (value, messages) = run_at(13, source, &[]);
            assert_eq!(value, 0.0);
            assert_eq!(messages.len(), 1, "{source}: {messages:?}");
            assert!(
                messages[0].ends_with(&format!("'variable.{first}'")),
                "{source}: {messages:?}"
            );
        }
    }
}

mod operators {
    use crate::common::{P, PLATFORMS, on};
    use crate::common::{
        Samples,
        compile_support::{server_at, server_expr, server_expr_at},
        eval_with_samples,
    };
    use molang::compile::{CompileFailure, CompileOptions, compile};
    use molang::numeric::{self, PostOp};
    use molang::rng::FixedRng;
    use molang::stdlib::math;
    use molang::version::MolangVersion;
    use molang::vm::NoHostEnv;

    fn eval(source: &str) -> f32 {
        server_expr(source).eval_f32(&mut NoHostEnv::new().cx())
    }

    /// The platform is fixed at compile time; on every one an ordered comparison with a NaN is
    /// false.
    #[test]
    fn nan_comparisons_are_false() {
        for source in [
            "v.n < 1", "v.n <= 1", "v.n > 1", "v.n >= 1", "1 < v.n", "1 > v.n",
        ] {
            for platform in PLATFORMS {
                let options = on(platform, CompileOptions::server(MolangVersion::LATEST));
                let expr = compile(&format!("v.n = math.sqrt(-1); return {source};"), &options)
                    .expr()
                    .cloned()
                    .expect("an expression");
                assert_eq!(
                    expr.eval_f32(&mut NoHostEnv::new().cx()),
                    0.0,
                    "{source} on {platform:?}"
                );
            }
        }
    }

    /// A NaN run-time divisor does not take the guard; a constant NaN divisor folds to 0.
    #[test]
    fn a_nan_divisor_at_run_time_passes_the_guard() {
        let source = "v.o = 1; v.n = math.sqrt(-1); return v.o / v.n;";
        let expr = compile(source, &CompileOptions::server(MolangVersion::LATEST))
            .expr()
            .cloned()
            .expect("an expression");
        let value = expr.eval_f32(&mut NoHostEnv::new().cx());
        assert!(value.is_nan());
        assert_eq!(eval("v.o = 1; return v.o / math.sqrt(-1);"), 0.0);
        assert_eq!(eval("1 / math.sqrt(-1)"), 0.0);
    }

    /// `math.random` with two literal bounds is folded into the node's own post-op, whose offset
    /// `lo·S + O` is rounded twice; a draw of 0 returns that offset. With `lo` = 0.1, `S` = 3.1 and
    /// `O` = 1.1 a fused offset would differ in the last bit.
    #[test]
    fn a_folded_random_draw_of_zero_returns_the_twice_rounded_offset() {
        let expr = compile(
            "math.random(0.1, 0.5) * 3.1 + 1.1",
            &CompileOptions::server(MolangVersion::LATEST),
        )
        .expr()
        .cloned()
        .expect("an expression");
        let mut env = NoHostEnv::new();
        let mut rng = FixedRng::ZERO;
        let mut cx = env.cx();
        cx.rng = &mut rng;
        assert_eq!(0.1_f32.mul_add(3.1, 1.1).to_bits(), 0x3fb4_7ae1);
        assert_eq!((0.1_f32 * 3.1 + 1.1).to_bits(), 0x3fb4_7ae2);
        assert_eq!(expr.eval_f32(&mut cx).to_bits(), 0x3fb4_7ae2);
    }

    /// A generator's samples lie in `[0, 1]`; the random functions clamp a raw sample.
    #[test]
    fn samples_are_clamped() {
        let expr = server_expr("math.random(10, 20)");
        let mut env = NoHostEnv::new();
        for (mut rng, expected) in [
            (FixedRng::ONE, 20.0),
            (FixedRng::from_word(u32::MAX), 20.0),
            (FixedRng::ZERO, 10.0),
            (FixedRng::from_word(0x8000_0000), 10.0),
            (FixedRng::HALF, 15.0),
        ] {
            let mut cx = env.cx();
            cx.rng = &mut rng;
            assert_eq!(expr.eval_f32(&mut cx), expected, "{rng:?}");
        }
        for (sample, expected) in [(2.0, 20.0), (-1.0, 10.0), (f32::NAN, 10.0), (0.5, 15.0)] {
            assert_eq!(
                math::random(P, 10.0, 20.0, sample, PostOp::IDENTITY),
                expected,
                "{sample}"
            );
        }
    }

    /// A random operator draws after its bounds are evaluated, one draw each, `die_roll` one per
    /// roll.
    #[test]
    fn evaluation_order_fixes_the_draws() {
        // Divisor first: 0.25·4 = 1 for the divisor, then 0.5·8 = 4 for the numerator.
        assert_eq!(
            eval_with_samples("math.random(0, 8) / math.random(0, 4)", &[0.25, 0.5]),
            (4.0, 2)
        );
        // Comparison: left first.
        assert_eq!(
            eval_with_samples("math.random(0, 1) < math.random(0, 1)", &[0.2, 0.7]),
            (1.0, 2)
        );
        assert_eq!(
            eval_with_samples("math.random(0, 1) < math.random(0, 1)", &[0.7, 0.2]),
            (0.0, 2)
        );
        // Function arguments in order.
        assert_eq!(
            eval_with_samples(
                "math.max(math.random(0, 10), 0) - math.random(0, 1)",
                &[0.5, 1.0]
            ),
            (4.0, 2)
        );
        // The bound's draw comes first, then the operator's own.
        assert_eq!(
            eval_with_samples("math.random(math.random(0, 1), 10)", &[0.5, 0.0]),
            (0.5, 2)
        );
        assert_eq!(
            eval_with_samples("math.die_roll(3, 0, 10)", &[0.1, 0.2, 0.3]).1,
            3
        );
        assert_eq!(
            eval_with_samples("math.die_roll_integer(2, 1, 6)", &[0.0, 1.0]),
            (7.0, 2)
        );
    }

    /// A run-time divisor below `f32::EPSILON` (2^-23) ends the division with 0: the numerator is
    /// never evaluated and the post-op not applied.
    #[test]
    fn division_guard_skips_the_numerator_and_the_post_op() {
        let run = |source: &str| {
            let expr = server_expr_at(source, 13);
            let mut env = NoHostEnv::new();
            let mut rng = Samples::repeat(0.5);
            let value = {
                let mut cx = env.cx();
                cx.rng = &mut rng;
                expr.eval_f32(&mut cx)
            };
            (value, rng.draws)
        };
        assert_eq!(
            run("v.g = 0.0000001; return math.random(0, 8) / v.g;"),
            (0.0, 0)
        );
        assert_eq!(run("v.g = 0.0000001; return (5 / v.g) * 2 + 3;"), (0.0, 0));
        assert_eq!(run("v.g = -0.0000001; return 5 / v.g + 3;"), (0.0, 0));
        assert_eq!(
            run("v.g = 0.25; return math.random(0, 8) / v.g;"),
            (16.0, 1)
        );
        // Exactly `f32::EPSILON` (2^-23, folded from `1 / 8388608`) is not guarded.
        assert_eq!(run("v.g = 1 / 8388608; return 1 / v.g;"), (8_388_608.0, 0));
        // A NaN divisor passes the guard.
        let nan = compile(
            "v.n = math.sqrt(-1); return (1 / v.n) * 2 + 3;",
            &server_at(13),
        )
        .expr()
        .cloned()
        .expect("compiles");
        let value = nan.eval_f32(&mut NoHostEnv::new().cx());
        assert!(value.is_nan());
    }

    /// The crate has no array resolver, so an expression with an array stops at the link; the index
    /// rule `numeric::array_index` is `max(0, trunc(i))` modulo the length, with no element for an
    /// empty array.
    #[test]
    fn array_index_rule() {
        assert_eq!(numeric::array_index(P, 1.9, 3), Some(1));
        assert_eq!(numeric::array_index(P, 4.0, 3), Some(1));
        assert_eq!(numeric::array_index(P, -2.5, 3), Some(0));
        assert_eq!(numeric::array_index(P, f32::NAN, 3), Some(0));
        assert_eq!(numeric::array_index(P, 0.0, 0), None);
        assert_eq!(
            compile("array.a[v.i]", &server_at(13)).failure(),
            Some(CompileFailure::UsesArrays)
        );
    }
}

mod division_per_version {
    use crate::common::compile_support::server_expr_at;
    use molang::vm::{NoHostEnv, Value, VariableName};

    fn divide(version: i16, a: f32, d: f32) -> f32 {
        let expr = server_expr_at("v.a / v.d", version);
        let mut env = NoHostEnv::new();
        env.variables_mut()
            .set(VariableName::new("a"), Value::Float(a));
        env.variables_mut()
            .set(VariableName::new("d"), Value::Float(d));
        expr.eval_f32(&mut env.cx())
    }

    #[test]
    fn a_positive_divisor_divides_the_same_at_every_version() {
        for version in [-1, 0, 6, 7, 13] {
            assert_eq!(divide(version, 6.0, 2.0), 3.0, "v{version} ");
            assert_eq!(divide(version, 1.0, 4.0), 0.25, "v{version} ");
        }
    }

    #[test]
    fn a_negative_divisor_is_divided_by_in_absolute_value_up_to_version_6() {
        assert_eq!(divide(6, 6.0, -2.0), 3.0);
        assert_eq!(divide(0, 6.0, -2.0), 3.0);
        assert_eq!(divide(7, 6.0, -2.0), -3.0);
        assert_eq!(divide(13, 6.0, -2.0), -3.0);
    }

    #[test]
    fn a_tiny_divisor_ends_the_division_with_zero_at_every_version() {
        for version in [0, 6, 7, 13] {
            assert_eq!(divide(version, 5.0, 0.0), 0.0, "v{version} ");
            assert_eq!(divide(version, 5.0, 1.0e-8), 0.0, "v{version} ");
            assert_eq!(divide(version, 5.0, -1.0e-8), 0.0, "v{version} ");
        }
    }
}

mod fixed_generator {
    use crate::common::compile_support::server_expr;
    use molang::rng::FixedRng;
    use molang::vm::NoHostEnv;

    fn draw(source: &str, sample: f32) -> f32 {
        let expr = server_expr(source);
        let mut env = NoHostEnv::new();
        let mut rng = FixedRng::from_sample(sample).expect("a word's sample");
        let mut cx = env.cx();
        cx.rng = &mut rng;
        expr.eval_f32(&mut cx)
    }

    #[test]
    fn random_spans_its_closed_interval() {
        assert_eq!(draw("math.random(10, 20)", 0.0), 10.0);
        assert_eq!(draw("math.random(10, 20)", 0.5), 15.0);
        assert_eq!(draw("math.random(10, 20)", 1.0), 20.0);
        // Inverted bounds are sorted.
        assert_eq!(draw("math.random(20, 10)", 0.5), 15.0);
    }

    #[test]
    fn random_integer_stays_within_its_bounds() {
        for sample in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let value = draw("math.random_integer(1, 6)", sample);
            assert_eq!(value.fract(), 0.0, "{sample}");
            assert!((1.0..=6.0).contains(&value), "{sample}: {value}");
        }
        assert_eq!(draw("math.random_integer(1, 6)", 0.0), 1.0);
        assert_eq!(draw("math.random_integer(1, 6)", 1.0), 6.0);
    }

    #[test]
    fn a_die_roll_sums_its_rolls() {
        assert_eq!(draw("math.die_roll(4, 10, 20)", 0.0), 40.0);
        assert_eq!(draw("math.die_roll(4, 10, 20)", 1.0), 80.0);
        assert_eq!(draw("math.die_roll(4, 10, 20)", 0.5), 60.0);
        assert_eq!(draw("math.die_roll(0, 10, 20)", 0.5), 0.0);
        assert_eq!(draw("math.die_roll_integer(3, 1, 6)", 0.0), 3.0);
        assert_eq!(draw("math.die_roll_integer(3, 1, 6)", 1.0), 18.0);
    }
}

mod generator {
    use molang::rng::{FixedRng, RandomSource, unit_31};

    use crate::common::rng::Xorshift128;

    #[test]
    fn first_outputs_from_the_standard_seeds() {
        // t = x ^ (x << 11); x = y; y = z; z = w; w = w ^ (w >> 19) ^ t ^ (t >> 8)
        let mut raw = Xorshift128::new();
        let words: Vec<u32> = (0..6).map(|_| raw.next_u32()).collect();
        assert_eq!(
            words,
            [
                3_701_687_786,
                458_299_110,
                2_500_872_618,
                3_633_119_408,
                516_391_518,
                2_377_269_574
            ]
        );

        // sample = (w & 0x7fffffff) · 2^-31 as f32
        let mut rng = Xorshift128::new();
        let samples: Vec<u32> = (0..6).map(|_| rng.sample().to_bits()).collect();
        assert_eq!(
            samples,
            [
                0x3f39_468c,
                0x3e5a_88b7,
                0x3e28_824d,
                0x3f31_1a01,
                0x3e76_3c13,
                0x3ddb_2414
            ]
        );
    }

    #[test]
    fn samples_stay_in_the_closed_unit_interval() {
        let mut rng = Xorshift128::new();
        for _ in 0..100_000 {
            let sample = rng.sample();
            assert!((0.0..=1.0).contains(&sample), "{sample}");
        }
    }

    #[test]
    fn a_sample_can_round_to_exactly_one() {
        // The low 31 bits of 0xffffffff round up to 2^31 when converted to f32.
        assert_eq!(unit_31(0xffff_ffff), 1.0);
    }

    #[test]
    fn fixed_sources() {
        for (mut source, expected) in [
            (FixedRng::ZERO, 0.0),
            (FixedRng::HALF, 0.5),
            (FixedRng::ONE, 1.0),
        ] {
            assert_eq!(source.sample(), expected);
        }
        let mut fixed = FixedRng::from_sample(0.25).expect("a word's sample");
        assert_eq!((fixed.sample(), fixed.sample()), (0.25, 0.25));
        assert_eq!(FixedRng::from_sample(f32::NAN), None);
    }

    #[test]
    fn a_closure_or_a_dynamic_reference_is_a_source() {
        fn draw_two(mut rng: impl RandomSource) -> (f32, f32) {
            (rng.sample(), rng.sample())
        }
        let mut rng = Xorshift128::new();
        let first = draw_two(|| rng.sample());
        let dynamic: &mut dyn RandomSource = &mut rng;
        let second = draw_two(dynamic);
        let mut reference = Xorshift128::new();
        assert_eq!(first, (reference.sample(), reference.sample()));
        assert_eq!(second, (reference.sample(), reference.sample()));
    }
}

mod math_library {
    //! The functions of `molang::stdlib::math` called directly, on the default platform.

    // Expected values are written with nine significant digits.
    #![allow(clippy::excessive_precision)]
    use molang::numeric::arith;
    use molang::stdlib::math;

    use molang::numeric::{Platform, PostOp};

    use crate::common::{P, assert_bits};
    use molang::rng::RandomSource;

    /// `|actual − expected| <= tolerance`; non-finite expectations must match to the bit.
    fn within(actual: f32, expected: f32, tolerance: f32) -> bool {
        if !expected.is_finite() {
            actual.to_bits() == expected.to_bits()
        } else {
            (actual - expected).abs() <= tolerance
        }
    }

    const ID: PostOp = PostOp::IDENTITY;

    const NAN: f32 = f32::NAN;

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            within(actual, expected, tolerance),
            "{actual:e} is not {expected:e} ± {tolerance:e}"
        );
    }

    struct Scripted {
        samples: Vec<f32>,
        draws: usize,
    }

    /// Plays back `samples` in a loop and counts its draws.
    impl RandomSource for Scripted {
        fn sample(&mut self) -> f32 {
            let sample = self.samples[self.draws % self.samples.len()];
            self.draws += 1;
            sample
        }
    }

    #[test]
    fn constants() {
        assert_eq!(math::PI, 3.141_592_74);
        assert_eq!(math::DEG_TO_RAD.to_bits(), 0x3c8e_fa35);
        assert_eq!(math::RAD_TO_DEG.to_bits(), 0x4265_2ee0);
        assert_eq!(math::INVERSE_TRIG_TOLERANCE.to_bits(), 0x3f80_1062);
        assert_eq!(math::pi(P, ID), 3.141_592_74);
        assert_eq!(math::pi(P, PostOp::new(2.0, 0.0)), 6.283_185_5);
    }

    #[test]
    fn inverse_trigonometry_in_degrees_with_the_tolerance_window() {
        close(math::acos(P, 1.0005, ID), 0.0, 5e-4);
        close(math::acos(P, -1.0005, ID), 180.0, 5e-4);
        close(math::acos(P, -1.0001, ID), 180.0, 5e-4);
        assert!(math::acos(P, 1.0006, ID).is_nan());
        assert!(math::acos(P, -1.0006, ID).is_nan());
        close(math::asin(P, 1.0005, ID), 90.0, 5e-4);
        close(math::asin(P, -1.0, ID), -90.0, 5e-4);
        assert!(math::asin(P, 1.0006, ID).is_nan());
        close(math::atan(P, 1.0, ID), 45.0, 5e-4);
        close(math::atan2(P, -7.0, -7.0, ID), -135.0, 5e-4);
        assert_eq!(math::atan2(P, 0.0, 0.0, ID), 0.0);
        // The post-op scale is merged into the degree conversion: rad·(57.29578·S) + O.
        close(math::asin(P, 1.0, PostOp::new(2.0, 1.0)), 181.0, 5e-4);
        close(
            math::atan2(P, 1.0, 1.0, PostOp::new(-2.0, 0.5)),
            -89.5,
            5e-4,
        );
        // A NaN argument gives NaN.
        assert!(math::acos(P, NAN, ID).is_nan());
        assert!(math::asin(P, NAN, PostOp::new(3.0, 0.5)).is_nan());
        // The post-op: ((rad·57.29578)·S) + O, each step rounded.
        let (x, post) = (
            f32::from_bits(0xbe85_e4b5),
            PostOp::new(f32::from_bits(0xbfa0_820c), f32::from_bits(0x3ffb_1d93)),
        );
        let rad = crate::common::library::atan(x);
        let grouped = (rad * math::RAD_TO_DEG) * post.scale + post.offset;
        let merged = rad * (math::RAD_TO_DEG * post.scale) + post.offset;
        assert_eq!(math::atan(P, x, post), grouped);
        assert_bits(
            grouped,
            crate::common::library::pinned(0x41a2_b669, 0x41a2_b669, grouped),
            "((rad·57.29578)·S) + O",
        );
        assert_bits(
            merged,
            crate::common::library::pinned(0x41a2_b66a, 0x41a2_b66a, merged),
            "rad·(57.29578·S) + O",
        );
    }

    #[test]
    fn trigonometry_takes_degrees() {
        close(math::sin(P, 90.0, ID), 1.0, 1e-6);
        close(math::cos(P, 180.0, ID), -1.0, 1e-6);
        close(math::sin(P, 180.0, ID), -8.742_278e-8, 1e-12);
        assert_eq!(math::sin(P, 0.0, ID), 0.0);
        assert_eq!(math::cos(P, 0.0, ID), 1.0);
        close(math::cos(P, 0.0, PostOp::new(-2.0, 1.1)), -0.9, 1e-6);
    }

    #[test]
    fn rounding_functions() {
        assert_eq!(math::ceil(P, -0.5, ID).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(math::ceil(P, 1.1, ID), 2.0);
        assert_eq!(math::floor(P, -0.5, ID), -1.0);
        assert_eq!(math::floor(P, 1.9, ID), 1.0);
        assert_eq!(math::round(P, 0.5, ID), 1.0);
        assert_eq!(math::round(P, -0.5, ID), -1.0);
        assert_eq!(math::round(P, -1.5, ID), -2.0);
        assert_eq!(math::round(P, -2.5, ID), -3.0);
        assert_eq!(math::round(P, 2.4, ID), 2.0);
        assert_eq!(math::trunc(P, 1.7, ID), 1.0);
        assert_eq!(math::trunc(P, -1.7, ID), -1.0);
        assert_eq!(math::ceil(P, 1.1, PostOp::new(-2.0, -1.0)), -5.0);
    }

    #[test]
    fn simple_functions() {
        assert_eq!(math::abs(P, -2.0, ID), 2.0);
        assert_eq!(math::abs(P, -0.0, ID).to_bits(), 0.0_f32.to_bits());
        assert_eq!(math::clamp(P, 3.0, 2.0, 1.0, ID), 1.0);
        assert_eq!(math::clamp(P, -1.0, -2.0, -3.0, ID), -3.0);
        assert_eq!(math::clamp(P, 1.0, 2.0, 3.0, ID), 2.0);
        assert_eq!(
            math::clamp(P, 2.1, 0.0, 1.1, PostOp::new(2.0, 1.0)),
            arith::mul_add(P, 1.1, 2.0, 1.0)
        );
        assert_eq!(math::copy_sign(P, -1.1, 3.1, ID), 1.1);
        assert_eq!(math::copy_sign(P, 2.0, -0.0, ID), -2.0);
        assert_eq!(
            math::copy_sign(P, 0.0, -2.0, ID).to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(math::exp(P, 0.0, ID), 1.0);
        assert_eq!(math::ln(P, 1.0, ID), 0.0);
        assert_eq!(math::ln(P, 0.0, ID), f32::NEG_INFINITY);
        assert!(math::sqrt(P, -1.0, ID).is_nan());
        assert_eq!(math::sqrt(P, 16.0, ID), 4.0);
        assert_eq!(math::pow(P, 2.0, 10.0, ID), 1024.0);
        assert_eq!(math::hermite_blend(P, 0.5, ID), 0.5);
        assert_eq!(math::hermite_blend(P, 2.0, ID), -4.0);
        assert_eq!(math::hermite_blend(P, 0.0, ID), 0.0);
        assert_eq!(math::hermite_blend(P, 1.0, ID), 1.0);
        assert_eq!(math::max(P, 0.0, 1.0, ID), 1.0);
        assert_eq!(math::min(P, 0.0, 1.0, ID), 0.0);
        assert_eq!(math::max(P, 1.0, 2.0, PostOp::new(3.0, 1.0)), 7.0);
    }

    #[test]
    fn hermite_blend_is_term_by_term() {
        let t = f32::from_bits(0x3fde_6363);
        assert_eq!(
            math::hermite_blend(P, t, ID),
            (3.0 * t) * t - ((t + t) * t) * t
        );
        assert_eq!(math::hermite_blend(P, t, ID).to_bits(), 0xbfb7_7588);
    }

    #[test]
    fn interpolation_and_angles() {
        assert_eq!(math::lerp(P, 0.0, 10.0, 0.5, ID), 5.0);
        assert_eq!(math::lerp(P, 0.0, 10.0, 2.0, ID), 20.0);
        assert_eq!(math::lerp(P, 0.0, 10.0, 0.5, PostOp::new(2.0, 1.0)), 11.0);
        assert_eq!(math::lerprotate(P, 350.0, 10.0, 0.5, ID), 360.0);
        assert_eq!(math::lerprotate(P, 10.0, 350.0, 0.5, ID), 0.0);
        assert_eq!(math::inverse_lerp(P, 1.0, 5.0, 3.0, ID), 0.5);
        assert_eq!(math::inverse_lerp(P, 1.0, 5.0, 9.0, ID), 2.0);
        assert_eq!(math::inverse_lerp(P, 5.0, 5.0, 4.0, ID), f32::NEG_INFINITY);
        assert!(math::inverse_lerp(P, 5.0, 5.0, 5.0, ID).is_nan());
        for (x, wrapped) in [
            (180.0, -180.0),
            (-180.0, -180.0),
            (540.0, -180.0),
            (370.0, 10.0),
            (-370.0, -10.0),
            (0.0, 0.0),
            (179.0, 179.0),
        ] {
            assert_eq!(math::min_angle(P, x, ID), wrapped, "min_angle({x})");
        }
    }

    #[test]
    fn easing_endpoints_and_special_cases() {
        // Expo has no endpoint special cases.
        assert_eq!(math::ease_in_expo(P, 1.0, 5.0, 0.0, ID), 1.003_906_25);
        assert_eq!(math::ease_out_expo(P, 1.0, 5.0, 1.0, ID), 4.996_093_75);
        assert_eq!(math::ease_in_out_expo(P, 1.0, 5.0, 0.0, ID), 1.001_953_12);
        // Elastic special-cases t == 0 and t == 1 (2t == 2 for in-out).
        for ease in [
            math::ease_in_elastic,
            math::ease_out_elastic,
            math::ease_in_out_elastic,
        ] {
            assert_eq!(ease(P, 1.0, 5.0, 0.0, ID), 1.0);
            assert_eq!(ease(P, 1.0, 5.0, 1.0, ID), 5.0);
            assert_eq!(ease(P, 1.0, 5.0, -0.0, ID), 1.0);
        }
        close(
            math::ease_in_out_elastic(P, 1.0, 5.0, 0.3, ID),
            0.937_503_457,
            1e-6,
        );
        // t is not clamped.
        assert_eq!(math::ease_in_quad(P, 0.0, 1.0, 2.0, ID), 4.0);
        assert_eq!(math::ease_in_cubic(P, 0.0, 1.0, -2.0, ID), -8.0);
        assert_eq!(math::ease_out_quad(P, 0.0, 1.0, 2.0, ID), 0.0);
        // The post-op applies to the eased value.
        assert_eq!(
            math::ease_in_quad(P, 0.0, 1.0, 2.0, PostOp::new(2.0, 1.0)),
            9.0
        );
        assert_eq!(
            math::ease_in_elastic(P, 1.0, 5.0, 1.0, PostOp::new(2.0, 1.0)),
            11.0
        );
    }

    #[test]
    fn the_easing_formulas_round_left_to_right() {
        // c·t·t·t·t rounded left to right.
        assert_eq!(
            math::ease_in_quart(P, 0.0, 1.0, 0.1, ID).to_bits(),
            0x38d1_b718
        );
        assert_eq!(
            math::ease_in_quart(P, 0.0, 1.0, 0.1, ID),
            (((1.0_f32 * 0.1) * 0.1) * 0.1) * 0.1
        );
        // The elastic angle is ((x − 0.075)·2π)/0.3.
        assert_eq!(
            math::ease_in_elastic(P, 1.0, 5.0, 0.9, ID).to_bits(),
            0x386a_4000
        );
        // The elastic end point is start + (end − start), not `end`.
        let (start, end) = (0.3_f32, 0.1_f32);
        assert_ne!(start + (end - start), end);
        for ease in [
            math::ease_in_elastic,
            math::ease_out_elastic,
            math::ease_in_out_elastic,
        ] {
            assert_eq!(ease(P, start, end, 1.0, ID), start + (end - start));
        }
        // Infinite and huge operands: the left-to-right form keeps what the reassociated form
        // loses.
        assert_eq!(math::ease_in_quad(P, -0.0, 0.0, -1.0e38, ID), 0.0);
        assert_eq!(
            math::ease_in_out_quad(P, 1.0, 5.0, f32::INFINITY, ID),
            f32::NEG_INFINITY
        );
    }

    /// In the second half of `in_out_bounce` the bounce product is zero-normalised before the
    /// halves are added. With `start = -0` and a `d` so small that `d * 0.5` underflows to -0, the
    /// sign of the result is the sign of that `+0`.
    #[test]
    fn in_out_bounce_second_half_with_a_negative_zero_start() {
        let tiny = f32::from_bits(1); // 1.4e-45
        assert_eq!(
            math::ease_in_out_bounce(P, -0.0, -tiny, 0.5, ID).to_bits(),
            0.0_f32.to_bits()
        );
    }

    type Ease = fn(Platform, f32, f32, f32, PostOp) -> f32;

    #[test]
    fn every_easing_runs_from_start_to_end() {
        let exact: [(&str, Ease); 21] = [
            ("in_quad", math::ease_in_quad),
            ("out_quad", math::ease_out_quad),
            ("in_out_quad", math::ease_in_out_quad),
            ("in_cubic", math::ease_in_cubic),
            ("out_cubic", math::ease_out_cubic),
            ("in_out_cubic", math::ease_in_out_cubic),
            ("in_quart", math::ease_in_quart),
            ("out_quart", math::ease_out_quart),
            ("in_out_quart", math::ease_in_out_quart),
            ("in_quint", math::ease_in_quint),
            ("out_quint", math::ease_out_quint),
            ("in_out_quint", math::ease_in_out_quint),
            ("in_circ", math::ease_in_circ),
            ("out_circ", math::ease_out_circ),
            ("in_out_circ", math::ease_in_out_circ),
            ("in_bounce", math::ease_in_bounce),
            ("out_bounce", math::ease_out_bounce),
            ("in_out_bounce", math::ease_in_out_bounce),
            ("in_elastic", math::ease_in_elastic),
            ("out_elastic", math::ease_out_elastic),
            ("in_out_elastic", math::ease_in_out_elastic),
        ];
        // Table quantisation, the 2^-10 residue of expo and the rounded back constants keep these
        // slightly off their endpoints.
        let approximate: [(&str, Ease); 9] = [
            ("in_sine", math::ease_in_sine),
            ("out_sine", math::ease_out_sine),
            ("in_out_sine", math::ease_in_out_sine),
            ("in_expo", math::ease_in_expo),
            ("out_expo", math::ease_out_expo),
            ("in_out_expo", math::ease_in_out_expo),
            ("in_back", math::ease_in_back),
            ("out_back", math::ease_out_back),
            ("in_out_back", math::ease_in_out_back),
        ];
        for (name, ease) in exact {
            assert!(within(ease(P, 1.0, 5.0, 0.0, ID), 1.0, 1e-6), "{name} at 0");
            assert!(within(ease(P, 1.0, 5.0, 1.0, ID), 5.0, 1e-6), "{name} at 1");
            assert!(ease(P, 1.0, 5.0, NAN, ID).is_nan(), "{name} of NaN");
        }
        for (name, ease) in approximate {
            assert!(within(ease(P, 1.0, 5.0, 0.0, ID), 1.0, 5e-3), "{name} at 0");
            assert!(within(ease(P, 1.0, 5.0, 1.0, ID), 5.0, 5e-3), "{name} at 1");
        }
        // Every in-out curve passes through the midpoint.
        let in_out: [(&str, Ease); 9] = [
            ("quad", math::ease_in_out_quad),
            ("cubic", math::ease_in_out_cubic),
            ("quart", math::ease_in_out_quart),
            ("quint", math::ease_in_out_quint),
            ("sine", math::ease_in_out_sine),
            ("expo", math::ease_in_out_expo),
            ("circ", math::ease_in_out_circ),
            ("bounce", math::ease_in_out_bounce),
            ("back", math::ease_in_out_back),
        ];
        for (name, ease) in in_out {
            assert!(
                within(ease(P, 1.0, 5.0, 0.5, ID), 3.0, 1e-3),
                "in_out_{name} at 0.5"
            );
        }
    }

    /// Entry `i` of the table is `sin(i / 10430.378)`.
    #[test]
    fn sine_easings_are_quantised_by_the_sine_table() {
        let scale = f32::from_bits(0x4622_f983);
        let half_pi = f32::from_bits(0x3fc9_0fdb);
        let sin = crate::common::library::sin;
        let mut differs_from_plain_sine = false;
        for step in 1..100_u8 {
            let t = f32::from(step) / 100.0;
            let radians = t * half_pi;
            // ease_out_sine(0, 1, t) is the table sine itself: the entry trunc(x·10430.378)
            // mod 65536.
            let index = ((radians * scale) as i32 & 0xffff) as f32;
            assert_eq!(math::ease_out_sine(P, 0.0, 1.0, t, ID), sin(index / scale));
            let eased = math::ease_out_sine(P, 0.0, 1.0, t, ID);
            close(eased, sin(radians), 1e-4);
            differs_from_plain_sine |= eased != sin(radians);
        }
        assert!(
            differs_from_plain_sine,
            "the table sine should not equal the plain sine everywhere"
        );
    }

    #[test]
    fn random_samples_are_clamped() {
        assert_eq!(math::clamp_sample(Platform::default(), 0.25), 0.25);
        assert_eq!(math::clamp_sample(Platform::default(), 2.0), 1.0);
        assert_eq!(math::clamp_sample(Platform::default(), -1.0), 0.0);
        assert_eq!(
            math::clamp_sample(Platform::default(), -0.0).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(math::clamp_sample(Platform::default(), NAN), 0.0);
        assert_eq!(math::clamp_sample(Platform::default(), 1.0), 1.0);
        assert_eq!(math::random(P, 2.0, 7.0, 9.0, ID), 7.0);
        assert_eq!(math::random(P, 2.0, 7.0, -9.0, ID), 2.0);
    }

    #[test]
    fn random_and_random_integer() {
        for (sample, real, integer) in [(0.0, 2.0, 2.0), (0.5, 4.5, 4.0), (1.0, 7.0, 7.0)] {
            assert_eq!(math::random(P, 2.0, 7.0, sample, ID), real);
            assert_eq!(
                math::random(P, 7.0, 2.0, sample, ID),
                real,
                "inverted bounds"
            );
            assert_eq!(math::random_integer(P, 2.0, 7.0, sample, ID), integer);
            assert_eq!(
                math::random_integer(P, 7.0, 2.0, sample, ID),
                integer,
                "inverted bounds"
            );
            assert_eq!(
                math::random_integer_const_bounds(P, 2.0, 7.0, sample, ID),
                integer
            );
            assert_eq!(
                math::random(P, 2.0, 7.0, sample, PostOp::new(2.0, 1.0)),
                real * 2.0 + 1.0
            );
            // Literal bounds fold into the post-op: S' = (hi − lo)·S, O' = lo·S + O.
            let folded = math::random_const_bounds(P, 7.0, 2.0, PostOp::new(2.0, 1.0));
            assert_eq!(folded, PostOp::new(10.0, 5.0));
            assert_eq!(math::random_folded(P, sample, folded), real * 2.0 + 1.0);
            assert_eq!(
                math::random_folded(P, sample, math::random_const_bounds(P, 2.0, 7.0, ID)),
                real
            );
        }
        // The bounds are not rounded: the sample floors below the lower bound, which is returned.
        assert_eq!(math::random_integer(P, 0.1, 1_000_001.0, 0.0, ID), 0.1);
        assert_eq!(
            math::random_integer_const_bounds(P, 0.1, 1_000_001.0, 0.0, ID),
            0.1
        );
        assert_eq!(
            math::random_integer(P, 0.0, 3.0, 1.0, PostOp::new(2.0, 1.0)),
            7.0
        );
    }

    #[test]
    fn die_rolls_draw_once_per_roll() {
        const NO_LIMIT: u32 = u32::MAX;
        for (sample, real, integer) in [(0.0, 3.0, 3.0), (0.5, 10.5, 9.0), (1.0, 18.0, 18.0)] {
            let mut rng = Scripted {
                samples: vec![sample],
                draws: 0,
            };
            assert_eq!(
                math::die_roll(P, 3.0, 1.0, 6.0, NO_LIMIT, &mut rng, ID),
                Some(real)
            );
            assert_eq!(rng.draws, 3);
            assert_eq!(
                math::die_roll(P, 3.0, 6.0, 1.0, NO_LIMIT, &mut rng, ID),
                Some(real),
                "inverted bounds"
            );
            assert_eq!(rng.draws, 6);
            assert_eq!(
                math::die_roll_integer(P, 3.0, 1.0, 6.0, NO_LIMIT, &mut rng, ID),
                Some(integer)
            );
            assert_eq!(rng.draws, 9);
        }
        // The count is truncated and the bounds are floored.
        let mut rng = Scripted {
            samples: vec![1.0],
            draws: 0,
        };
        assert_eq!(
            math::die_roll(P, 2.9, 1.9, 6.9, NO_LIMIT, &mut rng, ID),
            Some(12.0)
        );
        assert_eq!(rng.draws, 2);
        // Fewer than one roll: no draw, result 0 (then the post-op).
        for n in [0.9, 0.0, -1.0, NAN] {
            assert_eq!(
                math::die_roll(P, n, 1.0, 6.0, NO_LIMIT, &mut rng, ID),
                Some(0.0)
            );
            assert_eq!(
                math::die_roll_integer(P, n, 1.0, 6.0, 0, &mut rng, PostOp::new(2.0, 1.5)),
                Some(1.5)
            );
        }
        assert_eq!(rng.draws, 2);
        let mut rng = Scripted {
            samples: vec![0.0, 1.0],
            draws: 0,
        };
        assert_eq!(
            math::die_roll(P, 2.0, 1.0, 6.0, NO_LIMIT, &mut rng, ID),
            Some(7.0)
        );
        assert_eq!(
            math::die_roll_integer(P, 2.0, 1.0, 6.0, NO_LIMIT, &mut rng, PostOp::new(2.0, 1.0)),
            Some(15.0)
        );
        assert_eq!(math::die_roll_count(P, 3.9), 3);
        assert_eq!(math::die_roll_count(P, -3.9), 0);
        assert_eq!(math::die_roll_count(P, NAN), 0);
        assert_eq!(math::die_roll_count(P, 1.0e10), 0);
    }

    #[test]
    fn die_rolls_are_bounded_by_the_caller() {
        // More rolls than the limit: nothing is drawn and the caller decides what happens.
        let mut rng = Scripted {
            samples: vec![0.5],
            draws: 0,
        };
        assert_eq!(math::die_roll(P, 5.0, 1.0, 6.0, 4, &mut rng, ID), None);
        assert_eq!(
            math::die_roll_integer(P, 2_000_000_000.0, 1.0, 6.0, 1_024, &mut rng, ID),
            None
        );
        assert_eq!(rng.draws, 0);
        assert_eq!(
            math::die_roll(P, 5.0, 1.0, 6.0, 5, &mut rng, ID),
            Some(17.5)
        );
        assert_eq!(rng.draws, 5);

        // Rolling step by step gives the same bits as the one-call form, for both operators.
        let samples = vec![0.1, 0.9, 0.333_333_34, 0.5, 1.0, 0.0, 0.777];
        let post = PostOp::new(1.5, -0.25);
        for integer in [false, true] {
            let mut rng = Scripted {
                samples: samples.clone(),
                draws: 0,
            };
            let whole = if integer {
                math::die_roll_integer(P, 7.0, -2.5, 9.25, u32::MAX, &mut rng, post)
            } else {
                math::die_roll(P, 7.0, -2.5, 9.25, u32::MAX, &mut rng, post)
            };
            let mut roll = if integer {
                math::DieRoll::new_integer(P, 7.0, -2.5, 9.25)
            } else {
                math::DieRoll::new(P, 7.0, -2.5, 9.25)
            };
            assert_eq!(roll.remaining(), 7);
            for (made, sample) in samples.iter().enumerate() {
                roll.roll(*sample);
                assert_eq!(roll.remaining() as usize, 6 - made);
            }
            let before = roll;
            roll.roll(0.5);
            assert_eq!(roll, before);
            assert_eq!(Some(roll.finish(post).to_bits()), whole.map(f32::to_bits));
        }
        // A caller may stop early: the sum of the rolls made so far.
        let mut roll = math::DieRoll::new(P, 100.0, 1.0, 6.0);
        roll.roll(0.0);
        roll.roll(1.0);
        assert_eq!(roll.remaining(), 98);
        assert_eq!(roll.finish(ID), 7.0);
        // No rolls: 0, then the post-op.
        assert_eq!(math::DieRoll::new(P, 0.5, 1.0, 6.0).remaining(), 0);
        assert_eq!(
            math::DieRoll::new_integer(P, NAN, 1.0, 6.0).finish(PostOp::new(2.0, 1.5)),
            1.5
        );
    }
}

mod primitives {
    //! The scalar primitives called directly, on the default platform unless a test names others.

    // Expected values are written with nine significant digits.
    #![allow(clippy::excessive_precision)]
    use crate::common::{P, PLATFORMS, servers_or_android};
    use molang::numeric::{self, PostOp, arith};
    use molang::stdlib::math;

    const ID: PostOp = PostOp::IDENTITY;

    const NAN: f32 = f32::NAN;

    /// The literal `1 / 3` as it folds.
    fn third() -> f32 {
        numeric::fold_const_div(P, 1.0, 3.0)
    }

    #[test]
    fn identity_post_op_returns_the_raw_value() {
        assert_eq!(ID.apply(P, -0.0).to_bits(), (-0.0_f32).to_bits());
        // A real post-op of (1, 0) would turn -0 into +0; the plain instruction form does not.
        assert_eq!(math::ceil(P, -0.5, ID).to_bits(), (-0.0_f32).to_bits());
        assert!(ID.is_identity());
        assert!(PostOp::new(1.0, -0.0).is_identity());
        assert!(!PostOp::new(1.0, 1.0).is_identity());
        assert_eq!(PostOp::default(), ID);
    }

    #[test]
    fn negate_and_add_apply_the_post_op() {
        assert_eq!(numeric::negate(P, 2.5, ID), -2.5);
        assert_eq!(numeric::negate(P, 0.0, ID).to_bits(), (-0.0_f32).to_bits());
        // O − x·S
        assert_eq!(numeric::negate(P, 2.5, PostOp::new(2.0, 1.0)), -4.0);
        assert_eq!(numeric::add(P, 0.1, 0.2, ID), 0.300_000_01);
        assert_eq!(numeric::add(P, 1.5, 2.5, PostOp::new(2.0, 1.0)), 9.0);
        let x = third();
        assert_eq!(numeric::negate(P, x, PostOp::new(3.0, 1.0)), 1.0 - x * 3.0);
        assert_ne!(1.0 - x * 3.0, (-x).mul_add(3.0, 1.0));
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
        // The child's offset is scaled by the constant first: ((c·child.O)·S) + O, not
        // (S·c)·child.O + O. They differ once the node's own scale is not 1.
        let (parent, c, child) = (PostOp::new(0.1, 0.25), 0.3_f32, PostOp::new(2.0, 0.7));
        assert_eq!(
            parent.fold_scaled(P, c, child).offset,
            (c * child.offset) * parent.scale + parent.offset
        );
        assert_ne!(
            (c * child.offset) * parent.scale,
            (parent.scale * c) * child.offset
        );
        assert_eq!(
            parent.fold_scaled(P, c, child).scale,
            (parent.scale * c) * child.scale
        );
        // Each step of the offset is rounded.
        let (parent, child) = (PostOp::new(3.0, -1.0), PostOp::new(1.0, third()));
        assert_eq!(parent.fold_scaled(P, 1.0, child).offset, 0.0);
        // The constant multiplies the child's offset before the node's scale: with S = 0.1, O =
        // 0.1, c = 0.3 and O' = 0.7 that is 0.12100001, where the other grouping gives 0.121.
        let (parent, child) = (PostOp::new(0.1, 0.1), PostOp::new(1.0, 0.7));
        assert_eq!(
            parent.fold_scaled(P, 0.3, child).offset.to_bits(),
            0x3df7_ceda
        );
    }

    #[test]
    fn nan_comparisons_are_false() {
        for (a, b) in [(NAN, 1.0), (1.0, NAN), (NAN, NAN)] {
            assert!(!numeric::lt(a, b) && !numeric::le(a, b));
            assert!(!numeric::gt(a, b) && !numeric::ge(a, b));
            assert!(!numeric::eq(a, b) && numeric::ne(a, b));
        }
    }

    #[test]
    fn ordinary_comparisons_are_the_float_ones() {
        let samples = [
            -2.0_f32,
            -0.0,
            0.0,
            1.0,
            1.000_000_1,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for a in samples {
            for b in samples {
                assert_eq!(numeric::lt(a, b), a < b);
                assert_eq!(numeric::le(a, b), a <= b);
                assert_eq!(numeric::gt(a, b), a > b);
                assert_eq!(numeric::ge(a, b), a >= b);
            }
        }
    }

    #[test]
    fn comparison_results_are_precomputed_constants() {
        // F = O, T = S + O; no multiplication happens at run time.
        assert_eq!(ID.select(P, true), 1.0);
        assert_eq!(ID.select(P, false), 0.0);
        let post = PostOp::new(2.0, 1.1);
        assert_eq!(post.select(P, true), 2.0 + 1.1);
        assert_eq!(post.select(P, false), 1.1);
        assert_eq!(post.truthy_value(P), 3.1);
        assert_eq!(post.falsy_value(), 1.1);
    }

    #[test]
    fn min_max_give_the_second_operand_for_a_nan_or_two_zeros() {
        assert!(arith::max(1.0, NAN).is_nan());
        assert_eq!(arith::max(NAN, 1.0), 1.0);
        assert!(arith::min(1.0, NAN).is_nan());
        assert_eq!(arith::min(NAN, 1.0), 1.0);
        assert_eq!(arith::max(0.0, -0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(arith::min(-0.0, 0.0).to_bits(), 0);
    }

    #[test]
    fn sign_post_op_quirk() {
        // v.x = -1; math.sign(v.x) + 1 → −2 (not 0); math.sign(v.x) * -2 + 1 → 1 (not 3).
        assert_eq!(math::sign(P, -1.0, PostOp::new(1.0, 1.0)), -2.0);
        assert_eq!(math::sign(P, -1.0, PostOp::new(-2.0, 1.0)), 1.0);
        assert_eq!(math::sign(P, 1.0, PostOp::new(1.0, 1.0)), 2.0);
        assert_eq!(math::sign(P, 1.0, PostOp::new(-2.0, 1.0)), -1.0);
    }

    #[test]
    fn random_interpolation_is_hi_r_plus_one_minus_r_lo() {
        let (lo, hi, r) = (
            third(),
            f32::from_bits(0x3f91_bc0d),
            f32::from_bits(0x3ecf_a978),
        );
        assert_eq!(math::random(P, lo, hi, r, ID), hi * r + (1.0 - r) * lo);
        assert_eq!(math::random(P, lo, hi, r, ID).to_bits(), 0x3f28_f09f);
        // The shape `(hi − lo)·r + lo` would be one ulp lower.
        assert_eq!(((hi - lo) * r + lo).to_bits(), 0x3f28_f09e);
        // An infinite bound survives, where the difference hi − lo would not.
        assert_eq!(
            math::random(P, -180.0, f32::NEG_INFINITY, 0.0, ID),
            f32::NEG_INFINITY
        );
        // The integer draw: floor((1 − r)·lo + r·top) with top = (hi·(−ε) + 1) + hi.
        let (lo, hi, r) = (-3.0_f32, 2.0_f32, 0.5_f32);
        let top = (-f32::EPSILON * hi + 1.0) + hi;
        assert_eq!(
            math::random_integer(P, lo, hi, r, ID),
            ((1.0 - r) * lo + top * r).floor()
        );
        assert_eq!(
            math::random_integer_const_bounds(P, lo, hi, r, ID),
            ((1.0 - r) * lo + top * r).floor()
        );
        // 0.5·(−3) + 0.5·2.9999998 is just below 0, where `lo + r·span` would give 0.
        assert_eq!(math::random_integer(P, lo, hi, r, ID), -1.0);
        let mut half = molang::rng::FixedRng::HALF;
        assert_eq!(
            math::die_roll_integer(P, 1.0, lo, hi, 1, &mut half, ID),
            Some(-1.0)
        );
        // One roll of a die is one such draw.
        let mut rng = molang::rng::FixedRng::from_word(crate::common::word(third()));
        assert_eq!(
            math::die_roll(P, 1.0, 1.1, 7.3, 1, &mut rng, ID),
            Some(0.0 + (third() * 7.0 + (1.0 - third()) * 1.0))
        );
    }

    #[test]
    fn float_to_int_conversion() {
        for p in PLATFORMS {
            assert_eq!(arith::to_int(p, 2.9), 2);
            assert_eq!(arith::to_int(p, -2.9), -2);
            assert_eq!(arith::to_int(p, 2_147_483_520.0), 2_147_483_520);
            assert_eq!(arith::to_int(p, -2_147_483_648.0), i32::MIN);
            // The servers answer i32::MIN for NaN and out-of-range values; Android saturates and
            // maps NaN to 0.
            let (nan, high, low) =
                servers_or_android(p, (i32::MIN, i32::MIN, i32::MIN), (0, i32::MAX, i32::MIN));
            assert_eq!(arith::to_int(p, NAN), nan, "{p:?}");
            assert_eq!(arith::to_int(p, 3.0e9), high, "{p:?}");
            assert_eq!(arith::to_int(p, -3.0e9), low, "{p:?}");
            let rolls = servers_or_android(p, 0, i32::MAX as u32);
            assert_eq!(math::die_roll_count(p, 3.0e9), rolls, "{p:?}");
        }
    }

    #[test]
    fn array_index_wraps_and_never_goes_out_of_range() {
        assert_eq!(numeric::array_index(P, 0.0, 3), Some(0));
        assert_eq!(numeric::array_index(P, 2.9, 3), Some(2));
        assert_eq!(numeric::array_index(P, 3.0, 3), Some(0));
        assert_eq!(numeric::array_index(P, 7.5, 3), Some(1));
        assert_eq!(numeric::array_index(P, -1.0, 3), Some(0));
        assert_eq!(numeric::array_index(P, NAN, 3), Some(0));
        assert_eq!(numeric::array_index(P, f32::NEG_INFINITY, 3), Some(0));
        // An empty array has no element to select, whatever the index.
        for index in [
            0.0,
            2.0,
            -1.0,
            NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0e30,
        ] {
            assert_eq!(numeric::array_index(P, index, 0), None);
        }
        assert_eq!(numeric::array_index(P, 1.0e30, 1), Some(0));
        assert_eq!(numeric::array_index(P, f32::INFINITY, 3), Some(0));
    }

    #[test]
    fn multiply_adds_round_twice() {
        let x = third();
        assert_eq!(arith::mul_add(P, x, 3.0, -1.0), 0.0);
        assert_eq!(arith::mul_sub(P, x, 3.0, 1.0), 0.0);
        assert_eq!(arith::mul_add(P, 2.0, 3.0, 1.0), 7.0);
        assert_eq!(arith::mul_sub(P, 2.0, 3.0, 1.0), -5.0);
        assert_eq!(arith::max(1.0, 2.0), 2.0);
        assert_eq!(arith::min(1.0, 2.0), 1.0);
    }
}

mod nan_signs {
    //! Which NaN comes out where two NaNs meet, and the NaN of `math.ln`. The sign is what
    //! `math.copy_sign` shows of a NaN.
    use crate::common::{PLATFORMS, on, servers_or_android};
    use molang::compile::{CompileOptions, compile};
    use molang::numeric::Platform;
    use molang::version::MolangVersion;
    use molang::vm::{NoHostEnv, Value, VariableName};

    /// The quiet NaN with its sign bit set.
    const NEG: u32 = 0xffc0_0000;
    /// The same NaN with its sign bit clear.
    const POS: u32 = 0x7fc0_0000;

    /// The bits `source` evaluates to on `platform` with `v.n` = `NEG`, `v.p` = `POS`, `v.h` = 0.5.
    fn bits_on(platform: Platform, source: &str) -> u32 {
        let options = on(platform, CompileOptions::server(MolangVersion::LATEST));
        let compiled = compile(source, &options);
        assert_eq!(
            compiled.failure(),
            None,
            "{source:?}: {:?}",
            compiled.diagnostics()
        );
        let mut env = NoHostEnv::new();
        for (name, value) in [
            ("n", f32::from_bits(NEG)),
            ("p", f32::from_bits(POS)),
            ("h", 0.5),
            ("one", 1.0),
        ] {
            env.variables_mut()
                .set(VariableName::new(name), Value::Float(value));
        }
        compiled
            .expr()
            .cloned()
            .expect("an expression")
            .eval_f32(&mut env.cx())
            .to_bits()
    }

    /// The bits every platform gives.
    #[track_caller]
    fn bits(source: &str) -> u32 {
        let value = bits_on(PLATFORMS[0], source);
        for platform in PLATFORMS {
            assert_eq!(bits_on(platform, source), value, "{source} on {platform:?}");
        }
        value
    }

    /// In `+`, `-` and `*` of two quiet NaNs the left one wins, at run time and in the constant
    /// fold, and in a longer sum or product the leftmost.
    #[test]
    fn the_left_quiet_nan_wins_in_sums_and_products() {
        for (source, expected) in [
            ("v.n + v.p", NEG),
            ("v.p + v.n", POS),
            ("v.n - v.p", NEG),
            ("v.p - v.n", POS),
            ("v.n * v.p", NEG),
            ("v.p * v.n", POS),
            ("v.one + v.n + v.p", NEG),
            ("v.one + v.p + v.n", POS),
            ("v.p * v.one * v.n", POS),
            ("(v.n + v.p) * 2 + 1", NEG),
            ("-(v.p + v.n)", POS),
        ] {
            assert_eq!(bits(source), expected, "{source}");
        }
        // `math.sqrt(-1)` is the platform's invalid NaN, `NEG` on the servers and `POS` on Android.
        for platform in PLATFORMS {
            let sqrt = servers_or_android(platform, NEG, POS);
            for (source, expected) in [
                ("math.sqrt(-1) + math.copy_sign(math.sqrt(-1), 1)", sqrt),
                ("math.copy_sign(math.sqrt(-1), 1) + math.sqrt(-1)", POS),
                ("math.sqrt(-1) * math.copy_sign(math.sqrt(-1), 1)", sqrt),
                ("math.copy_sign(math.sqrt(-1), 1) * math.sqrt(-1)", POS),
                ("1 + math.sqrt(-1) + math.copy_sign(math.sqrt(-1), 1)", sqrt),
            ] {
                assert_eq!(
                    bits_on(platform, source),
                    expected,
                    "{source} on {platform:?}"
                );
            }
        }
    }

    /// The NaN an interpolation or easing returns depends on its formula's operand order.
    #[test]
    fn interpolation_and_easing_operand_order() {
        for (source, expected) in [
            ("math.lerp(v.h, v.n, v.p)", NEG),
            ("math.lerp(v.n, v.h, v.p)", NEG),
            ("math.lerp(v.p, v.h, v.n)", POS),
            ("math.lerprotate(v.h, v.p, v.n)", POS),
            ("math.ease_in_quad(v.n, v.p, v.h)", POS),
            ("math.ease_out_cubic(v.n, v.h, v.p)", POS),
            ("math.ease_in_out_sine(v.n, v.p, v.h)", POS),
            ("math.ease_in_out_circ(v.h, v.n, v.p)", NEG),
            ("math.ease_in_back(v.n, v.h, v.p)", POS),
            ("math.ease_out_quad(v.n, v.p, v.h)", NEG),
            ("math.ease_out_elastic(v.n, v.p, v.h)", NEG),
            ("math.ease_in_out_bounce(v.n, v.p, v.h)", NEG),
            ("math.ease_in_out_elastic(v.n, v.p, 0.25)", NEG),
            ("math.ease_in_out_elastic(v.n, v.p, 0.75)", NEG),
        ] {
            assert_eq!(bits(source), expected, "{source}");
        }
    }

    /// Run time and folded: `math.ln` of a negative number is the platform's NaN; a NaN argument
    /// comes back unchanged.
    #[test]
    fn the_logarithm_of_a_negative_number_has_the_platforms_nan() {
        for (platform, expected) in [
            (Platform::BdsLinux, NEG),
            (Platform::BdsWindows, NEG),
            (Platform::AndroidArm64, POS),
        ] {
            for source in [
                "math.ln(-1)",
                "math.ln(math.ln(0))",
                "math.ln(-1e-30)",
                "math.ln(-v.h)",
            ] {
                assert_eq!(
                    bits_on(platform, source),
                    expected,
                    "{source} on {platform:?}"
                );
            }
        }
        assert_eq!(bits("math.ln(v.n)"), NEG);
        assert_eq!(bits("math.ln(v.p)"), POS);
        assert_eq!(f32::from_bits(bits("math.ln(0)")), f32::NEG_INFINITY);
        assert_eq!(
            f32::from_bits(bits("math.ln(math.copy_sign(0, -1))")),
            f32::NEG_INFINITY
        );
    }
}
