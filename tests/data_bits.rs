//! Single operations from `numeric_bits.json`, compared to the bit on every platform.
//!
//! The `X86_64` set holds the results of the servers' arithmetic, its transcendental rows those of
//! the `libm` crate (`strict-math`): BDS Linux gives every row, and without `strict-math` every
//! transcendental row within [`HOST_ULPS`]. The `Arm64` set holds results with fused multiply-adds,
//! which no platform gives; only its transcendental samples are checked. For BDS Windows and
//! Android this file checks only which functions differ from the `X86_64` set; their bits are
//! pinned in `tests/numerics.rs` (`platforms`).
//!
//! ```text
//! {schema, counts{<set>: {rows, transcendental_rows, families{}}}, profiles{X86_64[], Arm64[]}}
//! profiles.<set>[] = {family, function, rows[{args[], samples[]?, post_op: null | {scale, offset}, result_bits | null,
//!                                               transcendental, transcendental_samples[]?}]}
//! ```
//!
//! Every float is its bit pattern as eight hex digits. `function` names the molang function and
//! `args` are its arguments in order (`div_guard_*`: `result_bits` null when the guard fired).
//! `samples` are the random draws in order. `transcendental: true` marks a row that evaluates a
//! transcendental function; `transcendental_samples` lists the `[function, [argument bits…], result
//! bits]` the row relies on, each also checked alone against the platform's function. `counts`
//! states the file's own size per set.

#![cfg(all(feature = "compiler", feature = "stdlib"))]

mod common;

use common::{Samples, hex_f32};

use std::collections::{BTreeMap, BTreeSet};

use common::PLATFORMS;
use molang::numeric::{self, Platform, PostOp};
use molang::rng::FixedRng;
use molang::stdlib::math;
use serde_json::Value;

const ID: PostOp = PostOp::IDENTITY;

/// `math.die_roll*` stepped with the row's raw samples. When every sample is a word's, the one-call
/// form over a generator playing those words back must agree to the bit and draw once per roll.
fn die_roll_row(
    platform: Platform,
    integer: bool,
    a: &[f32],
    samples: &[f32],
    post: PostOp,
) -> f32 {
    let mut roll = if integer {
        math::DieRoll::new_integer(platform, a[0], a[1], a[2])
    } else {
        math::DieRoll::new(platform, a[0], a[1], a[2])
    };
    let rolls = roll.remaining() as usize;
    for &sample in &samples[..rolls] {
        roll.roll(sample);
    }
    let stepped = roll.finish(post);
    if samples
        .iter()
        .all(|&sample| FixedRng::from_sample(sample).is_some())
    {
        let mut rng = Samples::new(samples);
        let whole = if integer {
            math::die_roll_integer(platform, a[0], a[1], a[2], u32::MAX, &mut rng, post)
        } else {
            math::die_roll(platform, a[0], a[1], a[2], u32::MAX, &mut rng, post)
        };
        assert_eq!(
            whole.map(f32::to_bits),
            Some(stepped.to_bits()),
            "die_roll over {samples:?}"
        );
        assert_eq!(rng.draws, rolls, "one draw per roll");
    }
    stepped
}

/// `None` means the division guard fired.
fn evaluate_row(
    platform: Platform,
    function: &str,
    a: &[f32],
    samples: &[f32],
    post: PostOp,
) -> Option<f32> {
    type Ease = fn(Platform, f32, f32, f32, PostOp) -> f32;
    let ease: Option<Ease> = match function {
        "ease_in_quad" => Some(math::ease_in_quad),
        "ease_out_quad" => Some(math::ease_out_quad),
        "ease_in_out_quad" => Some(math::ease_in_out_quad),
        "ease_in_cubic" => Some(math::ease_in_cubic),
        "ease_out_cubic" => Some(math::ease_out_cubic),
        "ease_in_out_cubic" => Some(math::ease_in_out_cubic),
        "ease_in_quart" => Some(math::ease_in_quart),
        "ease_out_quart" => Some(math::ease_out_quart),
        "ease_in_out_quart" => Some(math::ease_in_out_quart),
        "ease_in_quint" => Some(math::ease_in_quint),
        "ease_out_quint" => Some(math::ease_out_quint),
        "ease_in_out_quint" => Some(math::ease_in_out_quint),
        "ease_in_sine" => Some(math::ease_in_sine),
        "ease_out_sine" => Some(math::ease_out_sine),
        "ease_in_out_sine" => Some(math::ease_in_out_sine),
        "ease_in_expo" => Some(math::ease_in_expo),
        "ease_out_expo" => Some(math::ease_out_expo),
        "ease_in_out_expo" => Some(math::ease_in_out_expo),
        "ease_in_circ" => Some(math::ease_in_circ),
        "ease_out_circ" => Some(math::ease_out_circ),
        "ease_in_out_circ" => Some(math::ease_in_out_circ),
        "ease_in_bounce" => Some(math::ease_in_bounce),
        "ease_out_bounce" => Some(math::ease_out_bounce),
        "ease_in_out_bounce" => Some(math::ease_in_out_bounce),
        "ease_in_back" => Some(math::ease_in_back),
        "ease_out_back" => Some(math::ease_out_back),
        "ease_in_out_back" => Some(math::ease_in_out_back),
        "ease_in_elastic" => Some(math::ease_in_elastic),
        "ease_out_elastic" => Some(math::ease_out_elastic),
        "ease_in_out_elastic" => Some(math::ease_in_out_elastic),
        _ => None,
    };
    if let Some(ease) = ease {
        return Some(ease(platform, a[0], a[1], a[2], post));
    }
    let value = match function {
        "add" => numeric::add(platform, a[0], a[1], post),
        "mul" => numeric::mul(platform, a[0], a[1], post),
        "div" => numeric::div(platform, a[0], a[1], post),
        "negate" => numeric::negate(platform, a[0], post),
        "div_guard_signed" => return numeric::div_guard(true, a[0]),
        "div_guard_abs" => return numeric::div_guard(false, a[0]),
        "lt" => post.select(platform, numeric::lt(a[0], a[1])),
        "le" => post.select(platform, numeric::le(a[0], a[1])),
        "ge" => post.select(platform, numeric::ge(a[0], a[1])),
        "gt" => post.select(platform, numeric::gt(a[0], a[1])),
        "abs" => math::abs(platform, a[0], post),
        "acos" => math::acos(platform, a[0], post),
        "asin" => math::asin(platform, a[0], post),
        "atan" => math::atan(platform, a[0], post),
        "ceil" => math::ceil(platform, a[0], post),
        "cos" => math::cos(platform, a[0], post),
        "exp" => math::exp(platform, a[0], post),
        "floor" => math::floor(platform, a[0], post),
        "hermite_blend" => math::hermite_blend(platform, a[0], post),
        "ln" => math::ln(platform, a[0], post),
        "min_angle" => math::min_angle(platform, a[0], post),
        "round" => math::round(platform, a[0], post),
        "sin" => math::sin(platform, a[0], post),
        "sqrt" => math::sqrt(platform, a[0], post),
        "trunc" => math::trunc(platform, a[0], post),
        "atan2" => math::atan2(platform, a[0], a[1], post),
        "copy_sign" => math::copy_sign(platform, a[0], a[1], post),
        "pow" => math::pow(platform, a[0], a[1], post),
        "min" => math::min(platform, a[0], a[1], post),
        "max" => math::max(platform, a[0], a[1], post),
        "clamp" => math::clamp(platform, a[0], a[1], a[2], post),
        "lerp" => math::lerp(platform, a[0], a[1], a[2], post),
        "lerprotate" => math::lerprotate(platform, a[0], a[1], a[2], post),
        "inverse_lerp" => math::inverse_lerp(platform, a[0], a[1], a[2], post),
        "mod_const" => math::mod_const(platform, a[0], a[1], post),
        "mod_runtime" => math::mod_runtime(platform, a[0], a[1], post),
        "sign" => math::sign(platform, a[0], post),
        "random" => math::random(platform, a[0], a[1], samples[0], post),
        "random_const" => math::random_folded(
            platform,
            samples[0],
            math::random_const_bounds(platform, a[0], a[1], post),
        ),
        "random_integer" => math::random_integer(platform, a[0], a[1], samples[0], post),
        "random_integer_const" => {
            math::random_integer_const_bounds(platform, a[0], a[1], samples[0], post)
        }
        "die_roll" => die_roll_row(platform, false, a, samples, post),
        "die_roll_integer" => die_roll_row(platform, true, a, samples, post),
        other => panic!("the file names a function this test does not know: {other}"),
    };
    Some(value)
}

/// The ulps a transcendental row or sample may miss its bits by without `strict-math`, where the
/// host's C library computes it; the set holds the `libm` crate's results.
const HOST_ULPS: u32 = 2;

/// `x`'s place in the order of the floats, −0 and +0 adjacent.
fn ordinal(x: f32) -> i64 {
    let bits = i64::from(x.to_bits());
    if bits & 0x8000_0000 == 0 {
        bits
    } else {
        0x8000_0000 - bits
    }
}

/// The ulps between `a` and `b`; `None` unless both are numbers or both are the same NaN.
fn ulps(a: f32, b: f32) -> Option<u32> {
    if a.is_nan() || b.is_nan() {
        return (a.to_bits() == b.to_bits()).then_some(0);
    }
    u32::try_from((ordinal(a) - ordinal(b)).unsigned_abs()).ok()
}

/// The bits are equal, NaN payload and sign included; without `strict-math` a transcendental
/// result may miss them by [`HOST_ULPS`].
fn agrees(actual: Option<f32>, expected: Option<f32>, transcendental: bool) -> bool {
    match (actual, expected) {
        (Some(a), Some(e)) if transcendental && !cfg!(feature = "strict-math") => {
            ulps(a, e).is_some_and(|d| d <= HOST_ULPS)
        }
        _ => actual.map(f32::to_bits) == expected.map(f32::to_bits),
    }
}

/// A row's post-op, or the identity.
fn post_of(row: &Value) -> PostOp {
    if row["post_op"].is_null() {
        ID
    } else {
        PostOp::new(
            hex_f32(&row["post_op"]["scale"]),
            hex_f32(&row["post_op"]["offset"]),
        )
    }
}

/// A row's arguments, samples and result (`None`: the guard fired).
fn row_values(row: &Value) -> (Vec<f32>, Vec<f32>, Option<f32>) {
    let args = row["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(hex_f32)
        .collect();
    let samples = row
        .get("samples")
        .and_then(Value::as_array)
        .map(|s| s.iter().map(hex_f32).collect())
        .unwrap_or_default();
    let expected = if row["result_bits"].is_null() {
        None
    } else {
        Some(hex_f32(&row["result_bits"]))
    };
    (args, samples, expected)
}

/// The functions of `set` whose rows `platform` does not reproduce.
fn differing_functions(file: &Value, set: &str, platform: Platform) -> BTreeSet<String> {
    let mut differing = BTreeSet::new();
    for group in file["profiles"][set].as_array().unwrap() {
        let function = group["function"].as_str().unwrap();
        for row in group["rows"].as_array().unwrap() {
            let (args, samples, expected) = row_values(row);
            if function.starts_with("die_roll")
                && samples.len() != math::die_roll_count(platform, args[0]) as usize
            {
                differing.insert(function.to_owned());
                continue;
            }
            let actual = evaluate_row(platform, function, &args, &samples, post_of(row));
            if !agrees(actual, expected, row["transcendental"].as_bool().unwrap()) {
                differing.insert(function.to_owned());
            }
        }
    }
    differing
}

#[test]
fn bds_linux_matches_the_x86_64_rows() {
    let file = common::data_json("numeric_bits.json");
    let differing = differing_functions(&file, "X86_64", Platform::BdsLinux);
    assert!(differing.is_empty(), "BdsLinux differs in {differing:?}");
    let (mut rows, mut nan_rows) = (0_u64, 0_u64);
    let mut most = BTreeMap::<String, u32>::new();
    for group in file["profiles"]["X86_64"].as_array().unwrap() {
        let function = group["function"].as_str().unwrap();
        for row in group["rows"].as_array().unwrap() {
            let (args, samples, expected) = row_values(row);
            rows += 1;
            nan_rows += u64::from(expected.is_some_and(f32::is_nan));
            let actual = evaluate_row(Platform::BdsLinux, function, &args, &samples, post_of(row));
            if let (Some(a), Some(e)) = (actual, expected) {
                let d = ulps(a, e).unwrap_or(u32::MAX);
                let entry = most.entry(function.to_owned()).or_default();
                *entry = (*entry).max(d);
            }
        }
    }
    most.retain(|_, d| *d > 0);
    println!("numeric_bits.json [X86_64]: {rows} rows, {nan_rows} NaN; ulps off {most:?}");
}

/// The functions of the `X86_64` set each other platform computes differently: the invalid
/// operations' NaN, the NaN an operation of two NaNs keeps, and the number of rolls. The bits
/// themselves are pinned in `tests/numerics.rs` (`platforms`).
#[test]
fn each_other_platform_differs_from_the_x86_64_set_in_its_functions() {
    let file = common::data_json("numeric_bits.json");
    let pinned: [(Platform, &[&str]); 2] = [
        (Platform::BdsWindows, &[]),
        (
            Platform::AndroidArm64,
            &[
                "cos",
                "die_roll",
                "die_roll_integer",
                "ease_in_bounce",
                "ease_in_circ",
                "ease_in_elastic",
                "ease_in_out_circ",
                "ease_in_out_sine",
                "ease_in_sine",
                "ease_out_circ",
                "ease_out_cubic",
                "ease_out_elastic",
                "ease_out_sine",
                "hermite_blend",
                "inverse_lerp",
                "lerprotate",
                "ln",
                "min_angle",
                "mod_const",
                "mod_runtime",
                "pow",
                "random",
                "random_const",
                "sin",
                "sqrt",
            ],
        ),
    ];
    for (platform, expected) in pinned {
        let differing = differing_functions(&file, "X86_64", platform);
        let expected: BTreeSet<String> = expected.iter().map(|&f| f.to_owned()).collect();
        assert_eq!(differing, expected, "{platform:?}");
    }
}

/// A reader that loses rows must fail.
#[test]
fn both_sets_hold_the_rows_they_count() {
    let file = common::data_json("numeric_bits.json");
    for name in ["X86_64", "Arm64"] {
        let (mut rows, mut transcendental_rows) = (0_u64, 0_u64);
        let mut families = BTreeMap::<String, u64>::new();
        for group in file["profiles"][name].as_array().unwrap() {
            for row in group["rows"].as_array().unwrap() {
                rows += 1;
                transcendental_rows += u64::from(row["transcendental"].as_bool().unwrap());
                *families
                    .entry(group["family"].as_str().unwrap().to_owned())
                    .or_default() += 1;
            }
        }
        let counts = &file["counts"][name];
        assert_eq!(Some(rows), counts["rows"].as_u64(), "{name}");
        assert_eq!(
            Some(transcendental_rows),
            counts["transcendental_rows"].as_u64(),
            "{name}"
        );
        for (family, count) in counts["families"].as_object().unwrap() {
            assert_eq!(
                families.get(family).copied(),
                count.as_u64(),
                "{name} {family}"
            );
        }
        assert_eq!(
            families.len(),
            counts["families"].as_object().unwrap().len(),
            "{name}"
        );
    }
}

/// `x` and its `n` neighbouring floats on each side.
fn around(x: f32, n: i32) -> impl Iterator<Item = f32> {
    (-n..=n).map(move |k| f32::from_bits(x.to_bits().wrapping_add_signed(k)))
}

/// The degrees whose product with `DEG_TO_RAD` is `radians`.
fn degrees_of(radians: f32) -> Option<f32> {
    if !radians.is_finite() {
        return Some(radians);
    }
    around(radians / math::DEG_TO_RAD, 8).find(|&d| d * math::DEG_TO_RAD == radians)
}

/// `sin(radians)` through the sine table of `ease_out_sine`, when `radians` is one of its angles:
/// entry `i` is `sin(i / SCALE)`.
fn sine_table(platform: Platform, radians: f32) -> Option<f32> {
    const SCALE: f32 = f32::from_bits(0x4622_f983);
    const HALF_PI: f32 = f32::from_bits(0x3fc9_0fdb);
    let entry = |i: i32| i as f32 / SCALE;
    let guess = (radians * SCALE).round() as i32;
    let index =
        (guess - 1..=guess + 1).find(|&i| (0..65_536).contains(&i) && entry(i) == radians)?;
    let t = around(index as f32 / 16_384.0, 64)
        .find(|&t| ((t * HALF_PI) * SCALE) as i32 & 0xffff == index)?;
    Some(math::ease_out_sine(platform, 0.0, 1.0, t, ID))
}

/// `exp2(x)` through `ease_in_expo` from 0 to 1, when some `t` gives the exponent `(t − 1)·10`.
fn exp2_through_an_easing(platform: Platform, x: f32) -> Option<f32> {
    if !x.is_finite() {
        return Some(math::ease_in_expo(platform, 0.0, 1.0, x, ID));
    }
    around(x / 10.0 + 1.0, 32)
        .find(|&t| (t - 1.0) * 10.0 == x)
        .map(|t| math::ease_in_expo(platform, 0.0, 1.0, t, ID))
}

/// The `exp2` exponents no `ease_in_expo` reaches; the unit tests of `exp2` pin them.
const EXP2_UNREACHED: [u32; 5] = [
    0xbf80_0000,
    0xc080_0000,
    0xbfd6_ab74,
    0xbf13_7db9,
    0xc08c_be3f,
];

/// Every `transcendental_samples` entry, `[function, [argument bits…], result bits]`, is what the
/// crate's function gives, to the bit under `strict-math` and else within [`HOST_ULPS`]:
/// trigonometry through the degree conversion, `exp2` through `ease_in_expo`, on every platform.
/// The NaN of an invalid operation is compared to the bit only on BDS Linux against the `X86_64`
/// set.
#[test]
fn every_transcendental_sample_is_the_crates_function() {
    let file = common::data_json("numeric_bits.json");
    let (mut checked, mut failures, mut unreached) = (0_u32, Vec::new(), BTreeSet::new());
    let mut most = BTreeMap::<&str, u32>::new();
    let in_degrees = |radians: f32| radians * math::RAD_TO_DEG;
    for (platform, set) in PLATFORMS
        .into_iter()
        .flat_map(|p| ["X86_64", "Arm64"].map(|set| (p, set)))
    {
        let exact = set == "X86_64" && platform == Platform::BdsLinux;
        for group in file["profiles"][set].as_array().unwrap() {
            for sample in group["rows"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|row| row.get("transcendental_samples"))
                .flat_map(|s| s.as_array().unwrap())
            {
                let a: Vec<f32> = sample[1].as_array().unwrap().iter().map(hex_f32).collect();
                let expected = hex_f32(&sample[2]);
                let function = sample[0].as_str().unwrap();
                let (actual, expected) = match (function, a.as_slice()) {
                    ("sin", &[x]) => (
                        degrees_of(x)
                            .map(|d| math::sin(platform, d, ID))
                            .or_else(|| sine_table(platform, x)),
                        expected,
                    ),
                    ("cos", &[x]) => (degrees_of(x).map(|d| math::cos(platform, d, ID)), expected),
                    ("asin", &[x]) => (Some(math::asin(platform, x, ID)), in_degrees(expected)),
                    ("acos", &[x]) => (Some(math::acos(platform, x, ID)), in_degrees(expected)),
                    ("atan", &[x]) => (Some(math::atan(platform, x, ID)), in_degrees(expected)),
                    ("atan2", &[y, x]) => {
                        (Some(math::atan2(platform, y, x, ID)), in_degrees(expected))
                    }
                    ("exp", &[x]) => (Some(math::exp(platform, x, ID)), expected),
                    ("exp2", &[x]) => (exp2_through_an_easing(platform, x), expected),
                    ("ln", &[x]) => (Some(math::ln(platform, x, ID)), expected),
                    ("pow", &[x, y]) => (Some(math::pow(platform, x, y, ID)), expected),
                    (other, _) => {
                        panic!("the file names a function this test does not know: {other}")
                    }
                };
                let Some(actual) = actual else {
                    unreached.insert((function.to_owned(), a[0].to_bits()));
                    continue;
                };
                checked += 1;
                let invalid_of_the_other =
                    !exact && expected.is_nan() && !a.iter().any(|x| x.is_nan());
                let agrees = if invalid_of_the_other {
                    actual.is_nan()
                } else {
                    if let Some(d) = ulps(actual, expected) {
                        let entry = most.entry(function).or_default();
                        *entry = (*entry).max(d);
                    }
                    agrees(Some(actual), Some(expected), true)
                };
                if !agrees {
                    failures.push(format!(
                        "{platform:?} {set} {sample} gives {:08x}, expected {:08x}",
                        actual.to_bits(),
                        expected.to_bits()
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} sample(s) disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
    most.retain(|_, d| *d > 0);
    println!("transcendental samples: ulps off {most:?}");
    let expected_unreached: BTreeSet<_> = EXP2_UNREACHED
        .iter()
        .map(|&bits| ("exp2".to_owned(), bits))
        .collect();
    assert_eq!(unreached, expected_unreached);
    assert_eq!(checked, 3 * (1468 - 18));
}
