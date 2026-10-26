//! Evaluates an expression that is one bare `math.<fn>(…)` call through `molang::stdlib::math`
//! directly, so the math library is checked without the parser. Arguments are literals or variables
//! assigned literals by earlier statements; a post-op (`math.ceil(1.1f) + 1`), a nested call or
//! `math.pi` is not a bare call.

use std::collections::HashMap;

use molang::numeric::{Platform, PostOp};
use molang::rng::RandomSource;
use molang::stdlib::math;

/// `literal` picks between the constant-operand and the run-time form of `mod`, `random` and
/// `random_integer`.
#[derive(Copy, Clone, Debug)]
pub struct Arg {
    pub value: f32,
    pub literal: bool,
}

#[derive(Debug)]
pub struct Call {
    /// Lower case, without `math.`.
    pub name: String,
    pub args: Vec<Arg>,
}

/// `-?digits[.digits][e[+-]digits][f]`, optionally in redundant parentheses.
fn parse_literal(text: &str) -> Option<f32> {
    let mut text = text.trim();
    while let Some(inner) = text.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        text = inner.trim();
    }
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, text),
    };
    let digits = digits.strip_suffix('f').unwrap_or(digits);
    let first = digits.chars().next()?;
    if !(first.is_ascii_digit() || first == '.')
        || !digits
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-'))
    {
        return None;
    }
    let value: f32 = digits.parse().ok()?;
    Some(if negative { -value } else { value })
}

fn variable_key(text: &str) -> Option<String> {
    let text = text.trim();
    let (namespace, name) = text.split_once('.')?;
    let namespace = match namespace {
        "v" | "variable" => "variable",
        "t" | "temp" => "temp",
        _ => return None,
    };
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some(format!("{namespace}.{name}"))
}

/// Recognises `[<var> = <literal>;]* [return] math.<fn>(<literal or var>, …)[;]`.
pub fn parse_call(expr: &str) -> Option<Call> {
    let lowered = expr.to_ascii_lowercase();
    let mut statements: Vec<&str> = lowered.split(';').map(str::trim).collect();
    let complex = statements.len() > 1;
    if complex && statements.pop() != Some("") {
        // A complex expression must end with `;`.
        return None;
    }
    let last = statements.pop()?;
    let mut variables = HashMap::new();
    for statement in statements {
        let (target, value) = statement.split_once('=')?;
        variables.insert(variable_key(target)?, parse_literal(value)?);
    }
    let call = if complex {
        last.strip_prefix("return")?.trim()
    } else {
        last
    };
    let call = call.strip_prefix("math.")?;
    let (name, rest) = call.split_once('(')?;
    let inner = rest.strip_suffix(')')?;
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let mut args = Vec::new();
    for arg in inner.split(',') {
        if let Some(value) = parse_literal(arg) {
            args.push(Arg {
                value,
                literal: true,
            });
        } else {
            args.push(Arg {
                value: *variables.get(&variable_key(arg)?)?,
                literal: false,
            });
        }
    }
    Some(Call {
        name: name.to_owned(),
        args,
    })
}

/// `None` for a function name or arity the math library does not have.
pub fn evaluate_call(platform: Platform, call: &Call, rng: &mut dyn RandomSource) -> Option<f32> {
    let post = PostOp::IDENTITY;
    let values: Vec<f32> = call.args.iter().map(|a| a.value).collect();
    let all_literal = call.args.iter().all(|a| a.literal);
    Some(match (call.name.as_str(), values.as_slice()) {
        ("abs", &[x]) => math::abs(platform, x, post),
        ("acos", &[x]) => math::acos(platform, x, post),
        ("asin", &[x]) => math::asin(platform, x, post),
        ("atan", &[x]) => math::atan(platform, x, post),
        ("atan2", &[y, x]) => math::atan2(platform, y, x, post),
        ("ceil", &[x]) => math::ceil(platform, x, post),
        ("clamp", &[v, lo, hi]) => math::clamp(platform, v, lo, hi, post),
        ("copy_sign", &[a, b]) => math::copy_sign(platform, a, b, post),
        ("cos", &[x]) => math::cos(platform, x, post),
        ("die_roll", &[n, a, b]) => math::die_roll(platform, n, a, b, u32::MAX, rng, post)?,
        ("die_roll_integer", &[n, a, b]) => {
            math::die_roll_integer(platform, n, a, b, u32::MAX, rng, post)?
        }
        ("exp", &[x]) => math::exp(platform, x, post),
        ("floor", &[x]) => math::floor(platform, x, post),
        ("hermite_blend", &[t]) => math::hermite_blend(platform, t, post),
        ("lerp", &[a, b, t]) => math::lerp(platform, a, b, t, post),
        ("lerprotate", &[a, b, t]) => math::lerprotate(platform, a, b, t, post),
        ("inverse_lerp", &[a, b, v]) => math::inverse_lerp(platform, a, b, v, post),
        ("ln", &[x]) => math::ln(platform, x, post),
        ("max", &[a, b]) => math::max(platform, a, b, post),
        ("min", &[a, b]) => math::min(platform, a, b, post),
        ("min_angle", &[x]) => math::min_angle(platform, x, post),
        ("mod", &[a, b]) => {
            if call.args[1].literal {
                math::mod_const(platform, a, b, post)
            } else {
                math::mod_runtime(platform, a, b, post)
            }
        }
        ("pow", &[a, b]) => math::pow(platform, a, b, post),
        ("random", &[a, b]) => {
            let sample = rng.sample();
            if all_literal {
                math::random_folded(
                    platform,
                    sample,
                    math::random_const_bounds(platform, a, b, post),
                )
            } else {
                math::random(platform, a, b, sample, post)
            }
        }
        ("random_integer", &[a, b]) => {
            let sample = rng.sample();
            if all_literal {
                math::random_integer_const_bounds(platform, a, b, sample, post)
            } else {
                math::random_integer(platform, a, b, sample, post)
            }
        }
        ("round", &[x]) => math::round(platform, x, post),
        ("sign", &[x]) => math::sign(platform, x, post),
        ("sin", &[x]) => math::sin(platform, x, post),
        ("sqrt", &[x]) => math::sqrt(platform, x, post),
        ("trunc", &[x]) => math::trunc(platform, x, post),
        ("ease_in_quad", &[s, e, t]) => math::ease_in_quad(platform, s, e, t, post),
        ("ease_out_quad", &[s, e, t]) => math::ease_out_quad(platform, s, e, t, post),
        ("ease_in_out_quad", &[s, e, t]) => math::ease_in_out_quad(platform, s, e, t, post),
        ("ease_in_cubic", &[s, e, t]) => math::ease_in_cubic(platform, s, e, t, post),
        ("ease_out_cubic", &[s, e, t]) => math::ease_out_cubic(platform, s, e, t, post),
        ("ease_in_out_cubic", &[s, e, t]) => math::ease_in_out_cubic(platform, s, e, t, post),
        ("ease_in_quart", &[s, e, t]) => math::ease_in_quart(platform, s, e, t, post),
        ("ease_out_quart", &[s, e, t]) => math::ease_out_quart(platform, s, e, t, post),
        ("ease_in_out_quart", &[s, e, t]) => math::ease_in_out_quart(platform, s, e, t, post),
        ("ease_in_quint", &[s, e, t]) => math::ease_in_quint(platform, s, e, t, post),
        ("ease_out_quint", &[s, e, t]) => math::ease_out_quint(platform, s, e, t, post),
        ("ease_in_out_quint", &[s, e, t]) => math::ease_in_out_quint(platform, s, e, t, post),
        ("ease_in_sine", &[s, e, t]) => math::ease_in_sine(platform, s, e, t, post),
        ("ease_out_sine", &[s, e, t]) => math::ease_out_sine(platform, s, e, t, post),
        ("ease_in_out_sine", &[s, e, t]) => math::ease_in_out_sine(platform, s, e, t, post),
        ("ease_in_expo", &[s, e, t]) => math::ease_in_expo(platform, s, e, t, post),
        ("ease_out_expo", &[s, e, t]) => math::ease_out_expo(platform, s, e, t, post),
        ("ease_in_out_expo", &[s, e, t]) => math::ease_in_out_expo(platform, s, e, t, post),
        ("ease_in_circ", &[s, e, t]) => math::ease_in_circ(platform, s, e, t, post),
        ("ease_out_circ", &[s, e, t]) => math::ease_out_circ(platform, s, e, t, post),
        ("ease_in_out_circ", &[s, e, t]) => math::ease_in_out_circ(platform, s, e, t, post),
        ("ease_in_bounce", &[s, e, t]) => math::ease_in_bounce(platform, s, e, t, post),
        ("ease_out_bounce", &[s, e, t]) => math::ease_out_bounce(platform, s, e, t, post),
        ("ease_in_out_bounce", &[s, e, t]) => math::ease_in_out_bounce(platform, s, e, t, post),
        ("ease_in_back", &[s, e, t]) => math::ease_in_back(platform, s, e, t, post),
        ("ease_out_back", &[s, e, t]) => math::ease_out_back(platform, s, e, t, post),
        ("ease_in_out_back", &[s, e, t]) => math::ease_in_out_back(platform, s, e, t, post),
        ("ease_in_elastic", &[s, e, t]) => math::ease_in_elastic(platform, s, e, t, post),
        ("ease_out_elastic", &[s, e, t]) => math::ease_out_elastic(platform, s, e, t, post),
        ("ease_in_out_elastic", &[s, e, t]) => math::ease_in_out_elastic(platform, s, e, t, post),
        _ => return None,
    })
}

pub fn uses_random(call: &Call) -> bool {
    matches!(
        call.name.as_str(),
        "random" | "random_integer" | "die_roll" | "die_roll_integer"
    )
}
