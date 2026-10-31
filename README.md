# molang

[![Crates.io](https://img.shields.io/crates/v/molang.svg)](https://crates.io/crates/molang) [![docs.rs](https://img.shields.io/docsrs/molang)](https://docs.rs/molang) [![CI](https://github.com/bedrock-crustaceans/molang/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bedrock-crustaceans/molang/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/bedrock-crustaceans/molang/blob/main/LICENSE)

A compiler and bytecode VM for [Molang](https://learn.microsoft.com/en-us/minecraft/creator/reference/content/molangreference/examples/molangconcepts/molangintroduction), the expression language of Minecraft: Bedrock Edition, in pure Rust. It aims to behave like the game does, at every Molang version, down to the error messages. Part of [bedrock-crustaceans](https://github.com/bedrock-crustaceans).

The crate compiles text to bytecode and evaluates it. It knows every standard query and `math.*` function, but it implements no query: your embedding fills in a `QueryTable`, and a query without a function returns its declared default.

## Installation

Edition 2024, Rust 1.92 or newer, `std`.

```toml
[dependencies]
molang = { version = "1", features = ["vm"] }
```

## Quick start

`compile` never fails. A rejected expression becomes the constant 0 with its diagnostics attached, and `into_result` turns that into a `Result`. `NoHostEnv` evaluates without a world.

```rust
use molang::compile::{CompileOptions, compile};
use molang::version::MolangVersion;
use molang::vm::{NoHostEnv, Value, VariableName};

fn main() -> Result<(), molang::compile::CompileError> {
    let options = CompileOptions::server(MolangVersion::LATEST);
    let (expr, diagnostics) = compile("v.speed * 2 + math.clamp(1 + 2 * 3, 0, 5)", &options).into_result()?;
    assert!(diagnostics.is_empty());

    let mut env = NoHostEnv::new();
    env.variables_mut().set(VariableName::new("speed"), Value::Float(1.5));
    assert_eq!(expr.eval_f32(&mut env.cx()), 8.0);
    Ok(())
}
```

Pick `CompileOptions::server` or `client`; there is no default side. The version comes from the pack: a `MolangSource` carries text plus version and `compile_source` compiles it at its own version. `MolangVersion::from_engine_version_str` maps an engine version such as `"1.18.10"`.

## Embedding

Everything the game provides, you provide. Each piece has an example under [`examples/`](https://github.com/bedrock-crustaceans/molang/tree/main/examples).

- **Queries.** Declare your own in a catalogue extended from `stdlib::queries(side)`, compile against it, and put the functions in a `QueryTable`. `QueryCatalog::overriding` changes a standard declaration.
- **Variables and contexts.** `NoHostEnv` holds maps for `variable.*`, `temp.*` and `context.*`. A `HostEnv` wraps your own `Host`, `VariableStore` and `ContextProvider` over a real world.
- **Render-controller arrays.** `array.<name>[i]` reads an `ArrayTable` at evaluation under `CompileOptions::client`. A server has no arrays, so under `server` the compile ends with `CompileOutcome::UsesArrays`.
- **Math functions.** A `MathCatalog` adds or replaces `math.*` functions. Pure ones fold at compile time; volatile ones draw from the random source.
- **Random.** Any `rng::RandomSource` works, including a closure. The default is `ProcessRng`, one xorshift128 per process like the game's entity contexts. Worldgen uses a Mersenne Twister per chunk and rule; the `rng` module docs give the seeding.
- **Limits.** `EvalLimits` budgets loops, steps, struct size and query depth. Keep the defaults for untrusted content.

```rust
use molang::catalog::{Arity, QueryDecl, QueryShape, ReturnType, Side};
use molang::compile::{CompileOptions, compile};
use molang::stdlib;
use molang::version::MolangVersion;
use molang::vm::{NoHost, NoHostEnv, QueryCx, QueryResult, QueryTable, Value};

fn double(cx: &mut QueryCx<'_, '_, NoHost>) -> QueryResult<NoHost> {
    Ok(Value::Float(cx.arg_f32(0).unwrap_or(0.0) * 2.0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = QueryShape::DEFAULT.with_args(Arity::exactly(1)).with_returns(ReturnType::FLOAT);
    let catalog = stdlib::queries(Side::Server).extended([QueryDecl::new("query.double", shape)?])?;
    let options = CompileOptions::new(catalog.clone(), MolangVersion::LATEST);
    let (expr, _) = compile("query.double(21)", &options).into_result()?;

    let mut queries = QueryTable::new(&catalog);
    queries.set("query.double", double)?;
    let mut env = NoHostEnv::new().with_queries(queries);
    assert_eq!(expr.eval_f32(&mut env.cx()), 42.0);
    Ok(())
}
```

## Reading a result

`eval` returns a `Value`. The game reads the same float differently depending on the field, so `Value` has one reading per field kind. `eval_f32` is the plain float.

| Field | Reading |
| --- | --- |
| permutation and event conditions, `bone_visibility` | `rounds_to_nonzero`: round, then `!= 0` |
| `tags`, bool property defaults | `is_nonzero` |
| ambient-sound conditions | `exceeds_epsilon` |
| anger-sound conditions | `is_positive` |
| int property defaults, experience rewards | `to_property_int`, `round_to_int` |
| float property defaults | `to_property_float`: clamped, NaN is the minimum |
| animations, colours, everything else | `to_f32` |

## Exact bits

By default the crate uses whatever your machine does: `std`'s `f32` functions, so the C library of the host, and the host CPU's NaNs. That matches the game closely, since the game does the same on its own machine, but the last bit of a `math.sin` and the sign of a NaN can differ between hosts. Two features pin them down:

- **`strict-math`** takes the transcendental functions from the `libm` crate instead of the host's C library. Same bits on every host.
- **`platform-quirks`** adds `Platform::BdsLinux`, `BdsWindows` and `AndroidArm64` to `numeric::Platform`. An expression compiled for one of them reproduces that machine's NaNs and float-to-int conversions exactly, so for instance `array.frames[1e10]` picks the element the server picks, and `math.copy_sign(1, math.ln(-1))` is −1 like on the server rather than whatever the host gives.

Both are additive and combine. With both on, the recorded server data in `tests/` replays bit for bit on any host. Molang's own rules always hold: comparisons with NaN are false, `math.min` and `math.max` return the second operand on NaN, there is no fused multiply-add.

```rust
use molang::compile::{CompileOptions, compile};
use molang::numeric::Platform;
use molang::version::MolangVersion;

# #[cfg(feature = "platform-quirks")]
fn main() -> Result<(), molang::compile::CompileError> {
    let options = CompileOptions::server(MolangVersion::LATEST).with_platform(Platform::AndroidArm64);
    let (expr, _) = compile("math.copy_sign(1, math.ln(-1))", &options).into_result()?;
    assert_eq!(expr.as_constant(), Some(1.0));
    Ok(())
}
# #[cfg(not(feature = "platform-quirks"))]
# fn main() {}
```

## Features

`stdlib` is on by default. Every public item lives in exactly one module; docs.rs shows the feature each one needs.

| Feature | What it adds |
| --- | --- |
| `stdlib` | the standard queries and `math.*` functions, `CompileOptions::server` / `client` |
| `compiler` | `compile`, `numeric`, `rng`, `diag`, host math catalogues |
| `vm` | evaluation: `Value`, the host traits, `EvalLimits` (implies `compiler`) |
| `cache` | `CompileCache`, a concurrent cache keyed by version, options and source |
| `facet` | `Facet` impls for the JSON forms |
| `strict-math` | transcendentals from `libm` |
| `platform-quirks` | the emulated server and client platforms |
| `fuzz` | internals for the repository's fuzz crate, no SemVer guarantee |

Without `stdlib` the crate is the bare engine: no query and no `math.*` name is known until you declare them.

## Examples

Each one prints what it shows. Start with `cargo run --example hello_world --features vm`.

| Example | Covers |
| --- | --- |
| `hello_world` | compile, evaluate, variables, a loop, a missing read |
| `versions` | engine versions and how one text parses differently across Molang versions |
| `json_forms` | the JSON forms a Molang field can take |
| `diagnostics` | error texts, spans, lints |
| `restricted` | query sets, allow-lists, side-effect-free contexts |
| `host` | a `Host` over a toy world, evaluated through `HostEnv` |
| `variables` | `v.` / `t.` / `c.`, `??`, `->`, public variables |
| `math` | host math functions, pure and volatile |
| `cache` | one text compiled at several versions through `CompileCache` |

## Contributing

See [CONTRIBUTING.md](https://github.com/bedrock-crustaceans/molang/blob/main/CONTRIBUTING.md).

## License

[Apache-2.0](https://github.com/bedrock-crustaceans/molang/blob/main/LICENSE)
