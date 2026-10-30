# molang

[![Crates.io](https://img.shields.io/crates/v/molang.svg)](https://crates.io/crates/molang) [![docs.rs](https://img.shields.io/docsrs/molang)](https://docs.rs/molang) [![CI](https://github.com/bedrock-crustaceans/molang/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bedrock-crustaceans/molang/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/bedrock-crustaceans/molang/blob/main/LICENSE)

A pure-Rust compiler and bytecode VM for [Molang](https://learn.microsoft.com/en-us/minecraft/creator/reference/content/molangreference/examples/molangconcepts/molangintroduction), the expression language of Minecraft: Bedrock Edition, at every Molang version. Part of the [bedrock-crustaceans](https://github.com/bedrock-crustaceans) ecosystem.

molang compiles Molang text to bytecode and evaluates it. Queries are declared, not implemented: a `QueryCatalog` holds their names, argument counts, version windows and query sets, so the compiler can tell that `query.does_not_exist` does not resolve, while the implementations come from a `QueryTable` your embedding fills in, one function per query name; a query without one returns the default of its declaration. The crate is not a JSON loader either: a field that holds Molang reaches it through `MolangValueRepr` as a plain `JsonScalar`.

## Installation

molang uses edition 2024 and needs Rust 1.92 or newer and `std`:

```toml
[dependencies]
molang = { version = "1", features = ["vm"] }
```

`vm` is everything needed to compile and evaluate; the other flags are listed under [Features](#features). Without `stdlib` (`default-features = false`) the crate is the engine alone: no `math.*` name and no query is known, so a host declares its queries with `QueryCatalog::new`, its math functions in a `MathCatalog`, and compiles with `CompileOptions::new`.

## Quick start

`compile` never fails: a rejected expression compiles to the constant 0 with its diagnostics attached (`Compiled::expr_or_zero`), and `Compiled::into_result` gives a `Result` instead. `NoHostEnv` evaluates without a world (numbers, strings, `variable.` / `temp.` / `context.`, `math.*`); an embedding with a world uses `HostEnv` (see the [`host`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/host.rs) example).

```rust
use molang::compile::{CompileOptions, compile};
use molang::version::MolangVersion;
use molang::vm::{NoHostEnv, Value, VariableName};

fn main() -> Result<(), molang::compile::CompileError> {
    let options = CompileOptions::server(MolangVersion::LATEST);
    // A success carries what the compile logged (warnings, notes) along with the expression.
    let source = "v.speed * 2 + math.clamp(1 + 2 * 3, 0, 5)";
    let (expr, diagnostics) = compile(source, &options).into_result()?;
    for diagnostic in &diagnostics {
        eprintln!("{diagnostic}");
    }
    assert!(diagnostics.is_empty());

    let mut env = NoHostEnv::new();
    env.variables.set(VariableName::new("speed"), Value::Float(1.5));
    assert_eq!(expr.eval_f32(&mut env.cx()), 8.0);
    Ok(())
}
```

## Reading a result

`Expr::eval` returns a `Value` and the field it came from decides how to read it; `eval_f32` is the plain float reading. The readings the game's fields use are named on `Value`. The ones that truncate to an int take the `numeric::Platform` the expression was compiled for (`Expr::platform`), because out of range or NaN gives `i32::MIN` on a server and saturates (NaN to 0) on Android.

| Field | Reading |
| --- | --- |
| block permutation conditions, block and item event trigger conditions, `bone_visibility` | `rounds_to_nonzero(platform)`: `(int)round(x) != 0`, so 0.4 is false and 0.5 true |
| item and block descriptor `tags`, bool entity-property defaults | `is_nonzero()`: `x != 0` |
| ambient-sound conditions | `exceeds_epsilon()`: `x > 2^-23` |
| anger-sound conditions | `is_positive()`: `x > 0` |
| experience reward entries | `round_to_int(platform)` per entry, which the caller sums |
| int entity-property defaults | `to_property_int(min..=max, platform)`: `clamp(trunc(x + 2^-23), min, max)` |
| float entity-property defaults | `to_property_float(min..=max)`: `x > max ? max : (x > min ? x : min)`, so NaN is `min` |
| animation values, render-controller colours and the other float fields | `to_f32()` |

```rust
use molang::compile::{CompileOptions, compile};
use molang::numeric::Platform;
use molang::version::MolangVersion;
use molang::vm::NoHostEnv;

let options = CompileOptions::server(MolangVersion::LATEST);
let (expr, _) = compile("0.4", &options).into_result().unwrap();
let value = expr.eval(&mut NoHostEnv::new().cx());
assert!(value.is_nonzero());
assert!(!value.rounds_to_nonzero(expr.platform()));

let (expr, _) = compile("1e10", &options).into_result().unwrap();
let value = expr.eval(&mut NoHostEnv::new().cx());
assert_eq!(value.to_property_int(0..=10, Platform::default()), Some(0));
assert_eq!(value.to_property_int(0..=10, Platform::AndroidArm64), Some(10));
```

## Versions and sides

`MolangVersion` covers `-1` (`Invalid`) and `V0` to `V13`. `MolangVersion::from_engine_version_str` maps an engine-version text (`"1.18.10"` is `V5`), and an `EngineVersion` is `"*"` or a `semver::Version`. Queries resolve against the raw version (`RawVersion`): above 13 a text parses like 13 and resolves no query.

A `MolangSource` is text plus version. `compile_source(&source, &options)` compiles it at its own version, and `CompileCache::compile_source` does the same through the cache; `compile(text, &options)` uses the options' version instead.

`CompileOptions::server` and `CompileOptions::client` compile against `stdlib::queries(Side::Server)` and `stdlib::queries(Side::Client)`; there is no default side. For the catalogue of an older release, `stdlib::queries_at(side, &release)` takes a `semver::Version` from the `semver` crate, which this crate does not re-export.

## Queries of your own

Declare them in an extended catalogue, compile against it, and install the functions in a `QueryTable` built from it. `NoHostEnv::new()` and the default `HostEnv` hold no `QueryTable`, so every query returns its declared default. To change a query the catalogue already declares (a standard query's argument count or return type), build a catalogue with `QueryCatalog::overriding`, which replaces each declaration of the same name in place.

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
    let shape = QueryShape {
        args: Arity::exactly(1),
        returns: ReturnType::FLOAT,
        ..QueryShape::DEFAULT
    };
    let catalog = stdlib::queries(Side::Server).extended([QueryDecl::new("query.double", shape)?])?;
    let options = CompileOptions::new(catalog.clone(), MolangVersion::LATEST);
    let (expr, _) = compile("query.double(21)", &options).into_result()?;

    let mut queries = QueryTable::new(&catalog);
    queries.set("query.double", double)?;
    let mut env = NoHostEnv { queries: Some(queries), ..NoHostEnv::new() };
    assert_eq!(expr.eval_f32(&mut env.cx()), 42.0);
    Ok(())
}
```

## Contexts of your own

`context.*` is read-only and filled by the caller for one evaluation. Nothing is declared at compile time: any name compiles, and a read the provider does not answer is a missing variable, which ends the expression with 0 and logs a message unless `??` catches it. `NoHostEnv` and the default `HostEnv` hold a `ContextMap`, filled before each evaluation; `HostEnv::with_context` swaps in any `ContextProvider` of your own, one that reads straight from the host's state for instance (see the [`host`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/host.rs) example), and `NoContext` answers nothing.

```rust
use molang::compile::{CompileOptions, compile};
use molang::version::MolangVersion;
use molang::vm::{ContextName, NoHostEnv, Value};

fn main() -> Result<(), molang::compile::CompileError> {
    let options = CompileOptions::server(MolangVersion::LATEST);
    let (expr, _) = compile("c.damage * 0.5 + (c.bonus ?? 1)", &options).into_result()?;

    let mut env = NoHostEnv::new();
    env.context.set(ContextName::new("damage"), Value::Float(4.0));
    assert_eq!(expr.eval_f32(&mut env.cx()), 3.0);

    env.context.set(ContextName::new("bonus"), Value::Float(2.0));
    assert_eq!(expr.eval_f32(&mut env.cx()), 4.0);
    Ok(())
}
```

## Render-controller arrays

`array.<name>[index]` reads an element of one of a render controller's arrays: its `geometries`, `textures` or `materials`. With `CompileOptions::client` (`ArrayAccess::FromHost`) it compiles to a read at evaluation. The index is truncated toward zero as the platform converts a float to an int, a negative index is 0, and the index wraps by the array's length. An x86-64 platform converts as the servers do, since only the arm64 client's conversion was observed (the crate's choice).

The element is a `Value::Resource`. Math on it is the compile error `MathOnArray` at the root; below the root it is only a diagnostic, the expression compiles and the element reads as 0. `NoHostEnv` and the default `HostEnv` hold an `ArrayTable`; `HostEnv::with_arrays` swaps in any `ArrayProvider`. An unknown array reads as 0 and logs a `molang:` message; so does an empty one (the crate's choice: what an empty array gives is unknown).

A server has no render controllers, so under `CompileOptions::server` (`ArrayAccess::Unresolved`) the compile ends with `CompileOutcome::UsesArrays` and no program. A bare `array.<name>` does so under either.

```rust
use molang::compile::{CompileOptions, compile};
use molang::version::MolangVersion;
use molang::vm::{ArrayName, NoHostEnv, ResourceRef, Value, VariableName};

fn main() -> Result<(), molang::compile::CompileError> {
    let options = CompileOptions::client(MolangVersion::LATEST);
    let (expr, _) = compile("array.skins[v.variant]", &options).into_result()?;

    let mut env = NoHostEnv::new();
    let skins = [ResourceRef::new("texture.default"), ResourceRef::new("texture.red")];
    env.arrays.set(ArrayName::new("skins"), skins);
    env.variables.set(VariableName::new("variant"), Value::Float(3.0));
    assert_eq!(expr.eval(&mut env.cx()), Value::Resource(skins[1]));
    Ok(())
}
```

## Math functions of your own

Declare them in a `MathCatalog` and pass it with `CompileOptions::with_math`; `math.<name>(…)` then calls them like the standard functions, with float arguments (at most `MAX_MATH_ARGS`) and the argument count checked at compile time. A pure function must be deterministic: a call of constants is folded at compile time and equal calls are merged. A volatile function gets the evaluation's random source (`&mut dyn RandomSource`; its `sample()` draws a sample as the standard functions do), is never folded or merged, and is forbidden by `OpSet::without_assignments_or_random`. A function must not panic. A function may take a standard function's name (all but the constant `math.pi`) and then replaces the standard function while the catalogue is in the options, as an ordinary host call.

```rust
use molang::catalog::{Arity, MathCatalog, MathDecl};
use molang::compile::{CompileOptions, compile};
use molang::version::MolangVersion;
use molang::vm::NoHostEnv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let math = MathCatalog::new([
        MathDecl::pure("math.hypot", Arity::exactly(2), |_, a| a[0].hypot(a[1]))?,
        MathDecl::volatile("math.jitter", Arity::exactly(1), |_, rng, a| a[0] * rng.sample())?,
    ])?;
    let options = CompileOptions::server(MolangVersion::LATEST).with_math(Some(math));
    let hypot = compile("math.hypot(3, 4)", &options);
    assert_eq!(hypot.expr().and_then(|e| e.as_constant()), Some(5.0));
    let (expr, _) = compile("math.jitter(10) + math.hypot(v.x ?? 3, 4)", &options).into_result()?;
    assert!((5.0..=15.0).contains(&expr.eval_f32(&mut NoHostEnv::new().cx())));
    Ok(())
}
```

## Features

`stdlib` is on by default. Every public item has one path, the module this table names (`molang::compile::CompileOptions`); [docs.rs](https://docs.rs/molang) shows the feature each item needs.

| Feature | Contents | Implies |
| --- | --- | --- |
| (always) | `version`: `MolangVersion`, `RawVersion`, `EngineVersion` and the engine-version map, experiments; `json`: `MolangSource`, `MolangValueRepr` and its readers; `hash`: `HashedStr`; `catalog`: the query catalogue and declaration types; `ops`: the operator table and `OpSet` | |
| `stdlib` | `stdlib`: the queries (`stdlib::queries`, `stdlib::query`), the `math.*` table (`MathFn`) and, with `compiler`, the math library (`stdlib::math`) and `CompileOptions::server` / `client` | |
| `compiler` | `compile`: `compile`, `CompileOptions`, `Compiled`, `Expr`, `ProgramFlags` and the limits (`MAX_DEPTH`, `MAX_SOURCE_LEN`, `MAX_DIAGNOSTICS`); `diag`: `Diagnostic` and `LanguageMessage`; `numeric`: the float semantics; `rng`: the random sources and the mappings of a generator word; `catalog::MathCatalog`: host math functions | |
| `vm` | `vm`: `Value`, the `Host` traits, variable storage and `EvalLimits`; `Expr::eval` | `compiler` |
| `cache` | `cache::CompileCache`, a concurrent compile cache keyed by version, options and source | `compiler` |
| `facet` | `Facet` impls for `json::MolangSource` and `json::MolangValueRepr` | |
| `fuzz` | test support, **no SemVer guarantee**: a hidden `internals` module read by the repository's fuzz crate | `stdlib` |

## Compile options

Start from `CompileOptions::new`, `server` or `client` and change a setting with its `with_*` setter (`with_catalog`, `with_deviations`, …); each setting has a getter of the same name.

| Setting | Meaning | Default |
| --- | --- | --- |
| `catalog` | The `QueryCatalog` queries resolve against. | Given to the constructor. |
| `raw_version` | The `RawVersion` queries resolve with; parser gates use its effective value. | Given to the constructor. |
| `admission` | Which queries of the catalogue may be called: query sets or an allow-list. | The `default` query set. |
| `allowed_ops` | The `OpSet` of allowed operations. | All. |
| `experiments` | The enabled experiments (`ExperimentMask`). No standard query is experiment-gated. | None. |
| `math` | The host's own `math.*` functions (`MathCatalog`). | None. |
| `keep_source` | Whether the compiled expression keeps the source text. | `false` |
| `deviations` | The compile switches below. | `Deviations::DEFAULT` |

`Deviations` is a bit set; `Deviations::NONE` turns every flag off.

| Flag | On | Off | Default |
| --- | --- | --- | --- |
| `SOURCE_LENGTH_LIMIT` | A source longer than `MAX_SOURCE_LEN` (65,536 bytes) is rejected with `DiagCode::SourceTooLong`. | No length limit. | on |
| `VALIDATE_NESTED` | Validation findings below the root are warnings. | They are errors. Either way the expression compiles. | on |
| `TRUE_FALSE_PREFIX_ADVANCE` | A prefix of `true` / `false` (`t`, `tr`, `fals`) is a boolean token and the lexer advances by its length. | The lexer advances four bytes for `true`, five for `false`. | on |
| `QUERY_ARITY_LINT` | A warning for a query call with an argument count outside its declared range. | No warning. | on |
| `QUERY_CLIENT_ONLY` | An informational note for a client-only query compiled against a server catalogue. | No note. | on |
| `OBJECT_VERSION_WARNING` | A warning for an object-form version outside -1..=13, a note for `Invalid`. | No diagnostic. | on |
| `DIAGNOSTIC_LIMIT` | At most `MAX_DIAGNOSTICS` (256) diagnostics, then one `DiagCode::DiagnosticLimit` note counting the rest. | Every diagnostic is kept. | on |

```rust
use molang::compile::{CompileOptions, Deviations};
use molang::version::MolangVersion;

let options = CompileOptions::server(MolangVersion::LATEST)
    .with_keep_source(true)
    .with_deviations(Deviations::DEFAULT.difference(Deviations::VALIDATE_NESTED));
assert_eq!(options.version(), MolangVersion::LATEST);
```

## Float behaviour

The host chooses whose float behaviour an expression reproduces with `CompileOptions::with_platform`, a `numeric::Platform`: `BdsLinux` (by release), `BdsWindows` or `AndroidArm64`. The default is `BdsLinux` at its latest release. The compiled expression records it, constant folding and evaluation both follow it, and the same expression gives the same bits on every host.

```rust
use molang::compile::{CompileOptions, compile};
use molang::numeric::Platform;
use molang::version::MolangVersion;

let options = CompileOptions::server(MolangVersion::LATEST).with_platform(Platform::BdsWindows);
let expr = compile("math.copy_sign(1, math.ln(-1))", &options).into_result().unwrap().0;
assert_eq!(expr.platform(), Platform::BdsWindows);
assert_eq!(expr.as_constant(), Some(-1.0));
```

Every platform rounds each multiply, add and divide on its own in formula order, makes every comparison with a NaN false and returns the second operand from `math.min` / `math.max` when either is NaN. The servers return the left NaN operand of an arithmetic operation quietened and `0xffc00000` for an invalid one; Android returns the first signalling NaN operand quietened, else the first quiet one, `0x7fc00000` for an invalid one, and saturates a float → int conversion (`loop` and die counts, array indices) with NaN as 0 where the servers give `i32::MIN`. The NaN of an invalid standard function differs too: `math.ln` of a negative number changed sign between Linux releases. The transcendental functions are glibc's single-precision ones on Linux. Android takes glibc's algorithms for `sin`, `cos`, `exp`, `exp2`, `ln` and `pow`, which its C library shares, and FreeBSD's for `asin`, `acos`, `atan` and `atan2`, which its C library builds from: a best effort, not checked against a device. On Windows they are evaluated in `f64` and rounded once. `numeric::Platform` lists every difference.

The arithmetic and the standard functions choose the sign and payload of the NaNs they return, so those are the same on every host too. On 32-bit x86 a value returned through the x87 register may have a signalling NaN quietened; 32-bit x86 without SSE2 does not compile.

## Evaluation

`EvalLimits` budgets one evaluation. Each budget is an `Option` (`None` is no limit); `EvalLimits::DEFAULT` is the column below and `EvalLimits::NONE` sets none. Keep a `total_steps` budget for untrusted content: a member store pays steps for the structs it copies, so the step budget bounds memory too. The operand stack is capped at 65,536 slots under every budget.

| Field | Meaning | Default |
| --- | --- | --- |
| `loop_iterations` | Iterations one `loop` / `for_each` may run before it is left. | 1,024 |
| `total_steps` | Steps one evaluation may take before it ends with 0. | 1,048,576 |
| `struct_depth` | Struct levels a member store may leave in the variable or temp it writes; a deeper store ends the evaluation with 0. | 32 |
| `struct_members` | Members a member store may leave in a struct it adds a member to; a wider store ends the evaluation with 0. | 256 |
| `query_depth` | How deep query arguments may nest at run time (the top level is 0); `Some(0)` lets no query evaluate an argument. | None |

```rust
use molang::vm::EvalLimits;

let limits = EvalLimits {
    loop_iterations: Some(5), // a tighter per-loop budget
    struct_depth: None,       // no struct depth budget
    query_depth: Some(0),     // no query may evaluate an argument
    ..EvalLimits::DEFAULT
};
assert!(!limits.is_unlimited());
```

`temp.*` starts empty in every evaluation; `temps: Temps::Kept(TempMap::new())` on a `HostEnv` or `NoHostEnv` keeps them in a map nothing clears.

The random source is any `rng::RandomSource`: each draw of `math.random*` and `math.die_roll*` is one `sample()`, and the source decides how its generator's word becomes the sample. A closure `FnMut() -> f32` is a source. `HostEnv` and `NoHostEnv` draw from `vm::ProcessRng` unless `with_rng` gives them another. Which source to use depends on where the expression runs:

| Where | Generator | Sample from one 32-bit word `u` | Source |
|---|---|---|---|
| Entities, items, blocks, commands, scripting, camera, and `math.random` in particles | one xorshift128 for the whole process, from the standard seeds | `rng::unit_31(u)`: `(u & 0x7fff_ffff) · 2^-31` in `f32` | `ProcessRng`, the default |
| The UI player preview (paper doll) | none: a constant | 0 | `FixedRng::ZERO` |
| A feature rule's expressions | MT19937 seeded per chunk | `rng::unit_32(u)`: `u · 2^-32` in `f64`, then rounded to `f32` | the host's MT19937, mapped with `unit_32` |
| A scatter feature placed by a feature rule | MT19937 seeded per rule, shared with the scatter's own placement draws | as above | as above |
| Expressions evaluated directly by sapling growth, `/place feature` and other helper-placed features | MT19937 per thread, seeded from the system's random device | as above | as above |

Both mappings can give exactly 1. The crate has no MT19937 of its own; a host seeds one (the `rand_mt` crate's `Mt`, say) as follows:

- **Chunk seed.** With `S` the level seed as 32 bits, take `a` and `b`, the first two words of `MT(S)` each shifted right by 1 and with the low bit set; the chunk `(cx, cz)` seeds `MT((cx·a + cz·b) ^ S)` in wrapping 32-bit arithmetic. That generator drives the `math.random` of every rule in the chunk's feature-rule pass, one state across the rules.
- **Rule seed.** Each rule seeds `MT(((s << 6) + (s >> 2) + h + 0x9e3779b9) ^ s)`, wrapping, with `s` the chunk generator's seed (not its state) and `h` the low 32 bits of the rule identifier's hash. It draws the rule's chance, iterations and coordinate distributions; when the iterations are more than 0, the placed scatter feature gets a second generator, a fresh one of the same seed.
- **Structure start.** A structure start seeds as the chunk does and then discards one word.

The host supplies `h`, and decides how a 64-bit level seed becomes `S`: taking its low 32 bits is this crate's assumption, not established. `/place featurerule` runs the feature-rule pass with the level's generator in place of the per-chunk one.

```rust
use molang::compile::{CompileOptions, compile};
use molang::rng::unit_32;
use molang::version::MolangVersion;
use molang::vm::NoHostEnv;
use rand_mt::Mt;

/// The seed of a chunk's feature-rule pass.
fn chunk_seed(level_seed: u32, chunk_x: i32, chunk_z: i32) -> u32 {
    let mut level = Mt::new(level_seed);
    let a = (level.next_u32() >> 1) | 1;
    let b = (level.next_u32() >> 1) | 1;
    (chunk_x.cast_unsigned().wrapping_mul(a)).wrapping_add(chunk_z.cast_unsigned().wrapping_mul(b))
        ^ level_seed
}

/// The seed of one rule, from the chunk's seed and the identifier hash's low 32 bits.
fn rule_seed(chunk_seed: u32, hash: u32) -> u32 {
    let s = chunk_seed;
    ((s << 6).wrapping_add(s >> 2).wrapping_add(hash).wrapping_add(0x9e37_79b9)) ^ s
}

let seed = chunk_seed(12_345, 3, -7);
assert_eq!((seed, rule_seed(seed, 0xf7a7_6dfd)), (1_197_795_863, 3_348_516_332));

let (expr, _) = compile("math.random(0, 10)", &CompileOptions::server(MolangVersion::LATEST))
    .into_result()
    .unwrap();
let mut chunk = Mt::new(seed);
let mut source = || unit_32(chunk.next_u32());
let mut env = NoHostEnv::new();
let mut cx = env.cx();
cx.rng = &mut source;
let value = expr.eval_f32(&mut cx);
assert!((0.0..=10.0).contains(&value));
```

The default message sink, `BoundedSink`, keeps the last 64 messages, each cut to 1 KiB; `LogOnce` drops repeats.

The deepest nesting `compile` accepts needs about 320 KiB of stack in a release build and 1.9 MiB in a debug build; evaluate on threads of at least 512 KiB / 4 MiB, or set `EvalLimits::query_depth`. Structs share members, so walk a stored value with `Value::distinct_structs` or bound the walk.

## Examples

Every example prints what it demonstrates. Start with `hello_world`: `cargo run --example hello_world --features vm`.

| Example | What it covers | Features |
| --- | --- | --- |
| [`hello_world`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/hello_world.rs) | Compile and evaluate with no host; variables, a loop and a missing read. | `vm` |
| [`versions`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/versions.rs) | `MolangVersion::from` an `EngineVersion`, and one text grouped differently at versions 4, 5 and 6. | `compiler` |
| [`json_forms`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/json_forms.rs) | `MolangValueRepr` round trip of the number, string, `{expression, version}` and bool forms. | none |
| [`diagnostics`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/diagnostics.rs) | Error texts, reject versus keep, byte spans, and the crate's own lints. | `compiler` |
| [`restricted`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/restricted.rs) | Query sets, allow-lists and the side-effect-free contexts (`without_assignments`, `without_assignments_or_random`). | `compiler` |
| [`host`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/host.rs) | Implementing `Host`, `VariableStore`, `ContextProvider` and queries over a toy world, evaluated through `HostEnv`. | `vm` |
| [`variables`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/variables.rs) | `v.` / `t.` / `c.` scope, `??`, the missing-variable abort, `->` and public variables. | `vm` |
| [`math`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/math.rs) | Host math functions in a `MathCatalog`: a pure one folded, a volatile one drawing, the argument-count error. | `vm` |
| [`cache`](https://github.com/bedrock-crustaceans/molang/blob/main/examples/cache.rs) | `CompileCache` compiling one text at several versions. | `cache` |

## Contributing

See [CONTRIBUTING.md](https://github.com/bedrock-crustaceans/molang/blob/main/CONTRIBUTING.md).

## License

Licensed under the **Apache License 2.0**; see [LICENSE](https://github.com/bedrock-crustaceans/molang/blob/main/LICENSE).
