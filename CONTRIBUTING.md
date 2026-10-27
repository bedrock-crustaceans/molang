# Contributing to molang

Fork the repository, branch from `main`, keep a branch to one change, and discuss larger changes in an issue first.

## Building and testing

```bash
cargo fmt --all
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
```

CI also runs the tests per feature set (each integration test gates itself on the features it needs; all but `engine_only.rs` need `stdlib`):

```bash
cargo test --no-default-features --features stdlib
cargo test --no-default-features --features stdlib,compiler
cargo test --no-default-features --features stdlib,vm
cargo test --no-default-features --features stdlib,cache
cargo test --no-default-features --features stdlib,facet
cargo test --no-default-features --features fuzz
cargo test --no-default-features --features compiler --test engine_only
cargo test --no-default-features --features vm --test engine_only
```

and clippy with each feature alone (`--no-default-features --features <feature>`) and with `compiler,vm` without `stdlib`. The unit tests need `stdlib`. CI pins one Rust release (`RUST_TOOLCHAIN` in `.github/workflows/ci.yml`), since clippy's lints change between releases; use it locally when clippy disagrees with CI. The minimum supported version is `rust-version` in `Cargo.toml`.

Hostile-input tests can allocate a lot; run them under a memory cap, for example `systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 cargo test …`.

### The glibc oracle

`src/stdlib/math/glibc/` reproduces glibc's single-precision functions bit for bit, as glibc 2.41 computes them on an x86-64 CPU with FMA. `sinf`, `cosf`, `expf`, `exp2f`, `logf`, `log2f` and `powf` are ports of the build glibc selects there, whose fused multiply-adds each port mirrors with `mul_add`; `atan2f` is a port of its correctly rounded code. `asinf`, `acosf` and `atanf` are correctly rounded too, so they take the `f64` functions rounded once, with a table of the four arguments where that rounding is off. The unit tests pin reference results on every host, among them arguments that glibc's build without FMA rounds the other way.

`tests/glibc_oracle.rs` (features `fuzz` and `compiler`) compares them with the host's own glibc. It detects drift against one glibc build and is not a regression test: the file is gated on x86-64 Linux with glibc, and its tests skip, saying why, unless the host runs glibc 2.41 on a CPU with FMA. glibc 2.40 and earlier compute the inverse trigonometric functions differently, and glibc's build without FMA rounds 12 `sinf` and 22 `cosf` arguments the other way.

An ignored test sweeps every `f32` through each one-argument function and 2³² random pairs through `powf` and `atan2f`; `MOLANG_EXHAUSTIVE` limits it to a comma-separated list of names:

```bash
cargo test --release --features fuzz,compiler --test glibc_oracle -- --ignored
MOLANG_EXHAUSTIVE=sinf,powf cargo test --release --features fuzz,compiler --test glibc_oracle -- --ignored
```

### The fuzz crate

`fuzz/` (`molang-fuzz`, not published) is a separate package with its own empty `[workspace]`. It depends on `molang` with the features `vm`, `fuzz` and `stdlib` and reads crate-private items through the hidden `molang::internals` module. Its library (`fuzz/src/`) holds `tree_walker`, a second evaluator that walks the optimised tree, and `generator`, the program generator. The library and its tests run on stable; running a fuzz target needs nightly and `cargo-fuzz`.

```bash
cargo fmt --manifest-path fuzz/Cargo.toml --all
cargo test --locked --manifest-path fuzz/Cargo.toml
cargo clippy --locked --manifest-path fuzz/Cargo.toml --all-targets -- -D warnings
cargo +nightly fuzz run compile_eval -- -max_total_time=600 -timeout=10
```

The targets (`fuzz/fuzz_targets/`) are `lex_parse` (arbitrary text through the front end), `compile_eval` (generated programs on the VM and the tree walker) and `repr_roundtrip` (the serialised forms).

## Platforms

The float behaviour is a run-time `numeric::Platform` in the compile options, the same on every host. The primitives with the processor's NaN rules and float → int conversion are in `src/numeric/fpu.rs`, the C-library seam (which body and which invalid NaN) in `src/stdlib/math/cmath.rs`, and the formula shapes in `src/stdlib/math/formula/`; each takes the platform as its first argument. Unit tests check every platform through `numeric::test_support::PLATFORMS`, integration tests through `tests/common/mod.rs`'s `PLATFORMS`. No row and no `#[test]` is behind a `cfg`, and every host runs the same checks:

- `tests/data_bits.rs`: BDS Linux 1.26.36 reproduces the `X86_64` set of `tests/numeric_bits.json` to the bit, 1.26.52 too with the `ln` rows of negative numbers against `0x7fc00000`; the other platforms pin the functions where they differ from it.
- Server runs replay each release on its own platform and every probe on the other platforms, which must log other lines exactly on the probes `ServerRun::differs_on` lists.
- Run rows (`RunGroup`): the rows a platform lists in `RunGroup::differs_on` differ there and the rest hold.
- `tests/data_parse.rs`: every row on every platform; the rows of `TREE_DIFFERS` check their outcome and messages only.

CI runs the whole suite on an arm64 runner too, which must give the same bits as on x86-64. An x86-64 machine runs it under user-mode emulation, with a cross linker and `qemu-aarch64` (`gcc-aarch64-linux-gnu`, `qemu-user`):

```bash
rustup target add aarch64-unknown-linux-gnu
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export MOLANG_TIME_SCALE=10
cargo test --locked --target aarch64-unknown-linux-gnu --all-features --no-fail-fast
```

The all-features run takes several minutes, half of it `tests/limits.rs`. Everything is uniformly slower under the emulator: `MOLANG_TIME_SCALE` multiplies every wall-clock bound in `tests/limits.rs`, and `MOLANG_HOSTILE_STACK_KIB` the 512 KiB stack of `hostile_input_compiles_on_a_small_stack`. Doc tests and examples run under the emulator as well.

The fuzz crate's `libfuzzer-sys` builds C++, which needs an aarch64 C++ toolchain; its three `#![no_main]` targets only need to link, so a stub archive stands in for libFuzzer:

```bash
mkdir -p target/no-libfuzzer
printf 'int main(void) { return 2; }\n' > target/no-libfuzzer/stub_main.c
aarch64-linux-gnu-gcc -c target/no-libfuzzer/stub_main.c -o target/no-libfuzzer/stub_main.o
aarch64-linux-gnu-ar rcs target/no-libfuzzer/libfuzzer.a target/no-libfuzzer/stub_main.o
CUSTOM_LIBFUZZER_PATH=$PWD/target/no-libfuzzer/libfuzzer.a CUSTOM_LIBFUZZER_STD_CXX=none \
  cargo test --locked --target aarch64-unknown-linux-gnu --manifest-path fuzz/Cargo.toml
```

Its build script reruns when the archive changes, not when `CUSTOM_LIBFUZZER_STD_CXX` does: `touch` the archive after changing only that. Without a cross linker, the arm64 build still compiles and lints (nothing is linked, so the C and C++ that two dependencies build are skipped):

```bash
export CARGO_FEATURE_PURE=1 CUSTOM_LIBFUZZER_PATH=$PWD/target/no-libfuzzer/libfuzzer.a
cargo clippy --locked --target aarch64-unknown-linux-gnu --all-targets --all-features -- -D warnings
cargo clippy --locked --target aarch64-unknown-linux-gnu --manifest-path fuzz/Cargo.toml --all-targets -- -D warnings
```

## Code conventions

- **Modules.** A module with child files is `name/mod.rs`, never `name.rs` beside `name/`.
- **One path per item.** Every public item has exactly one public path, its module's: the crate root declares modules only; a `pub use` names a whole external crate or an item of a private module, each once and never by a glob, and there is no `pub extern crate`. A `pub type` may name an instantiation of a public generic type (`vm::TempName` is `Name<Temp>`) or a `dyn` type; it may not rename a type, and no two aliases may name the same type.
- **Unit tests** are inline: one `#[cfg(test)] mod tests { … }` as the last item of the file it tests; no `tests.rs`, no out-of-line test modules. When a file grows too large, split the code and move each test with it. Helpers shared across a module's files go in an inline `#[cfg(test)] pub(crate) mod test_support` of its `mod.rs`; support shared more widely is `src/reference_catalog.rs`, also read by the fuzz crate. Unit tests and the seam files under `src/` read no files (the package ships only `src/`).
- **Invalid state is unrepresentable.** Fields that must agree become an enum whose variants carry what each state has; no in-band sentinels (`i16::MIN`, `u32::MAX`); a flag that makes other fields meaningless is a variant without them; values with different meanings get different types; a check the type cannot express is made once, in a constructor returning `Result` or `Option`. A review treats a representable invalid state as a defect.
- **Tables.** The query, operator and math tables in `src/stdlib/{query_table,math_fn}.rs` and `src/ops/table.rs` are maintained by hand; `tests/operator_rules.rs` and `tests/public_api.rs` pin their invariants.
- **No `unsafe`**, and `clippy::pedantic` with only the allowances listed in `src/lib.rs`, each with its reason.

## Comments and wording

Comments explain only non-obvious behaviour: a reason, an invariant, an ordering constraint, a trap, a unit or limit. They state what the code does, in as few words as possible, and never say where a behaviour comes from. Identifiers and prose do not use the usual name for unmodded behaviour. Message texts in strings and data files are data and stay byte for byte.

## Tests

A test asserts specific values (the bits of an `f32`, the exact message text). Something reachable through the public API gets an integration test, and the private logic behind it a unit test. Integration tests use the public API only; shared helpers live in `tests/common/`. A behaviour change comes with a row that pins it. Tolerances are never widened to make a test pass: when a row fails, the code or the row is wrong.

| File | What it holds |
| --- | --- |
| `compile.rs` | compile results, folded constants, options |
| `diagnostics.rs` | messages and diagnostics |
| `evaluation.rs` | values and control flow |
| `numerics.rs` | the math library and the two float behaviours |
| `host.rs` | embedding: queries, sinks, variables |
| `host_math.rs` | host math functions: declaration, folding, merging, evaluation |
| `limits.rs` | evaluation budgets and hostile input |
| `cache.rs` | the compile cache |
| `serialised_forms.rs` | `MolangValueRepr` and `MolangSource` forms |
| `public_api.rs` | the public surface |
| `properties.rs` | property tests without the program generator |
| `measured_variables.rs`, `measured_arithmetic.rs`, `measured_math.rs`, `measured_rounding_and_random.rs`, `measured_logic.rs`, `measured_control_flow.rs`, `measured_queries.rs`, `measured_parsing.rs` | rows by topic |
| `measured_declarations.rs` | rows the query declarations alone decide; runs with `stdlib` alone |
| `data_parse.rs` | `tests/parse_vectors.json` |
| `data_bits.rs` | `tests/numeric_bits.json` |
| `data_sweeps.rs` | `tests/server_sweeps.json` |
| `operator_rules.rs` | the operator table's invariants |
| `corpus.rs` | the pack-corpus tests (ignored) |
| `alloc_free.rs` | allocation-free evaluation (own allocator) |
| `compile_memory.rs` | memory and time of a hostile compile (own allocator) |
| `engine_only.rs` | the engine with host catalogues; passes with and without `stdlib` |

The fuzz crate's tests (helpers in `fuzz/tests/common/`):

| File | What it holds |
| --- | --- |
| `fuzz/tests/tree_walker.rs` | the VM against the tree walker: unmodelled jumps, budgets, the pack corpus (ignored) |
| `fuzz/tests/properties.rs` | property tests over generated programs |
| `fuzz/tests/recorded.rs` | the VM against the tree walker on the data files of `tests/` |

### Adding a row

A row is one input (expression, version, optional setup) with its expected result. Rows in `tests/measured_*.rs` are statements calling the helpers of `tests/common/measured/` (or `tests/common/declared.rs` for `VersionWindowCase` and `AllowListCase`). To add one:

1. Add it to the case or group it belongs to in its topic file (`EvalCase`, `RunGroup`, `ParseGroup`, `LoopCapGroup`, `SmokeGroup`, a `ServerRun` probe), or start a new one with an id no other row has.
2. Raise the count its terminal call takes (`case.check(3)`, `group.check(29)`, `run_19().replay(9)`): a list row (`is_constant`, `all_parse`, `evaluates_to`, `fails_evaluation`, `has_disallowed_queries`) counts one per item, every other row one; setup steps and case-level calls do not count.
3. A `parse_fails` row whose first message is about a string operand or an unresolved query states it with `.because(ParseFailure::…)`.
4. An `EvalCase` with no setup whose `eval` rows are all at the latest version carries `case.also_on_a_fresh_state()`; drop it when the case gains setup.
5. A new `ServerRun` (`fn run_NN()`) gets a `#[test]` in the same file calling `run_NN().replay(<probes>)`, and its console logs in `tests/server_logs/`. A new run in `tests/server_sweeps.json` gets a `replay(NN, <probes>)` test in `tests/data_sweeps.rs`, and `the_sweep_file_is_complete` its new counts.

Run the topic file's tests.

## Content

Never commit Minecraft content: pack files, game files, the pack corpus, or text copied from Mojang's documentation. The corpus tests (`tests/corpus.rs`, ignored) read `MOLANG_CORPUS_DIR/corpus_pairs.json`, built locally and never committed; keep the corpus strings out of tests, comments and commit messages.

## Commits and pull requests

Commit subjects follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<scope>): <description>`, with the type one of `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `build`, `ci`, `chore` and the scope the module or area touched, for example `fix(parse): keep unary minus tighter than addition`. Keep commits small and do not mix formatting with behaviour changes.

A pull request says what changed, why, how it was validated, and any breaking change; it links its issues (`Closes #123`). Update the documentation and examples when behaviour changes.

## Reporting bugs

Give the Molang expression and its context (version, options, features), the expected and actual result, your OS and Rust version, and a minimal example if possible.
