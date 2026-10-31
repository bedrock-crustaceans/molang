# Contributing

Branch from `main`, one change per branch. Open an issue first for anything large.

## Checks

```bash
cargo fmt --all
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked                       # host maths, host floats
cargo test --locked --all-features        # strict-math and platform-quirks on
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
```

CI additionally runs the tests per feature set and clippy per feature, on x86-64 and arm64; the matrix is `.github/workflows/ci.yml`. The unit tests need `stdlib`. CI pins one Rust release (`RUST_TOOLCHAIN` in the workflow), so use it locally when clippy disagrees.

The hostile-input tests allocate a lot. Run them under a cap: `systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 cargo test …`.

The fuzz crate in `fuzz/` is its own package. Its library (a tree-walking second evaluator and a program generator) and tests run on stable; the targets need nightly and `cargo-fuzz`.

```bash
cargo test --locked --manifest-path fuzz/Cargo.toml
cargo test --locked --manifest-path fuzz/Cargo.toml --features platform-quirks
cargo +nightly fuzz run compile_eval -- -max_total_time=600 -timeout=10
```

## Where things are

| Path | What |
| --- | --- |
| `src/compile/` | lexer, parser, semantic passes, codegen |
| `src/vm/` | the bytecode VM, values, hosts, environments |
| `src/stdlib/` | the query and `math.*` tables, hand maintained, and the math library |
| `src/numeric/` | float semantics: `Platform`, the FPU rules (`fpu.rs`), post-ops |
| `src/rng/` | random sources |
| `tests/` | integration tests; `tests/measured_*.rs` hold the rows recorded from the game, `tests/common/` the helpers |
| `tests/*.json`, `tests/server_logs/` | recorded data, replayed by `data_*.rs` and the server runs |
| `fuzz/` | the fuzz crate |

## Floats

Two seams decide the bits:

- `src/stdlib/math/transcendental.rs` calls `std` or, under `strict-math`, `libm`.
- `src/numeric/fpu.rs` has a `native` module (host NaNs, `as i32`) and a `quirks` module (the emulated platforms). Without `platform-quirks` the only platform is `Native`.

Tests follow the same split. A test that asserts a NaN's bits or an `i32::MIN` conversion is `#[cfg(feature = "platform-quirks")]` and has a native counterpart that asserts NaN-ness or Rust's conversion. The recorded server bits replay only with `platform-quirks`; without `strict-math` the transcendentals replay only on the host the logs were taken on (Linux x86-64, glibc 2.41 with FMA) and hold within two ulps elsewhere. A skipped replay prints why. Never widen a tolerance to make a row pass: either the code or the row is wrong.

Two sets of expected failures are listed explicitly and counted: `math.ln` of a negative number on 1.26.52.3, which logs the positive NaN, and under `strict-math` the angles `libm` rounds one ulp low.

### arm64 under emulation

CI runs the suite on an arm64 runner. On an x86-64 machine with `gcc-aarch64-linux-gnu` and `qemu-user`:

```bash
rustup target add aarch64-unknown-linux-gnu
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu"
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export MOLANG_TIME_SCALE=10
cargo test --locked --target aarch64-unknown-linux-gnu --all-features --no-fail-fast
```

`MOLANG_TIME_SCALE` multiplies the wall-clock bounds in `tests/limits.rs`, `MOLANG_HOSTILE_STACK_KIB` the small stack of the hostile-input test. The fuzz crate's `libfuzzer-sys` needs a C++ toolchain; a stub archive stands in, since the targets only have to link:

```bash
mkdir -p target/no-libfuzzer
printf 'int main(void) { return 2; }\n' > target/no-libfuzzer/stub_main.c
aarch64-linux-gnu-gcc -c target/no-libfuzzer/stub_main.c -o target/no-libfuzzer/stub_main.o
aarch64-linux-gnu-ar rcs target/no-libfuzzer/libfuzzer.a target/no-libfuzzer/stub_main.o
CUSTOM_LIBFUZZER_PATH=$PWD/target/no-libfuzzer/libfuzzer.a CUSTOM_LIBFUZZER_STD_CXX=none \
  cargo test --locked --target aarch64-unknown-linux-gnu --manifest-path fuzz/Cargo.toml
```

## Conventions

- **One path per item.** A public item is reachable through its module and nowhere else. No glob re-exports, no `pub extern crate`.
- **Modules with children** are `name/mod.rs`.
- **Unit tests are inline**, one `mod tests` at the end of the file. When a file grows too large, split the code and take the tests along. Integration tests use the public API only.
- **Invalid state is unrepresentable.** Correlated fields become an enum, sentinels become `Option` or a newtype, checks happen once in a constructor.
- **No `unsafe`.** `clippy::pedantic` is on; the allowances in `src/lib.rs` each carry a reason.
- **Comments** explain only what the code cannot: a reason, an invariant, a trap. They do not say where a behaviour comes from, and identifiers and prose do not use the usual name for unmodded behaviour.
- **Message texts** in strings and data files are data. They stay byte for byte.
- **A test asserts specific values**: the bits of an `f32`, the exact message. A behaviour change comes with a row that pins it.

## Rows

A row is one expression with its version, setup and expected result. Rows live in `tests/measured_*.rs` by topic and call the helpers in `tests/common/measured/`. Add the row to the case or group it belongs to, raise the count that case's `check(n)` takes, and run the file. A new server run gets its console log in `tests/server_logs/` and a `replay(<probes>)` test next to the others.

## Content

Never commit game content: pack files, game files, the pack corpus, or text from Mojang's documentation. The corpus tests are ignored and read `MOLANG_CORPUS_DIR`, which stays local.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<scope>): <description>`, with `!` for a breaking change, for example `fix(parse): keep unary minus tighter than addition`. Small commits, formatting separate from behaviour. A pull request says what changed, why, and how it was checked, and updates docs and examples with it.

## Bugs

Give the expression, its version, options and features, what you expected and what you got, and your OS and Rust version.
