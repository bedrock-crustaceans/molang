//! A pure Rust implementation of the Molang expression language of the 1.26 releases: grammar,
//! version gates, the standard library's queries and functions, error model and messages.
//!
//! Features (`stdlib` is the default):
//! - `stdlib`: the standard library, the `math.*` functions and the queries a script can name out
//!   of the box, and [`CompileOptions::server`] / [`client`][`CompileOptions::client`]. Without it
//!   no `math.*` name is known and every query comes from a host catalogue.
//! - `compiler`: the modules [`compile`][compile-mod], [`diag`], [`numeric`] and [`rng`], host math
//!   functions ([`catalog::MathCatalog`]) and, with `stdlib`, [`stdlib::math`].
//! - `vm`, `cache`: the modules of the same name ([`vm`], [`cache`]; each implies `compiler`).
//! - `facet`: `Facet` impls for [`json::MolangSource`] and [`json::MolangValueRepr`]; the optional
//!   `facet` dependency is a pre-release and is not re-exported.
//! - `strict-math`: the transcendental `math.*` functions from the `libm` crate, the same bits on
//!   every host, in place of the host's C library.
//! - `fuzz`: a hidden `internals` module for the repository's fuzz crate, with no `SemVer`
//!   guarantee (implies `stdlib`).
//!
//! Every public item has one path: the module it lives in. Modules, with the feature a module or
//! part of it needs:
//! - [`catalog`]: what a script may call and how a host declares it: [`QueryCatalog`],
//!   [`QueryDecl`], [`QueryShape`], [`Side`], and host `math.*` functions in a
//!   [`MathCatalog`][`catalog::MathCatalog`] (`compiler`).
//! - [`ops`]: the operator table and the sets of operations a compilation allows ([`OpSet`]).
//! - [`version`]: what content targets: [`MolangVersion`], [`RawVersion`], [`EngineVersion`], the
//!   game release (a `semver::Version`) and experiments ([`ExperimentMask`]).
//! - [`json`]: Molang fields of JSON documents: [`MolangSource`], [`MolangValueRepr`] and their
//!   readers.
//! - [`hash`]: the string hash a Molang string value is ([`HashedStr`]).
//! - [`stdlib`] (`stdlib`): the standard `math.*` functions and queries ([`queries`], [`query`]).
//! - [`compile`][compile-mod] (`compiler`): [`compile`], [`compile_source`], [`CompileOptions`],
//!   [`Compiled`], [`Expr`], [`ProgramFlags`] and the limits [`MAX_DEPTH`], [`MAX_SOURCE_LEN`]
//!   and [`MAX_DIAGNOSTICS`].
//! - [`diag`] (`compiler`): [`Diagnostic`], [`DiagCode`], [`Severity`] and [`LanguageMessage`].
//! - [`numeric`], [`rng`] (`compiler`): the float semantics and the random sources.
//! - [`vm`] (`vm`): evaluation, [`Value`], the host traits, variable storage and [`EvalLimits`].
//! - [`cache`] (`cache`): [`CompileCache`].
//!
//! The library does not implement queries. Every `query.*` call resolves at compile time against a
//! [`QueryCatalog`] ([`stdlib::queries`] plus the host's own declarations) and runs through the
//! host's [`vm::QueryTable`]. A host adds `math.*` functions of its own in a
//! [`catalog::MathCatalog`] passed with [`CompileOptions::with_math`].
//!
//! # Compiling pack content
//!
//! [`compile_source(&source, &field_options)`][`compile_source`] compiles a [`MolangSource`] at the
//! source's version; [`compile(text, &options)`][`compile`] uses the options' version.
//!
//! The catalogue's [`Side`] has no default. [`CompileOptions::server`] leaves a query the server
//! catalogue lacks (`query.is_on_screen`) unresolved and lints a client-only query; resource-pack
//! Molang compiles with [`CompileOptions::client`].
//!
//! # Quick start
//!
//! [`compile`] never fails: a rejected expression is the constant 0 with its [`Diagnostic`]s
//! attached, and [`Compiled::into_result`] turns that into a `Result`. [`NoHostEnv`] evaluates
//! without a world: numbers, strings, `variable.` / `temp.` / `context.` values and `math.*`.
//!
//! ```
//! # #[cfg(all(feature = "vm", feature = "stdlib"))]
//! # {
//! use molang::compile::{CompileOptions, compile};
//! use molang::version::MolangVersion;
//! use molang::vm::{NoHostEnv, Value, VariableName};
//!
//! let options = CompileOptions::server(MolangVersion::LATEST);
//! // A success carries what the compile logged (warnings, notes) along with the expression.
//! let (expr, diagnostics) =
//!     compile("v.speed * 2 + math.clamp(1 + 2 * 3, 0, 5)", &options).into_result()?;
//! for diagnostic in &diagnostics {
//!     eprintln!("{diagnostic}");
//! }
//! assert!(diagnostics.is_empty());
//!
//! let mut env = NoHostEnv::new();
//! env.variables_mut().set(VariableName::new("speed"), Value::Float(1.5));
//! assert_eq!(expr.eval_f32(&mut env.cx()), 8.0);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! For untrusted content keep the default [`vm::EvalLimits`] and bounded sink, seed each
//! environment's random source, and evaluate on a thread with the stack [`Expr::eval`] documents.
//!
//! # Float behaviour
//!
//! The host chooses whose float behaviour an expression reproduces: [`compile::CompileOptions`]
//! holds a [`numeric::Platform`] (BDS Linux, BDS Windows or the Android arm64 client), the compiled
//! expression records it, and folding and evaluation follow it. The platforms differ in the NaN an
//! operation returns and in float → int conversion; [`numeric::Platform`] lists the differences.
//!
//! The transcendental functions (`math.sin`, `math.pow`, the easings' `exp2` and sine table, …)
//! call the host's C library through `std`, as the game calls the one of the system it runs on, so
//! the last bits can differ between hosts: the recorded BDS Linux data is reproduced bit for bit on
//! a Linux x86-64 host with glibc 2.41 and FMA, and on other hosts within two ulps. With
//! `strict-math` they call the `libm` crate instead and give the same bits on every host.
//!
//! [`QueryCatalog`]: catalog::QueryCatalog
//! [`QueryDecl`]: catalog::QueryDecl
//! [`QueryShape`]: catalog::QueryShape
//! [`Side`]: catalog::Side
//! [`OpSet`]: ops::OpSet
//! [`MolangVersion`]: version::MolangVersion
//! [`RawVersion`]: version::RawVersion
//! [`EngineVersion`]: version::EngineVersion
//! [`ExperimentMask`]: version::ExperimentMask
//! [`MolangSource`]: json::MolangSource
//! [`MolangValueRepr`]: json::MolangValueRepr
//! [`HashedStr`]: hash::HashedStr
#![cfg_attr(
    feature = "compiler",
    doc = "[compile-mod]: compile
[`compile`]: compile::compile
[`compile_source`]: compile::compile_source
[`CompileOptions`]: compile::CompileOptions
[`compile::CompileOptions`]: compile::CompileOptions
[`CompileOptions::with_math`]: compile::CompileOptions::with_math
[`Compiled`]: compile::Compiled
[`Compiled::into_result`]: compile::Compiled::into_result
[`Expr`]: compile::Expr
[`ProgramFlags`]: compile::ProgramFlags
[`MAX_DEPTH`]: compile::MAX_DEPTH
[`MAX_SOURCE_LEN`]: compile::MAX_SOURCE_LEN
[`MAX_DIAGNOSTICS`]: compile::MAX_DIAGNOSTICS
[`diag`]: diag
[`Diagnostic`]: diag::Diagnostic
[`DiagCode`]: diag::DiagCode
[`Severity`]: diag::Severity
[`LanguageMessage`]: diag::LanguageMessage
[`numeric`]: numeric
[`numeric::Platform`]: numeric::Platform
[`rng`]: rng
[`catalog::MathCatalog`]: catalog::MathCatalog"
)]
#![cfg_attr(
    not(feature = "compiler"),
    doc = "[compile-mod]: https://docs.rs/molang/1/molang/compile/index.html
[`compile`]: https://docs.rs/molang/1/molang/compile/fn.compile.html
[`compile_source`]: https://docs.rs/molang/1/molang/compile/fn.compile_source.html
[`CompileOptions`]: https://docs.rs/molang/1/molang/compile/struct.CompileOptions.html
[`compile::CompileOptions`]: https://docs.rs/molang/1/molang/compile/struct.CompileOptions.html
[`CompileOptions::with_math`]: https://docs.rs/molang/1/molang/compile/struct.CompileOptions.html#method.with_math
[`Compiled`]: https://docs.rs/molang/1/molang/compile/struct.Compiled.html
[`Compiled::into_result`]: https://docs.rs/molang/1/molang/compile/struct.Compiled.html#method.into_result
[`Expr`]: https://docs.rs/molang/1/molang/compile/struct.Expr.html
[`ProgramFlags`]: https://docs.rs/molang/1/molang/compile/struct.ProgramFlags.html
[`MAX_DEPTH`]: https://docs.rs/molang/1/molang/compile/constant.MAX_DEPTH.html
[`MAX_SOURCE_LEN`]: https://docs.rs/molang/1/molang/compile/constant.MAX_SOURCE_LEN.html
[`MAX_DIAGNOSTICS`]: https://docs.rs/molang/1/molang/compile/constant.MAX_DIAGNOSTICS.html
[`diag`]: https://docs.rs/molang/1/molang/diag/index.html
[`Diagnostic`]: https://docs.rs/molang/1/molang/diag/struct.Diagnostic.html
[`DiagCode`]: https://docs.rs/molang/1/molang/diag/enum.DiagCode.html
[`Severity`]: https://docs.rs/molang/1/molang/diag/enum.Severity.html
[`LanguageMessage`]: https://docs.rs/molang/1/molang/diag/enum.LanguageMessage.html
[`numeric`]: https://docs.rs/molang/1/molang/numeric/index.html
[`numeric::Platform`]: https://docs.rs/molang/1/molang/numeric/enum.Platform.html
[`rng`]: https://docs.rs/molang/1/molang/rng/index.html
[`catalog::MathCatalog`]: https://docs.rs/molang/1/molang/catalog/struct.MathCatalog.html"
)]
#![cfg_attr(
    feature = "stdlib",
    doc = "[`stdlib`]: stdlib
[`stdlib::queries`]: stdlib::queries
[`queries`]: stdlib::queries
[`query`]: stdlib::query"
)]
#![cfg_attr(
    not(feature = "stdlib"),
    doc = "[`stdlib`]: https://docs.rs/molang/1/molang/stdlib/index.html
[`stdlib::queries`]: https://docs.rs/molang/1/molang/stdlib/fn.queries.html
[`queries`]: https://docs.rs/molang/1/molang/stdlib/fn.queries.html
[`query`]: https://docs.rs/molang/1/molang/stdlib/query/index.html"
)]
#![cfg_attr(
    all(feature = "stdlib", feature = "compiler"),
    doc = "[`stdlib::math`]: stdlib::math
[`CompileOptions::server`]: compile::CompileOptions::server
[`CompileOptions::client`]: compile::CompileOptions::client"
)]
#![cfg_attr(
    not(all(feature = "stdlib", feature = "compiler")),
    doc = "[`stdlib::math`]: https://docs.rs/molang/1/molang/stdlib/math/index.html
[`CompileOptions::server`]: https://docs.rs/molang/1/molang/compile/struct.CompileOptions.html#method.server
[`CompileOptions::client`]: https://docs.rs/molang/1/molang/compile/struct.CompileOptions.html#method.client"
)]
#![cfg_attr(
    feature = "vm",
    doc = "[`vm`]: vm
[`Value`]: vm::Value
[`EvalLimits`]: vm::EvalLimits
[`vm::EvalLimits`]: vm::EvalLimits
[`vm::QueryTable`]: vm::QueryTable
[`NoHostEnv`]: vm::NoHostEnv
[`Expr::eval`]: compile::Expr::eval"
)]
#![cfg_attr(
    not(feature = "vm"),
    doc = "[`vm`]: https://docs.rs/molang/1/molang/vm/index.html
[`Value`]: https://docs.rs/molang/1/molang/vm/enum.Value.html
[`EvalLimits`]: https://docs.rs/molang/1/molang/vm/struct.EvalLimits.html
[`vm::EvalLimits`]: https://docs.rs/molang/1/molang/vm/struct.EvalLimits.html
[`vm::QueryTable`]: https://docs.rs/molang/1/molang/vm/struct.QueryTable.html
[`NoHostEnv`]: https://docs.rs/molang/1/molang/vm/struct.NoHostEnv.html
[`Expr::eval`]: https://docs.rs/molang/1/molang/compile/struct.Expr.html#method.eval"
)]
#![cfg_attr(
    feature = "cache",
    doc = "[`cache`]: cache
[`CompileCache`]: cache::CompileCache"
)]
#![cfg_attr(
    not(feature = "cache"),
    doc = "[`cache`]: https://docs.rs/molang/1/molang/cache/index.html
[`CompileCache`]: https://docs.rs/molang/1/molang/cache/struct.CompileCache.html"
)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
// Deliberate: integer literals wrap, array indices truncate, the exponent goes through `f64`.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
// Truthiness and `==` are bit-exact f32 comparisons.
#![allow(clippy::float_cmp)]
// `#[must_use]` goes on methods returning a changed copy and on values whose loss is a bug, not
// on every pure getter.
#![allow(clippy::must_use_candidate)]
// The error enums' variants document each failure.
#![allow(clippy::missing_errors_doc)]
// Names such as `MolangVersion` in `version` stay descriptive.
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// Every public item has its path in its module; none resolves at the crate root.
///
/// ```compile_fail,E0432
/// use molang::CompileOptions;
/// ```
/// ```compile_fail,E0423
/// let _ = molang::compile;
/// ```
/// ```compile_fail,E0432
/// use molang::EvalLimits;
/// ```
/// ```compile_fail,E0432
/// use molang::Side;
/// ```
#[cfg(doctest)]
struct NoItemAtTheRoot;

/// Compiles the README's code blocks as doctests.
#[cfg(all(doctest, feature = "vm", feature = "stdlib"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod bitmask;
#[cfg(feature = "cache")]
#[cfg_attr(docsrs, doc(cfg(feature = "cache")))]
pub mod cache;
pub mod catalog;
#[cfg(feature = "compiler")]
#[cfg_attr(docsrs, doc(cfg(feature = "compiler")))]
pub mod compile;
#[cfg(feature = "compiler")]
#[cfg_attr(docsrs, doc(cfg(feature = "compiler")))]
pub mod diag;
pub mod hash;
#[cfg(feature = "fuzz")]
#[doc(hidden)]
pub mod internals;
pub mod json;
#[cfg(feature = "compiler")]
#[cfg_attr(docsrs, doc(cfg(feature = "compiler")))]
pub mod numeric;
pub mod ops;
#[cfg(all(any(test, feature = "fuzz"), feature = "stdlib"))]
mod reference_catalog;
#[cfg(feature = "compiler")]
#[cfg_attr(docsrs, doc(cfg(feature = "compiler")))]
pub mod rng;
#[cfg(feature = "stdlib")]
#[cfg_attr(docsrs, doc(cfg(feature = "stdlib")))]
pub mod stdlib;
pub mod version;
#[cfg(feature = "vm")]
#[cfg_attr(docsrs, doc(cfg(feature = "vm")))]
pub mod vm;

#[cfg(all(test, not(feature = "stdlib")))]
compile_error!("the unit tests need the `stdlib` feature; without it run `--test engine_only`");
