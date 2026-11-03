//! A compile with a failure policy: [`check`] compiles a source and decides whether the caller
//! keeps, rejects or drops it.

use crate::compile::{CompileFailure, CompileOptions, Compiled, Expr, compile, compile_source};
use crate::diag::{DiagCode, Diagnostic};
use crate::json::MolangSource;
use std::sync::Arc;

/// What [`check`] does with a source whose compile failed.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OnFailure {
    /// [`CheckDecision::Rejected`]: no expression, the diagnostics say why.
    #[default]
    Reject,
    /// [`CheckDecision::Kept`]: the expression of the failed compile (the constant 0 of a
    /// rejection, none for arrays or resources) with the diagnostics.
    Keep,
    /// [`CheckDecision::Dropped`]: no expression; the diagnostics are still reported.
    Drop,
}

/// What [`check`] decided.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CheckDecision {
    /// The source is kept: it compiled, or it failed under [`OnFailure::Keep`].
    Kept,
    /// It failed under [`OnFailure::Reject`].
    Rejected,
    /// It failed under [`OnFailure::Drop`].
    Dropped,
}

/// Which [`DiagCode::QueryClientOnly`](crate::diag::DiagCode::QueryClientOnly) notes a check
/// reports: the side filter. The compile logs them; the filter applies after it, so a cached
/// compile serves every filter.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClientOnlyNotes {
    /// Every note.
    #[default]
    All,
    /// The first note of the expression.
    First,
    /// None.
    None,
}

/// Where the version a check compiles at comes from.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VersionFrom {
    /// The source's own version, as [`compile_source`] compiles.
    #[default]
    Source,
    /// The options' [raw version](CompileOptions::raw_version), whatever the source's.
    Options,
}

/// How [`check`] compiles and what it decides: the [`CompileOptions`], the failure policy, the
/// side filter, whether arrays and resources are accepted and where the version comes from.
///
/// The policy is kept apart from the compile options so that it is not part of the compile
/// cache's key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CheckOptions {
    compile: CompileOptions,
    on_failure: OnFailure,
    client_only: ClientOnlyNotes,
    accept_arrays: bool,
    accept_resources: bool,
    version_from: VersionFrom,
}

impl CheckOptions {
    /// `compile` with [`OnFailure::Reject`], [`ClientOnlyNotes::All`], arrays and resources
    /// accepted and [`VersionFrom::Source`].
    pub const fn new(compile: CompileOptions) -> Self {
        Self {
            compile,
            on_failure: OnFailure::Reject,
            client_only: ClientOnlyNotes::All,
            accept_arrays: true,
            accept_resources: true,
            version_from: VersionFrom::Source,
        }
    }

    /// The compile options.
    pub const fn compile(&self) -> &CompileOptions {
        &self.compile
    }

    /// What a failed compile becomes (by default [`OnFailure::Reject`]).
    pub const fn on_failure(&self) -> OnFailure {
        self.on_failure
    }

    /// Which client-only notes are reported (by default [`ClientOnlyNotes::All`]).
    pub const fn client_only(&self) -> ClientOnlyNotes {
        self.client_only
    }

    /// Whether [`CompileFailure::UsesArrays`] is kept rather than a failure (by default it is).
    pub const fn accept_arrays(&self) -> bool {
        self.accept_arrays
    }

    /// Whether [`CompileFailure::UsesResources`] is kept rather than a failure (by default it
    /// is).
    pub const fn accept_resources(&self) -> bool {
        self.accept_resources
    }

    /// Where the version comes from (by default [`VersionFrom::Source`]).
    pub const fn version_from(&self) -> VersionFrom {
        self.version_from
    }

    /// These options with other [compile options](Self::compile).
    #[must_use]
    pub fn with_compile(self, compile: CompileOptions) -> Self {
        Self { compile, ..self }
    }

    /// These options with another [failure policy](Self::on_failure).
    #[must_use]
    pub fn with_on_failure(self, on_failure: OnFailure) -> Self {
        Self { on_failure, ..self }
    }

    /// These options with another [side filter](Self::client_only).
    #[must_use]
    pub fn with_client_only(self, client_only: ClientOnlyNotes) -> Self {
        Self {
            client_only,
            ..self
        }
    }

    /// These options accepting [arrays](Self::accept_arrays) or not.
    #[must_use]
    pub fn with_accept_arrays(self, accept_arrays: bool) -> Self {
        Self {
            accept_arrays,
            ..self
        }
    }

    /// These options accepting [resources](Self::accept_resources) or not.
    #[must_use]
    pub fn with_accept_resources(self, accept_resources: bool) -> Self {
        Self {
            accept_resources,
            ..self
        }
    }

    /// These options taking the version [from](Self::version_from) elsewhere.
    #[must_use]
    pub fn with_version_from(self, version_from: VersionFrom) -> Self {
        Self {
            version_from,
            ..self
        }
    }
}

impl From<CompileOptions> for CheckOptions {
    fn from(compile: CompileOptions) -> Self {
        Self::new(compile)
    }
}

/// What [`check`] returns: the compile, the decision and the diagnostics the side filter reports.
#[derive(Clone, Debug)]
pub struct Checked {
    compiled: Arc<Compiled>,
    decision: CheckDecision,
    client_only: ClientOnlyNotes,
}

impl Checked {
    /// `compiled` under the policy of `options`.
    pub(crate) fn decide(compiled: Arc<Compiled>, options: &CheckOptions) -> Self {
        let failed = match compiled.failure() {
            None => false,
            Some(CompileFailure::UsesArrays) => !options.accept_arrays,
            Some(CompileFailure::UsesResources) => !options.accept_resources,
            Some(CompileFailure::Rejected) => true,
        };
        let decision = match (failed, options.on_failure) {
            (false, _) | (true, OnFailure::Keep) => CheckDecision::Kept,
            (true, OnFailure::Reject) => CheckDecision::Rejected,
            (true, OnFailure::Drop) => CheckDecision::Dropped,
        };
        Self {
            compiled,
            decision,
            client_only: options.client_only,
        }
    }

    /// The decision.
    pub const fn decision(&self) -> CheckDecision {
        self.decision
    }

    /// Whether the source is kept ([`CheckDecision::Kept`]).
    pub const fn is_kept(&self) -> bool {
        matches!(self.decision, CheckDecision::Kept)
    }

    /// The expression of a kept source: the compiled one, or the constant 0 of a rejection kept
    /// under [`OnFailure::Keep`]; `None` for a rejected or dropped source and for one that
    /// uses arrays or resources.
    pub fn expr(&self) -> Option<&Expr> {
        match self.decision {
            CheckDecision::Kept => self.compiled.expr_or_zero(),
            CheckDecision::Rejected | CheckDecision::Dropped => None,
        }
    }

    /// The diagnostics of the compile the side filter reports, in order. A rejected or dropped
    /// source reports them as a kept one does.
    pub fn diagnostics(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        let mut client_only_seen = false;
        self.compiled
            .diagnostics()
            .iter()
            .filter(move |diagnostic| {
                if diagnostic.code() != DiagCode::QueryClientOnly {
                    return true;
                }
                match self.client_only {
                    ClientOnlyNotes::All => true,
                    ClientOnlyNotes::First => !std::mem::replace(&mut client_only_seen, true),
                    ClientOnlyNotes::None => false,
                }
            })
    }

    /// How the compile failed, whatever the decision; `None` for a success.
    pub fn failure(&self) -> Option<CompileFailure> {
        self.compiled.failure()
    }

    /// The whole compile, every diagnostic included.
    pub fn compiled(&self) -> &Compiled {
        &self.compiled
    }
}

/// Compiles `src` under `options` and applies the failure policy.
///
/// The compile is [`compile_source`] (or [`compile`](super::compile) of the text at the options'
/// version under [`VersionFrom::Options`]). A rejection fails; so do
/// [`CompileFailure::UsesArrays`] and [`CompileFailure::UsesResources`] when the options do not
/// accept them. A failure becomes the decision [`CheckOptions::on_failure`] names.
#[must_use = "the result carries the decision, the expression and the diagnostics"]
pub fn check(src: &MolangSource, options: &CheckOptions) -> Checked {
    let compiled = match options.version_from {
        VersionFrom::Source => compile_source(src, &options.compile),
        VersionFrom::Options => compile(src.as_str(), &options.compile),
    };
    Checked::decide(Arc::new(compiled), options)
}
