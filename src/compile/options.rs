//! What a caller configures: the catalogue, the deviations and the compile options.

#[cfg(feature = "stdlib")]
use crate::catalog::Side;
use crate::catalog::{MathCatalog, QueryAdmission, QueryCatalog, QuerySetMask};
use crate::json::MolangSource;
use crate::numeric::Platform;
use crate::ops::OpSet;
use crate::version::{ExperimentMask, MolangVersion, RawVersion};

bitflags::bitflags! {
    /// Behaviour switches, as a set: a flag in the set turns its behaviour on.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Deviations: u8 {
        /// A prefix of `true` / `false` is a complete boolean token and the lexer advances by its
        /// length. Off, it advances four bytes for `true` and five for `false` (`t;` loses its
        /// `;`).
        const TRUE_FALSE_PREFIX_ADVANCE = 1 << 0;
        /// Reject a source longer than [`MAX_SOURCE_LEN`](super::MAX_SOURCE_LEN) with
        /// [`DiagCode::SourceTooLong`](crate::diag::DiagCode::SourceTooLong).
        const SOURCE_LENGTH_LIMIT = 1 << 1;
        /// Report the validation messages of sub-expressions below the root as warnings; off,
        /// they are [`Severity::Error`](crate::diag::Severity::Error). The expression compiles
        /// either way.
        const VALIDATE_NESTED = 1 << 2;
        /// Warn ([`DiagCode::QueryArity`](crate::diag::DiagCode::QueryArity)) when a query is
        /// called with an argument count outside its declared range. Such a call still compiles
        /// and runs.
        const QUERY_ARITY_LINT = 1 << 3;
        /// Inform ([`DiagCode::QueryClientOnly`](crate::diag::DiagCode::QueryClientOnly)) when a
        /// client-only query is compiled against a
        /// [`Side::Server`](crate::catalog::Side::Server) catalogue, where it still resolves.
        const QUERY_CLIENT_ONLY = 1 << 4;
        /// Warn ([`DiagCode::InvalidVersion`](crate::diag::DiagCode::InvalidVersion)) when the
        /// raw version is outside −1..=13, and inform when it is `Invalid` (−1).
        const OBJECT_VERSION_WARNING = 1 << 5;
        /// Keep at most [`MAX_DIAGNOSTICS`](super::MAX_DIAGNOSTICS) diagnostics per compile, then
        /// one [`DiagCode::DiagnosticLimit`](crate::diag::DiagCode::DiagnosticLimit) note with the
        /// number left out. Off, every message is kept. The compile itself is unchanged.
        const DIAGNOSTIC_LIMIT = 1 << 6;
    }
}

impl Deviations {
    /// The default: [`Deviations::ALL`].
    pub const DEFAULT: Self = Self::ALL;
    /// Every deviation on.
    pub const ALL: Self = Self::all();
    /// Every deviation off.
    pub const NONE: Self = Self::empty();
}

impl Default for Deviations {
    /// [`Deviations::DEFAULT`].
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What `array.<name>[index]` compiles to.
///
/// A bare `array.<name>` and an expression array end the compile with
/// [`CompileOutcome::UsesArrays`](super::CompileOutcome::UsesArrays) either way.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArrayAccess {
    /// No program: the compile ends with
    /// [`CompileOutcome::UsesArrays`](super::CompileOutcome::UsesArrays), as on a server, which
    /// has no render controllers.
    #[default]
    Unresolved,
    /// An instruction that reads the element at evaluation, from
    /// [`EvalCx::arrays`](crate::vm::EvalCx::arrays) by the array's name. The array need not be
    /// known when compiling.
    FromHost,
}

/// How an expression is compiled: query catalogue, version, admitted queries, allowed operations,
/// host math functions, deviations and the platform whose float behaviour it reproduces.
///
/// The [`QueryCatalog`] decides which queries exist and whether client-only ones are
/// flagged. The admitted queries ([`CompileOptions::admission`]) are set independently of it: an
/// allow-list admits the queries of the compile's catalogue that it names.
///
/// [`CompileOptions::version`] (parser gates) is derived from
/// [`CompileOptions::raw_version`] (query resolution). For a document field, compile its
/// [`MolangSource`] with [`compile_source`](super::compile_source), which applies the source's
/// version to the field's options.
///
/// Each setting has a getter and a consuming `with_*` setter.
///
/// Equality is the compile cache's key: a new field must compare unequal whenever it can change a
/// compile's result.
///
/// ```
/// # #[cfg(feature = "stdlib")]
/// # {
/// use molang::catalog::{QueryAdmission, QueryAllowList, Side};
/// use molang::compile::{CompileFailure, CompileOptions, compile};
/// use molang::ops::OpSet;
/// use molang::stdlib::{self, query};
/// use molang::version::MolangVersion;
///
/// let list = QueryAllowList::new(stdlib::queries(Side::Server), [query::BLOCK_STATE])?;
/// let block = CompileOptions::server(MolangVersion::LATEST)
///     .with_admission(QueryAdmission::Only(list))
///     .with_allowed_ops(OpSet::all().without_assignments_or_random());
/// assert!(compile("q.block_state('facing') == 'west'", &block).is_success());
/// assert_eq!(compile("q.is_baby", &block).failure(), Some(CompileFailure::Rejected));
/// # }
/// # Ok::<(), molang::catalog::AllowListError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    /// The catalogue queries resolve against. An allow-list in `admission` admits the queries of
    /// this catalogue that it names.
    pub(crate) catalog: QueryCatalog,
    /// The raw version queries resolve with. Parser gates use
    /// [`RawVersion::effective`](crate::version::RawVersion::effective) of it; above 13 it gates
    /// like 13 and resolves no query.
    pub(crate) raw_version: RawVersion,
    /// Which queries of the catalogue may be called: query sets (by default `default`) or an
    /// allow-list.
    pub(crate) admission: QueryAdmission,
    /// The allowed operations (by default all).
    pub(crate) allowed_ops: OpSet,
    /// The enabled experiments (by default none). No standard query is experiment-gated.
    pub(crate) experiments: ExperimentMask,
    /// The host's own `math.*` functions (by default none), callable like the standard ones.
    pub(crate) math: Option<MathCatalog>,
    /// Whether the compiled expression keeps the source text (by default not).
    pub(crate) keep_source: bool,
    /// The active deviations (by default [`Deviations::DEFAULT`], every one on).
    pub(crate) deviations: Deviations,
    /// The platform whose float behaviour constant folding and evaluation reproduce (by default
    /// [`Platform::default`]).
    pub(crate) platform: Platform,
    /// What `array.<name>[index]` compiles to (by default [`ArrayAccess::Unresolved`]).
    pub(crate) arrays: ArrayAccess,
}

impl CompileOptions {
    /// Default settings at `version` against `catalog`: the `default` query set, every operation
    /// allowed, no experiments, no host math functions, source not kept, [`Deviations::DEFAULT`],
    /// [`Platform::default`], [`ArrayAccess::Unresolved`].
    pub const fn new(catalog: QueryCatalog, version: MolangVersion) -> Self {
        Self::from_raw_version(catalog, RawVersion(version.to_i16()))
    }

    /// Default settings at a raw version: parser gates use
    /// [`RawVersion::effective`](crate::version::RawVersion::effective), queries resolve with the
    /// raw value.
    pub const fn from_raw_version(catalog: QueryCatalog, raw: RawVersion) -> Self {
        Self {
            catalog,
            raw_version: raw,
            admission: QueryAdmission::Sets(QuerySetMask::DEFAULT),
            allowed_ops: OpSet::all(),
            experiments: ExperimentMask::empty(),
            math: None,
            keep_source: false,
            deviations: Deviations::DEFAULT,
            platform: Platform::DEFAULT,
            arrays: ArrayAccess::Unresolved,
        }
    }

    /// [`CompileOptions::from_raw_version`] at the source's raw version; `None` for a string source
    /// whose context version was never applied.
    pub fn for_source(catalog: QueryCatalog, source: &MolangSource) -> Option<Self> {
        source
            .raw_version()
            .map(|raw| Self::from_raw_version(catalog, raw))
    }

    /// The version parser gates use:
    /// [`RawVersion::effective`](crate::version::RawVersion::effective) of
    /// [`CompileOptions::raw_version`].
    pub const fn version(&self) -> MolangVersion {
        self.raw_version.effective()
    }

    /// The catalogue queries resolve against. An allow-list in
    /// [`admission`](Self::admission) admits the queries of this catalogue that it names.
    pub const fn catalog(&self) -> &QueryCatalog {
        &self.catalog
    }

    /// The raw version queries resolve with. Parser gates use
    /// [`RawVersion::effective`](crate::version::RawVersion::effective) of it; above 13 it gates
    /// like 13 and resolves no query.
    pub const fn raw_version(&self) -> RawVersion {
        self.raw_version
    }

    /// Which queries of the catalogue may be called: query sets (by default `default`) or an
    /// allow-list.
    pub const fn admission(&self) -> &QueryAdmission {
        &self.admission
    }

    /// The allowed operations (by default all).
    pub const fn allowed_ops(&self) -> OpSet {
        self.allowed_ops
    }

    /// The enabled experiments (by default none). No standard query is experiment-gated.
    pub const fn experiments(&self) -> ExperimentMask {
        self.experiments
    }

    /// The host's own `math.*` functions (by default none), callable like the standard ones.
    pub const fn math(&self) -> Option<&MathCatalog> {
        self.math.as_ref()
    }

    /// Whether the compiled expression keeps the source text (by default not).
    pub const fn keep_source(&self) -> bool {
        self.keep_source
    }

    /// The active deviations (by default [`Deviations::DEFAULT`], every one on).
    pub const fn deviations(&self) -> Deviations {
        self.deviations
    }

    /// The platform whose float behaviour constant folding and evaluation reproduce (by default
    /// [`Platform::default`]).
    pub const fn platform(&self) -> Platform {
        self.platform
    }

    /// What `array.<name>[index]` compiles to (by default [`ArrayAccess::Unresolved`]).
    pub const fn arrays(&self) -> ArrayAccess {
        self.arrays
    }

    /// These options against another [catalogue](Self::catalog).
    #[must_use]
    pub fn with_catalog(self, catalog: QueryCatalog) -> Self {
        Self { catalog, ..self }
    }

    /// These options at another [raw version](Self::raw_version).
    #[must_use]
    pub fn with_raw_version(self, raw_version: RawVersion) -> Self {
        Self {
            raw_version,
            ..self
        }
    }

    /// These options with other [admitted queries](Self::admission).
    #[must_use]
    pub fn with_admission(self, admission: QueryAdmission) -> Self {
        Self { admission, ..self }
    }

    /// These options with other [allowed operations](Self::allowed_ops).
    #[must_use]
    pub fn with_allowed_ops(self, allowed_ops: OpSet) -> Self {
        Self {
            allowed_ops,
            ..self
        }
    }

    /// These options with other [experiments](Self::experiments).
    #[must_use]
    pub fn with_experiments(self, experiments: ExperimentMask) -> Self {
        Self {
            experiments,
            ..self
        }
    }

    /// These options with other [host math functions](Self::math); `None` for none.
    #[must_use]
    pub fn with_math(self, math: Option<MathCatalog>) -> Self {
        Self { math, ..self }
    }

    /// These options keeping the source text or not ([`keep_source`](Self::keep_source)).
    #[must_use]
    pub fn with_keep_source(self, keep_source: bool) -> Self {
        Self {
            keep_source,
            ..self
        }
    }

    /// These options with other [deviations](Self::deviations).
    #[must_use]
    pub fn with_deviations(self, deviations: Deviations) -> Self {
        Self { deviations, ..self }
    }

    /// These options for another [platform](Self::platform).
    #[must_use]
    pub fn with_platform(self, platform: Platform) -> Self {
        Self { platform, ..self }
    }

    /// These options with another [array access](Self::arrays).
    #[must_use]
    pub fn with_arrays(self, arrays: ArrayAccess) -> Self {
        Self { arrays, ..self }
    }
}

#[cfg(feature = "stdlib")]
#[cfg_attr(docsrs, doc(cfg(feature = "stdlib")))]
impl CompileOptions {
    /// [`CompileOptions::new`] against the standard library's server queries.
    pub fn server(version: MolangVersion) -> Self {
        Self::new(crate::stdlib::queries(Side::Server).clone(), version)
    }

    /// [`CompileOptions::new`] against the standard library's client queries, with
    /// [`ArrayAccess::FromHost`].
    pub fn client(version: MolangVersion) -> Self {
        Self::new(crate::stdlib::queries(Side::Client).clone(), version)
            .with_arrays(ArrayAccess::FromHost)
    }
}

#[cfg(test)]
mod tests {

    use crate::catalog::{
        QueryAdmission, QueryAllowList, QueryCatalog, QueryDecl, QuerySetMask, QueryShape, Side,
    };
    use crate::compile::{CompileOptions, Deviations, test_support::*};
    use crate::ops::{ExpressionOp as Op, OpSet};
    use crate::stdlib::query;
    use crate::version::{Experiment, ExperimentMask};

    use crate::json::MolangSource;
    use crate::numeric::Platform;
    use crate::version::{MolangVersion, RawVersion};

    fn server() -> &'static QueryCatalog {
        crate::stdlib::queries(Side::Server)
    }

    fn client() -> &'static QueryCatalog {
        crate::stdlib::queries(Side::Client)
    }

    #[test]
    fn all_has_every_deviation_on_and_none_has_none() {
        for flag in Deviations::all().iter() {
            assert!(Deviations::ALL.contains(flag), "{flag:?}");
            assert!(!Deviations::NONE.intersects(flag), "{flag:?}");
        }
        assert_eq!(Deviations::all().iter().count(), 7);
        assert_ne!(Deviations::ALL, Deviations::NONE);
        assert!(Deviations::NONE.is_empty());
    }

    #[test]
    fn the_default_deviations_are_all() {
        assert_eq!(Deviations::default(), Deviations::ALL);
        assert_eq!(Deviations::DEFAULT, Deviations::ALL);
    }

    #[test]
    fn new_is_the_default_configuration() {
        let o = CompileOptions::server(MolangVersion::LATEST);
        assert_eq!(o.version(), MolangVersion::LATEST);
        assert_eq!(o.raw_version, RawVersion(13));
        assert_eq!(o.admission, QueryAdmission::Sets(QuerySetMask::DEFAULT));
        assert_eq!(o.allowed_ops, OpSet::all());
        assert_eq!(o.experiments, ExperimentMask::empty());
        assert!(!o.keep_source);
        assert_eq!(o.catalog, server().clone());
        assert_eq!(o.catalog.side(), Side::Server);
        assert_eq!(
            CompileOptions::client(MolangVersion::LATEST).catalog,
            client().clone()
        );
        assert_eq!(
            o,
            CompileOptions::new(server().clone(), MolangVersion::LATEST)
        );
        assert_eq!(o.deviations, Deviations::ALL);
        assert_eq!(o.platform, Platform::default());
    }

    #[test]
    fn options_for_another_platform_compare_unequal() {
        let o = CompileOptions::server(MolangVersion::LATEST);
        let windows = o.clone().with_platform(Platform::BdsWindows);
        assert_ne!(o, windows);
        assert_eq!(windows.platform, Platform::BdsWindows);
    }

    #[test]
    fn new_at_each_version_records_the_version_and_its_number() {
        for raw in -1..=13 {
            let version = MolangVersion::from_i16(raw).expect("a defined version");
            let o = CompileOptions::server(version);
            assert_eq!(o.version(), version);
            assert_eq!(o.raw_version, RawVersion(raw));
        }
    }

    #[test]
    fn the_version_is_the_effective_raw_version() {
        for (raw, gate) in [
            (i16::MIN, MolangVersion::Invalid),
            (-2, MolangVersion::Invalid),
            (-1, MolangVersion::Invalid),
            (0, MolangVersion::V0),
            (7, MolangVersion::V7),
            (13, MolangVersion::LATEST),
            (14, MolangVersion::LATEST),
            (i16::MAX, MolangVersion::LATEST),
        ] {
            let o = CompileOptions::from_raw_version(server().clone(), RawVersion(raw));
            assert_eq!(o.version(), gate, "raw {raw}");
            assert_eq!(o.raw_version, RawVersion(raw), "raw {raw}");
            let set = opts().with_raw_version(RawVersion(raw));
            assert_eq!(set, o, "raw {raw}");
            assert_eq!(set.version(), RawVersion(raw).effective(), "raw {raw}");
        }
    }

    #[test]
    fn for_source_uses_the_raw_version_of_the_source() {
        for (source, raw, gate) in [
            (MolangSource::string("1", 2), 2, MolangVersion::V2),
            (MolangSource::object("1", 99), 99, MolangVersion::LATEST),
            (MolangSource::string("1", -5), -5, MolangVersion::Invalid),
            (
                MolangSource::object("1", i16::MIN),
                i16::MIN,
                MolangVersion::Invalid,
            ),
        ] {
            let o = CompileOptions::for_source(server().clone(), &source)
                .expect("a source with a version");
            assert_eq!(o.raw_version, RawVersion(raw));
            assert_eq!(o.version(), gate);
            assert_eq!(
                o,
                CompileOptions::from_raw_version(server().clone(), RawVersion(raw))
            );
        }
        assert_eq!(
            CompileOptions::for_source(
                server().clone(),
                &MolangSource::string_without_context("1")
            ),
            None
        );
    }

    #[test]
    fn without_a_list_the_sets_decide() {
        let o = opts();
        assert_eq!(o.admission, QueryAdmission::Sets(QuerySetMask::DEFAULT));
        let decl = |name| o.catalog.get(name).unwrap();
        assert!(o.admission.admits(decl(query::IS_BABY)));
        assert!(!o.admission.admits(decl(query::ANY_TAG)));
        assert!(QueryAdmission::Sets(QuerySetMask::TAGS).admits(decl(query::ANY_TAG)));
    }

    /// Options compare their catalogue by identity.
    #[test]
    fn options_compare_the_catalogue_by_identity_and_a_list_never_changes_it() {
        let list = QueryAllowList::new(client(), [query::IS_BABY]).unwrap();
        let own = server()
            .extended([QueryDecl::new("query.mine", QueryShape::DEFAULT).unwrap()])
            .unwrap();
        let o = opts()
            .with_catalog(own.clone())
            .with_admission(QueryAdmission::Only(list));
        assert_eq!(o.catalog, own);
        assert_ne!(o, o.clone().with_catalog(server().clone()));
        assert_eq!(
            CompileOptions::from_raw_version(own, RawVersion(13)),
            o.with_admission(opts().admission)
        );
    }

    #[test]
    fn the_op_set_presets_remove_the_assignment_and_the_random_functions() {
        let ops = OpSet::all().without_assignments();
        assert_eq!(ops, OpSet::all().without(Op::Assignment));
        assert!(ops.contains(Op::Random) && ops.contains(Op::RandomInt));
        assert_eq!(ops.len(), OpSet::all().len() - 1);
        let ops = OpSet::all().without_assignments_or_random();
        assert_eq!(
            ops,
            OpSet::all()
                .without(Op::Assignment)
                .without(Op::Random)
                .without(Op::RandomInt)
                .without(Op::HostMathVolatile)
        );
        assert!(
            ops.contains(Op::DieRoll) && ops.contains(Op::DieRollInt) && ops.contains(Op::HostMath)
        );
        assert_eq!(ops.len(), OpSet::all().len() - 4);
    }

    #[test]
    fn no_host_math_by_default_and_a_catalogue_by_struct_update() {
        use crate::catalog::{Arity, MathCatalog, MathDecl};
        assert_eq!(opts().math, None);
        assert_eq!(CompileOptions::client(MolangVersion::LATEST).math, None);
        let math =
            MathCatalog::new([MathDecl::pure("math.f", Arity::exactly(1), |_, a| a[0]).unwrap()])
                .unwrap();
        let o = opts().with_math(Some(math.clone()));
        assert_eq!(o.math.as_ref(), Some(&math));
        assert_ne!(o, opts());
        let copy =
            MathCatalog::new([MathDecl::pure("math.f", Arity::exactly(1), |_, a| a[0]).unwrap()])
                .unwrap();
        assert_ne!(o, o.clone().with_math(Some(copy)), "compared by identity");
    }

    #[test]
    fn struct_update_sets_only_the_fields_it_names() {
        let experiments =
            ExperimentMask::empty().with(Experiment::new(5).expect("a valid experiment"));
        let o = opts()
            .with_catalog(client().clone())
            .with_deviations(Deviations::NONE)
            .with_experiments(experiments)
            .with_keep_source(true);
        assert_eq!(o.catalog.side(), Side::Client);
        assert_eq!(
            (o.deviations, o.experiments, o.keep_source),
            (Deviations::NONE, experiments, true)
        );
        assert_eq!(
            (o.raw_version, o.admission, o.allowed_ops),
            (opts().raw_version, opts().admission, opts().allowed_ops)
        );
        let deviations = Deviations::ALL.difference(Deviations::VALIDATE_NESTED);
        assert!(
            !deviations.contains(Deviations::VALIDATE_NESTED)
                && deviations.contains(Deviations::QUERY_ARITY_LINT)
        );
    }
}
