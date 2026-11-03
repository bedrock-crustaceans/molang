//! `compile::check`: a compile with a failure policy, the side filter and the arrays and
//! resources rules.

#![cfg(all(feature = "compiler", feature = "stdlib"))]

use molang::catalog::{QueryAdmission, QuerySetMask, QuerySide, Side};
use molang::compile::{
    CheckDecision, CheckOptions, ClientOnlyNotes, CompileFailure, CompileOptions, OnFailure,
    VersionFrom, check,
};
use molang::diag::DiagCode;
use molang::json::MolangSource;
use molang::version::{ExperimentMask, MolangVersion, RawVersion};

/// A source the parser rejects.
fn rejected_source() -> MolangSource {
    MolangSource::string("1 +", 13)
}

fn server(on_failure: OnFailure) -> CheckOptions {
    CheckOptions::new(CompileOptions::server(MolangVersion::LATEST)).with_on_failure(on_failure)
}

/// A query of the server catalogue that is client-only and resolves at version 13.
fn client_only_query() -> &'static str {
    molang::stdlib::queries(Side::Server)
        .iter()
        .find(|decl| {
            decl.shape().side() == QuerySide::CLIENT
                && decl
                    .resolve(
                        RawVersion(13),
                        &QueryAdmission::Sets(QuerySetMask::DEFAULT),
                        ExperimentMask::empty(),
                    )
                    .is_some()
        })
        .expect("the catalogue has client-only queries")
        .name()
}

#[test]
fn reject_returns_no_expression_and_the_diagnostics() {
    let checked = check(&rejected_source(), &server(OnFailure::Reject));
    assert_eq!(checked.decision(), CheckDecision::Rejected);
    assert!(
        checked.expr().is_none(),
        "a rejected source has no expression"
    );
    assert_eq!(checked.failure(), Some(CompileFailure::Rejected));
    assert!(checked.diagnostics().count() > 0, "the diagnostics say why");
}

#[test]
fn keep_returns_the_zero_expression_and_the_diagnostics() {
    let checked = check(&rejected_source(), &server(OnFailure::Keep));
    assert_eq!(checked.decision(), CheckDecision::Kept);
    let expr = checked.expr().expect("a kept rejection has its expression");
    assert_eq!(expr.as_constant(), Some(0.0));
    assert!(checked.diagnostics().count() > 0);
}

#[test]
fn drop_returns_no_expression_and_reports_it_was_dropped() {
    let checked = check(&rejected_source(), &server(OnFailure::Drop));
    assert_eq!(checked.decision(), CheckDecision::Dropped);
    assert!(!checked.is_kept());
    assert!(
        checked.expr().is_none(),
        "a dropped source has no expression"
    );
    // The caller still logs why its entry was dropped.
    assert!(checked.diagnostics().count() > 0);
}

#[test]
fn a_success_is_kept_under_every_policy() {
    let source = MolangSource::string("1 + 2", 13);
    for policy in [OnFailure::Reject, OnFailure::Keep, OnFailure::Drop] {
        let checked = check(&source, &server(policy));
        assert_eq!(checked.decision(), CheckDecision::Kept, "{policy:?}");
        assert_eq!(checked.expr().and_then(|e| e.as_constant()), Some(3.0));
    }
}

#[test]
fn arrays_and_resources_are_kept_unless_the_options_refuse_them() {
    let cases = [
        ("array.skins[0]", CompileFailure::UsesArrays),
        ("geometry.default", CompileFailure::UsesResources),
    ];
    for (text, failure) in cases {
        let source = MolangSource::string(text, 13);
        let accepted = check(&source, &server(OnFailure::Reject));
        assert_eq!(accepted.failure(), Some(failure), "{text}");
        assert_eq!(accepted.decision(), CheckDecision::Kept, "{text}");
        assert!(accepted.expr().is_none(), "{text}: no program to evaluate");

        let refusing = server(OnFailure::Reject)
            .with_accept_arrays(failure != CompileFailure::UsesArrays)
            .with_accept_resources(failure != CompileFailure::UsesResources);
        assert_eq!(
            check(&source, &refusing).decision(),
            CheckDecision::Rejected,
            "{text}"
        );
        // Refusing the other kind does not touch this one.
        let other = server(OnFailure::Reject)
            .with_accept_arrays(failure == CompileFailure::UsesArrays)
            .with_accept_resources(failure == CompileFailure::UsesResources);
        assert_eq!(
            check(&source, &other).decision(),
            CheckDecision::Kept,
            "{text}"
        );
    }
}

#[test]
fn the_side_filter_keeps_all_the_first_or_none_of_the_client_only_notes() {
    let name = client_only_query();
    let source = MolangSource::string(format!("{name} + {name}"), 13);
    let notes = |filter| {
        check(&source, &server(OnFailure::Reject).with_client_only(filter))
            .diagnostics()
            .filter(|d| d.code() == DiagCode::QueryClientOnly)
            .count()
    };
    assert_eq!(notes(ClientOnlyNotes::All), 2, "{name}: one note per call");
    assert_eq!(notes(ClientOnlyNotes::First), 1, "{name}");
    assert_eq!(notes(ClientOnlyNotes::None), 0, "{name}");
    // The whole compile still has every note.
    let checked = check(
        &source,
        &server(OnFailure::Reject).with_client_only(ClientOnlyNotes::None),
    );
    assert_eq!(
        checked
            .compiled()
            .diagnostics()
            .iter()
            .filter(|d| d.code() == DiagCode::QueryClientOnly)
            .count(),
        2
    );
}

#[test]
fn version_from_options_ignores_the_source_version() {
    // Version 2 predates the string-arithmetic error of version 3.
    let source = MolangSource::string("'a' + 1", 2);
    let options = server(OnFailure::Reject);
    assert_eq!(check(&source, &options).decision(), CheckDecision::Kept);
    let at_options = options.with_version_from(VersionFrom::Options);
    assert_eq!(
        check(&source, &at_options).decision(),
        CheckDecision::Rejected
    );
}
