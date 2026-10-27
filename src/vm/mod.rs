//! Evaluation: values, the host traits, variable storage and run-time diagnostics.
//!
//! An embedding implements [`Host`] once and passes an [`EvalCx`] to every evaluation;
//! [`NoHostEnv`] and [`HostEnv`] own every part of one. The crate implements no query: a
//! [`Query`] is registered by name in a [`QueryTable`].
//!
//! ```
//! # #[cfg(feature = "stdlib")]
//! # {
//! use molang::compile::{CompileOptions, compile};
//! use molang::version::MolangVersion;
//! use molang::vm::NoHostEnv;
//!
//! let (expr, _) = compile(
//!     "math.clamp(1 + 2 * 3, 0, 5)",
//!     &CompileOptions::server(MolangVersion::LATEST),
//! )
//! .into_result()
//! .unwrap();
//! assert_eq!(expr.eval_f32(&mut NoHostEnv::new().cx()), 5.0);
//! # }
//! ```
//!
//! # Reading a result
//!
//! [`Expr::eval`](crate::compile::Expr::eval) returns a [`Value`] and the caller decides how a
//! field reads it: [`Value::as_f32`] for a plain float field, [`Value::is_nonzero`],
//! [`Value::is_positive`], [`Value::exceeds_epsilon`], [`Value::rounds_to_nonzero`],
//! [`Value::round_to_int`], [`Value::to_int`] and [`Value::to_property_int`] and [`Value::to_property_float`] for the others. The
//! ones that truncate take the [`Platform`](crate::numeric::Platform) the expression was compiled
//! for.
//!
//! # Stack
//!
//! An evaluation needs the stack [`Expr::eval`](crate::compile::Expr::eval) documents;
//! [`EvalLimits::query_depth`] caps it.
//!
//! # Walking stored values
//!
//! Structs share members, so walk stored values with [`Value::distinct_structs`] or bound the walk.
//!
//! # Contexts
//!
//! How the game wires an evaluation up differs by site; the host builds each one.
//!
//! | Context | What the host does |
//! |---|---|
//! | Ambient-sound and anger-sound conditions | No variable store: use [`NoVariables`], so `variable.*` reads are missing and end the evaluation with 0 |
//! | `context.*` | The game keeps one map per thread, shared by every evaluation; some sites clear it afterwards and others do not, so a later evaluation may see a stale value. A host that wants that keeps one [`ContextMap`] across evaluations, otherwise it fills a fresh one each time |
//! | Particles | One `variable.*` map per emitter, shared by the emitter's expressions and all its particles, never the owner actor's map |
//! | Client rotation and walk queries | Body and head rotations (including rider and ride), `walk_distance`, `modified_distance_moved`, `modified_move_speed`, `swell_amount`, `head_roll_angle` and `shield_blocking_bob` are interpolated by the frame alpha; the query implementation applies it |
//!
//! # Deviations
//!
//! | Behaviour | Default | Alternative |
//! |---|---|---|
//! | Temp lifetime | [`Temps::PerEvaluation`]: temps start empty in every evaluation | [`Temps::Kept`] in [`EvalCx::temps`], [`HostEnv`] or [`NoHostEnv`] |
//! | Loop and step budgets | [`EvalLimits::DEFAULT`]: 1,024 iterations per loop, 1,048,576 steps per evaluation | [`EvalLimits::NONE`], or `loop_iterations: None` keeping the step budget |
//! | Struct depth budget | [`EvalLimits::DEFAULT`]: a member store nesting structs deeper than 32 ends the evaluation with 0 | `struct_depth: None` |
//! | Struct member budget | [`EvalLimits::DEFAULT`]: a member store adding a 257th member ends the evaluation with 0 | `struct_members: None` |
//! | Operand-stack cap | operands left behind by `break` / `continue` past [`EvalLimits::OPERAND_STACK_CAP`] slots end the evaluation | none |
//! | Random source | [`ProcessRng`]: one xorshift128 for the whole process | any [`RandomSource`](crate::rng::RandomSource): a closure mapping a generator's words with [`unit_32`](crate::rng::unit_32) for world generation, [`FixedRng`](crate::rng::FixedRng) |

mod arrays;
mod cx;
mod error;
mod eval;
mod global_rng;
mod host;
mod limits;
mod name;
mod query;
mod sink;
mod value;
mod vars;

pub use arrays::{ArrayProvider, ArrayTable, NoArrays};
pub use cx::{EvalCx, HostEnv, NoHostEnv, Temps};
pub use error::{DuplicateMember, QueryError};
pub use global_rng::ProcessRng;
pub use host::{
    ContextMap, ContextProvider, Host, HostAccess, NoContext, NoHost, Subjects, WorldGenPos,
};
pub use limits::EvalLimits;
pub use name::{
    AnyName, ArrayName, ContextName, Name, ParseNameError, TempName, VariableName, namespace,
};
pub use query::{Query, QueryBackend, QueryCx, QueryResult, QueryTable, UnknownQuery};
pub use sink::{BoundedSink, CollectSink, LogLevel, LogOnce, NullSink, RuntimeMsg, RuntimeSink};
#[cfg(feature = "fuzz")]
pub(crate) use value::MemberStoreCheck;
pub use value::{DistinctStructs, ResourceRef, StructValue, Value, ValueKind};
pub use vars::{Access, NoVariables, TempMap, VariableMap, VariableStorage, VariableStore};

#[cfg(test)]
pub(crate) mod test_support {
    use super::host::{Host, HostAccess, Subjects};

    /// A host whose actors are numbers; actors 100 and above are dead / null.
    #[derive(Debug)]
    pub(crate) struct TestHost;

    impl Host for TestHost {
        type ActorRef = u32;
        type ItemRef = u16;
        type BlockRef = (i32, i32, i32);
        type Access<'w> = TestHost;
    }

    impl HostAccess<TestHost> for TestHost {
        fn resolve_actor(&self, _from: &Subjects<TestHost>, actor: u32) -> Option<u32> {
            alive(actor)
        }
    }

    /// The actor `actor` refers to, or `None` for the dead / null ones (100 and above).
    pub(crate) fn alive(actor: u32) -> Option<u32> {
        (actor < 100).then_some(actor)
    }
}

#[cfg(test)]
mod tests {
    // The import list itself checks that every re-export resolves.
    #![allow(unused_imports)]

    use super::test_support::TestHost;
    use super::{
        Access, AnyName, ArrayName, ArrayProvider, ArrayTable, BoundedSink, CollectSink,
        ContextMap, ContextName, ContextProvider, DistinctStructs, DuplicateMember, EvalCx,
        EvalLimits, Host, HostAccess, HostEnv, LogLevel, LogOnce, Name, NoArrays, NoContext,
        NoHost, NoHostEnv, NoVariables, NullSink, ProcessRng, Query, QueryBackend, QueryCx,
        QueryError, QueryTable, ResourceRef, RuntimeMsg, RuntimeSink, StructValue, Subjects,
        TempMap, TempName, Value, ValueKind, VariableMap, VariableName, VariableStorage,
        VariableStore, WorldGenPos,
    };

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn the_values_and_tables_a_host_stores_are_send_and_sync() {
        assert_send_sync::<Value<NoHost>>();
        assert_send_sync::<Value<TestHost>>();
        assert_send_sync::<StructValue<TestHost>>();
        assert_send_sync::<VariableMap<TestHost>>();
        assert_send_sync::<VariableStorage<TestHost>>();
        assert_send_sync::<ContextMap<TestHost>>();
        assert_send_sync::<ArrayTable>();
        assert_send_sync::<NoArrays>();
        assert_send_sync::<QueryTable<TestHost>>();
        assert_send_sync::<Subjects<TestHost>>();
        assert_send_sync::<EvalLimits>();
        assert_send_sync::<VariableName>();
        assert_send_sync::<TempName>();
        assert_send_sync::<ContextName>();
        assert_send_sync::<NoHostEnv>();
        assert_send_sync::<BoundedSink>();
        assert_send_sync::<CollectSink>();
        assert_send_sync::<ProcessRng>();
    }

    #[test]
    fn the_re_exported_names_are_the_types_of_their_modules() {
        assert_eq!(ValueKind::Float, Value::<NoHost>::ZERO.kind());
        assert!(LogLevel::Error > LogLevel::Warn);
        assert_eq!(Access::default(), Access::Private);
        assert_eq!(EvalLimits::default(), EvalLimits::DEFAULT);
        assert_eq!(
            VariableName::new("x").hashed(),
            crate::hash::HashedStr::new("variable.x")
        );
        assert_eq!(WorldGenPos::default(), WorldGenPos { x: 0, y: 0, z: 0 });
        assert_eq!(size_of::<Value<NoHost>>(), 16);
    }
}
