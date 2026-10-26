//! Compile `math.clamp(1 + 2 * 3, 0, 5)` and evaluate it with no host, then an expression with
//! variables, a loop and a missing read.
//!
//! Run with `cargo run --example hello_world --features vm`.

use molang::compile::{CompileError, CompileOptions, compile};
use molang::numeric::Platform;
use molang::version::MolangVersion;
use molang::vm::{NoHostEnv, Value, VariableName};

fn main() -> Result<(), CompileError> {
    let options = CompileOptions::server(MolangVersion::LATEST);

    let (expr, diagnostics) = compile("math.clamp(1 + 2 * 3, 0, 5)", &options).into_result()?;
    for diagnostic in &diagnostics {
        println!("  compile: {diagnostic}");
    }
    let mut env = NoHostEnv::new();
    println!(
        "math.clamp(1 + 2 * 3, 0, 5) = {}",
        expr.eval_f32(&mut env.cx())
    );

    // Variables persist in the environment between evaluations; temps do not.
    env.variables
        .set(VariableName::new("speed"), Value::Float(2.5));
    let source = "t.d = 0; loop(4, {t.d = t.d + v.speed;}); v.distance = t.d; return v.distance;";
    let (expr, _) = compile(source, &options).into_result()?;
    println!("{source} = {}", expr.eval_f32(&mut env.cx()));
    println!(
        "variable.distance is now {:?}",
        env.variables.get(VariableName::new("distance"))
    );

    // Reading a variable nobody set ends the expression with 0 and logs a message.
    let (expr, _) = compile("v.a = 1; return v.never_set + 1;", &options).into_result()?;
    println!(
        "v.a = 1; return v.never_set + 1; = {}",
        expr.eval_f32(&mut env.cx())
    );
    for message in env.sink.take() {
        println!("  logged: {message}");
    }

    // The caller decides how a result is read; the truncating readings follow the platform.
    let (expr, _) = compile("0.4", &options).into_result()?;
    let value = expr.eval(&mut env.cx());
    println!(
        "0.4: is_nonzero = {}, rounds_to_nonzero = {}",
        value.is_nonzero(),
        value.rounds_to_nonzero(expr.platform())
    );
    let (expr, _) = compile("1e10", &options).into_result()?;
    let value = expr.eval(&mut env.cx());
    for platform in [Platform::default(), Platform::AndroidArm64] {
        println!(
            "1e10 as an int property in 0..=10 on {platform:?} = {}",
            value.to_property_int(0..=10, platform)
        );
    }
    Ok(())
}
