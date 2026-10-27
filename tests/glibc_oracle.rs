//! The ports of glibc's single-precision functions against the host's glibc, bit for bit.
//!
//! Runs on x86-64 Linux with glibc only; the ports' own unit tests pin known results everywhere.
//! A failure names the host's glibc version: the ports follow glibc 2.41 on a CPU with FMA, so a
//! different glibc or a CPU without FMA can fail here without the port being wrong.
//!
//! The ignored `exhaustive` test sweeps every `f32` through each one-argument function and 2³²
//! random pairs through `powf` and `atan2f`; `MOLANG_EXHAUSTIVE`, a comma-separated list of names
//! such as `sinf,powf`, limits it to those.
#![cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::hint::black_box;
use std::sync::OnceLock;

use molang::internals::glibc as port;

// libm's functions are pure for every argument.
#[link(name = "m")]
unsafe extern "C" {
    safe fn sinf(x: f32) -> f32;
    safe fn cosf(x: f32) -> f32;
    safe fn expf(x: f32) -> f32;
    safe fn exp2f(x: f32) -> f32;
    safe fn logf(x: f32) -> f32;
    safe fn log2f(x: f32) -> f32;
    safe fn powf(x: f32, y: f32) -> f32;
    safe fn asinf(x: f32) -> f32;
    safe fn acosf(x: f32) -> f32;
    safe fn atanf(x: f32) -> f32;
    safe fn atan2f(y: f32, x: f32) -> f32;
}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlvsym(handle: *mut c_void, symbol: *const c_char, version: *const c_char) -> *mut c_void;
    fn gnu_get_libc_version() -> *const c_char;
}

const RTLD_NOW: c_int = 2;

type Unary = fn(f32) -> f32;
type Binary = fn(f32, f32) -> f32;
/// A host function.
type HostUnary = Box<dyn Fn(f32) -> f32 + Sync>;
/// A host function of two arguments.
type HostBinary = Box<dyn Fn(f32, f32) -> f32 + Sync>;

fn glibc_version() -> String {
    // SAFETY: glibc returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(gnu_get_libc_version()) }
        .to_string_lossy()
        .into_owned()
}

/// The `GLIBC_2.2.5` version of the libm symbol `name`.
fn compat_symbol(name: &CStr) -> *mut c_void {
    static LIBM: OnceLock<usize> = OnceLock::new();
    let handle = *LIBM.get_or_init(|| {
        // SAFETY: the name is NUL-terminated; libm is already loaded.
        let handle = unsafe { dlopen(c"libm.so.6".as_ptr(), RTLD_NOW) };
        assert!(!handle.is_null(), "dlopen(libm.so.6) failed");
        handle as usize
    });
    // SAFETY: the handle is libm's; both strings are NUL-terminated.
    let symbol = unsafe {
        dlvsym(
            handle as *mut c_void,
            name.as_ptr(),
            c"GLIBC_2.2.5".as_ptr(),
        )
    };
    assert!(!symbol.is_null(), "{name:?}@GLIBC_2.2.5 is missing");
    symbol
}

fn compat_unary(name: &CStr) -> extern "C" fn(f32) -> f32 {
    // SAFETY: the symbol is a libm function `float f(float)`.
    unsafe { std::mem::transmute::<*mut c_void, extern "C" fn(f32) -> f32>(compat_symbol(name)) }
}

fn compat_binary(name: &CStr) -> extern "C" fn(f32, f32) -> f32 {
    // SAFETY: the symbol is a libm function `float f(float, float)`.
    unsafe {
        std::mem::transmute::<*mut c_void, extern "C" fn(f32, f32) -> f32>(compat_symbol(name))
    }
}

/// Each one-argument port with the host function it reproduces.
fn unary() -> Vec<(&'static str, Unary, HostUnary)> {
    vec![
        ("sinf", port::sinf, Box::new(|x| sinf(x))),
        ("cosf", port::cosf, Box::new(|x| cosf(x))),
        ("expf", port::expf, Box::new(|x| expf(x))),
        ("exp2f", port::exp2f, Box::new(|x| exp2f(x))),
        ("logf", port::logf, Box::new(|x| logf(x))),
        ("log2f", port::log2f, Box::new(|x| log2f(x))),
        ("asinf", port::asinf, Box::new(|x| asinf(x))),
        ("acosf", port::acosf, Box::new(|x| acosf(x))),
        ("atanf", port::atanf, Box::new(|x| atanf(x))),
        ("expf@GLIBC_2.2.5", port::expf, {
            let f = compat_unary(c"expf");
            Box::new(move |x| f(x))
        }),
        ("exp2f@GLIBC_2.2.5", port::exp2f, {
            let f = compat_unary(c"exp2f");
            Box::new(move |x| f(x))
        }),
        ("logf@GLIBC_2.2.5", port::logf_compat, {
            let f = compat_unary(c"logf");
            Box::new(move |x| f(x))
        }),
    ]
}

/// Each two-argument port with the host function it reproduces.
fn binary() -> Vec<(&'static str, Binary, HostBinary)> {
    vec![
        ("powf", port::powf, Box::new(|x, y| powf(x, y))),
        ("atan2f", port::atan2f, Box::new(|y, x| atan2f(y, x))),
        ("powf@GLIBC_2.2.5", port::powf, {
            let f = compat_binary(c"powf");
            Box::new(move |x, y| f(x, y))
        }),
    ]
}

/// Collects the arguments a port gets wrong.
struct Mismatches {
    name: &'static str,
    checked: u64,
    count: u64,
    examples: Vec<String>,
}

impl Mismatches {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            checked: 0,
            count: 0,
            examples: Vec::new(),
        }
    }

    fn check(&mut self, args: &[f32], got: f32, expected: f32) {
        self.checked += 1;
        if got.to_bits() != expected.to_bits() {
            self.count += 1;
            if self.examples.len() < 20 {
                let args: Vec<String> = args
                    .iter()
                    .map(|a| format!("{:#010x}", a.to_bits()))
                    .collect();
                self.examples.push(format!(
                    "{}({}) = {:#010x}, glibc {:#010x}",
                    self.name,
                    args.join(", "),
                    got.to_bits(),
                    expected.to_bits()
                ));
            }
        }
    }

    fn merge(&mut self, other: Self) {
        self.checked += other.checked;
        self.count += other.count;
        let room = 20usize.saturating_sub(self.examples.len());
        self.examples.extend(other.examples.into_iter().take(room));
    }

    fn assert_none(&self) {
        assert!(
            self.count == 0,
            "{} of {} arguments differ from the host's glibc {} (the ports follow 2.41 with FMA):\n{}",
            self.count,
            self.checked,
            glibc_version(),
            self.examples.join("\n")
        );
    }
}

/// SplitMix64, fixed-seeded.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn bits(&mut self) -> f32 {
        f32::from_bits(self.next() as u32)
    }

    /// Uniform in `[lo, hi)`.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

const SPECIALS: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7f80_0001,
    0xff80_0001,
    0x7fa0_0000,
    0x7fc1_2345,
    0xffbf_ffff,
    0x7fff_ffff,
    0x0000_0001,
    0x8000_0001,
    0x0000_0002,
    0x0040_0000,
    0x007f_ffff,
    0x807f_ffff,
    0x0080_0000,
    0x8080_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x3f80_0000,
    0xbf80_0000,
    0x3f00_0000,
    0xbf00_0000,
    0x4000_0000,
    0xc000_0000,
];

/// Arguments near each function's branch thresholds, of both signs.
fn thresholds(name: &str) -> Vec<u32> {
    let mut t: Vec<u32> = match name.split('@').next().unwrap() {
        "sinf" | "cosf" => {
            let mut t = vec![
                0x3f40_0000,
                0x3f49_0fdb,
                0x3980_0000,
                0x0080_0000,
                0x42f0_0000,
            ];
            // Quadrant boundaries below 120 and a few above.
            for k in 1..=200 {
                t.push((k as f32 * std::f32::consts::FRAC_PI_4).to_bits());
            }
            for e in 7..=127 {
                t.push(f32::from_bits((e + 127) << 23).to_bits());
            }
            t
        }
        "expf" => vec![
            0x42b0_0000,
            0x42b1_7217,
            0xc2cf_f1b4,
            0xc2ce_8ecf,
            0x0000_0000,
            0x3380_0000,
        ],
        "exp2f" => vec![
            0x4300_0000,
            0xc316_0000,
            0xc315_0000,
            0x0000_0000,
            0x3380_0000,
        ],
        "logf" | "log2f" => {
            let mut t = vec![0x3f80_0000, 0x0080_0000, 0x7f80_0000];
            for j in 0..32 {
                t.push(0x3f33_0000 + (j << 19));
            }
            t
        }
        "asinf" => vec![
            0x3f61_4800,
            0x3980_0000,
            0x3f00_0000,
            0x3f80_0000,
            0x3f2a_b445,
            0x3f08_3a1a,
        ],
        "acosf" => vec![
            0x3f61_50ee,
            0x2000_0000,
            0x3f00_0000,
            0x3f80_0000,
            0x3288_85a3,
            0x3982_6222,
        ],
        "atanf" => vec![0x4c70_0518, 0x3900_0000, 0x3300_0000, 0x3f80_0000],
        _ => Vec::new(),
    };
    let negated: Vec<u32> = t.iter().map(|b| b ^ 0x8000_0000).collect();
    t.extend(negated);
    t
}

/// The 65,536 arguments of the easings' sine table, `i / 10430.378`.
fn sine_table_arguments() -> impl Iterator<Item = f32> {
    let scale = f32::from_bits(0x4622_f983);
    (0..=u16::MAX).map(move |i| f32::from(i) / scale)
}

/// A range a function's ordinary arguments come from.
fn domain(name: &str) -> (f32, f32) {
    match name.split('@').next().unwrap() {
        // Degree arguments in [−720, 720] after the degree conversion, and beyond.
        "sinf" | "cosf" => (-200.0, 200.0),
        "expf" => (-110.0, 95.0),
        "exp2f" => (-155.0, 135.0),
        "logf" | "log2f" => (0.0, 100.0),
        "asinf" | "acosf" => (-1.0, 1.0),
        _ => (-50.0, 50.0),
    }
}

const RANDOM_PER_FUNCTION: u64 = 2_000_000;
const GRID_HALF_WIDTH: u32 = 4096;

fn check_unary(index: usize) {
    let cases = unary();
    let (name, port_fn, host) = &cases[index];
    let mut m = Mismatches::new(name);
    let mut check = |x: f32| m.check(&[x], port_fn(x), host(x));
    for &b in SPECIALS {
        check(f32::from_bits(b));
    }
    for t in thresholds(name) {
        for d in 0..=2 * GRID_HALF_WIDTH {
            check(f32::from_bits(
                t.wrapping_add(d).wrapping_sub(GRID_HALF_WIDTH),
            ));
        }
    }
    for x in sine_table_arguments() {
        check(x);
    }
    let (lo, hi) = domain(name);
    let mut rng = Rng(0x6d6f_6c61_6e67 ^ index as u64);
    for _ in 0..RANDOM_PER_FUNCTION / 2 {
        check(rng.bits());
        check(rng.uniform(lo, hi));
    }
    m.assert_none();
}

fn check_binary(index: usize) {
    let cases = binary();
    let (name, port_fn, host) = &cases[index];
    let mut m = Mismatches::new(name);
    let mut check = |a: f32, b: f32| m.check(&[a, b], port_fn(a, b), host(a, b));
    for &a in SPECIALS {
        for &b in SPECIALS {
            check(f32::from_bits(a), f32::from_bits(b));
        }
    }
    let mut rng = Rng(0x706f_7766 ^ index as u64);
    for _ in 0..RANDOM_PER_FUNCTION / 4 {
        check(rng.bits(), rng.bits());
        let a = rng.uniform(-10.0, 10.0);
        check(a, rng.uniform(-40.0, 40.0));
        let integer = (rng.uniform(-50.0, 50.0)).round();
        check(a, integer);
        let small = rng.uniform(0.0, 2.0);
        check(small, rng.uniform(-200.0, 200.0));
    }
    if name.starts_with("powf") {
        // The expo easings' `pow(2, x)`, and the overflow and underflow bounds of `y·log2 x`.
        for d in 0..=2 * GRID_HALF_WIDTH {
            for y in [128.0_f32, -149.0, -150.0, 126.0, 10.0] {
                let y = f32::from_bits(y.to_bits().wrapping_add(d).wrapping_sub(GRID_HALF_WIDTH));
                check(2.0, y);
                check(-2.0, y);
                check(0.5, y);
            }
        }
    }
    if name.starts_with("atan2f") {
        // Subnormal quotients, where an exact `y/x` can be a midpoint of two results.
        let divisors = [1.5_f32, 2.0, 3.0, 5.0, 7.0, 10.0, 1e3, 2e20, 3e30];
        for b in (1..0x0080_0000_u32).step_by(97) {
            for x in divisors {
                check(f32::from_bits(b), x);
                check(f32::from_bits(b | 0x8000_0000), x);
            }
        }
    }
    m.assert_none();
}

/// The host functions are glibc's FMA build: they differ from an `f64` body rounded once on
/// `math.sin(13)` and on 893 entries of the sine table, and from the build without FMA on
/// `sinf(0x4255b0a9)`.
#[test]
fn the_oracle_is_glibc() {
    // Opaque arguments: an optimised build folds a call on a constant itself.
    let degrees = black_box(13.0_f32 * f32::from_bits(0x3c8e_fa35));
    assert_eq!(sinf(degrees).to_bits(), 0x3e66_5992);
    assert_eq!((f64::from(degrees).sin() as f32).to_bits(), 0x3e66_5991);
    let differing = sine_table_arguments()
        .filter(|&x| sinf(x).to_bits() != (f64::from(x).sin() as f32).to_bits())
        .count();
    assert_eq!(differing, 893, "glibc {}", glibc_version());
    assert_eq!(
        sinf(black_box(f32::from_bits(0x4255_b0a9))).to_bits(),
        0xbc7d_08a9,
        "the host's glibc {} is not running its FMA build",
        glibc_version()
    );
}

#[test]
fn sinf_matches_the_host() {
    check_unary(0);
}

#[test]
fn cosf_matches_the_host() {
    check_unary(1);
}

#[test]
fn expf_matches_the_host() {
    check_unary(2);
}

#[test]
fn exp2f_matches_the_host() {
    check_unary(3);
}

#[test]
fn logf_matches_the_host() {
    check_unary(4);
}

#[test]
fn log2f_matches_the_host() {
    check_unary(5);
}

#[test]
fn asinf_matches_the_host() {
    check_unary(6);
}

#[test]
fn acosf_matches_the_host() {
    check_unary(7);
}

#[test]
fn atanf_matches_the_host() {
    check_unary(8);
}

#[test]
fn the_compatibility_symbols_match_the_host() {
    check_unary(9);
    check_unary(10);
    check_unary(11);
    check_binary(2);
}

#[test]
fn powf_matches_the_host() {
    check_binary(0);
}

#[test]
fn atan2f_matches_the_host() {
    check_binary(1);
}

/// Every `f32` through each one-argument function, 2³² random pairs through each two-argument one.
#[test]
#[ignore = "a sweep of 2³² arguments per function; run it in release"]
fn exhaustive() {
    let selected: Option<Vec<String>> = std::env::var("MOLANG_EXHAUSTIVE")
        .ok()
        .map(|names| names.split(',').map(|n| n.trim().to_owned()).collect());
    let wanted = |name: &str| {
        selected
            .as_ref()
            .is_none_or(|s| s.iter().any(|n| n == name))
    };
    let threads = std::thread::available_parallelism().map_or(4, usize::from) as u64;
    let chunk = (1u64 << 32) / threads;
    let mut failures = Vec::new();
    for (name, port_fn, host) in &unary() {
        if !wanted(name) {
            continue;
        }
        let mut m = Mismatches::new(name);
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|t| {
                    s.spawn(move || {
                        let mut m = Mismatches::new(name);
                        let end = if t + 1 == threads {
                            1 << 32
                        } else {
                            (t + 1) * chunk
                        };
                        for b in t * chunk..end {
                            let x = f32::from_bits(b as u32);
                            m.check(&[x], port_fn(x), host(x));
                        }
                        m
                    })
                })
                .collect();
            for p in parts {
                m.merge(p.join().unwrap());
            }
        });
        eprintln!("{name}: {} of {} differ", m.count, m.checked);
        if m.count != 0 {
            failures.push(m);
        }
    }
    for (index, (name, port_fn, host)) in binary().iter().enumerate() {
        if !wanted(name) {
            continue;
        }
        let mut m = Mismatches::new(name);
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|t| {
                    s.spawn(move || {
                        let mut m = Mismatches::new(name);
                        let mut rng = Rng((index as u64) << 32 | t);
                        for _ in 0..chunk {
                            let (a, b) = if rng.next() & 1 == 0 {
                                (rng.bits(), rng.bits())
                            } else {
                                (rng.uniform(-10.0, 10.0), rng.uniform(-60.0, 60.0))
                            };
                            m.check(&[a, b], port_fn(a, b), host(a, b));
                        }
                        m
                    })
                })
                .collect();
            for p in parts {
                m.merge(p.join().unwrap());
            }
        });
        eprintln!("{name}: {} of {} differ", m.count, m.checked);
        if m.count != 0 {
            failures.push(m);
        }
    }
    for m in &failures {
        eprintln!("{}", m.examples.join("\n"));
    }
    if let Some(m) = failures.first() {
        m.assert_none();
    }
}
