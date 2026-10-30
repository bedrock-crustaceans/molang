//! The C library functions the crate calls for the transcendentals: the `libm` crate's under
//! `strict-math`, else `std`'s, so an expected value is the crate's own in either mode.

macro_rules! unary {
    ($($name:ident: $strict:ident, $host:ident;)*) => {$(
        pub fn $name(x: f32) -> f32 {
            if cfg!(feature = "strict-math") {
                libm::$strict(x)
            } else {
                x.$host()
            }
        }
    )*};
}

unary! {
    sin: sinf, sin;
    cos: cosf, cos;
    asin: asinf, asin;
    acos: acosf, acos;
    atan: atanf, atan;
    exp: expf, exp;
    exp2: exp2f, exp2;
    ln: logf, ln;
}

pub fn atan2(y: f32, x: f32) -> f32 {
    if cfg!(feature = "strict-math") {
        libm::atan2f(y, x)
    } else {
        y.atan2(x)
    }
}

pub fn pow(a: f32, b: f32) -> f32 {
    if cfg!(feature = "strict-math") {
        libm::powf(a, b)
    } else {
        a.powf(b)
    }
}

/// Whether the crate gives the transcendental bits BDS Linux logs: always under `strict-math`
/// (with the listed `libm` departures), else only on the host the logs were reproduced on.
pub fn reproduces_the_server() -> bool {
    cfg!(feature = "strict-math") || is_the_reference_host()
}

/// Linux x86-64 with glibc 2.41 and FMA.
#[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
pub fn is_the_reference_host() -> bool {
    unsafe extern "C" {
        fn gnu_get_libc_version() -> *const std::ffi::c_char;
    }
    // SAFETY: glibc returns a static NUL-terminated string.
    let version = unsafe { std::ffi::CStr::from_ptr(gnu_get_libc_version()) };
    version.to_bytes() == b"2.41" && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
pub fn is_the_reference_host() -> bool {
    false
}

/// The pinned bits: `strict` under `strict-math`, `host` on the reference host, else `otherwise`,
/// the crate's own result on a host whose C library is not pinned.
pub fn pinned(strict: u32, host: u32, otherwise: f32) -> f32 {
    if cfg!(feature = "strict-math") {
        f32::from_bits(strict)
    } else if is_the_reference_host() {
        f32::from_bits(host)
    } else {
        otherwise
    }
}
