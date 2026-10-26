//! The NaN / signed-zero / infinity sweeps of `server_sweeps.json`, replayed on both releases.
//!
//! Each expression evaluates one math function or operator on NaN of either sign, ±0, ±infinity or
//! ordinary values inside `v.one ? (…) : 0` and reads the class of the result: ±infinity, finite
//! (for some split into equal to a stated value with either sign, and not equal), or NaN with the
//! sign bit clear or set. A record holds `run`, `id`, `expr`, an optional `version`, the
//! `question`, the `branches` it can answer, the branch each release took (`observed`) and
//! `predicted_x86_64`, which the replay ignores. Each release is replayed on its own platform, and
//! every probe on the other platforms too.

#![cfg(all(feature = "compiler", feature = "stdlib"))]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use common::measured::*;
use common::{data_json, data_path};
use molang::numeric::Platform;
use serde_json::Value;

/// The probes of `math.ln` of a negative number, which takes the NaN sign of its release: `nann` on
/// 1.26.36.1, `nanp` on 1.26.52.3. The two releases differ on these only.
const LN_OF_A_NEGATIVE: [(u64, &str); 8] = [
    (44, "s0038"),
    (44, "s0039"),
    (44, "s0042"),
    (44, "s0043"),
    (44, "s0046"),
    (44, "s0047"),
    (44, "s0056"),
    (44, "s0057"),
];

/// The probes each other platform answers differently from the servers: the NaN of an invalid
/// operation, the NaN an operation of two NaNs keeps, and what a NaN converts or compares to.
const OTHER_PLATFORMS_DIFFER: &[(u64, Platform, &[&str])] = &[
    (
        44,
        Platform::AndroidArm64,
        &[
            "s0000", "s0001", "s0010", "s0011", "s0014", "s0015", "s0018", "s0019", "s0028",
            "s0029", "s0038", "s0039", "s0042", "s0043", "s0046", "s0047", "s0056", "s0057",
            "s0058", "s0059", "s0088", "s0089", "s0126", "s0127", "s0164", "s0165", "s0172",
            "s0173", "s0174", "s0175", "s0200", "s0201", "s0208", "s0209", "s0210", "s0211",
            "s0236", "s0237", "s0270", "s0271", "s0298", "s0299", "s0326", "s0327", "s0352",
            "s0353", "s0360", "s0361", "s0362", "s0363", "s0386", "s0387", "s0394", "s0395",
        ],
    ),
    (
        44,
        Platform::BdsWindows,
        &[
            "s0096", "s0097", "s0098", "s0099", "s0108", "s0109", "s0110", "s0111", "s0120",
            "s0121", "s0122", "s0123", "s0124", "s0125", "s0134", "s0135", "s0136", "s0137",
            "s0146", "s0147", "s0148", "s0149", "s0158", "s0159", "s0160", "s0161", "s0162",
            "s0163", "s0038", "s0039", "s0042", "s0043", "s0046", "s0047", "s0056", "s0057",
        ],
    ),
    (
        45,
        Platform::AndroidArm64,
        &[
            "s0408", "s0409", "s0413", "s0414", "s0415", "s0416", "s0491", "s0492", "s0493",
            "s0494", "s0515", "s0516", "s0517", "s0518", "s0539", "s0540", "s0541", "s0542",
            "s0543", "s0544", "s0545", "s0546", "s0547", "s0548", "s0549", "s0550", "s0563",
            "s0564", "s0575", "s0576", "s0587", "s0588", "s0599", "s0600", "s0635", "s0636",
            "s0637", "s0638", "s0659", "s0660", "s0671", "s0672", "s0683", "s0684", "s0695",
            "s0696", "s0707", "s0708", "s0719", "s0720", "s0734", "s0738", "s0779", "s0780",
            "s0781", "s0783", "s0784", "s0785",
        ],
    ),
    (
        46,
        Platform::AndroidArm64,
        &[
            "s0827", "s0828", "s0839", "s0840", "s0851", "s0852", "s0863", "s0864", "s0875",
            "s0876", "s0887", "s0888", "s0902", "s0906", "s0947", "s0948", "s0949", "s0951",
            "s0952", "s0953", "s0995", "s0996", "s1007", "s1008", "s1019", "s1020", "s1031",
            "s1032", "s1043", "s1044", "s1055", "s1056", "s1067", "s1068", "s1069", "s1070",
            "s1071", "s1072", "s1073", "s1074", "s1099", "s1100", "s1101", "s1102", "s1103",
            "s1104", "s1105", "s1106", "s1107", "s1108", "s1109", "s1110", "s1111", "s1112",
            "s1113", "s1114", "s1115", "s1116", "s1117", "s1118", "s1119", "s1120", "s1121",
            "s1122", "s1132", "s1133", "s1136", "s1137", "s1140", "s1141", "s1144", "s1145",
            "s1163", "s1164", "s1165", "s1166", "s1168", "s1170", "s1171", "s1172", "s1173",
            "s1174", "s1187", "s1188", "s1192", "s1194", "s1199", "s1200", "s1204",
        ],
    ),
    (
        47,
        Platform::AndroidArm64,
        &[
            "s1206", "s1211", "s1212", "s1216", "s1218", "s1219", "s1220", "s1221", "s1222",
            "s1223", "s1224", "s1228", "s1230", "s1231", "s1232", "s1233", "s1234", "s1299",
            "s1300", "s1301", "s1302", "s1343", "s1344", "s1345", "s1346", "s1351", "s1352",
            "s1353", "s1354", "s1367", "s1368", "s1379", "s1380", "s1391", "s1392", "s1403",
            "s1404", "s1415", "s1416", "s1435", "s1436", "s1446", "s1447", "s1458", "s1459",
            "s1460", "s1462", "s1463", "s1464", "s1465", "s1466", "s1467", "s1468", "s1469",
            "s1470", "s1471", "s1483", "s1494", "s1495", "s1506", "s1507", "s1508", "s1510",
            "s1516", "s1517", "s1518", "s1519", "s1520", "s1522", "s1524", "s1526", "s1527",
            "s1528", "s1530", "s1532", "s1539", "s1540", "s1563", "s1564", "s1565", "s1566",
            "s1567", "s1568", "s1569", "s1570", "s1571", "s1572", "s1573", "s1574",
        ],
    ),
    (
        48,
        Platform::AndroidArm64,
        &[
            "s1655", "s1656", "s1687", "s1688", "s1695", "s1696", "s1697", "s1698", "s1699",
            "s1700", "s1711", "s1712", "s1723", "s1724", "s1725", "s1726", "s1727", "s1728",
            "s1735", "s1736", "s1743", "s1744", "s1745", "s1746", "s1747", "s1748", "s1759",
            "s1760", "s1773", "s1774", "s1781", "s1782", "s1789", "s1790", "s1791", "s1792",
            "s1793", "s1794", "s1801", "s1802", "s1803", "s1804", "s1805", "s1806", "s1823",
            "s1824", "s1840", "s1841", "s1844", "s1846", "s1848", "s1849", "s1865", "s1866",
            "s1869", "s1871", "s1873", "s1874", "s1890", "s1891", "s1894", "s1896", "s1898",
            "s1899", "s1915", "s1916", "s1919", "s1921", "s1923", "s1924", "s1940", "s1941",
            "s1944", "s1946", "s1948", "s1949", "s1965", "s1966", "s1969", "s1971", "s1973",
            "s1974", "s1990", "s1991", "s1994", "s1996", "s1998", "s1999",
        ],
    ),
    (
        49,
        Platform::AndroidArm64,
        &[
            "s2015", "s2016", "s2019", "s2021", "s2023", "s2024", "s2040", "s2041", "s2044",
            "s2046", "s2048", "s2049", "s2065", "s2066", "s2069", "s2071", "s2073", "s2074",
            "s2090", "s2091", "s2094", "s2096", "s2098", "s2099", "s2115", "s2116", "s2119",
            "s2121", "s2128", "s2129", "s2140", "s2141", "s2144", "s2145", "s2146", "s2147",
            "s2153", "s2154", "s2165", "s2166", "s2169", "s2171", "s2178", "s2179", "s2190",
            "s2191", "s2194", "s2196", "s2198", "s2199", "s2215", "s2216", "s2219", "s2221",
            "s2223", "s2224", "s2240", "s2241", "s2244", "s2246", "s2248", "s2249", "s2265",
            "s2266", "s2269", "s2271", "s2273", "s2274", "s2278", "s2279", "s2280", "s2288",
            "s2290", "s2291", "s2294", "s2296", "s2298", "s2299", "s2303", "s2304", "s2305",
            "s2314", "s2315", "s2316", "s2319", "s2321", "s2323", "s2324", "s2328", "s2329",
            "s2330", "s2338", "s2340", "s2341", "s2344", "s2346", "s2348", "s2349", "s2365",
            "s2366", "s2373", "s2374", "s2390", "s2391", "s2394", "s2396", "s2398", "s2399",
        ],
    ),
    (49, Platform::BdsWindows, &["s2290", "s2291"]),
    (
        50,
        Platform::AndroidArm64,
        &[
            "s2415", "s2416", "s2419", "s2421", "s2423", "s2424", "s2428", "s2429", "s2440",
            "s2441", "s2448", "s2449", "s2455", "s2465", "s2466", "s2469", "s2471", "s2473",
            "s2474", "s2490", "s2491", "s2494", "s2495", "s2496", "s2497", "s2498", "s2499",
            "s2515", "s2516", "s2519", "s2520", "s2521", "s2522", "s2523", "s2524", "s2540",
            "s2541", "s2544", "s2546", "s2548", "s2549", "s2565", "s2566", "s2569", "s2570",
            "s2571", "s2572", "s2573", "s2574", "s2575", "s2576", "s2577", "s2578", "s2579",
            "s2580", "s2581", "s2582", "s2593", "s2603", "s2613", "s2618", "s2623", "s2627",
            "s2633", "s2643", "s2653", "s2663", "s2673", "s2674", "s2675", "s2676", "s2677",
            "s2678", "s2679", "s2680", "s2681", "s2682", "s2683", "s2684", "s2685", "s2686",
            "s2687", "s2688", "s2705", "s2706", "s2721", "s2722", "s2737", "s2738", "s2745",
            "s2746", "s2753", "s2754", "s2759", "s2760", "s2769", "s2770", "s2786", "s2787",
            "s2788", "s2789", "s2790", "s2791", "s2792", "s2793", "s2794", "s2805",
        ],
    ),
    (
        51,
        Platform::AndroidArm64,
        &[
            "s2815", "s2825", "s2835", "s2845", "s2855", "s2865", "s2875", "s2885", "s2886",
            "s2887", "s2888", "s2889", "s2890", "s2891", "s2892", "s2893", "s2894", "s2895",
            "s2896", "s2897", "s2898", "s2899", "s2900", "s2902", "s2917", "s2918", "s2933",
            "s2934", "s2949", "s2950", "s2955", "s2956", "s2965", "s2966", "s2973", "s2974",
            "s2981", "s2982", "s2997", "s2998", "s2999", "s3000", "s3001", "s3002", "s3003",
            "s3004", "s3005", "s3006", "s3017", "s3021", "s3022", "s3027", "s3031", "s3032",
            "s3037", "s3039", "s3040", "s3047", "s3049", "s3050", "s3057", "s3067", "s3077",
            "s3087", "s3097", "s3098", "s3099", "s3100", "s3101", "s3102", "s3103", "s3104",
            "s3113", "s3116", "s3117", "s3121", "s3124", "s3125", "s3129", "s3130", "s3131",
            "s3132", "s3133", "s3137", "s3138", "s3139", "s3140", "s3141", "s3145", "s3148",
            "s3149", "s3153", "s3154", "s3157", "s3158", "s3159", "s3160", "s3173", "s3183",
            "s3193", "s3197", "s3198", "s3203", "s3207", "s3208", "s3213",
        ],
    ),
    (
        52,
        Platform::AndroidArm64,
        &[
            "s3223", "s3233", "s3243", "s3253", "s3254", "s3256", "s3259", "s3262", "s3263",
            "s3264", "s3265", "s3266", "s3267", "s3268", "s3269", "s3270", "s3271", "s3274",
            "s3302", "s3326", "s3349", "s3350", "s3352", "s3355", "s3358", "s3359", "s3360",
            "s3361", "s3362", "s3363", "s3367", "s3370", "s3373", "s3374", "s3376", "s3379",
            "s3382", "s3383", "s3384", "s3385", "s3386", "s3387", "s3391", "s3394", "s3398",
        ],
    ),
    (
        53,
        Platform::AndroidArm64,
        &[
            "s3853", "s3854", "s3893", "s3894", "s3913", "s3914", "s3933", "s3934", "s3953",
            "s3954", "s3961", "s3962", "s3963", "s3964", "s3973", "s3974", "s3993", "s3994",
        ],
    ),
    (
        54,
        Platform::AndroidArm64,
        &[
            "u0000", "u0001", "u0004", "u0005", "u0008", "u0009", "u0012", "u0013", "u0016",
            "u0017", "u0020", "u0021", "u0024", "u0025", "u0028", "u0029", "u0032", "u0033",
            "u0036", "u0037", "u0040", "u0041", "u0044", "u0045", "u0048", "u0049", "u0052",
            "u0053", "u0056", "u0057", "u0060", "u0061", "u0064", "u0065", "u0068", "u0069",
            "u0072", "u0073", "u0076", "u0077", "u0080", "u0081", "u0084", "u0085", "u0088",
            "u0089", "u0092", "u0093", "u0096", "u0098", "u0100", "u0102", "u0104", "u0106",
            "u0108", "u0110", "u0112", "u0114", "u0115", "u0122", "u0123", "u0124", "u0125",
            "u0128", "u0129", "u0134", "u0135", "u0136", "u0137", "u0146", "u0147", "u0148",
            "u0149", "u0152", "u0153", "u0156", "u0157", "u0162", "u0163", "u0166", "u0167",
            "u0170", "u0171", "u0172", "u0173", "u0176", "u0177", "u0182", "u0183", "u0186",
            "u0187", "u0190", "u0191", "u0194", "u0195", "u0196", "u0197", "u0200", "u0201",
            "u0204", "u0205", "u0209", "u0212", "u0214",
        ],
    ),
    (
        55,
        Platform::AndroidArm64,
        &[
            "e0001", "e0002", "e0004", "e0006", "e0009", "e0010", "e0013", "e0015", "e0017",
            "e0019", "e0020", "e0022", "e0025", "e0027", "e0029", "e0031", "e0033", "e0035",
            "e0037", "e0038", "e0040", "e0042", "e0045", "e0046", "e0049", "e0051", "e0053",
            "e0055", "e0056", "e0058", "e0061", "e0063", "e0065", "e0067", "e0069", "e0071",
            "e0073", "e0074", "e0076", "e0079", "e0080", "e0082", "e0085", "e0086", "e0088",
            "e0091", "e0093", "e0095", "e0097", "e0099", "e0101", "e0103", "e0105", "e0107",
            "e0109", "e0111", "e0113", "e0115", "e0116", "e0118", "e0121", "e0122", "e0124",
            "e0127", "e0129", "e0131", "e0133", "e0135", "e0137", "e0139", "e0141", "e0143",
            "e0144", "e0146", "e0148", "e0150", "e0153", "e0154", "e0156", "e0159", "e0160",
            "e0163", "e0164", "e0166", "e0169", "e0170", "e0172", "e0174", "e0177", "e0178",
            "e0196", "e0197", "e0199",
        ],
    ),
    (
        55,
        Platform::BdsWindows,
        &[
            "e0012", "e0013", "e0014", "e0015", "e0016", "e0017", "e0030", "e0031", "e0032",
            "e0033", "e0034", "e0035", "e0048", "e0049", "e0050", "e0051", "e0052", "e0053",
            "e0066", "e0067", "e0068", "e0069", "e0070", "e0071", "e0102", "e0103", "e0106",
            "e0107", "e0110", "e0111", "e0112", "e0113", "e0114", "e0115", "e0120", "e0121",
            "e0138", "e0139", "e0142", "e0143", "e0176", "e0177", "e0180", "e0181", "e0182",
            "e0183", "e0184", "e0185", "e0186", "e0187", "e0190", "e0191", "e0192", "e0193",
            "e0194", "e0195", "e0198",
        ],
    ),
];

/// Parsed once for every test of the file.
fn sweeps() -> &'static Value {
    static SWEEPS: LazyLock<Value> = LazyLock::new(|| data_json("server_sweeps.json"));
    &SWEEPS
}

fn str_field<'a>(record: &'a Value, key: &str) -> &'a str {
    record[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not a string in {record}"))
}

fn run_number(record: &Value) -> u64 {
    record["run"]
        .as_u64()
        .unwrap_or_else(|| panic!("run is not a number in {record}"))
}

fn sweep_run(run: u64) -> ServerRun {
    let data = sweeps();
    let meta = data["runs"]
        .as_array()
        .expect("runs")
        .iter()
        .find(|r| run_number(r) == run)
        .unwrap_or_else(|| panic!("run {run} is not in tests/server_sweeps.json"));
    let mut server_run = ServerRun::new(
        &format!("run_{run:02}"),
        BOTH_RELEASES,
        str_field(meta, "purpose"),
    );
    for record in data["probes"]
        .as_array()
        .expect("probes")
        .iter()
        .filter(|p| run_number(p) == run)
    {
        let id = str_field(record, "id");
        let probe = server_run.probe(id, str_field(record, "expr"));
        if let Some(version) = record.get("version") {
            let version = version
                .as_i64()
                .and_then(|v| i16::try_from(v).ok())
                .expect("version");
            probe.at(version);
        }
        for release in BOTH_RELEASES {
            probe.answers_on(*release, str_field(&record["observed"], release.text()));
        }
    }
    for (_, platform, ids) in OTHER_PLATFORMS_DIFFER.iter().filter(|(r, _, _)| *r == run) {
        server_run.differs_on(*platform, ids);
    }
    server_run
}

/// Replays one run of `probes` entries and checks that every expected failure was met.
fn replay(run: u64, probes: usize) {
    let report = sweep_run(run).replay(probes);
    assert_eq!(
        report.not_run, 0,
        "run_{run}: every sweep probe was answered"
    );
}

/// `math.sqrt`, `math.ln`, `math.exp`, `math.asin`, `math.acos`, `math.sin`, `math.cos`,
/// `math.round`, `math.floor`, `math.ceil`, `math.trunc`, `math.min_angle`, `math.hermite_blend`.
#[test]
fn roots_logarithms_trigonometry_and_rounding_sweep_run_44() {
    replay(44, 406);
}

/// `math.hermite_blend`, `math.atan`, `math.abs`, `math.sign`, `math.atan2`, `math.copy_sign`,
/// `math.max`.
#[test]
fn abs_sign_atan2_copy_sign_and_max_sweep_run_45() {
    replay(45, 406);
}

/// `math.max`, `math.min`, `math.mod`.
#[test]
fn max_min_and_mod_sweep_run_46() {
    replay(46, 406);
}

/// `math.mod`, `math.pow`, `math.random`, `math.random_integer`.
#[test]
fn mod_pow_and_random_sweep_run_47() {
    replay(47, 406);
}

/// `math.random_integer`, `math.clamp`, `math.lerp`, `math.inverse_lerp`, `math.lerprotate`, and
/// the quadratic, cubic and quartic easings.
#[test]
fn clamp_lerps_and_polynomial_easings_sweep_run_48() {
    replay(48, 406);
}

/// The easings from `math.ease_out_quart` to `math.ease_in_out_back`.
#[test]
fn quartic_to_back_easings_sweep_run_49() {
    replay(49, 406);
}

/// `math.ease_in_out_back`, the elastic and bounce easings, then the binary `+` and `-`.
#[test]
fn elastic_and_bounce_easings_and_addition_sweep_run_50() {
    replay(50, 406);
}

/// The binary `-`, `*` and `/`.
#[test]
fn subtraction_multiplication_and_division_sweep_run_51() {
    replay(51, 406);
}

/// `/`, `==`, `!=`, `<`, `<=`.
#[test]
fn division_and_comparisons_sweep_run_52() {
    replay(52, 406);
}

/// `<=`, `>`, `>=`, `&&`, `||`, the unary `-` and `!`, and `0 - x`.
#[test]
fn comparisons_logical_operators_and_negation_sweep_run_53() {
    replay(53, 399);
}

/// Chains of `+`, `-`, `*` and `/`, sums and products inside other operations, lerps, clamp,
/// easings, `math.die_roll` and `math.die_roll_integer`.
#[test]
fn operator_chains_lerps_and_die_rolls_follow_up_sweep_run_54() {
    replay(54, 248);
}

/// Every easing with a NaN of each sign in two of its three arguments, in every order.
#[test]
fn easing_pairs_sweep_run_55() {
    replay(55, 204);
}

/// Ids are unique per run and every entry has both releases' branch; the releases differ exactly on
/// [`LN_OF_A_NEGATIVE`].
#[test]
fn the_sweep_file_is_complete() {
    let data = sweeps();
    let releases: Vec<&str> = data["releases"]
        .as_array()
        .expect("releases")
        .iter()
        .map(|r| r.as_str().expect("release"))
        .collect();
    assert_eq!(
        releases,
        BOTH_RELEASES.iter().map(|r| r.text()).collect::<Vec<_>>()
    );

    let runs: Vec<u64> = data["runs"]
        .as_array()
        .expect("runs")
        .iter()
        .map(run_number)
        .collect();
    assert_eq!(runs, (44..=55).collect::<Vec<_>>());
    for run in &runs {
        for release in BOTH_RELEASES {
            let path = data_path(&format!("server_logs/run_{run:02}_{}.log", release.text()));
            assert!(path.is_file(), "missing {}", path.display());
        }
    }

    let probes = data["probes"].as_array().expect("probes");
    assert_eq!(probes.len(), 4_505);
    let mut ids: BTreeMap<u64, BTreeSet<&str>> = BTreeMap::new();
    let mut differ = Vec::new();
    let mut previous_run = 0;
    for record in probes {
        let run = run_number(record);
        assert!(runs.contains(&run), "probe of an unknown run: {record}");
        assert!(run >= previous_run, "probes out of run order: {record}");
        previous_run = run;
        let id = str_field(record, "id");
        assert!(
            ids.entry(run).or_default().insert(id),
            "run_{run}: id {id} repeats"
        );
        let observed = record["observed"].as_object().expect("observed");
        assert_eq!(
            observed.len(),
            2,
            "run_{run} {id}: observed on both releases"
        );
        let first = str_field(&record["observed"], Release::V1_26_36_1.text());
        let second = str_field(&record["observed"], Release::V1_26_52_3.text());
        if first != second {
            assert_eq!(first, format!("{id}_nann"), "run_{run} {id}");
            assert_eq!(second, format!("{id}_nanp"), "run_{run} {id}");
            differ.push((run, id));
        }
    }
    assert_eq!(differ, LN_OF_A_NEGATIVE);
}
