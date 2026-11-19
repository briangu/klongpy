//! `kg fpga source` (src/emit_hls.rs): the golden files of tests/static/test_emit_hls.py and test_hls_stream.py, regenerated from
//! the programs and compared with the copies in tests/data/ (copied from tests/static/golden/hls_* and hlsb_*, so that this proof
//! survives the deletion of the Python tests), and the error texts the Python tests pin.
use klong_static::compile::{Failure, EmitFailure};
use klong_static::emit_hls::{emit_hls_program, json, HlsFailure, HlsRun};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn data(name: &str) -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)).unwrap_or_else(|e| panic!("{}: {}", name, e))
}

fn same(name: &str, text: &str) {
    let want = data(name);
    assert!(text == want, "{} differs from the golden file in tests/data", name);
}

fn emit(path: &str, text: Option<&str>, prefix: Option<&str>, wrapped: bool, lanes: Option<usize>) -> HlsRun {
    match emit_hls_program(path, None, text, 32, prefix, 300, wrapped, lanes, false) {
        Ok(r) => r,
        Err(HlsFailure::Emit(EmitFailure::Compile(Failure::Check(r)))) => panic!("check: {:?}", r.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>()),
        Err(HlsFailure::Emit(_)) => panic!("emit failed"),
        Err(HlsFailure::Hls(e, _)) => panic!("hls: {:?}", e),
    }
}

fn embedded(name: &str) -> String {
    root().join("examples/embedded").join(format!("{}.kg", name)).to_string_lossy().into_owned()
}

fn hls_error(path: &str, text: &str, wrapped: bool, lanes: Option<usize>) -> String {
    match emit_hls_program(path, None, Some(text), 32, None, 300, wrapped, lanes, false) {
        Err(f) => f.render(path).0.join("\n"),
        Ok(_) => panic!("accepted"),
    }
}

const SUMMARY: &str = "
.param(\"gain\";\"int\";2)
tot::0
cnt::0
step::{tot::tot+x*gain;cnt::cnt+1;0}
summary::{tot,cnt}
.entry(\"step\";\"int -> void\")
.entry(\"summary\";\"-> int[2]\")
";

const LANES_EMA: &str = "
.param(\"alpha\";\"fixed<32,16>\";0.5)
.param(\"depth\";\"int\";2)
acc::.fixed(\"fixed<32,16>\";0)
count::0
step::{acc::acc+((x-acc)*alpha);count::(count+1)!8;acc+.fixed(\"fixed<32,16>\";count&depth)}
.entry(\"step\";\"fixed<32,16> -> fixed<32,16>\")
";

#[test]
fn stage_a_golden_files() {
    for (name, entry) in [("mavg", "step"), ("fir", "run")] {
        let r = emit(&embedded(name), None, None, true, None);
        let f = &r.out.files;
        same(&format!("hls_{}_{}_top.cpp", name, entry), &f[&format!("{}_{}_top.cpp", name, entry)]);
        same(&format!("hls_{}_{}_tb.cpp", name, entry), &f[&format!("{}_{}_tb.cpp", name, entry)]);
        same(&format!("hls_{}_{}.tcl", name, entry), &f[&format!("hls_{}.tcl", entry)]);
        same(&format!("hls_{}_manifest.json", name), &f["manifest.json"]);
    }
}

#[test]
fn stage_b_golden_files() {
    let fir_fixed = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/fir_fixed.kg");
    for (name, entry, path) in [("mavg", "step", embedded("mavg")), ("fir", "run", embedded("fir")), ("firfx", "run", fir_fixed.to_string_lossy().into_owned())] {
        let r = emit(&path, None, Some(name), false, None);
        let f = &r.out.files;
        same(&format!("hlsb_{}_{}_top.cpp", name, entry), &f[&format!("{}_{}_top.cpp", name, entry)]);
        same(&format!("hlsb_{}.c", name), &f[&format!("{}.c", name)]);
        same(&format!("hlsb_{}_manifest.json", name), &f["manifest.json"]);
    }
    let r = emit("ema.kg", Some(LANES_EMA), Some("ema"), false, Some(4));
    let f = &r.out.files;
    same("hlsb_ema4_step_top.cpp", &f["ema_step_top.cpp"]);
    same("hlsb_ema4_step_tb.cpp", &f["ema_step_tb.cpp"]);
    same("hlsb_ema4_manifest.json", &f["manifest.json"]);
    let r = emit("sm.kg", Some(SUMMARY), Some("sm"), false, Some(3));
    let f = &r.out.files;
    same("hlsb_summary3_step_top.cpp", &f["sm_step_top.cpp"]);
    same("hlsb_summary3_step_tb.cpp", &f["sm_step_tb.cpp"]);
    same("hlsb_summary3_manifest.json", &f["manifest.json"]);
}

#[test]
fn the_summary_pair_with_lanes_is_one_process_per_lane() {
    let r = emit("sm.kg", Some(SUMMARY), Some("sm"), false, Some(3));
    assert!(r.out.files.contains_key("lane_0.c") && r.out.files.contains_key("lane_2.c"));
    assert_eq!(r.out.entries.len(), 1, "the reader is not a kernel of its own");
    assert_eq!(r.out.entries[0].out_words, 6);
}

#[test]
fn rejections() {
    let e = hls_error("t.kg", "t::0\nstep::{t::t+x;0}\n.entry(\"step\";\"int -> void\")\n", false, None);
    assert!(e.contains("entry step (int -> void) cannot be a hardware kernel: it returns nothing, so there is no result record to read back [hls]"), "{}", e);
    let two = format!("{}other::{{cnt}}\n.entry(\"other\";\"-> int\")\n", SUMMARY);
    let e = hls_error("t.kg", &two, false, None);
    assert!(e.contains("returns nothing"), "{}", e);
    let e = hls_error("t.kg", ".param(\"n\";\"int\";2)\nf::{x+n}\n.entry(\"f\";\"int -> int\")\n", false, None);
    assert!(e.starts_with("static error: the parameter n cannot be a kernel register"), "{}", e);
    assert!(e.ends_with("\n  hint: rename the parameter, for example n_p"), "{}", e);
}

#[test]
fn json_is_python_dumps() {
    use json::J;
    let v = J::Obj(vec![
        ("a".into(), J::Arr(vec![])),
        ("b".into(), J::Arr(vec![J::Int(1), J::Flt(0.5), J::Flt(1e16), J::Flt(3.0)])),
        ("c".into(), J::Obj(vec![])),
        ("d".into(), J::Str("q\"\\\n\u{e9}\u{1f600}".into())),
        ("e".into(), J::Bool(true)),
    ]);
    let want = "{\n  \"a\": [],\n  \"b\": [\n    1,\n    0.5,\n    1e+16,\n    3.0\n  ],\n  \"c\": {},\n  \"d\": \"q\\\"\\\\\\n\\u00e9\\ud83d\\ude00\",\n  \"e\": true\n}";
    assert_eq!(json::dumps(&v), want);
}
