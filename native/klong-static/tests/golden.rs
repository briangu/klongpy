//! The golden IR dumps and C of the reference programs (copied from tests/static/golden so that these tests do not depend on the Python
//! implementation or its test tree), and the run of the emitted C through the host C compiler for the reference programs.
use klong_static::compile::{compile_program, emit_program};
use klong_static::ir::dump;

fn golden(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/data/golden/{}", env!("CARGO_MANIFEST_DIR"), name)).unwrap()
}

fn embedded(name: &str) -> String {
    format!("{}/../../examples/embedded/{}.kg", env!("CARGO_MANIFEST_DIR"), name)
}

const DOT: &str = "f::{+/x*y}\n.entry(\"f\";\"float[256] float[256] -> float\")";
const AXPY: &str = "f::{x+y*z}\n.entry(\"f\";\"float[256] float[256] float[256] -> float[256]\")";

fn c_of_text(text: &str) -> String {
    let r = emit_program("t.kg", None, Some(text), 32, 32, false, None, None).ok().expect("compiles");
    r.emitted.c
}

fn c_of_file(name: &str) -> String {
    let r = emit_program(&embedded(name), None, None, 32, 32, false, None, None).ok().expect("compiles");
    r.emitted.c
}

#[test]
fn ir_dumps_of_the_reference_programs() {
    for name in ["mavg", "pid", "fir"] {
        let p = compile_program(&embedded(name), None, 32, None).ok().expect("compiles").program;
        assert_eq!(dump(&p), golden(&format!("{}.ir", name)), "{}.ir", name);
    }
}

#[test]
fn golden_c() {
    assert_eq!(c_of_text(DOT), golden("dot.c"));
    assert_eq!(c_of_text(AXPY), golden("axpy.c"));
    for name in ["mavg", "pid"] {
        assert_eq!(c_of_file(name), golden(&format!("{}.c", name)), "{}.c", name);
    }
    let fir = c_of_file("fir");
    let at = fir.find("static void kg_run__1(fir_t *p, const kg_float *a0_x, kg_float *out);").expect("the run function");
    assert_eq!(&fir[at..], golden("fir_run.c"));
}

#[test]
fn the_runtime_header_is_the_one_of_klongpy() {
    // (skipped when the Python package is gone)
    let p = format!("{}/../../klongpy/static/klong_rt.h", env!("CARGO_MANIFEST_DIR"));
    if let Ok(py) = std::fs::read_to_string(p) {
        assert_eq!(klong_static::compile::RUNTIME_HEADER, py);
    }
}

fn find_cc() -> Option<String> {
    for c in ["cc", "clang", "gcc"] {
        if std::process::Command::new(c).arg("--version").output().is_ok() {
            return Some(c.to_string());
        }
    }
    None
}

#[test]
fn the_emitted_c_compiles_without_warnings() {
    let Some(cc) = find_cc() else { return };
    let dir = std::env::temp_dir().join(format!("kgs_c_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (i, (path, text, bits64)) in [(Some("pid"), None, false), (Some("mavg"), None, true), (Some("fir"), None, false), (None, Some(DOT), false), (None, Some(AXPY), true)].into_iter().enumerate() {
        let out = dir.join(format!("p{}", i));
        let (prefix, c, h) = {
            let r = match (path, text) {
                (Some(n), _) => emit_program(&embedded(n), None, None, if bits64 { 64 } else { 32 }, 32, false, Some(&format!("p{}", i)), None),
                (_, Some(t)) => emit_program("t.kg", None, Some(t), if bits64 { 64 } else { 32 }, 32, false, Some(&format!("p{}", i)), None),
                _ => unreachable!(),
            }
            .ok()
            .expect("compiles");
            (r.emitted.prefix.clone(), r.emitted.c.clone(), r.emitted.h.clone())
        };
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join(format!("{}.c", prefix)), c).unwrap();
        std::fs::write(out.join(format!("{}.h", prefix)), h).unwrap();
        std::fs::write(out.join("klong_rt.h"), klong_static::compile::RUNTIME_HEADER).unwrap();
        let o = std::process::Command::new(&cc).args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-c", "-o"]).arg(out.join("x.o")).arg(out.join(format!("{}.c", prefix))).output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}
