//! The C ABI, tested from C: compiles `tests/host.c` with the system C compiler against the
//! host build of the static library and runs it. The program checks a console transcript
//! (banner, `1+2`, an upload with `^A put`, `^A run`, `^A ver`, the stack and heap limits,
//! a timer, `.x`) through the functions and callbacks of `include/klong.h`.
use std::path::PathBuf;
use std::process::Command;

#[test]
fn c_program_drives_the_console() {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // The test program is target/<profile>/deps/c_abi-<hash>; cargo has put the static
    // library next to it as libklong_ffi-<hash>.a (the newest, if old builds linger).
    let exe = std::env::current_exe().expect("test path");
    let deps = exe.parent().expect("deps directory");
    let lib = std::fs::read_dir(deps)
        .expect("deps directory is readable")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with("libklong_ffi-") && name.ends_with(".a")
        })
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
        .expect("libklong_ffi-*.a beside the test: cargo builds it with this test");
    let out = deps.join("klong-ffi-host");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-I"])
        .arg(here.join("include"))
        .arg(here.join("tests/host.c"))
        .arg(&lib)
        .arg("-o")
        .arg(&out)
        .status()
        .expect("the C compiler runs");
    assert!(status.success(), "compiling tests/host.c failed");
    let run = Command::new(&out).output().expect("the C program runs");
    let text = String::from_utf8_lossy(&run.stdout);
    println!("{text}");
    assert!(run.status.success(), "the C program failed:\n{text}\n{}", String::from_utf8_lossy(&run.stderr));
    assert!(text.contains("<< 3\n") && text.trim_end().ends_with("-- ok"), "{text}");
}
