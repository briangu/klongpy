use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/gpu_metal.m");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var_os("CARGO_FEATURE_METAL").is_none()
    {
        return;
    }
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        _ => panic!("unsupported macOS architecture"),
    };
    // Match rustc's minimum OS, including its architecture-specific default and any wheel override.
    let deployment = Command::new(env::var_os("RUSTC").expect("Cargo rustc"))
        .args(["--print", "deployment-target", "--target"])
        .arg(env::var_os("TARGET").expect("Cargo target triple"))
        .output()
        .expect("query rustc's macOS deployment target");
    assert!(
        deployment.status.success(),
        "query rustc's macOS deployment target"
    );
    let deployment =
        std::str::from_utf8(&deployment.stdout).expect("rustc deployment target output");
    let version = deployment
        .trim()
        .strip_prefix("MACOSX_DEPLOYMENT_TARGET=")
        .expect("rustc macOS deployment version");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    let object = out.join("gpu_metal.o");
    let archive = out.join("libklong_metal.a");
    let status = Command::new("xcrun")
        .args([
            "--sdk",
            "macosx",
            "clang",
            "-arch",
            arch,
            "-fobjc-arc",
            "-O2",
            "-Wall",
            "-Wextra",
        ])
        .arg(format!("-mmacosx-version-min={version}"))
        .args(["-c", "src/gpu_metal.m", "-o"])
        .arg(&object)
        .status()
        .expect("launch system clang for the Metal bridge");
    assert!(status.success(), "compile the Klong Metal bridge");
    let status = Command::new("xcrun")
        .args(["--sdk", "macosx", "libtool", "-static", "-o"])
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("launch system libtool for the Metal bridge");
    assert!(status.success(), "archive the Klong Metal bridge");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=klong_metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}
