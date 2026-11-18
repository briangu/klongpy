//! The names a fresh interpreter binds: what W002 treats as defined.
//!
//! They are the globals of a runtime as the `klong` binary builds it (`klong_rt::Runtime` on the host platform): the system functions
//! and variables it registers, read from the interpreter itself, so the list cannot drift from what the runtime does. A program that
//! runs in an embedding (the Python package adds `.py`, `.fpga`, ...) names those through `Options::extra_builtins`.
use std::collections::BTreeSet;
use std::sync::OnceLock;

use klong_core::Config;
use klong_host::{Console, HostPlatform};
use klong_rt::Runtime;

/// Names of the language itself that the parser handles and the runtime does not register as globals: `.comment("END")` hides text up
/// to the marker, and `.f` is the function that is running.
const PARSER_NAMES: &[&str] = &[".comment", ".f"];

/// The names bound in a fresh runtime, and the parser's own.
pub fn builtin_names() -> &'static BTreeSet<String> {
    static NAMES: OnceLock<BTreeSet<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let rt = Runtime::new(HostPlatform::new(Console::Buffer(Vec::new())), Config::default(), None);
        let mut set: BTreeSet<String> = rt.interp.global_names().into_iter().collect();
        set.extend(PARSER_NAMES.iter().map(|s| s.to_string()));
        set
    })
}
