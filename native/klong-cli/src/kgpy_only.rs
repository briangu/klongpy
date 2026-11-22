//! The system functions that only `kgpy`, the Python-hosted command, has (principle 2: a function of the host is named as one).
//! `kg` answers a call of one with a `name` error that says where it is, instead of "undefined function".
//!
//! One table, from what the Python package registers (`klongpy/sys_fn*.py`, `klongpy/db`, `klongpy/web`, `klongpy/ws`):
//! * Python: `.py .pya .pyc .pyf`, `.sys`, `.rls`;
//! * torch and autograd: `.grad .gradcheck .jacobian .export .bkf .bkd .bkp`;
//! * legacy database helpers: `.kvs .tables`. Native table and SQL functions are listed only if disabled in this build;
//! * the web and web sockets: `.web .webc .ws .wsc`;
//! * the FPGA card: `.fpga` and `.fpga.free .fpga.get .fpga.info .fpga.lanes .fpga.put .fpga.reset .fpga.set`.
//!
//! A name that `kg` has itself must not be here (the test below checks it against the runtime).

pub const NAMES: &[&str] = &[
    ".bkd", ".bkf", ".bkp",
    #[cfg(not(feature = "duckdb"))]
    ".db",
    ".export", ".fpga", ".fpga.free", ".fpga.get", ".fpga.info", ".fpga.lanes", ".fpga.put", ".fpga.reset", ".fpga.set", ".grad", ".gradcheck",
    #[cfg(not(feature = "tables"))]
    ".index",
    #[cfg(not(feature = "tables"))]
    ".insert",
    ".jacobian", ".kvs", ".py", ".pya", ".pyc", ".pyf",
    #[cfg(not(feature = "tables"))]
    ".rindex",
    ".rls",
    #[cfg(not(feature = "tables"))]
    ".schema",
    ".sys",
    #[cfg(not(feature = "tables"))]
    ".table",
    ".tables", ".web", ".webc", ".ws", ".wsc",
];

pub fn is(name: &str) -> bool {
    // (a module-qualified name carries its module after a backquote)
    let name = name.split('`').next().unwrap_or(name);
    NAMES.binary_search(&name).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_and_names_nothing_that_kg_has() {
        assert!(NAMES.windows(2).all(|w| w[0] < w[1]));
        let mut rt = klong_rt::Runtime::new(klong_host::HostPlatform::new(klong_host::Console::Stdout), klong_core::Config::default(), None);
        rt.interp.kgpy_only = None;
        for n in NAMES {
            let own = rt.interp.global(n).is_some();
            assert!(!own, "{n} is registered by kg");
        }
    }
}
