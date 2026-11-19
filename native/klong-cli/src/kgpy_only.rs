//! The system functions that only `kgpy`, the Python-hosted command, has (principle 2: a function of the host is named as one).
//! `kg` answers a call of one with a `name` error that says where it is, instead of "undefined function".
//!
//! One table, from what the Python package registers (`klongpy/sys_fn*.py`, `klongpy/db`, `klongpy/web`, `klongpy/ws`):
//! * Python: `.py .pya .pyc .pyf`, `.sys`, `.rls`;
//! * torch and autograd: `.grad .gradcheck .jacobian .bkx .bkf .bkd .bkp`;
//! * tables and SQL: `.db .table .tables .insert .index .rindex .schema .kvs`;
//! * the web and web sockets: `.web .webc .ws .wsc`;
//! * the FPGA card: `.fpga` and `.fpga.free .fpga.get .fpga.info .fpga.lanes .fpga.put .fpga.reset .fpga.set`.
//!
//! A name that `kg` has itself must not be here (the test below checks it against the runtime).

pub const NAMES: &[&str] = &[
    ".bkd", ".bkf", ".bkp", ".bkx", ".db", ".fpga", ".fpga.free", ".fpga.get", ".fpga.info", ".fpga.lanes", ".fpga.put", ".fpga.reset", ".fpga.set", ".grad", ".gradcheck", ".index",
    ".insert", ".jacobian", ".kvs", ".py", ".pya", ".pyc", ".pyf", ".rindex", ".rls", ".schema", ".sys", ".table", ".tables", ".web", ".webc", ".ws", ".wsc",
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
