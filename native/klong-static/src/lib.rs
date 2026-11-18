//! klong-static: the static profile (docs/static-profile.md): the checker, ported from `klongpy/static/checker.py`. `klong check`.
pub mod ast;
pub mod check;
pub mod cli;
pub mod sig;
pub mod tables;
pub mod util;

pub use check::{check_source, Options, Report};
pub use cli::{check_main, run_check};
