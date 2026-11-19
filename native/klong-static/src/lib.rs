//! klong-static: the static profile (docs/static-profile.md): the checker, ported from `klongpy/static/checker.py`. `klong check`.
pub mod argp;
pub mod ast;
pub mod check;
pub mod cli;
pub mod compile;
pub mod emit_c;
pub mod emit_hls;
pub mod eval_ir;
pub mod fixed;
mod fixed_ext;
pub mod host;
pub mod infer;
pub mod ir;
pub mod ir_json;
pub mod profile_doc;
pub mod sig;
pub mod streamable;
pub mod tables;
pub mod util;

pub use check::{check_source, Options, Report};
pub use cli::{check_main, run_check};
