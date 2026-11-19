//! The driver of the static compiler (`klongpy/static/compile.py`): check, evaluate the compile-time definitions under the native
//! interpreter (strict level 2), check again with the values, lower.
use crate::ast::{Conv, Lit};
use crate::check::{check_source, Diag, LitVal, Options, Report};
use crate::cli::render;
use crate::fixed as fx;
use crate::infer::{lower, Fail, StageValue};
use crate::ir::{dump, Program, V};
use crate::sig::{parse_type, Ty, C_KEYWORDS};
use klong_core::tree::Tree;
use klong_core::value::{Arr, WILDCARD};
use klong_core::{Config, Error, Value};
use klong_host::{Console, HostPlatform};
use klong_rt::Runtime;
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;

pub struct Compiled {
    pub report: Report,
    pub values: HashMap<String, StageValue>,
    pub program: Program,
}

pub enum Failure {
    /// The program is outside the profile (before or after stage one): every diagnostic of the report.
    Check(Report),
    /// Lowering stopped at the first problem.
    Lower(Fail, Report),
    /// The interpreter failed while evaluating stage one.
    Run(Error),
    Read(String),
}

impl Failure {
    /// The error texts, one per diagnostic, as `KlongError.render` lays them out.
    pub fn render(&self) -> Vec<String> {
        match self {
            Failure::Check(r) => r.diagnostics.iter().map(|d| render(d, r)).collect(),
            Failure::Lower(f, r) => {
                let d = Diag { rule: f.rule, span: f.span, src: f.src, message: f.message.clone(), hint: f.hint.clone(), function: None };
                vec![render(&d, r)]
            }
            Failure::Run(e) => vec![e.render()],
            Failure::Read(m) => vec![m.clone()],
        }
    }
}

/// The default C prefix of a program: its file name without the extension, as a C identifier.
pub fn c_prefix(path: &str) -> String {
    let base = path.rsplit('/').next().unwrap_or(path);
    let stem = match base.rfind('.') {
        Some(p) if p > 0 => &base[..p],
        _ => base,
    };
    let mut stem: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if stem.is_empty() {
        stem = "prog".to_string();
    }
    if !stem.chars().next().unwrap().is_alphabetic() {
        stem = format!("prog_{}", stem);
    }
    if stem == "kg" || stem.starts_with("kg_") || C_KEYWORDS.contains(&stem.as_str()) {
        stem = format!("prog_{}", stem);
    }
    stem
}

fn arr_shape<T: Copy>(a: &Arr<T>) -> Vec<u32> {
    (0..a.rank()).map(|i| a.dim(i) as u32).collect()
}

/// A fixed-point value at compile time (the interpreter has no such kind): the raw scaled integers, their shape and type.
pub struct FixedBox {
    pub elem: String,
    pub shape: Vec<usize>,
    pub raw: Vec<i64>,
}

fn fixed_box(v: &Value) -> Option<&FixedBox> {
    if let Value::Ext(e) = v {
        e.0.as_any().downcast_ref::<FixedBox>()
    } else {
        None
    }
}

fn shape_of(v: &Value) -> Option<(Vec<usize>, Vec<f64>, bool)> {
    match v {
        Value::Int(i) => Some((Vec::new(), vec![*i as f64], true)),
        Value::Flt(f) => Some((Vec::new(), vec![*f as f64], false)),
        Value::Ints(a) => Some(((0..a.rank()).map(|i| a.dim(i)).collect(), a.data.iter().map(|x| *x as f64).collect(), true)),
        Value::Flts(a) => Some(((0..a.rank()).map(|i| a.dim(i)).collect(), a.data.iter().map(|x| *x as f64).collect(), false)),
        _ => None,
    }
}

fn ints_of(v: &Value) -> Option<Vec<i64>> {
    match v {
        Value::Int(i) => Some(vec![*i as i64]),
        Value::Ints(a) => Some(a.data.iter().map(|x| *x as i64).collect()),
        _ => None,
    }
}

/// `.fixed(type;x)` at compile time: the raw scaled integers of x, in the shape of x, as a `FixedBox`.
fn fixed_standin(_ip: &mut klong_core::Interp, a: &[Value]) -> Result<Value, Error> {
    let text = match a.first() {
        Some(Value::Str(s)) => s.as_str().to_string(),
        _ => return Err(Error::typ(".fixed: the type must be a string")),
    };
    let ty = parse_type(&text).map_err(|e| Error::typ(format!(".fixed: {}", e.message)))?;
    if !crate::sig::is_fixed(&ty.elem) || !ty.shape.is_empty() {
        return Err(Error::typ(".fixed: the type must be fixed<W,I>, for example \"fixed<32,16>\""));
    }
    let spec = fx::spec_of(&ty.elem);
    let bad = || Error::typ(".fixed: the value must be a number or an array of numbers");
    let x = a.get(1).ok_or_else(bad)?;
    let (shape, raw) = if let Some(fb) = fixed_box(x) {
        let s = fx::spec_of(&fb.elem);
        (fb.shape.clone(), fb.raw.iter().map(|r| fx::convert(*r, s, spec)).collect())
    } else {
        let (shape, vals, is_int) = shape_of(x).ok_or_else(bad)?;
        let ints = ints_of(x);
        let raw = match (is_int, ints) {
            (true, Some(iv)) => iv.iter().map(|v| fx::from_int(*v, spec)).collect(),
            _ => vals.iter().map(|v| fx::from_float(*v, spec)).collect(),
        };
        (shape, raw)
    };
    Ok(Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(FixedBox { elem: ty.elem, shape, raw })))))
}

/// `.float(x)` at compile time: a fixed-point value, an int, a float or an array of them, as floats.
fn float_standin(_ip: &mut klong_core::Interp, a: &[Value]) -> Result<Value, Error> {
    let bad = || Error::typ(".float: the argument must be a fixed-point number or an array of numbers");
    let x = a.first().ok_or_else(bad)?;
    if let Some(fb) = fixed_box(x) {
        let s = fx::spec_of(&fb.elem);
        let fl: Vec<f64> = fb.raw.iter().map(|r| fx::to_float(*r, s)).collect();
        return Ok(floats(&fb.shape, fl));
    }
    let (shape, vals, _) = shape_of(x).ok_or_else(bad)?;
    Ok(floats(&shape, vals))
}

fn floats(shape: &[usize], v: Vec<f64>) -> Value {
    if shape.is_empty() {
        return Value::Flt(v[0] as klong_core::value::Flt);
    }
    let d: Vec<klong_core::value::Flt> = v.iter().map(|x| *x as klong_core::value::Flt).collect();
    if shape.len() > 1 {
        let dims: Vec<u32> = shape.iter().map(|x| *x as u32).collect();
        Value::Flts(Rc::new(Arr::shaped(d, &dims)))
    } else {
        Value::flts(d)
    }
}

fn big_i128(b: &impl ToString) -> i128 {
    let t = b.to_string();
    t.parse::<i128>().unwrap_or(if t.starts_with('-') { i128::MIN } else { i128::MAX })
}

fn lit_value(l: &Lit) -> Option<V> {
    match l {
        Lit::Num { float, ints, flts, .. } => Some(if *float { V::AF(flts.clone()) } else { V::AI(ints.iter().map(|x| *x as i64).collect()) }),
        _ => None,
    }
}

/// A value of the interpreter as the checker is told it and as lowering reads it.
pub fn stage_value(v: Option<Value>, conv: &Conv, param: Option<&Ty>) -> StageValue {
    let Some(v) = v else { return StageValue { lit: LitVal::Undefined, value: None } };
    if let Some(fb) = fixed_box(&v) {
        let raw = if fb.shape.is_empty() { V::I(fb.raw[0]) } else { V::AI(fb.raw.clone()) };
        return StageValue { lit: LitVal::Fixed(Ty::new(&fb.elem, fb.shape.clone())), value: Some(raw) };
    }
    if let Some(t) = param {
        // a `.param` of a fixed-point type holds its default as a number
        if crate::sig::is_fixed(&t.elem) {
            let spec = fx::spec_of(&t.elem);
            let raw = match &v {
                Value::Int(i) => Some(fx::from_int(*i as i64, spec)),
                Value::Flt(f) => Some(fx::from_float(*f as f64, spec)),
                _ => None,
            };
            if let Some(r) = raw {
                return StageValue { lit: LitVal::Fixed(t.clone()), value: Some(V::I(r)) };
            }
        }
    }
    match &v {
        Value::Int(i) => StageValue { lit: LitVal::Int(*i as i128), value: Some(V::I(*i as i64)) },
        Value::Flt(f) => StageValue { lit: LitVal::Flt(*f as f64), value: Some(V::F(*f as f64)) },
        Value::Big(b) => {
            let n = big_i128(b);
            StageValue { lit: LitVal::Int(n), value: Some(V::I(n.clamp(i64::MIN as i128, i64::MAX as i128) as i64)) }
        }
        Value::Char(_) => StageValue { lit: LitVal::Char, value: None },
        Value::Sym(s) => StageValue { lit: LitVal::QSym(conv.syms.name(*s).to_string()), value: None },
        Value::Undefined => StageValue { lit: LitVal::Undefined, value: None },
        Value::Str(_) => StageValue { lit: LitVal::Str, value: None },
        Value::Dict(_) => StageValue { lit: LitVal::Dict, value: None },
        Value::Fn(_) | Value::Prim(_) | Value::Ext(_) => StageValue { lit: LitVal::Func, value: None },
        Value::Ints(_) | Value::Flts(_) | Value::List(_) => {
            let l = conv.literal(&v);
            let value = lit_value(&l);
            StageValue { lit: LitVal::Arr(l), value }
        }
    }
}

fn abspath(p: &str) -> String {
    if p.starts_with('/') {
        return p.to_string();
    }
    let cwd = std::env::current_dir().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
    klong_lint::recover::py_join(&cwd, p)
}

/// The interpreter of stage one: strict level 2, the files of the operating system, `.fixed` evaluated to raw integers.
pub fn stage_runtime() -> Runtime<HostPlatform> {
    stage_runtime_in(&[])
}

/// `stage_runtime` with library directories searched after KLONGPATH.
pub fn stage_runtime_in(lib_dirs: &[String]) -> Runtime<HostPlatform> {
    let cfg = Config { strict: 2, stack_limit: Some(192 << 20), ..Config::default() };
    let mut rt = Runtime::new(HostPlatform::new(Console::Stdout), cfg, None);
    let path = std::env::var("KLONGPATH").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| String::from(".:lib"));
    let mut dirs: Vec<String> = path.split(':').filter(|d| !d.is_empty()).map(String::from).collect();
    dirs.extend(lib_dirs.iter().cloned());
    rt.set_load_path(dirs);
    rt.interp.register(".fixed", WILDCARD, Rc::new(fixed_standin));
    rt.interp.register(".float", WILDCARD, Rc::new(float_standin));
    rt
}

/// Check, evaluate stage one, check again with the values, and lower.
pub fn compile_program(path: &str, text: Option<&str>, int_bits: u32, name: Option<&str>) -> Result<Compiled, Failure> {
    compile_program_with(path, text, int_bits, name, HashMap::new())
}

/// `compile_program` with host stand-ins for the externs that compile-time code calls.
pub fn compile_program_with(path: &str, text: Option<&str>, int_bits: u32, name: Option<&str>, externs: HashMap<String, klong_core::value::NativeFn>) -> Result<Compiled, Failure> {
    compile_program_in(path, text, int_bits, name, externs, &[])
}

/// `compile_program_with` where a library file `.l` loads is looked up in `lib_dirs` after KLONGPATH (and reported with the path found).
pub fn compile_program_in(path: &str, text: Option<&str>, int_bits: u32, name: Option<&str>, externs: HashMap<String, klong_core::value::NativeFn>, lib_dirs: &[String]) -> Result<Compiled, Failure> {
    let text: String = match text {
        Some(t) => t.to_string(),
        None => klong_lint::read_text(path).map_err(Failure::Read)?,
    };
    let is_file = std::fs::metadata(path).map(|m| m.is_file()).unwrap_or(false);
    let opts = Options { int_bits, lib_dirs: lib_dirs.to_vec(), ..Options::default() };
    let first = check_source(&text, path, &opts);
    if !first.diagnostics.is_empty() {
        return Err(Failure::Check(first));
    }
    let mut rt = stage_runtime_in(lib_dirs);
    for (n, f) in externs {
        rt.interp.register(&n, WILDCARD, f);
    }
    let run_name = if is_file { abspath(path) } else { path.to_string() };
    if is_file {
        rt.set_program(&run_name);
    }
    rt.run(&text, &run_name).map_err(Failure::Run)?;
    let tree = Tree::new(None);
    let conv = Conv { tree: &tree, syms: &rt.interp.syms };
    let mut values = HashMap::new();
    for g in first.globals.values() {
        let v = rt.interp.global(&g.name);
        values.insert(g.name.clone(), stage_value(v, &conv, first.params.items.iter().find(|(k, _)| *k == g.name).map(|(_, t)| t)));
    }
    let opts = Options { int_bits, lib_dirs: lib_dirs.to_vec(), globals: values.iter().map(|(k, v)| (k.clone(), v.lit.clone())).collect(), ..Options::default() };
    let report = check_source(&text, path, &opts);
    if !report.diagnostics.is_empty() {
        return Err(Failure::Check(report));
    }
    let pname = name.map(String::from).unwrap_or_else(|| c_prefix(path));
    match lower(&report, &values, &pname, int_bits) {
        Ok(program) => Ok(Compiled { report, values, program }),
        Err(f) => Err(Failure::Lower(f, report)),
    }
}

/// The body of `kgpy check --dump-ir FILE...`: the typed IR of each program. Returns 0, 1 (outside the profile or failing at compile
/// time) or 2 (a file cannot be read).
pub fn dump_ir_main(files: &[String], int_bits: u32, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    dump_ir_main_in(files, int_bits, &[], out, err)
}

/// `dump_ir_main` with library directories.
pub fn dump_ir_main_in(files: &[String], int_bits: u32, lib_dirs: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let mut status = 0;
    for path in files {
        match compile_program_in(path, None, int_bits, None, HashMap::new(), lib_dirs) {
            Ok(c) => {
                let _ = out.write_all(dump(&c.program).as_bytes());
            }
            Err(Failure::Read(m)) => {
                let _ = writeln!(err, "{}: cannot read: {}", path, m);
                status = status.max(2);
            }
            Err(f) => {
                for t in f.render() {
                    let _ = writeln!(err, "{}", t);
                }
                status = 1;
            }
        }
    }
    status
}

// ---------------------------------------------------------------------------
// compile c
// ---------------------------------------------------------------------------

/// The runtime header every compiled program includes (`klong_rt.h`), shipped with the tool.
pub const RUNTIME_HEADER: &str = include_str!("../klong_rt.h");

fn io_text(e: &std::io::Error) -> String {
    let t = e.to_string();
    match t.find(" (os error") {
        Some(p) => t[..p].to_string(),
        None => t,
    }
}

fn py_join(dir: &str, name: &str) -> String {
    klong_lint::recover::py_join(dir, name)
}

/// What `emit_program` of compile.py gives: the compiled program, the C and the report.
pub struct Emitting {
    pub compiled: Compiled,
    pub emitted: crate::emit_c::Emitted,
}

pub enum EmitFailure {
    Compile(Failure),
    Emit(crate::emit_c::EmitError, Report),
    Io(String),
}

impl EmitFailure {
    pub fn render(&self, path: &str) -> (Vec<String>, i32) {
        match self {
            EmitFailure::Compile(Failure::Read(m)) => (vec![format!("{}: cannot read: {}", path, m)], 2),
            EmitFailure::Compile(f) => (f.render(), 1),
            EmitFailure::Io(m) => (vec![format!("{}: cannot read: {}", path, m)], 2),
            EmitFailure::Emit(crate::emit_c::EmitError::Value(m), _) => (vec![format!("{}: {}", path, m)], 1),
            EmitFailure::Emit(crate::emit_c::EmitError::Unsupported(m), _) => (vec![format!("{}: {}", path, m)], 1),
            EmitFailure::Emit(crate::emit_c::EmitError::Static(msg, hint, span, src), r) => {
                let d = Diag { rule: "extern", span: *span, src: *src, message: msg.clone(), hint: Some(hint.clone()), function: None };
                (vec![render(&d, r)], 1)
            }
        }
    }
}

/// Compile a program to C. With `out_dir`, writes `<prefix>.c`, `<prefix>.h` and `klong_rt.h` there.
pub fn emit_program(path: &str, out_dir: Option<&str>, text: Option<&str>, int_bits: u32, float_bits: u32, line_directives: bool, prefix: Option<&str>, hls: Option<crate::emit_c::HlsOptions>) -> Result<Emitting, EmitFailure> {
    emit_program_in(path, out_dir, text, int_bits, float_bits, line_directives, prefix, hls, &[])
}

/// `emit_program` with library directories (see `compile_program_in`).
#[allow(clippy::too_many_arguments)]
pub fn emit_program_in(path: &str, out_dir: Option<&str>, text: Option<&str>, int_bits: u32, float_bits: u32, line_directives: bool, prefix: Option<&str>, hls: Option<crate::emit_c::HlsOptions>, lib_dirs: &[String]) -> Result<Emitting, EmitFailure> {
    let compiled = compile_program_in(path, text, int_bits, prefix, HashMap::new(), lib_dirs).map_err(EmitFailure::Compile)?;
    let emitted = match crate::emit_c::emit(&compiled.program, &compiled.report, prefix, float_bits, line_directives, hls) {
        Ok(e) => e,
        Err(e) => return Err(EmitFailure::Emit(e, compiled.report)),
    };
    if let Some(d) = out_dir {
        write_files(d, &[(format!("{}.c", emitted.prefix), emitted.c.clone()), (format!("{}.h", emitted.prefix), emitted.h.clone()), ("klong_rt.h".to_string(), RUNTIME_HEADER.to_string())]).map_err(EmitFailure::Io)?;
    }
    Ok(Emitting { compiled, emitted })
}

pub fn write_files(dir: &str, files: &[(String, String)]) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| io_text(&e))?;
    for (name, text) in files {
        std::fs::write(std::path::Path::new(dir).join(name), text).map_err(|e| io_text(&e))?;
    }
    Ok(())
}

/// The body of `kgpy compile c FILE... -o DIR` (and of `--dump-ir`).
#[allow(clippy::too_many_arguments)]
pub fn emit_main(files: &[String], out_dir: Option<&str>, int_bits: u32, float_bits: u32, line_directives: bool, prefix: Option<&str>, dump_ir: bool, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    emit_main_in(files, out_dir, int_bits, float_bits, line_directives, prefix, dump_ir, &[], out, err)
}

/// `emit_main` with library directories.
#[allow(clippy::too_many_arguments)]
pub fn emit_main_in(files: &[String], out_dir: Option<&str>, int_bits: u32, float_bits: u32, line_directives: bool, prefix: Option<&str>, dump_ir: bool, lib_dirs: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let mut status = 0;
    if prefix.is_some() && files.len() > 1 {
        let _ = writeln!(err, "--prefix names one program: give one FILE");
        return 2;
    }
    for path in files {
        let res = match emit_program_in(path, Some(out_dir.unwrap_or(".")), None, int_bits, float_bits, line_directives, prefix, None, lib_dirs) {
            Ok(r) => r,
            Err(f) => {
                let (lines, st) = f.render(path);
                for l in lines {
                    let _ = writeln!(err, "{}", l);
                }
                status = if st == 2 { status.max(2) } else { 1 };
                continue;
            }
        };
        if dump_ir {
            let _ = out.write_all(dump(&res.compiled.program).as_bytes());
        }
        let where_ = out_dir.unwrap_or(".");
        let pfx = &res.emitted.prefix;
        let _ = writeln!(out, "{}: wrote {}, {}.h and klong_rt.h", path, py_join(where_, &format!("{}.c", pfx)), pfx);
        let _ = writeln!(out, "{}", res.emitted.report_text());
    }
    status
}

const COMPILE_C_HELP: &str = "\
Compile FILEs of the static profile to C99 (<name>.c, <name>.h, klong_rt.h) in
the directory given by -o and print the size report.

positional arguments:
  FILE                  the programs to compile

options:
  -h, --help            show this help message and exit
  -o DIR, --output DIR  the directory to write to (default: .)
  --int64               kg_int is 64 bits wide (the default is 32)
  --float64             kg_float is double (the default is float)
  --line-directives     emit #line so a debugger steps through the Klong
                        source
  --prefix NAME         the C prefix of the public names (default: the file
                        name)
  --dump-ir             evaluate the compile-time definitions and print the
                        typed IR
";

pub fn compile_c_spec(prog: &str) -> crate::argp::Spec {
    use crate::argp::{Opt, Spec};
    let mut s = Spec {
        prog: prog.to_string(),
        opts: vec![
            Opt { short: Some("-o"), long: "--output", metavar: Some("DIR"), int: false },
            Opt { short: None, long: "--int64", metavar: None, int: false },
            Opt { short: None, long: "--float64", metavar: None, int: false },
            Opt { short: None, long: "--line-directives", metavar: None, int: false },
            Opt { short: None, long: "--prefix", metavar: Some("NAME"), int: false },
            Opt { short: None, long: "--dump-ir", metavar: None, int: false },
        ],
        positional: "FILE [FILE ...]",
        help: String::new(),
    };
    s.help = format!("{}\n{}", crate::argp::usage(&s), COMPILE_C_HELP);
    s
}

/// `kg compile c ...`: `args` follow the words `compile c`.
pub fn compile_c_cli(prog: &str, args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let spec = compile_c_spec(prog);
    let p = match crate::argp::parse(&spec, args, out, err) {
        crate::argp::Outcome::Run(p) => p,
        crate::argp::Outcome::Exit(c) => return c,
    };
    emit_main(&p.files, p.get("--output").map(|s| s.as_str()), if p.flag("--int64") { 64 } else { 32 }, if p.flag("--float64") { 64 } else { 32 }, p.flag("--line-directives"), p.get("--prefix").map(|s| s.as_str()), p.flag("--dump-ir"), out, err)
}
