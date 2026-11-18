//! Running a static program on the native interpreter (`klongpy/static/host.py`, and `conform` of `klongpy/static/signature.py`): the
//! host half of `.extern` and `.entry`.
//!
//! ```ignore
//! let mut externs = HashMap::new();
//! externs.insert("read.adc".to_string(), native_fn(|a| Ok(Value::Flt(0.3))));
//! let mut p = load_program("examples/embedded/pid.kg", None, externs)?;
//! call_entry(&mut p, "step", &[HostValue::Float(0.31)], None, false)?;     // Float(0.12005)
//! ```
//!
//! `load_program` runs the file at strict level 2 with the host functions bound as its externs; `call_entry` calls an entry the way the C
//! caller would, with the arguments and the result checked against the `.entry` signature. A program that uses fixed-point numbers
//! (`fixed<` or `.fixed(` in its text) runs on the typed-IR evaluator (`eval_ir.rs`), as in Python: the interpreter has no fixed-point
//! type; its state globals live in the interpreter between calls.
use crate::compile::{compile_program, stage_runtime, FixedBox};
use crate::eval_ir::{Arr, Data, IrEvaluator, Trap, TRAP_BOUNDS};
use crate::fixed as fx;
use crate::ir::Program;
use crate::sig::{is_fixed, parse_signature, parse_type, Signature, Ty};
use klong_core::value::{Arr as KArr, Flt, Int, NativeFn, Value, WILDCARD};
use klong_core::{Error, Interp};
use klong_host::HostPlatform;
use klong_rt::Runtime;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A host function, bound as an extern: the interpreter's native function (`Rc<dyn Fn(&mut Interp, &[Value]) -> Result<Value, Error>>`).
pub type HostFn = NativeFn;

/// A value at the boundary of the program: what an entry takes and returns (`conform` of signature.py: "the canonical host form").
#[derive(Clone, Debug, PartialEq)]
pub enum HostValue {
    Int(i64),
    Float(f64),
    /// an int array of any rank: shape, row-major data
    Ints(Vec<usize>, Vec<i64>),
    Floats(Vec<usize>, Vec<f64>),
    /// a fixed-point array (or scalar: empty shape) of the type `elem` (`"fixed<32,16>"`), as the raw scaled integers
    Fixed { elem: String, shape: Vec<usize>, raw: Vec<i64> },
    Str(String),
    /// the result of a `void` entry
    Void,
    /// something a host cannot hold (a list, a function, ...), described the way Python's `_describe` words it ("a symbol or a character")
    Other(String),
}

impl HostValue {
    pub fn ints(shape: &[usize], data: Vec<i64>) -> HostValue {
        HostValue::Ints(shape.to_vec(), data)
    }
    pub fn floats(shape: &[usize], data: Vec<f64>) -> HostValue {
        HostValue::Floats(shape.to_vec(), data)
    }
    pub fn fixed(elem: &str, shape: &[usize], raw: Vec<i64>) -> HostValue {
        HostValue::Fixed { elem: elem.to_string(), shape: shape.to_vec(), raw }
    }
    /// A value of the interpreter as a host value.
    pub fn from_value(v: &Value) -> HostValue {
        let shape = |a: &dyn Fn(usize) -> usize, rank: usize| -> Vec<usize> { (0..rank).map(a).collect() };
        match v {
            Value::Int(i) => HostValue::Int(*i as i64),
            Value::Flt(f) => HostValue::Float(*f as f64),
            Value::Ints(a) => HostValue::Ints(shape(&|i| a.dim(i), a.rank()), a.data.iter().map(|x| *x as i64).collect()),
            Value::Flts(a) => HostValue::Floats(shape(&|i| a.dim(i), a.rank()), a.data.iter().map(|x| *x as f64).collect()),
            Value::Str(s) => HostValue::Str(s.as_str().to_string()),
            Value::Char(_) | Value::Sym(_) => HostValue::Other("a symbol or a character".to_string()),
            Value::Undefined => HostValue::Other("nothing".to_string()),
            Value::List(_) => HostValue::Other("a list".to_string()),
            Value::Dict(_) => HostValue::Other("a dict".to_string()),
            Value::Fn(_) | Value::Prim(_) => HostValue::Other("a function".to_string()),
            _ => match fixed_box(v) {
                Some(fb) => HostValue::Fixed { elem: fb.elem.clone(), shape: fb.shape.clone(), raw: fb.raw.clone() },
                None => HostValue::Other("an object".to_string()),
            },
        }
    }
    /// The value in the interpreter. A fixed-point value is a `FixedBox`.
    pub fn to_value(&self) -> Value {
        match self {
            HostValue::Int(i) => Value::Int(*i as Int),
            HostValue::Float(f) => Value::Flt(*f as Flt),
            HostValue::Ints(s, d) => Value::Ints(Rc::new(karr(d.iter().map(|x| *x as Int).collect(), s))),
            HostValue::Floats(s, d) => Value::Flts(Rc::new(karr(d.iter().map(|x| *x as Flt).collect(), s))),
            HostValue::Fixed { elem, shape, raw } => fixed_value(elem, shape, raw),
            HostValue::Str(s) => Value::str(s.clone()),
            HostValue::Void | HostValue::Other(_) => Value::Undefined,
        }
    }
    /// The number as a float (an int, a float, or a one-element float).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            HostValue::Int(i) => Some(*i as f64),
            HostValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

fn karr<T: Copy>(d: Vec<T>, shape: &[usize]) -> KArr<T> {
    if shape.len() <= 1 {
        KArr::vec(d)
    } else {
        let dims: Vec<u32> = shape.iter().map(|x| *x as u32).collect();
        KArr::shaped(d, &dims)
    }
}

fn fixed_value(elem: &str, shape: &[usize], raw: &[i64]) -> Value {
    Value::Ext(Rc::new(klong_core::value::ExtBox(Box::new(FixedBox { elem: elem.to_string(), shape: shape.to_vec(), raw: raw.to_vec() }))))
}

fn fixed_box(v: &Value) -> Option<&FixedBox> {
    if let Value::Ext(e) = v {
        e.0.as_any().downcast_ref::<FixedBox>()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// conform
// ---------------------------------------------------------------------------

/// `conform` failed: `kind` is "type" or "shape", `message` one line.
#[derive(Clone, Debug, PartialEq)]
pub struct Mismatch {
    pub kind: &'static str,
    pub message: String,
}

impl Mismatch {
    pub fn error(&self) -> Error {
        if self.kind == "shape" {
            Error::shape(self.message.clone())
        } else {
            Error::typ(self.message.clone())
        }
    }
    fn typ(m: String) -> Mismatch {
        Mismatch { kind: "type", message: m }
    }
    fn shape(m: String) -> Mismatch {
        Mismatch { kind: "shape", message: m }
    }
}

fn dims(shape: &[usize]) -> String {
    shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
}

/// `_describe`: how a value is worded in a message.
pub fn describe(v: &HostValue) -> String {
    match v {
        HostValue::Int(_) => "an integer".to_string(),
        HostValue::Float(_) => "a float".to_string(),
        HostValue::Str(_) => "a string".to_string(),
        HostValue::Ints(s, _) if s.is_empty() => "an integer".to_string(),
        HostValue::Floats(s, _) if s.is_empty() => "a float".to_string(),
        HostValue::Ints(s, _) => format!("int[{}]", dims(s)),
        HostValue::Floats(s, _) => format!("float[{}]", dims(s)),
        HostValue::Fixed { shape, .. } if !shape.is_empty() => format!("int[{}]", dims(shape)),
        HostValue::Fixed { .. } => "a FixedArray".to_string(),
        HostValue::Void => "nothing".to_string(),
        HostValue::Other(s) => s.clone(),
    }
}

fn conform_fixed(ty: &Ty, value: &HostValue, what: &str) -> Result<HostValue, Mismatch> {
    let spec = fx::spec_of(&ty.elem);
    let (shape, raw): (Vec<usize>, Vec<i64>) = match value {
        HostValue::Fixed { elem, shape, raw } => {
            if elem != &ty.elem {
                let d = if shape.is_empty() { String::new() } else { format!("[{}]", dims(shape)) };
                return Err(Mismatch::typ(format!("{} must be {}, got {}{}", what, ty, elem, d)));
            }
            (shape.clone(), raw.clone())
        }
        HostValue::Int(i) => (Vec::new(), vec![fx::from_int(*i, spec)]),
        HostValue::Float(f) => (Vec::new(), vec![fx::from_float(*f, spec)]),
        HostValue::Ints(s, d) => (s.clone(), d.iter().map(|x| fx::from_int(*x, spec)).collect()),
        HostValue::Floats(s, d) => (s.clone(), d.iter().map(|x| fx::from_float(*x, spec)).collect()),
        _ => return Err(Mismatch::typ(format!("{} must be {}, got {}", what, ty, describe(value)))),
    };
    if shape != ty.shape {
        let kind = if !ty.shape.is_empty() && !shape.is_empty() { "shape" } else { "type" };
        return Err(Mismatch { kind, message: format!("{} must be {}, got {}", what, ty, describe(value)) });
    }
    Ok(HostValue::Fixed { elem: ty.elem.clone(), shape, raw })
}

/// Check `value` against `ty` and return it in the canonical host form: `int` an Int; `float` a Float (an int is accepted and converted);
/// `int[..]` Ints of exactly that shape; `float[..]` Floats of exactly that shape (ints converted); `fixed<W,I>[..]` Fixed; `str` Str;
/// `void` Void, whatever the value. `what` names the value in the message ("argument 1 of read.adc").
pub fn conform(ty: &Ty, value: &HostValue, what: &str) -> Result<HostValue, Mismatch> {
    if ty.elem == "void" {
        return Ok(HostValue::Void);
    }
    if ty.elem == "str" {
        return match value {
            HostValue::Str(_) => Ok(value.clone()),
            _ => Err(Mismatch::typ(format!("{} must be str, got {}", what, describe(value)))),
        };
    }
    if is_fixed(&ty.elem) {
        return conform_fixed(ty, value, what);
    }
    // a 0-d array is a number
    let v = match value {
        HostValue::Ints(s, d) if s.is_empty() => HostValue::Int(d[0]),
        HostValue::Floats(s, d) if s.is_empty() => HostValue::Float(d[0]),
        v => v.clone(),
    };
    let bad = |v: &HostValue| Mismatch::typ(format!("{} must be {}, got {}", what, ty, describe(v)));
    if ty.shape.is_empty() {
        return match (&v, ty.elem.as_str()) {
            (HostValue::Int(i), "int") => Ok(HostValue::Int(*i)),
            (HostValue::Int(i), _) => Ok(HostValue::Float(*i as f64)),
            (HostValue::Float(f), "float") => Ok(HostValue::Float(*f)),
            _ => Err(bad(&v)),
        };
    }
    match &v {
        HostValue::Ints(s, d) => {
            if s != &ty.shape {
                return Err(Mismatch::shape(format!("{} must be {}, got {}", what, ty, describe(&v))));
            }
            Ok(if ty.elem == "int" { HostValue::Ints(s.clone(), d.clone()) } else { HostValue::Floats(s.clone(), d.iter().map(|x| *x as f64).collect()) })
        }
        HostValue::Floats(s, d) => {
            if ty.elem == "int" {
                return Err(bad(&v));
            }
            if s != &ty.shape {
                return Err(Mismatch::shape(format!("{} must be {}, got {}", what, ty, describe(&v))));
            }
            Ok(HostValue::Floats(s.clone(), d.clone()))
        }
        _ => Err(bad(&v)),
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// A loaded program: the interpreter that holds its state, and (for a fixed-point program) the typed-IR evaluator.
pub struct Loaded {
    pub rt: Runtime<HostPlatform>,
    pub path: String,
    pub text: String,
    pub externs: HashMap<String, HostFn>,
    evaluator: Option<IrEvaluator<Rc<Program>>>,
}

/// Does the program text use fixed-point numbers? Such a program runs through `call_entry` on the typed-IR evaluator.
pub fn uses_fixed(text: &str) -> bool {
    text.contains("fixed<") || text.contains(".fixed(")
}

impl Loaded {
    pub fn uses_fixed(&self) -> bool {
        uses_fixed(&self.text)
    }
    /// The value of a global of the program.
    pub fn global(&self, name: &str) -> Option<Value> {
        self.rt.interp.global(name)
    }
    /// A global as a host value.
    pub fn get(&self, name: &str) -> Option<HostValue> {
        self.global(name).map(|v| HostValue::from_value(&v))
    }
}

fn abspath(p: &str) -> String {
    if p.starts_with('/') {
        return p.to_string();
    }
    let cwd = std::env::current_dir().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
    klong_lint::recover::py_join(&cwd, p)
}

/// Run a program and return what holds its state.
///
/// `path` is the file to run (or, with `text`, the name the text is known by); `externs` are the stand-ins, bound before the program
/// runs. The program runs at strict level 2 (what the static profile requires), its top-level definitions are evaluated in order,
/// `.extern` binds each declared name to its stand-in and `.entry` validates each signature. `.fixed` evaluates to a fixed-point value
/// (see `compile::stage_runtime`).
pub fn load_program(path: &str, text: Option<&str>, externs: HashMap<String, HostFn>) -> Result<Loaded, Error> {
    let text: String = match text {
        Some(t) => t.to_string(),
        None => klong_lint::read_text(path).map_err(Error::system)?,
    };
    let is_file = std::fs::metadata(path).map(|m| m.is_file()).unwrap_or(false);
    let mut rt = stage_runtime();
    let run_name = if is_file { abspath(path) } else { path.to_string() };
    if is_file {
        rt.set_program(&run_name);
    }
    // a function the host registered before the declaration is what `.extern` binds
    for (name, f) in &externs {
        rt.interp.register(name, WILDCARD, f.clone());
    }
    rt.run(&text, &run_name)?;
    // an extern declared by a function that ran later (or one whose name the program defined itself) is bound now
    let declared: Vec<String> = rt.declarations.externs().into_iter().map(|(n, _, _)| n).collect();
    for name in declared {
        if let Some(f) = externs.get(&name) {
            rt.declarations.bind(&mut rt.interp, &name, f.clone())?;
        }
    }
    Ok(Loaded { rt, path: path.to_string(), text, externs, evaluator: None })
}

/// Python's `Ty` of a declared type of the interpreter.
fn host_ty(t: &klong_rt::declare::Ty) -> Ty {
    parse_type(&t.to_string()).expect("a declared type is canonical")
}

fn host_sig(s: &klong_rt::declare::Sig) -> Signature {
    parse_signature(&s.to_string(), false).expect("a declared signature is canonical")
}

/// Set the parameter `name` (declared with `.param`) between runs: the host side of the register that the C emitter exposes as
/// `<prog>_set_<name>` and the hardware as an s_axilite argument. The value is checked against the declared type and converted
/// ("int" takes an int, "float" an int or a float, "fixed<W,I>" an int, a float or a fixed-point value, truncating toward minus
/// infinity). Returns the value as converted.
pub fn set_param(p: &mut Loaded, name: &str, value: &HostValue) -> Result<HostValue, Error> {
    let decl = p.rt.declarations.params().into_iter().find(|(n, _, _)| n == name);
    let Some((_, dty, _)) = decl else {
        return Err(Error::name(format!("{} is not a parameter of the program", name)).with_hint(format!("declare it: .param(\"{}\";\"int\")", name)));
    };
    let ty = host_ty(&dty);
    let v = conform(&ty, value, &format!("the value of parameter {}", name)).map_err(|m| m.error())?;
    p.rt.interp.define_global(name, v.to_value())?;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Calling an entry
// ---------------------------------------------------------------------------

fn arr_of(ty: &Ty, v: &HostValue) -> Arr {
    let a = match v {
        HostValue::Int(i) => Arr::int(*i),
        HostValue::Float(f) => Arr::flt(*f),
        HostValue::Ints(s, d) => Arr::ints(s.clone(), d.clone()),
        HostValue::Floats(s, d) => Arr::flts(s.clone(), d.clone()),
        HostValue::Fixed { shape, raw, .. } => Arr::ints(shape.clone(), raw.clone()),
        _ => Arr::int(0),
    };
    a.as_elem(&ty.elem)
}

fn host_of(a: &Arr) -> HostValue {
    match (&a.data, a.shape.is_empty()) {
        (Data::I(d), true) => HostValue::Int(d[0]),
        (Data::F(d), true) => HostValue::Float(d[0]),
        (Data::I(d), false) => HostValue::Ints(a.shape.clone(), d.clone()),
        (Data::F(d), false) => HostValue::Floats(a.shape.clone(), d.clone()),
    }
}

/// A global of the interpreter as the evaluator's value (`_raw`): the raw values of a fixed-point global, a number or an array.
fn state_arr(ty: &Ty, v: &Value) -> Option<Arr> {
    if let Some(fb) = fixed_box(v) {
        return Some(Arr::ints(fb.shape.clone(), fb.raw.clone()).as_elem(&ty.elem));
    }
    let a = match HostValue::from_value(v) {
        HostValue::Int(i) if is_fixed(&ty.elem) => Arr::int(fx::from_int(i, fx::spec_of(&ty.elem))),
        HostValue::Float(f) if is_fixed(&ty.elem) => Arr::int(fx::from_float(f, fx::spec_of(&ty.elem))),
        h @ (HostValue::Int(_) | HostValue::Float(_) | HostValue::Ints(..) | HostValue::Floats(..)) => arr_of(ty, &h),
        _ => return None,
    };
    Some(a)
}

/// `_host_state`: the evaluator's value of a state global as the interpreter's.
fn state_value(ty: &Ty, a: &Arr) -> Value {
    if is_fixed(&ty.elem) {
        return fixed_value(&ty.elem, &a.shape, &a.i64s());
    }
    host_of(&a.clone().as_elem(&ty.elem)).to_value()
}

/// The extern callback of the evaluator: the host functions on a scratch interpreter, arguments and result checked against the signature.
fn extern_callback(externs: &HashMap<String, HostFn>, program: &Program) -> Option<crate::eval_ir::ExternCb> {
    if externs.is_empty() {
        return None;
    }
    let sigs: HashMap<String, Signature> = program.externs.iter().map(|x| (x.name.clone(), x.signature.clone())).collect();
    let externs = externs.clone();
    let ip: Rc<RefCell<Interp>> = Rc::new(RefCell::new(stage_runtime().interp));
    Some(Box::new(move |name: &str, args: &[Arr]| {
        let f = externs.get(name).ok_or_else(|| format!("extern {} has no stand-in on the host", name))?;
        let sig = sigs.get(name).ok_or_else(|| format!("{} is not a declared extern", name))?;
        let mut vals = Vec::new();
        for (i, (ty, a)) in sig.params.iter().zip(args).enumerate() {
            let h = if ty.elem == "str" {
                HostValue::Str(a.i64s().iter().filter_map(|c| char::from_u32(*c as u32)).collect())
            } else {
                host_of(&a.clone().as_elem(&ty.elem))
            };
            vals.push(conform(ty, &h, &format!("argument {} of {}", i + 1, name)).map_err(|m| format!("{} (declared \"{}\")", m.message, sig))?.to_value());
        }
        let r = f(&mut ip.borrow_mut(), &vals).map_err(|e| e.message.to_string())?;
        let h = conform(&sig.result, &HostValue::from_value(&r), &format!("the result of {}", name)).map_err(|m| format!("{} (declared \"{}\")", m.message, sig))?;
        Ok(match h {
            HostValue::Void => Arr::int(0),
            h => arr_of(&sig.result, &h),
        })
    }))
}

/// The typed-IR evaluator of a program that uses fixed-point numbers (compiled on first use).
fn evaluator(p: &mut Loaded) -> Result<&mut IrEvaluator<Rc<Program>>, Error> {
    if p.evaluator.is_none() {
        let compiled = compile_program(&p.path, Some(&p.text), 32, None).map_err(|f| match f {
            crate::compile::Failure::Run(e) => e,
            f => Error::system(f.render().join("\n")),
        })?;
        let program = Rc::new(compiled.program);
        let mut ev = IrEvaluator::new(program.clone());
        ev.extern_cb = extern_callback(&p.externs, &program);
        p.evaluator = Some(ev);
    }
    Ok(p.evaluator.as_mut().unwrap())
}

fn call_evaluated(p: &mut Loaded, name: &str, sig: &Signature, values: &[HostValue], raw: bool) -> Result<HostValue, Error> {
    let sigtext = sig.to_string();
    evaluator(p)?;
    let program = p.evaluator.as_ref().unwrap().program.clone();
    let entry = program.entries.iter().find(|e| e.decl.name == name && e.decl.signature.to_string() == sigtext);
    let Some(entry) = entry else {
        return Err(Error::name(format!("{} with the signature \"{}\" is not an entry of the compiled program", name, sigtext)));
    };
    for g in program.globals.iter().filter(|g| g.state) {
        if let Some(a) = p.rt.interp.global(&g.name).and_then(|v| state_arr(&g.ty, &v)) {
            p.evaluator.as_mut().unwrap().state.insert(g.name.clone(), a);
        }
    }
    let args: Vec<Arr> = sig.params.iter().zip(values).map(|(t, v)| arr_of(t, v)).collect();
    let ev = p.evaluator.as_mut().unwrap();
    let r = ev.run(entry, &args);
    // the state globals live in the interpreter between calls (also after a trap)
    for g in program.globals.iter().filter(|g| g.state) {
        if let Some(a) = ev.state.get(&g.name) {
            let v = state_value(&g.ty, a);
            p.rt.interp.define_global(&g.name, v)?;
        }
    }
    let r = match r {
        Ok(r) => r,
        Err(t) => return Err(trap_error(name, &t)),
    };
    if sig.result.elem == "void" {
        return Ok(HostValue::Void);
    }
    if is_fixed(&sig.result.elem) {
        let spec = fx::spec_of(&sig.result.elem);
        let raws = r.i64s();
        return Ok(if raw {
            if r.shape.is_empty() {
                HostValue::Int(raws[0])
            } else {
                HostValue::Ints(r.shape.clone(), raws.into_owned())
            }
        } else if r.shape.is_empty() {
            HostValue::Float(fx::to_float(raws[0], spec))
        } else {
            HostValue::Floats(r.shape.clone(), raws.iter().map(|x| fx::to_float(*x, spec)).collect())
        });
    }
    Ok(host_of(&r))
}

fn trap_error(name: &str, t: &Trap) -> Error {
    let m = format!("entry {}: {}", name, t);
    if t.code == TRAP_BOUNDS {
        Error::range(m)
    } else {
        Error::system(m)
    }
}

/// Call the entry `name` of a loaded program with host values, as the host of the compiled program would: one argument per parameter
/// of the `.entry` signature, checked and converted to the declared types; the result is checked the same way and returned
/// (`HostValue::Void` for `void`).
///
/// `signature` picks one entry when the function has several; otherwise the first whose parameters fit the arguments is used.
///
/// A program that uses fixed-point numbers runs on the typed-IR evaluator: a fixed-point argument is a number, an array or a `Fixed`;
/// a fixed-point result is returned as a float (exact up to 53 bits), or with `raw` as the scaled integers.
pub fn call_entry(p: &mut Loaded, name: &str, args: &[HostValue], signature: Option<&str>, raw: bool) -> Result<HostValue, Error> {
    let mut cands: Vec<Signature> = p.rt.declarations.entries().into_iter().filter(|(n, _)| n == name).map(|(_, s)| host_sig(&s)).collect();
    if cands.is_empty() {
        return Err(Error::name(format!("{} is not an entry of the program", name)).with_hint(format!("declare it: .entry(\"{}\";\"... -> ...\")", name)));
    }
    if let Some(s) = signature {
        let want = parse_signature(s, false).map_err(|e| Error::typ(e.message))?.to_string();
        match cands.iter().find(|c| c.to_string() == want) {
            Some(c) => cands = vec![c.clone()],
            // Python: a KeyError
            None => return Err(Error::name(format!("{} with the signature \"{}\" is not an entry of the program", name, want))),
        }
    }
    let mut err = None;
    for sig in &cands {
        if args.len() != sig.arity() {
            err = Some(Error::typ(format!("entry {} is declared \"{}\" and takes {} argument(s), got {}", name, sig, sig.arity(), args.len())));
            continue;
        }
        let mut values = Vec::new();
        let mut bad = None;
        for (i, (ty, a)) in sig.params.iter().zip(args).enumerate() {
            match conform(ty, a, &format!("argument {} of entry {}", i + 1, name)) {
                Ok(v) => values.push(v),
                Err(m) => {
                    bad = Some(Mismatch { kind: m.kind, message: format!("{} (declared \"{}\")", m.message, sig) });
                    break;
                }
            }
        }
        if let Some(m) = bad {
            err = Some(m.error());
            continue;
        }
        if p.uses_fixed() {
            return call_evaluated(p, name, sig, &values, raw);
        }
        let Some(f) = p.rt.interp.global(name) else {
            return Err(Error::name(format!("{} is not defined", name)));
        };
        let kvals: Vec<Value> = values.iter().map(|v| v.to_value()).collect();
        let r = p.rt.interp.apply(&f, kvals, None)?;
        return conform(&sig.result, &HostValue::from_value(&r), &format!("the result of entry {}", name)).map_err(|m| Mismatch { kind: m.kind, message: format!("{} (declared \"{}\")", m.message, sig) }.error());
    }
    Err(err.expect("at least one candidate"))
}
