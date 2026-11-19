//! The static profile checker: is a Klong program inside the subset that can be compiled ahead of time? A port of
//! `klongpy/static/checker.py` (docs/static-profile.md). The checker reads a program; it never runs it. It parses at strict level 2,
//! sorts the top-level statements into stages, and walks every run-time function with abstract values (an element type, a shape,
//! and whether the value is a compile-time constant) in place of real ones.
use crate::ast::{Bad, Conv, Kind, Lit, Node, Scalar, Sp};
use crate::sig::{self, fixed_spec, int_range, is_fixed, EntryDecl, ExternDecl, Ty, MAX_RANK};
use crate::tables::{self, lookup, Op};
use crate::util::plural;
use klong_lint::recover::{find_load, parse_recovering, Session, Source};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Abstract values
// ---------------------------------------------------------------------------

/// What is known of the value: not a constant, a constant whose value this pass does not know, or the value.
#[derive(Clone, Debug, PartialEq)]
pub enum Const {
    Nc,
    Cu,
    Int(i128),
    Flt(f64),
    /// an array of ints (values flattened) and its rank
    IArr(Vec<i128>, usize),
    FArr(Vec<f64>, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AKind {
    Num,
    Any,
    Void,
}

#[derive(Clone, Debug)]
pub struct At {
    pub kind: AKind,
    pub elem: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub c: Const,
}

pub type AtKey = (AKind, Option<String>, Option<Vec<usize>>);

impl At {
    fn any() -> At {
        At { kind: AKind::Any, elem: None, shape: None, c: Const::Nc }
    }
    fn void() -> At {
        At { kind: AKind::Void, elem: None, shape: None, c: Const::Nc }
    }
    fn unknown() -> At {
        At { kind: AKind::Num, elem: None, shape: None, c: Const::Nc }
    }
    fn num(elem: Option<&str>, shape: Option<Vec<usize>>, c: Const) -> At {
        At { kind: AKind::Num, elem: elem.map(String::from), shape, c }
    }
    fn n(elem: &str, shape: Vec<usize>) -> At {
        At::num(Some(elem), Some(shape), Const::Nc)
    }
    fn key(&self) -> AtKey {
        (self.kind, self.elem.clone(), self.shape.clone())
    }
    fn plain(&self) -> At {
        if self.c == Const::Nc {
            self.clone()
        } else {
            At { kind: self.kind, elem: self.elem.clone(), shape: self.shape.clone(), c: Const::Nc }
        }
    }
    fn known(&self) -> bool {
        self.kind == AKind::Num && self.elem.is_some() && self.shape.is_some()
    }
    pub fn ty(&self) -> Option<Ty> {
        if self.kind == AKind::Void {
            return Some(Ty::new("void", Vec::new()));
        }
        if self.known() {
            Some(Ty { elem: self.elem.clone().unwrap(), shape: self.shape.clone().unwrap() })
        } else {
            None
        }
    }
    fn is_int_const(&self) -> bool {
        matches!(self.c, Const::Int(_))
    }
    fn elem_is(&self, e: &str) -> bool {
        self.elem.as_deref() == Some(e)
    }
    fn fixed(&self) -> bool {
        self.elem.as_deref().map(is_fixed).unwrap_or(false)
    }
}

impl std::fmt::Display for At {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.kind == AKind::Void {
            return write!(f, "void");
        }
        if self.kind != AKind::Num || (self.elem.is_none() && self.shape.is_none()) {
            return write!(f, "a value of unknown type");
        }
        let elem = self.elem.clone().unwrap_or_else(|| "number".to_string());
        match &self.shape {
            None => write!(f, "{} of unknown shape", elem),
            Some(s) if !s.is_empty() => {
                let d: Vec<String> = s.iter().map(|d| d.to_string()).collect();
                write!(f, "{}[{}]", elem, d.join(","))
            }
            _ => write!(f, "{}", elem),
        }
    }
}

fn from_ty(ty: &Ty) -> At {
    if ty.elem == "void" {
        At::void()
    } else {
        At::n(&ty.elem, ty.shape.clone())
    }
}

/// `_promote` of two element types, either of which may be unknown.
fn promote_o(a: &Option<String>, b: &Option<String>) -> Option<String> {
    let fx = |x: &Option<String>| x.as_deref().map(is_fixed).unwrap_or(false);
    if fx(a) || fx(b) {
        return if a == b { a.clone() } else { None };
    }
    let is = |x: &Option<String>, w: &str| x.as_deref() == Some(w);
    if is(a, "float") || is(b, "float") {
        return Some("float".to_string());
    }
    if is(a, "int") && is(b, "int") {
        return Some("int".to_string());
    }
    None
}

fn promote(a: &str, b: &str) -> Option<String> {
    promote_o(&Some(a.to_string()), &Some(b.to_string()))
}

fn const_minmax(c: &Const) -> Option<(i128, i128)> {
    match c {
        Const::Int(v) => Some((*v, *v)),
        Const::IArr(v, _) if !v.is_empty() => Some((*v.iter().min().unwrap(), *v.iter().max().unwrap())),
        Const::Flt(f) => Some((f.trunc() as i128, f.trunc() as i128)),
        Const::FArr(v, _) if !v.is_empty() => {
            let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Some((lo.trunc() as i128, hi.trunc() as i128))
        }
        _ => None,
    }
}

/// Does the int constant fit the integer part of the fixed type `elem`? None when it is not a known constant.
fn fixed_holds(elem: &str, c: &Const) -> Option<bool> {
    if matches!(c, Const::Nc | Const::Cu) {
        return None;
    }
    let (_, i) = fixed_spec(elem)?;
    let (lo, hi) = (-(1i128 << (i - 1)), (1i128 << (i - 1)) - 1);
    match const_minmax(c) {
        None => Some(true),
        Some((a, b)) => Some(a >= lo && b <= hi),
    }
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

pub struct Diag {
    pub rule: &'static str,
    pub span: Sp,
    pub src: usize,
    pub message: String,
    pub hint: Option<String>,
    pub function: Option<String>,
}

pub struct SrcInfo {
    pub name: String,
    pub source: Source,
}

pub struct FunctionInfo {
    pub name: String,
    pub node: Rc<Node>,
    pub span: Sp,
    pub src: usize,
    pub params: Vec<String>,
    pub calls: Vec<String>,
    pub reachable: bool,
    pub in_profile: bool,
    /// argument types -> result type, in the order first seen
    pub specializations: Vec<(Vec<Option<Ty>>, Option<Ty>)>,
}

pub struct GlobalInfo {
    pub name: String,
    pub span: Sp,
    pub src: usize,
    pub at: At,
    pub assigned: bool,
    pub resolved: bool,
    pub param: bool,
}

impl GlobalInfo {
    fn new(name: &str, span: Sp, src: usize) -> GlobalInfo {
        GlobalInfo { name: name.to_string(), span, src, at: At::num(None, None, Const::Cu), assigned: false, resolved: false, param: false }
    }
    pub fn ty(&self) -> Option<Ty> {
        self.at.ty()
    }
}

/// A map that keeps the order its keys were first added in.
pub struct Ordered<T> {
    pub items: Vec<(String, T)>,
    index: HashMap<String, usize>,
}

impl<T> Ordered<T> {
    fn new() -> Ordered<T> {
        Ordered { items: Vec::new(), index: HashMap::new() }
    }
    fn get(&self, k: &str) -> Option<&T> {
        self.index.get(k).map(|&i| &self.items[i].1)
    }
    fn get_mut(&mut self, k: &str) -> Option<&mut T> {
        match self.index.get(k) {
            Some(&i) => Some(&mut self.items[i].1),
            None => None,
        }
    }
    fn contains(&self, k: &str) -> bool {
        self.index.contains_key(k)
    }
    fn set(&mut self, k: &str, v: T) {
        match self.index.get(k) {
            Some(&i) => self.items[i].1 = v,
            None => {
                self.index.insert(k.to_string(), self.items.len());
                self.items.push((k.to_string(), v));
            }
        }
    }
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.items.iter().map(|(_, v)| v)
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// A top-level statement of the program in the order the driver evaluates it: `kind` is "function", "global", "extern", "entry", "param" or
/// "load"; `name` the name defined (the file named, for a load); `span` the statement in the source `src`.
pub struct Stmt {
    pub kind: &'static str,
    pub name: String,
    pub span: Sp,
    pub src: usize,
}

pub struct Report {
    pub statements: Vec<Stmt>,
    pub sources: Vec<SrcInfo>,
    pub diagnostics: Vec<Diag>,
    pub functions: Ordered<FunctionInfo>,
    pub globals: Ordered<GlobalInfo>,
    pub externs: Ordered<ExternDecl>,
    pub entries: Vec<EntryDecl>,
    pub params: Ordered<Ty>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.diagnostics.is_empty()
    }
    pub fn unresolved(&self) -> Vec<String> {
        self.globals.values().filter(|g| !g.resolved).map(|g| g.name.clone()).collect()
    }
    pub fn summary(&self) -> String {
        let fns = self.functions.values().filter(|f| f.reachable).count();
        [plural(fns, "function", None), plural(self.entries.len(), "entry", Some("entries")), plural(self.externs.len(), "extern", None), plural(self.globals.len(), "global", None)].join(", ")
    }
}

// ---------------------------------------------------------------------------
// The checker
// ---------------------------------------------------------------------------

struct Var {
    ty: Option<At>,
    set: bool,
    hint: Option<String>,
}

struct Frame {
    vars: HashMap<String, Var>,
    parent: Option<usize>,
}

#[derive(Default)]
struct Refs {
    refs: Vec<(String, Sp)>,
    seen: HashSet<String>,
    assigns: Vec<(String, Sp)>,
    aseen: HashSet<String>,
}

impl Refs {
    fn r(&mut self, name: &str, sp: Sp) {
        if self.seen.insert(name.to_string()) {
            self.refs.push((name.to_string(), sp));
        }
    }
    fn a(&mut self, name: &str, sp: Sp) {
        if self.aseen.insert(name.to_string()) {
            self.assigns.push((name.to_string(), sp));
        }
    }
}

const MAX_PASSES: usize = 4;
const ARGS: [&str; 3] = ["x", "y", "z"];

#[derive(Clone, Copy, PartialEq)]
enum TK {
    Function,
    Global,
    Extern,
}

#[derive(Clone)]
enum Verb {
    Monad(String),
    Dyad(String),
    Func(String),
    Extern(String),
    Lit(Rc<Node>, usize),
}

#[derive(Clone)]
pub enum LitVal {
    /// a fixed-point value of this type (raw values are not kept)
    Fixed(Ty),
    Undefined,
    QSym(String),
    Char,
    Func,
    At(At),
    Int(i128),
    Flt(f64),
    Str,
    Arr(Lit),
    Dict,
}

enum Count {
    Cu,
    Val(i128),
}

pub struct Options {
    pub int_bits: u32,
    /// false: do not read the files named by `.l`
    pub resolve: bool,
    /// also check the functions no entry reaches, as roots with arguments of unknown type
    pub all_functions: bool,
    /// directory for `.l` lookups, instead of the directory of the name
    pub base_dir: Option<String>,
    /// library directories searched after KLONGPATH; empty: the library built into the binary (and `KLONG_LIB_DIRS`)
    pub lib_dirs: Vec<String>,
    /// the compile-time values of the globals (stage one of the compiler), by name
    pub globals: HashMap<String, LitVal>,
}

impl Default for Options {
    fn default() -> Options {
        Options { int_bits: 32, resolve: true, all_functions: false, base_dir: None, lib_dirs: Vec::new(), globals: HashMap::new() }
    }
}

/// What a module offers the compiler: its `.export` table, each value already named as the compiler knows it (the function or the
/// global `prefix.name`, or a number).
struct ModInfo {
    exports: HashMap<String, Rc<Node>>,
}

/// The module whose file is being read (the program itself is the first): the prefix its names get, the top-level names it defines,
/// the aliases it has bound, and the `.export` it has made.
struct ModCtx {
    prefix: String,
    names: HashSet<String>,
    aliases: HashMap<String, Rc<ModInfo>>,
    bad_aliases: HashSet<String>,
    export: Option<HashMap<String, Rc<Node>>>,
}

struct Checker {
    mods: Vec<ModCtx>,
    modules: HashMap<String, Option<Rc<ModInfo>>>,
    prefixes: HashSet<String>,
    r: Report,
    base_dir: String,
    int_bits: u32,
    resolve: bool,
    all_functions: bool,
    src: usize,
    mute: i32,
    seen: HashSet<(String, String, Sp, String)>,
    stack: Vec<String>,
    memo: HashMap<(String, Vec<AtKey>), At>,
    reported: HashSet<(String, Vec<AtKey>)>,
    loaded: HashSet<String>,
    kinds: HashMap<String, TK>,
    refs: HashMap<String, Refs>,
    broken: BTreeSet<String>,
    frames: Vec<Frame>,
    sess: Session,
    given: HashMap<String, LitVal>,
}

fn strip_module(msg: &str) -> String {
    // `[A-Za-z0-9.]+ after a backquote
    let c: Vec<char> = msg.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '`' {
            let mut j = i + 1;
            while j < c.len() && (c[j].is_ascii_alphanumeric() || c[j] == '.') {
                j += 1;
            }
            if j > i + 1 {
                i = j;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

fn is_plain_str(n: &Node) -> Option<&str> {
    match &n.k {
        Kind::Str(s) => Some(s),
        _ => None,
    }
}

impl Checker {
    // -- diagnostics ----------------------------------------------------------
    fn diag(&mut self, rule: &'static str, span: Sp, message: String, hint: Option<&str>) {
        let src = self.src;
        self.diag_in(rule, span, message, hint, src)
    }

    fn diag_in(&mut self, rule: &'static str, span: Sp, message: String, hint: Option<&str>, src: usize) {
        if self.mute > 0 {
            return;
        }
        let message = strip_module(&message);
        let key = (rule.to_string(), self.r.sources[src].name.clone(), span, message.clone());
        if !self.seen.insert(key) {
            return;
        }
        let hint = match hint {
            Some(h) => Some(h.to_string()),
            None => tables::rule(rule).and_then(|r| r.hint).map(String::from),
        };
        self.r.diagnostics.push(Diag { rule, span, src, message, hint, function: None });
    }

    // -- top level ------------------------------------------------------------------
    fn run(&mut self, name: &str, text: &str) {
        if std::fs::metadata(name).map(|m| m.is_file()).unwrap_or(false) {
            if let Ok(real) = std::fs::canonicalize(name) {
                self.loaded.insert(real.to_string_lossy().into_owned());
            }
        }
        let base = self.base_dir.clone();
        self.load(name, text, &base, 0);
        self.graph();
        self.roots();
        self.finish();
    }

    fn load(&mut self, name: &str, text: &str, base_dir: &str, depth: usize) {
        let src = self.r.sources.len();
        self.r.sources.push(SrcInfo { name: name.to_string(), source: Source::new(name, text.to_string()) });
        let rec = parse_recovering(name, text, 2);
        for d in &rec.problems {
            self.diag_in("strict-syntax", d.span, d.message.clone(), d.hint.as_deref(), src);
        }
        self.broken.extend(rec.broken.iter().cloned());
        let saved = self.src;
        self.src = src;
        let stmts: Vec<(Rc<Node>, Sp)> = rec
            .statements
            .iter()
            .map(|s| {
                let c = Conv { tree: &s.tree, syms: &rec.syms };
                (Rc::new(c.conv(s.node)), s.span)
            })
            .collect();
        self.collect_names(&stmts);
        for (n, span) in stmts {
            if self.module_statement(&n, span, src, base_dir, depth) {
                continue;
            }
            let n = self.rewrite(&n, &mut Vec::new(), span);
            self.statement(&n, span, src, base_dir, depth);
        }
        self.src = saved;
    }

    // -- modules (design point 11) ------------------------------------------------------
    //
    // `alias::.import("file")` at the top level reads the file as a module: its top-level names become the compiler's names
    // `prefix.name` (the file's own name, so the C name does not depend on the importer's alias), `.export` fixes what `alias.key` means,
    // and `alias.key` is rewritten to that name before the statement is checked. A function reached through two aliases is one function.

    fn collect_names(&mut self, stmts: &[(Rc<Node>, Sp)]) {
        let mut found = Vec::new();
        for (n, _) in stmts {
            if let Kind::Dyad(op, left, right) = &n.k {
                if op == "::" && !right.is_missing() {
                    match &left.k {
                        Kind::Sym(name) if !ARGS.contains(&name.as_str()) && !Self::is_import(right) => found.push(name.clone()),
                        Kind::Destructure(names) => found.extend(names.iter().cloned()),
                        _ => {}
                    }
                }
            }
        }
        if let Some(c) = self.mods.last_mut() {
            c.names.extend(found);
        }
    }

    fn is_import(n: &Node) -> bool {
        matches!(&n.k, Kind::Call { callee, .. } if matches!(&callee.k, Kind::Sym(s) if s == ".import"))
    }

    /// A statement that is the module system's own: `alias::.import("path")`, `.export(...)`, an `.entry` of a module. True when handled.
    fn module_statement(&mut self, n: &Rc<Node>, span: Sp, src: usize, base_dir: &str, depth: usize) -> bool {
        if let Kind::Dyad(op, left, right) = &n.k {
            if op == "::" && Self::is_import(right) {
                let lsp = n.arg_span(0, span);
                let Kind::Sym(alias) = &left.k else {
                    self.diag("redefinition", lsp, "the result of .import must be assigned to a name".to_string(), None);
                    return true;
                };
                let Kind::Call { args, .. } = &right.k else { return true };
                self.import_declaration(alias, args, n, span, lsp, base_dir, depth);
                return true;
            }
        }
        if let Kind::Call { callee, args, kcall: true } = &n.k {
            if let Kind::Sym(name) = &callee.k {
                if name == ".export" {
                    self.export_declaration(args, n, span, src);
                    return true;
                }
                // (a library's entries are not the program's)
                if name == ".entry" && self.mods.len() > 1 {
                    return true;
                }
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn import_declaration(&mut self, alias: &str, args: &[Rc<Node>], n: &Rc<Node>, span: Sp, lsp: Sp, base_dir: &str, depth: usize) {
        if args.len() != 1 || is_plain_str(&args[0]).is_none() {
            self.diag("decl-literal", n.arg_span(1, span), ".import in a compiled program takes one string literal: the compiler reads the file when it checks".to_string(), None);
            self.mods.last_mut().map(|c| c.bad_aliases.insert(alias.to_string()));
            return;
        }
        let spec = is_plain_str(&args[0]).unwrap().to_string();
        let again = self.mods.last().map(|c| c.aliases.contains_key(alias) || c.bad_aliases.contains(alias) || c.names.contains(alias)).unwrap_or(false);
        if again {
            self.diag("redefinition", lsp, format!("{} is assigned more than once; the compiler resolves alias.name only for a name assigned once, by .import", alias), None);
            return;
        }
        self.r.statements.push(Stmt { kind: "load", name: spec.clone(), span, src: self.src });
        if !self.resolve {
            return;
        }
        let Some(path) = klong_lint::recover::find_import(&self.sess, &spec, base_dir) else {
            self.diag("load", n.arg_span(1, span), format!("cannot find the file \"{}\" named by .import", spec), Some("a name without ./ or ../ is looked for in KLONGPATH and the library; ./name and ../name are relative to the importing file"));
            self.mods.last_mut().map(|c| c.bad_aliases.insert(alias.to_string()));
            return;
        };
        let real = std::fs::canonicalize(&path).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| path.clone());
        let info = match self.modules.get(&real) {
            Some(Some(i)) => Some(i.clone()),
            Some(None) => {
                self.diag("load", n.arg_span(1, span), format!("import cycle at \"{}\"", spec), None);
                None
            }
            None if depth >= 8 => {
                self.diag("load", n.arg_span(1, span), format!(".import nests too deeply at \"{}\"", spec), None);
                None
            }
            None => self.read_module(&real, &path, n.arg_span(1, span), depth),
        };
        match info {
            Some(i) => {
                if let Some(c) = self.mods.last_mut() {
                    c.aliases.insert(alias.to_string(), i);
                }
            }
            None => {
                self.mods.last_mut().map(|c| c.bad_aliases.insert(alias.to_string()));
            }
        }
    }

    fn read_module(&mut self, real: &str, path: &str, sp: Sp, depth: usize) -> Option<Rc<ModInfo>> {
        let text = match self.sess.read_loaded(path) {
            Ok(t) => t,
            Err(e) => {
                self.diag("load", sp, format!("cannot read \"{}\": {}", path, e), None);
                return None;
            }
        };
        let stem = path.rsplit('/').next().unwrap_or(path).trim_end_matches(".kg").to_string();
        let mut prefix = stem.clone();
        let mut k = 2;
        while self.prefixes.contains(&prefix) {
            prefix = format!("{}{}", stem, k);
            k += 1;
        }
        self.prefixes.insert(prefix.clone());
        self.modules.insert(real.to_string(), None);
        self.mods.push(ModCtx { prefix, names: HashSet::new(), aliases: HashMap::new(), bad_aliases: HashSet::new(), export: None });
        let dir = klong_lint::recover::py_dirname(path);
        let nsrc = self.r.sources.len();
        self.load(path, &text, &dir, depth + 1);
        let ctx = self.mods.pop()?;
        match ctx.export {
            Some(exports) => {
                let info = Rc::new(ModInfo { exports });
                self.modules.insert(real.to_string(), Some(info.clone()));
                Some(info)
            }
            None => {
                let saved = self.src;
                self.src = nsrc.min(self.r.sources.len().saturating_sub(1));
                self.diag("load", sp, format!("{} has no .export, so it is not a module", path), None);
                self.src = saved;
                self.modules.remove(real);
                None
            }
        }
    }

    fn export_declaration(&mut self, args: &[Rc<Node>], n: &Rc<Node>, span: Sp, src: usize) {
        let _ = src;
        let elems = match args {
            [a] => match &a.k {
                Kind::EDict(v) => Some(v.clone()),
                _ => None,
            },
            _ => None,
        };
        let Some(elems) = elems.filter(|e| e.len() % 2 == 0) else {
            self.diag("undefined-name", n.arg_span(0, span), ".export in a compiled program takes a dictionary written :{;\"key\";value;...}".to_string(), None);
            return;
        };
        let mut out = HashMap::new();
        for pair in elems.chunks(2) {
            let Some(key) = is_plain_str(&pair[0]) else {
                self.diag("undefined-name", span, "an export key must be a string literal".to_string(), None);
                continue;
            };
            let v = self.rewrite(&pair[1], &mut Vec::new(), span);
            match &v.k {
                Kind::Sym(_) | Kind::Int(_) | Kind::Flt(_) => {
                    out.insert(key.to_string(), v);
                }
                _ => self.diag("undefined-name", span, format!("the value of the export \"{}\" must be a name or a number for the compiler", key), None),
            }
        }
        if let Some(c) = self.mods.last_mut() {
            c.export = Some(out);
        }
    }

    /// The names of `n` as the compiler knows them: the running module's own names get its prefix, `alias.key` is what the module exports.
    fn rewrite(&mut self, n: &Rc<Node>, shadow: &mut Vec<String>, span: Sp) -> Rc<Node> {
        let mk = |k: Kind| Rc::new(Node { k, span: n.span, op_span: n.op_span, arg_spans: n.arg_spans.clone() });
        match &n.k {
            Kind::Sym(name) => {
                if shadow.contains(name) || ARGS.contains(&name.as_str()) || name.starts_with('.') {
                    return n.clone();
                }
                if let Some((base, rest)) = name.split_once('.') {
                    if shadow.iter().any(|s| s == base) {
                        return n.clone();
                    }
                    let found = self.mods.last().and_then(|c| c.aliases.get(base).cloned());
                    if let Some(info) = found {
                        return match info.exports.get(rest) {
                            Some(v) => Rc::new(Node { k: v.k.clone(), span: n.span, op_span: None, arg_spans: Vec::new() }),
                            None => {
                                let mut keys: Vec<&str> = info.exports.keys().map(|s| s.as_str()).collect();
                                keys.sort();
                                self.diag("undefined-name", span, format!("{} has no export \"{}\"", base, rest), Some(&format!("the module exports: {}", keys.join(" "))));
                                n.clone()
                            }
                        };
                    }
                    return n.clone();
                }
                match self.mods.last() {
                    Some(c) if !c.prefix.is_empty() && c.names.contains(name) => mk(Kind::Sym(format!("{}.{}", c.prefix, name))),
                    _ => n.clone(),
                }
            }
            Kind::Destructure(names) if shadow.is_empty() => match self.mods.last() {
                Some(c) if !c.prefix.is_empty() => mk(Kind::Destructure(names.iter().map(|x| if c.names.contains(x) { format!("{}.{}", c.prefix, x) } else { x.clone() }).collect())),
                _ => n.clone(),
            },
            Kind::Monad(g, a) => {
                let a = self.rewrite(a, shadow, span);
                mk(Kind::Monad(g.clone(), a))
            }
            Kind::Dyad(g, l, r) => {
                let (l, r) = (self.rewrite(l, shadow, span), self.rewrite(r, shadow, span));
                mk(Kind::Dyad(g.clone(), l, r))
            }
            Kind::Call { callee, args, kcall } => {
                let callee = self.rewrite(callee, shadow, span);
                let args = args.iter().map(|a| self.rewrite(a, shadow, span)).collect();
                mk(Kind::Call { callee, args, kcall: *kcall })
            }
            Kind::FnLit { stmts, spans, params, locals } => {
                let mark = shadow.len();
                shadow.extend(params.iter().cloned());
                shadow.extend(locals.iter().cloned());
                let stmts = stmts.iter().map(|s| self.rewrite(s, shadow, span)).collect();
                shadow.truncate(mark);
                mk(Kind::FnLit { stmts, spans: spans.clone(), params: params.clone(), locals: locals.clone() })
            }
            Kind::Body(v) => {
                let v = v.iter().map(|s| self.rewrite(s, shadow, span)).collect();
                mk(Kind::Body(v))
            }
            Kind::Chain { verb, advs, dyadic, l, r } => {
                let (verb, l, r) = (self.rewrite(verb, shadow, span), self.rewrite(l, shadow, span), self.rewrite(r, shadow, span));
                mk(Kind::Chain { verb, advs: advs.clone(), dyadic: *dyadic, l, r })
            }
            Kind::Cond(v) => {
                let v = v.iter().map(|s| self.rewrite(s, shadow, span)).collect();
                mk(Kind::Cond(v))
            }
            Kind::EList(v) => {
                let v = v.iter().map(|s| self.rewrite(s, shadow, span)).collect();
                mk(Kind::EList(v))
            }
            Kind::EDict(v) => {
                let v = v.iter().map(|s| self.rewrite(s, shadow, span)).collect();
                mk(Kind::EDict(v))
            }
            _ => n.clone(),
        }
    }

    fn statement(&mut self, n: &Rc<Node>, span: Sp, src: usize, base_dir: &str, depth: usize) {
        if let Kind::Dyad(op, left, right) = &n.k {
            if op == "::" && !right.is_missing() {
                let lsp = n.arg_span(0, span);
                match &left.k {
                    Kind::Destructure(names) => {
                        self.define_globals(names, right, n, span, lsp, src, false);
                        return;
                    }
                    Kind::Sym(name) => {
                        if ARGS.contains(&name.as_str()) {
                            self.diag("toplevel-effect", lsp, format!("{} is a function argument; it cannot be defined at top level", name), None);
                            return;
                        }
                        if right.is_fn_literal() {
                            self.define_function(name, right, span, lsp, src);
                        } else {
                            self.define_globals(&[name.clone()], right, n, span, lsp, src, true);
                        }
                        return;
                    }
                    _ => {}
                }
            }
        }
        if let Kind::Call { callee, kcall: true, .. } = &n.k {
            if let Kind::Sym(name) | Kind::QSym(name) = &callee.k {
                if [".extern", ".entry", ".l", ".param"].contains(&name.as_str()) {
                    let which = name.clone();
                    self.declaration(&which, n, span, src, base_dir, depth);
                    return;
                }
            }
        }
        let mut what = "this expression".to_string();
        if let Kind::Call { callee, kcall: true, .. } = &n.k {
            if let Kind::Sym(name) | Kind::QSym(name) = &callee.k {
                what = format!("the call of {}", name);
            }
        }
        self.diag("toplevel-effect", span, format!("{} is evaluated for its effect at top level, which has no meaning in a compiled program", what), None);
    }

    /// Record a top-level name; false (and a diagnostic) if it clashes with an earlier definition.
    fn taken(&mut self, name: &str, kind: TK, span: Sp) -> bool {
        let old = self.kinds.get(name).copied();
        if old.is_none() || (old == Some(TK::Global) && kind == TK::Global) {
            self.kinds.insert(name.to_string(), kind);
            return true;
        }
        let a = |k: TK| match k {
            TK::Function => "a function",
            TK::Global => "a global",
            TK::Extern => "an extern",
        };
        self.diag("redefinition", span, format!("{} is already defined as {}; it cannot be defined again as {}", name, a(old.unwrap()), a(kind)), None);
        false
    }

    fn define_function(&mut self, name: &str, node: &Rc<Node>, span: Sp, lsp: Sp, src: usize) {
        if !self.taken(name, TK::Function, lsp) {
            return;
        }
        let params = match &node.k {
            Kind::FnLit { params, .. } => params.clone(),
            _ => Vec::new(),
        };
        let f = FunctionInfo { name: name.to_string(), node: node.clone(), span, src, params, calls: Vec::new(), reachable: false, in_profile: true, specializations: Vec::new() };
        self.r.functions.set(name, f);
        self.r.statements.push(Stmt { kind: "function", name: name.to_string(), span, src });
    }

    #[allow(clippy::too_many_arguments)]
    fn define_globals(&mut self, names: &[String], right: &Rc<Node>, node: &Rc<Node>, span: Sp, lsp: Sp, src: usize, single: bool) {
        let known = self.literal(right);
        for (i, name) in names.iter().enumerate() {
            let given = self.given.get(name).cloned();
            if self.r.params.contains(name) {
                self.diag("redefinition", lsp, format!("{} is a parameter (.param); it cannot be defined again", name), None);
                continue;
            }
            if ARGS.contains(&name.as_str()) || !self.taken(name, TK::Global, lsp) {
                continue;
            }
            self.r.globals.set(name, GlobalInfo::new(name, span, src));
            self.r.statements.push(Stmt { kind: "global", name: name.clone(), span, src });
            if let Some(v) = &given {
                self.set_global(name, v, lsp, src);
            } else if let Some(v) = &known {
                let vsp = node.arg_span(1, span);
                if single {
                    self.set_global(name, v, vsp, src);
                } else if let LitVal::Arr(l) = v {
                    // a list of values, one for each name
                    if let Some(item) = row_of(l, names.len(), i) {
                        self.set_global(name, &item, vsp, src);
                    }
                }
            }
        }
        // a::b::3 defines b as well: any name the expression assigns exists once the host has run it
        let mut inner = Refs::default();
        self.scan(right, span, &HashSet::new(), &mut inner);
        for (name, _) in inner.assigns.clone() {
            if !self.kinds.contains_key(&name) && !ARGS.contains(&name.as_str()) && !name.starts_with('.') {
                self.kinds.insert(name.clone(), TK::Global);
                self.r.globals.set(&name, GlobalInfo::new(&name, span, src));
                self.r.statements.push(Stmt { kind: "global", name: name.clone(), span, src });
                if let Some(v) = self.given.get(&name).cloned() {
                    self.set_global(&name, &v, lsp, src);
                }
            }
        }
    }

    fn set_global(&mut self, name: &str, value: &LitVal, span: Sp, src: usize) {
        if let LitVal::At(at) = value {
            let g = self.r.globals.get_mut(name).unwrap();
            g.at = at.clone();
            g.resolved = at.known();
            return;
        }
        let res = validate_global(value, self.int_bits);
        self.r.globals.get_mut(name).unwrap().resolved = true;
        match res {
            Err((rule, message)) => {
                self.diag_in(rule, span, format!("{}: {}", name, message), None, src);
                self.r.globals.get_mut(name).unwrap().at = At::any();
            }
            Ok(ty) => {
                let c = if ty.shape.is_empty() {
                    match value {
                        LitVal::Int(v) => Const::Int(*v),
                        LitVal::Flt(f) => Const::Flt(*f),
                        _ => Const::Nc,
                    }
                } else if let LitVal::Arr(Lit::Num { shape, float, ints, flts }) = value {
                    let n: usize = shape.iter().product();
                    if n <= 4096 {
                        if *float {
                            Const::FArr(flts.clone(), shape.len())
                        } else {
                            Const::IArr(ints.clone(), shape.len())
                        }
                    } else {
                        Const::Nc
                    }
                } else {
                    Const::Nc
                };
                self.r.globals.get_mut(name).unwrap().at = At::num(Some(&ty.elem), Some(ty.shape.clone()), c);
            }
        }
    }

    /// The value of a top-level definition, when it can be read off the tree.
    fn literal(&self, n: &Node) -> Option<LitVal> {
        match &n.k {
            Kind::QSym(s) => Some(LitVal::QSym(s.clone())),
            Kind::Char => Some(LitVal::Char),
            Kind::Sym(name) => {
                if matches!(self.kinds.get(name), Some(TK::Function) | Some(TK::Extern)) {
                    return Some(LitVal::Func);
                }
                match self.r.globals.get(name) {
                    Some(g) if g.resolved => Some(LitVal::At(g.at.clone())),
                    _ => None,
                }
            }
            Kind::Int(v) => Some(LitVal::Int(*v)),
            Kind::Flt(f) => Some(LitVal::Flt(*f)),
            Kind::Str(_) => Some(LitVal::Str),
            Kind::Arr(l) => Some(LitVal::Arr(l.clone())),
            Kind::EDict(_) => Some(LitVal::Dict),
            Kind::DictLit(is_dict) => {
                if *is_dict {
                    Some(LitVal::Dict)
                } else {
                    None
                }
            }
            Kind::Monad(op, a) if op == "-" => match &a.k {
                Kind::Int(v) => Some(LitVal::Int(-*v)),
                Kind::Flt(f) => Some(LitVal::Flt(-*f)),
                _ => None,
            },
            Kind::FnLit { .. } => Some(LitVal::Func),
            Kind::Call { kcall: false, args, .. } if args.iter().any(|q| q.is_missing()) => Some(LitVal::Func),
            _ => None,
        }
    }

    fn declaration(&mut self, which: &str, n: &Rc<Node>, span: Sp, src: usize, base_dir: &str, depth: usize) {
        let args: Vec<Rc<Node>> = match &n.k {
            Kind::Call { args, .. } => args.clone(),
            _ => Vec::new(),
        };
        let want = match which {
            ".extern" | ".param" => (2, 3),
            ".entry" => (2, 2),
            _ => (1, 1),
        };
        if !(want.0 <= args.len() && args.len() <= want.1) {
            let takes = if which == ".extern" || which == ".param" { "2 or 3 arguments".to_string() } else { plural(want.0, "argument", None) };
            self.diag("decl-literal", span, format!("{} takes {}, got {}", which, takes, args.len()), None);
            return;
        }
        if which == ".param" {
            self.param_declaration(n, &args, span, src);
            return;
        }
        for (i, a) in args.iter().enumerate() {
            if is_plain_str(a).is_none() {
                self.diag("decl-literal", n.arg_span(i, span), format!("argument {} of {} must be a string literal", i + 1, which), None);
                return;
            }
        }
        if which == ".l" {
            let name = is_plain_str(&args[0]).unwrap().to_string();
            self.load_file(&name, n.arg_span(0, span), span, base_dir, depth);
            return;
        }
        let name = is_plain_str(&args[0]).unwrap().to_string();
        let text = is_plain_str(&args[1]).unwrap().to_string();
        let (nsp, ssp) = (n.arg_span(0, span), n.arg_span(1, span));
        let sig = match sig::parse_signature(&text, which == ".extern") {
            Ok(s) => s,
            Err(e) => {
                let at = self.inside_literal(src, ssp, &text, e.span);
                self.diag("signature", at, format!("{} signature: {}", which, e.message), None);
                return;
            }
        };
        if which == ".extern" {
            if let Some(bad) = sig::check_extern_name(&name) {
                self.diag("extern", nsp, bad, None);
                return;
            }
            let module = if args.len() == 3 { is_plain_str(&args[2]).unwrap().to_string() } else { "env".to_string() };
            if let Some(old) = self.r.externs.get(&name) {
                if old.signature != sig || old.module != module {
                    let first = old.signature.to_string();
                    self.diag("extern", span, format!("extern {} is declared again with a different signature (first as \"{}\")", name, first), None);
                }
                return;
            }
            if !self.taken(&name, TK::Extern, nsp) {
                return;
            }
            self.r.externs.set(&name, ExternDecl { name: name.clone(), signature: sig, module, span, src });
            self.r.statements.push(Stmt { kind: "extern", name: name.clone(), span, src });
            return;
        }
        if let Some(bad) = sig::check_entry_name(&name) {
            self.diag("entry", nsp, bad, None);
            return;
        }
        let Some(f) = self.r.functions.get(&name) else {
            match self.kinds.get(&name) {
                Some(TK::Global) => self.diag("entry", nsp, format!("{} is a global, not a function; only a function can be an entry", name), None),
                Some(TK::Extern) => self.diag("entry", nsp, format!("{} is an extern; only a function defined in the program can be an entry", name), None),
                _ => {
                    if !self.broken.contains(&name) {
                        self.diag("entry", nsp, format!("the function {} is not defined before this .entry", name), Some("define the function first; .entry checks the signature against it"));
                    }
                }
            }
            return;
        };
        let arity = f.params.len();
        if arity != sig.arity() {
            self.diag("entry", ssp, format!("{} takes {} but the signature has {}", name, plural(arity, "argument", None), plural(sig.arity(), "parameter", None)), None);
            return;
        }
        self.r.statements.push(Stmt { kind: "entry", name: name.clone(), span, src });
        self.r.entries.push(EntryDecl { name, signature: sig, span, src });
    }

    /// Where inside the string literal at `lit` (the span of the literal, quotes included) the characters `inner` of its value are, when
    /// the literal has no `""` escapes; else the literal.
    fn inside_literal(&self, src: usize, lit: Sp, text: &str, inner: (usize, usize)) -> Sp {
        let source = &self.r.sources[src].source;
        let shown = source.text.get(lit.0..lit.1).unwrap_or("");
        if shown == format!("\"{}\"", text) {
            // offsets are bytes here: convert the character offsets of the signature text
            let b = |c: usize| text.char_indices().nth(c).map(|(i, _)| i).unwrap_or(text.len());
            let start = lit.0 + 1 + b(inner.0);
            let end = (lit.0 + 1 + b(inner.1)).max(start + 1);
            (start, end)
        } else {
            lit
        }
    }

    fn param_declaration(&mut self, n: &Rc<Node>, args: &[Rc<Node>], span: Sp, src: usize) {
        for (i, a) in args.iter().take(2).enumerate() {
            if is_plain_str(a).is_none() {
                self.diag("decl-literal", n.arg_span(i, span), format!("argument {} of .param must be a string literal", i + 1), None);
                return;
            }
        }
        let name = is_plain_str(&args[0]).unwrap().to_string();
        let text = is_plain_str(&args[1]).unwrap().to_string();
        let (nsp, tsp) = (n.arg_span(0, span), n.arg_span(1, span));
        let ty = match sig::parse_type(&text) {
            Ok(t) => t,
            Err(e) => {
                let at = self.inside_literal(src, tsp, &text, e.span);
                self.diag("signature", at, format!(".param type: {}", e.message), None);
                return;
            }
        };
        if !ty.shape.is_empty() || (ty.elem != "int" && ty.elem != "float" && !is_fixed(&ty.elem)) {
            self.diag("param", tsp, format!("the type of a parameter is a scalar int, float or fixed<W,I>; {} is not", ty), Some("a parameter is one register: pass arrays as entry arguments"));
            return;
        }
        if let Some(bad) = sig::check_extern_name(&name) {
            self.diag("param", nsp, bad.replace("an extern", "a parameter"), None);
            return;
        }
        if args.len() == 3 {
            let d = &args[2];
            let num: Option<(bool, bool)> = match &d.k {
                Kind::Int(_) => Some((true, false)),
                Kind::Flt(_) => Some((true, true)),
                Kind::Monad(op, a) if op == "-" => match &a.k {
                    Kind::Int(_) => Some((true, false)),
                    Kind::Flt(_) => Some((true, true)),
                    _ => None,
                },
                _ => None,
            };
            match num {
                None => {
                    self.diag("param", n.arg_span(2, span), "the default of .param must be a number literal".to_string(), None);
                    return;
                }
                Some((_, is_float)) => {
                    if ty.elem == "int" && is_float {
                        self.diag("param", n.arg_span(2, span), format!("the default of the int parameter {} must be an int", name), None);
                        return;
                    }
                }
            }
        }
        if let Some(old) = self.r.params.get(&name) {
            if *old != ty {
                let o = old.to_string();
                self.diag("param", span, format!("the parameter {} is declared again as {}; it was {}", name, ty, o), None);
            }
            return;
        }
        if !self.taken(&name, TK::Global, nsp) {
            return;
        }
        let mut g = GlobalInfo::new(&name, span, src);
        g.at = At::n(&ty.elem, Vec::new());
        g.resolved = true;
        g.assigned = true;
        g.param = true;
        self.r.globals.set(&name, g);
        self.r.params.set(&name, ty);
        self.r.statements.push(Stmt { kind: "param", name, span, src });
    }

    fn load_file(&mut self, name: &str, nsp: Sp, span: Sp, base_dir: &str, depth: usize) {
        self.r.statements.push(Stmt { kind: "load", name: name.to_string(), span, src: self.src });
        if !self.resolve {
            return;
        }
        let Some(path) = find_load(&self.sess, name, base_dir) else {
            self.diag("load", nsp, format!("cannot find the file \"{}\" named by .l", name), Some("it is looked up like .l does: in the KLONGPATH directories (default .:lib) from the current directory, then klongpy/lib, then the directory of the program"));
            return;
        };
        let real = std::fs::canonicalize(&path).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| path.clone());
        if self.loaded.contains(&real) {
            return;
        }
        if depth >= 8 {
            self.diag("load", nsp, format!(".l nests too deeply at \"{}\"", name), None);
            return;
        }
        self.loaded.insert(real.clone());
        match self.sess.read_loaded(&path) {
            Ok(text) => {
                let dir = klong_lint::recover::py_dirname(&path);
                self.load(&path, &text, &dir, depth + 1);
            }
            Err(e) => {
                self.diag("load", nsp, format!("cannot read \"{}\": {}", path, e), None);
            }
        }
    }

    // -- call graph --------------------------------------------------------------------
    fn graph(&mut self) {
        let names: Vec<String> = self.r.functions.items.iter().map(|(k, _)| k.clone()).collect();
        for name in &names {
            let (node, span, params) = {
                let f = self.r.functions.get(name).unwrap();
                (f.node.clone(), f.span, f.params.clone())
            };
            let mut r = Refs::default();
            if let Kind::FnLit { stmts, spans, locals, .. } = &node.k {
                let mut bound: HashSet<String> = params.iter().cloned().collect();
                bound.extend(locals.iter().cloned());
                for (s, sp) in stmts.iter().zip(spans.iter()) {
                    self.scan(s, sp.unwrap_or(span), &bound, &mut r);
                }
            }
            let calls: Vec<String> = r.refs.iter().map(|(k, _)| k.clone()).filter(|k| self.r.functions.contains(k) || self.r.externs.contains(k)).collect();
            self.r.functions.get_mut(name).unwrap().calls = calls;
            self.refs.insert(name.clone(), r);
        }
        if !self.r.entries.is_empty() && !self.all_functions {
            let mut todo: Vec<String> = self.r.entries.iter().map(|e| e.name.clone()).collect();
            while let Some(k) = todo.pop() {
                let Some(f) = self.r.functions.get_mut(&k) else { continue };
                if f.reachable {
                    continue;
                }
                f.reachable = true;
                let calls = f.calls.clone();
                for c in calls {
                    if self.r.functions.contains(&c) {
                        todo.push(c);
                    }
                }
            }
        } else {
            for (_, f) in self.r.functions.items.iter_mut() {
                f.reachable = true;
            }
        }
        for name in &names {
            let (reachable, src, calls) = {
                let f = self.r.functions.get(name).unwrap();
                (f.reachable, f.src, f.calls.clone())
            };
            if !reachable {
                continue;
            }
            let (assigns, refs): (Vec<(String, Sp)>, Vec<(String, Sp)>) = {
                let r = &self.refs[name];
                (r.assigns.clone(), r.refs.clone())
            };
            for (a, asp) in &assigns {
                if let Some(g) = self.r.globals.get_mut(a) {
                    g.assigned = true;
                    if g.param {
                        self.diag_in("param", *asp, format!("{} is a parameter: a compiled function reads it and cannot assign it", a), Some("a parameter is written by the host between runs; keep the running value in another global"), src);
                    }
                }
            }
            for callee in &calls {
                if !self.r.functions.contains(callee) {
                    continue;
                }
                let Some(path) = self.path(callee, name) else { continue };
                let msg = if callee == name {
                    format!("{} calls itself", name)
                } else {
                    let mut chain = vec![name.clone()];
                    chain.extend(path);
                    format!("{} calls {}, which leads back to {} ({})", name, callee, name, chain.join(" -> "))
                };
                let sp = refs.iter().find(|(k, _)| k == callee).map(|(_, s)| *s).unwrap_or((0, 0));
                self.diag_in("recursion", sp, msg, None, src);
            }
        }
    }

    /// The names from start to goal along calls, or None.
    fn path(&self, start: &str, goal: &str) -> Option<Vec<String>> {
        let mut prev: HashMap<String, Option<String>> = HashMap::new();
        prev.insert(start.to_string(), None);
        let mut todo = std::collections::VecDeque::new();
        todo.push_back(start.to_string());
        while let Some(cur) = todo.pop_front() {
            if cur == goal {
                let mut out = Vec::new();
                let mut c = Some(cur);
                while let Some(x) = c {
                    c = prev[&x].clone();
                    out.push(x);
                }
                out.reverse();
                return Some(out);
            }
            for k in &self.r.functions.get(&cur).unwrap().calls {
                if self.r.functions.contains(k) && !prev.contains_key(k) {
                    prev.insert(k.clone(), Some(cur.clone()));
                    todo.push_back(k.clone());
                }
            }
        }
        None
    }

    /// Collect the free names a tree mentions and assigns (no types, no diagnostics).
    fn scan(&self, n: &Rc<Node>, sp: Sp, bound: &HashSet<String>, out: &mut Refs) {
        match &n.k {
            Kind::Missing | Kind::QSym(_) | Kind::Char | Kind::Op(_) | Kind::DictLit(_) | Kind::Int(_) | Kind::Flt(_) | Kind::Arr(_) | Kind::Str(_) | Kind::Destructure(_) => {}
            Kind::Sym(s) => {
                if !bound.contains(s) {
                    out.r(s, sp);
                }
            }
            Kind::Cond(items) | Kind::EList(items) | Kind::EDict(items) => {
                for (i, q) in items.iter().enumerate() {
                    let isp = match n.arg_spans.get(i) {
                        Some(Some(s)) => *s,
                        _ => sp,
                    };
                    self.scan(q, isp, bound, out);
                }
            }
            Kind::Body(items) => {
                for q in items {
                    self.scan(q, sp, bound, out);
                }
            }
            _ => {
                let nsp = n.span.unwrap_or(sp);
                match &n.k {
                    Kind::Chain { verb, dyadic, l, r, .. } => {
                        self.scan(verb, n.op_span.unwrap_or(nsp), bound, out);
                        if *dyadic {
                            self.scan(l, n.arg_span(0, nsp), bound, out);
                            self.scan(r, n.arg_span(1, nsp), bound, out);
                        } else {
                            self.scan(r, n.arg_span(0, nsp), bound, out);
                        }
                    }
                    Kind::Monad(_, a) => self.scan(a, n.arg_span(0, nsp), bound, out),
                    Kind::Dyad(op, l, r) => {
                        if op == "::" {
                            let targets: Vec<String> = match &l.k {
                                Kind::Destructure(names) => names.clone(),
                                Kind::Sym(s) => vec![s.clone()],
                                _ => Vec::new(),
                            };
                            for t in targets {
                                if !bound.contains(&t) {
                                    out.a(&t, n.arg_span(0, nsp));
                                }
                            }
                            self.scan(r, n.arg_span(1, nsp), bound, out);
                            return;
                        }
                        self.scan(l, n.arg_span(0, nsp), bound, out);
                        self.scan(r, n.arg_span(1, nsp), bound, out);
                    }
                    Kind::OpN(..) => {}
                    Kind::FnLit { stmts, spans, params, locals } => {
                        let mut inner = bound.clone();
                        inner.extend(params.iter().cloned());
                        inner.extend(locals.iter().cloned());
                        for (s, ssp) in stmts.iter().zip(spans.iter()) {
                            self.scan(s, ssp.unwrap_or(nsp), &inner, out);
                        }
                    }
                    Kind::Call { callee, args, .. } => {
                        let csp = n.op_span.unwrap_or(nsp);
                        match &callee.k {
                            Kind::Sym(_) => self.scan(callee, csp, bound, out),
                            Kind::Body(items) => {
                                for q in items {
                                    self.scan(q, csp, bound, out);
                                }
                            }
                            _ => self.scan(callee, csp, bound, out),
                        }
                        for (i, q) in args.iter().enumerate() {
                            self.scan(q, n.arg_span(i, nsp), bound, out);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // -- roots ----------------------------------------------------------------------
    fn roots(&mut self) {
        let entries: Vec<EntryDecl> = self.r.entries.clone();
        for e in &entries {
            let Some(f) = self.r.functions.get(&e.name) else { continue };
            let (fsrc, fspan) = (f.src, f.span);
            self.src = fsrc;
            let args: Vec<At> = e.signature.params.iter().map(from_ty).collect();
            let r = self.call_function(&e.name, &args, fspan);
            self.check_entry_result(e, &r);
        }
        if entries.is_empty() || self.all_functions {
            let names: Vec<String> = self.r.functions.items.iter().map(|(k, _)| k.clone()).collect();
            for name in names {
                let f = self.r.functions.get(&name).unwrap();
                let (fsrc, fspan, n) = (f.src, f.span, f.params.len());
                self.src = fsrc;
                self.call_function(&name, &vec![At::unknown(); n], fspan);
            }
        }
    }

    fn check_entry_result(&mut self, e: &EntryDecl, r: &At) {
        let want = &e.signature.result;
        if want.elem == "void" || r.kind == AKind::Any {
            return;
        }
        let ssp = e.span;
        let src = e.src;
        if r.kind == AKind::Void {
            self.diag_in("entry", ssp, format!("entry {} is declared to return {}, but the function returns nothing (its last statement is a call of a void extern)", e.name, want), None, src);
            return;
        }
        if (r.fixed() || is_fixed(&want.elem)) && r.elem.is_some() && r.elem.as_deref() != Some(want.elem.as_str()) {
            self.diag_in("entry", ssp, format!("entry {} is declared to return {}, but the function returns {}", e.name, want, r), Some("a fixed-point result must be exactly the declared type: convert it with .fixed(type;value)"), src);
            return;
        }
        if r.elem_is("float") && want.elem == "int" {
            self.diag_in("entry", ssp, format!("entry {} is declared to return {}, but the function returns {}", e.name, want, r), Some("an int result cannot hold a float: declare float, or floor the result with _"), src);
            return;
        }
        if let Some(s) = &r.shape {
            if *s != want.shape {
                self.diag_in("entry", ssp, format!("entry {} is declared to return {}, but the function returns {}", e.name, want, r), None, src);
            }
        }
    }

    fn finish(&mut self) {
        let names: Vec<String> = self.r.functions.items.iter().map(|(k, _)| k.clone()).collect();
        let mut order: HashMap<String, usize> = HashMap::new();
        for d in self.r.diagnostics.iter_mut() {
            let nm = self.r.sources[d.src].name.clone();
            let k = order.len();
            order.entry(nm).or_insert(k);
            for (name, f) in self.r.functions.items.iter_mut() {
                if f.src == d.src && f.span.0 <= d.span.0 && d.span.0 < f.span.1 {
                    d.function = Some(name.clone());
                    f.in_profile = false;
                    break;
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for name in &names {
                let f = self.r.functions.get(name).unwrap();
                if f.in_profile && f.calls.iter().any(|k| self.r.functions.get(k).map(|g| !g.in_profile).unwrap_or(false)) {
                    self.r.functions.get_mut(name).unwrap().in_profile = false;
                    changed = true;
                }
            }
        }
        let sources: Vec<String> = self.r.sources.iter().map(|s| s.name.clone()).collect();
        self.r.diagnostics.sort_by(|a, b| (order[&sources[a.src]], a.span.0, a.span.1, a.rule).cmp(&(order[&sources[b.src]], b.span.0, b.span.1, b.rule)));
    }

    // -- functions ---------------------------------------------------------------------
    /// The abstract result of calling function `name` with these argument values.
    fn call_function(&mut self, name: &str, args: &[At], sp: Sp) -> At {
        let (arity, fsrc, node) = {
            let f = self.r.functions.get(name).unwrap();
            (f.params.len(), f.src, f.node.clone())
        };
        if args.len() != arity {
            if args.len() < arity {
                self.diag("projection", sp, format!("{} takes {}, got {}: the call is a projection, which is a function value", name, plural(arity, "argument", None), args.len()), None);
            } else {
                self.diag("arity", sp, format!("{} takes {}, got {}", name, plural(arity, "argument", None), args.len()), None);
            }
            return At::any();
        }
        if self.stack.iter().any(|s| s == name) {
            return At::any(); // recursion: reported from the call graph
        }
        for a in args {
            if a.kind == AKind::Void {
                self.diag("void-value", sp, format!("an argument of {} is the result of a void extern", name), None);
                return At::any();
            }
        }
        let args: Vec<At> = args.iter().map(|a| a.plain()).collect();
        let key = (name.to_string(), args.iter().map(|a| a.key()).collect::<Vec<_>>());
        if let Some(m) = self.memo.get(&key) {
            if self.mute > 0 || self.reported.contains(&key) {
                return m.clone();
            }
        }
        let saved = self.src;
        self.src = fsrc;
        self.stack.push(name.to_string());
        let r = self.run_literal(&node, &args, None).plain();
        self.stack.pop();
        self.src = saved;
        self.memo.insert(key.clone(), r.clone());
        if self.mute == 0 {
            self.reported.insert(key);
            let tys: Vec<Option<Ty>> = args.iter().map(|a| a.ty()).collect();
            let f = self.r.functions.get_mut(name).unwrap();
            match f.specializations.iter_mut().find(|(k, _)| *k == tys) {
                Some(e) => e.1 = r.ty(),
                None => f.specializations.push((tys, r.ty())),
            }
        }
        r
    }

    /// Walk the body of a function literal with these argument values. The type of a local is the join of its assignments, so the body
    /// is walked (silently) until the types of its variables stop changing, then once more to report.
    fn run_literal(&mut self, node: &Rc<Node>, args: &[At], parent: Option<usize>) -> At {
        let Kind::FnLit { params, locals, .. } = &node.k else { return At::any() };
        let mut carried: HashMap<String, Option<String>> = HashMap::new();
        self.mute += 1;
        for _ in 0..MAX_PASSES {
            let fr = self.frame(params, locals, args, parent, &carried);
            self.body(node, fr);
            let mut now: HashMap<String, Option<String>> = HashMap::new();
            for (k, v) in &self.frames[fr].vars {
                if let Some(t) = &v.ty {
                    if t.kind == AKind::Num {
                        now.insert(k.clone(), t.elem.clone());
                    }
                }
            }
            if now == carried {
                break;
            }
            carried = now;
        }
        self.mute -= 1;
        let fr = self.frame(params, locals, args, parent, &carried);
        self.body(node, fr)
    }

    fn frame(&mut self, params: &[String], names: &[String], args: &[At], parent: Option<usize>, carried: &HashMap<String, Option<String>>) -> usize {
        let mut vars: HashMap<String, Var> = HashMap::new();
        for (p, a) in params.iter().zip(args.iter()) {
            let mut a = a.clone();
            if carried.get(p).map(|s| s.as_deref() == Some("float")).unwrap_or(false) && a.kind == AKind::Num && a.elem_is("int") {
                a = At::num(Some("float"), a.shape.clone(), Const::Nc);
            }
            vars.insert(p.clone(), Var { ty: Some(a), set: true, hint: None });
        }
        for q in names {
            if !vars.contains_key(q) {
                vars.insert(q.clone(), Var { ty: None, set: false, hint: carried.get(q).cloned().flatten() });
            }
        }
        self.frames.push(Frame { vars, parent });
        self.frames.len() - 1
    }

    fn body(&mut self, node: &Rc<Node>, fr: usize) -> At {
        let Kind::FnLit { stmts, spans, .. } = &node.k else { return At::any() };
        let mut r = At::any();
        for (s, sp) in stmts.iter().zip(spans.iter()) {
            let default = sp.or(node.span).unwrap_or((0, 0));
            r = self.ev(s, default, fr);
        }
        r
    }

    // -- expressions -------------------------------------------------------------------
    /// The abstract value of expression `n`; `sp` is its span when the node carries none.
    fn ev(&mut self, n: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let sp = n.span.unwrap_or(sp);
        match &n.k {
            Kind::Missing => At::any(),
            Kind::QSym(name) => {
                self.diag("symbol", sp, format!("the symbol :{} is not a run-time value", name), None);
                At::any()
            }
            Kind::Sym(name) => self.read(name, sp, fr),
            Kind::Char => {
                self.diag("string", sp, "a character is not a run-time value".to_string(), None);
                At::any()
            }
            Kind::Str(_) => {
                self.diag("string", sp, "a string is not a run-time value".to_string(), None);
                At::any()
            }
            Kind::Int(v) => self.int_literal(*v, sp),
            Kind::Flt(f) => At::num(Some("float"), Some(Vec::new()), Const::Flt(*f)),
            Kind::Arr(l) => self.list_literal(l, sp),
            Kind::Cond(_) => self.cond(n, sp, fr),
            Kind::EDict(_) | Kind::DictLit(_) => {
                self.diag("dict", sp, "a dictionary is not a run-time value".to_string(), None);
                At::any()
            }
            Kind::EList(_) => self.elist(n, sp, fr),
            Kind::Op(g) => {
                self.diag("fn-value", sp, format!("the operator {} is used as a value", g), None);
                At::any()
            }
            Kind::Chain { .. } => self.adverb(n, sp, fr),
            Kind::Monad(g, a) => {
                if g == "-" {
                    if let Kind::Int(v) = &a.k {
                        return self.int_literal(-*v, sp); // -2147483648 is in range; 2147483648 alone is not
                    }
                }
                let asp = n.arg_span(0, sp);
                let v = self.ev(a, asp, fr);
                self.ty_monad(g, &v, sp)
            }
            Kind::OpN(g, k) => {
                self.diag("unsupported-op", sp, format!("the operator {} is applied to {} operands", g, k), None);
                At::any()
            }
            Kind::Dyad(g, l, r) => {
                if g == "::" {
                    return self.assign(n, l, r, sp, fr);
                }
                if g == ":=" {
                    return self.amend(n, l, r, sp, fr);
                }
                if g == "@" {
                    if let Kind::Sym(name) | Kind::QSym(name) = &l.k {
                        if self.is_function(name, fr) {
                            let rsp = n.arg_span(1, sp);
                            self.ev(r, rsp, fr);
                            let lsp = n.arg_span(0, sp);
                            self.diag("fn-value", lsp, format!("{}@x applies the function {} as a value", name, name), Some(&format!("call it by name: {}(...)", name)));
                            return At::any();
                        }
                    }
                }
                let right = self.ev(r, n.arg_span(1, sp), fr); // the right operand is evaluated first
                let left = self.ev(l, n.arg_span(0, sp), fr);
                self.ty_dyad(g, &left, &right, sp)
            }
            Kind::FnLit { .. } => {
                self.diag("fn-value", sp, "a function literal is used as a value".to_string(), None);
                At::any()
            }
            Kind::Call { .. } => self.call(n, sp, fr),
            Kind::Body(_) | Kind::Destructure(_) => {
                self.diag("unsupported-op", sp, "a KGDestructure is not part of the profile".to_string(), None);
                At::any()
            }
        }
    }

    fn is_function(&self, name: &str, fr: usize) -> bool {
        let mut f = Some(fr);
        while let Some(i) = f {
            if self.frames[i].vars.contains_key(name) {
                return false;
            }
            f = self.frames[i].parent;
        }
        self.r.functions.contains(name) || self.r.externs.contains(name)
    }

    fn int_literal(&mut self, v: i128, sp: Sp) -> At {
        let (lo, hi) = int_range(self.int_bits);
        if v < lo || v > hi {
            self.diag("int-range", sp, format!("{} is outside the range of a {}-bit kg_int", v, self.int_bits), None);
            return At::n("int", Vec::new());
        }
        At::num(Some("int"), Some(Vec::new()), Const::Int(v))
    }

    fn list_literal(&mut self, l: &Lit, sp: Sp) -> At {
        match l {
            Lit::Scalar(_) => At::any(),
            Lit::Bad(b, _) => {
                match b {
                    Bad::Sym(name) => self.diag("symbol", sp, format!("the list holds the symbol :{}: a name inside [...] is not evaluated", name), None),
                    Bad::Char => self.diag("string", sp, "the list holds a character".to_string(), None),
                    Bad::Str(s) => {
                        let what = if [";", "(", ")", "{", "}", ":[", ":|"].contains(&s.as_str()) { format!("the string \"{}\"", s) } else { "a string".to_string() };
                        self.diag("string", sp, format!("the list holds {}", what), None);
                    }
                    Bad::Dict => self.diag("dict", sp, "the list holds a dictionary".to_string(), None),
                    Bad::Op(g) => self.diag("list-literal", sp, format!("the list holds the operator {}: inside [...] nothing is evaluated", g), Some("write the numbers only; a negative number is written -1 with no space")),
                    Bad::Ragged => self.diag("list-literal", sp, "the list is ragged: its rows have different lengths, or it mixes numbers and lists".to_string(), None),
                }
                At::any()
            }
            Lit::Num { shape, float, ints, flts } => {
                let size: usize = shape.iter().product();
                if size == 0 {
                    self.diag("array-shape", sp, "an empty list has a dimension of 0".to_string(), None);
                    return At::any();
                }
                if shape.len() > MAX_RANK {
                    self.diag("array-shape", sp, format!("the list has rank {}; the rank is 1 to {}", shape.len(), MAX_RANK), None);
                    return At::any();
                }
                if *float {
                    let c = if size <= 4096 { Const::FArr(flts.clone(), shape.len()) } else { Const::Nc };
                    return At::num(Some("float"), Some(shape.clone()), c);
                }
                let (lo, hi) = int_range(self.int_bits);
                let (mn, mx) = (*ints.iter().min().unwrap(), *ints.iter().max().unwrap());
                if mn < lo || mx > hi {
                    let worst = if mx > hi { mx } else { mn };
                    self.diag("int-range", sp, format!("{} is outside the range of a {}-bit kg_int", worst, self.int_bits), None);
                }
                let c = if size <= 4096 { Const::IArr(ints.clone(), shape.len()) } else { Const::Nc };
                At::num(Some("int"), Some(shape.clone()), c)
            }
        }
    }

    fn read(&mut self, name: &str, sp: Sp, fr: usize) -> At {
        let mut f = Some(fr);
        while let Some(i) = f {
            if let Some(v) = self.frames[i].vars.get(name) {
                if !v.set {
                    self.diag("unset-local", sp, format!("the local {} is read before it is assigned", name), None);
                    return At::any();
                }
                return v.ty.clone().unwrap_or_else(At::any);
            }
            f = self.frames[i].parent;
        }
        if let Some(g) = self.r.globals.get(name) {
            return if g.assigned { g.at.plain() } else { g.at.clone() };
        }
        if self.r.functions.contains(name) {
            self.diag("fn-value", sp, format!("the function {} is used as a value", name), None);
            return At::any();
        }
        if self.r.externs.contains(name) {
            self.diag("fn-value", sp, format!("the extern {} is used as a value", name), None);
            return At::any();
        }
        if name == ".f" {
            self.diag("recursion", sp, ".f is the running function: calling it is recursion".to_string(), None);
            return At::any();
        }
        if name.starts_with('.') {
            self.diag("sys-fn", sp, format!("{} is a system name; it does not exist in a compiled program", name), None);
            return At::any();
        }
        if ARGS.contains(&name) {
            self.diag("undefined-name", sp, format!("{} is not an argument of this function", name), None);
            return At::any();
        }
        if self.broken.contains(name) {
            return At::any();
        }
        self.diag("undefined-name", sp, format!("{} is not defined", name), Some("a run-time function sees its parameters, its locals, globals defined at top level, functions and externs"));
        At::any()
    }

    fn cond(&mut self, n: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let Kind::Cond(items) = &n.k else { return At::any() };
        let c = self.ev(&items[0], n.arg_span(0, sp), fr);
        if c.kind == AKind::Void {
            self.diag("void-value", n.arg_span(0, sp), "the condition is the result of a void extern".to_string(), None);
        } else if c.kind == AKind::Num && c.shape.as_ref().map(|s| !s.is_empty()).unwrap_or(false) {
            self.diag("cond-scalar", n.arg_span(0, sp), format!("the condition is {}; it must be a scalar", c), None);
        }
        let before: HashMap<String, bool> = self.frames[fr].vars.iter().map(|(k, v)| (k.clone(), v.set)).collect();
        let t = self.ev(&items[1], n.arg_span(1, sp), fr);
        let after_t: HashMap<String, bool> = self.frames[fr].vars.iter().map(|(k, v)| (k.clone(), v.set)).collect();
        for (k, v) in self.frames[fr].vars.iter_mut() {
            v.set = before[k];
        }
        let e = self.ev(&items[2], n.arg_span(2, sp), fr);
        for (k, v) in self.frames[fr].vars.iter_mut() {
            v.set = v.set && after_t.get(k).copied().unwrap_or(false);
        }
        if t.kind == AKind::Void && e.kind == AKind::Void {
            return At::void();
        }
        if t.kind == AKind::Any || e.kind == AKind::Any {
            return At::any();
        }
        if t.kind == AKind::Void || e.kind == AKind::Void {
            self.diag("cond-shape", sp, "one branch of the conditional is a value and the other is the result of a void extern".to_string(), None);
            return At::any();
        }
        if t.shape.is_some() && e.shape.is_some() && t.shape != e.shape {
            self.diag("cond-shape", sp, format!("the branches of the conditional are {} and {}; they must have the same shape", t, e), None);
            return At::any();
        }
        let elem = match self.join_elem(&[&t, &e], sp, "the branches of the conditional") {
            Err(()) => return At::any(),
            Ok(x) => x,
        };
        At::num(elem.as_deref(), if t.shape == e.shape { t.shape.clone() } else { None }, Const::Nc)
    }

    fn elist(&mut self, n: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let Kind::EList(nodes) = &n.k else { return At::any() };
        let items: Vec<At> = nodes.iter().enumerate().map(|(i, q)| self.ev(q, n.arg_span(i, sp), fr)).collect();
        if items.is_empty() {
            self.diag("array-shape", sp, "an empty list has a dimension of 0".to_string(), None);
            return At::any();
        }
        let mut shape: Option<Vec<usize>> = if items[0].kind == AKind::Num { items[0].shape.clone() } else { None };
        for (i, it) in items.iter().enumerate() {
            if it.kind == AKind::Void {
                self.diag("void-value", n.arg_span(i, sp), "a list element is the result of a void extern".to_string(), None);
                return At::any();
            }
            if it.kind != AKind::Num {
                return At::any();
            }
            match &it.shape {
                None => shape = None,
                Some(s) => {
                    if let Some(cur) = &shape {
                        if s != cur {
                            self.diag("list-literal", sp, format!("the elements of the list are {} and {}; they must have the same shape", items[0], it), None);
                            return At::any();
                        }
                    }
                }
            }
        }
        let refs: Vec<&At> = items.iter().collect();
        let elem = match self.join_elem(&refs, sp, "the list") {
            Err(()) => return At::any(),
            Ok(x) => x,
        };
        match shape {
            None => At::num(elem.as_deref(), None, Const::Nc),
            Some(s) => {
                let mut full = vec![items.len()];
                full.extend(s);
                self.mk(elem.as_deref(), Some(full), sp)
            }
        }
    }

    /// A number of this type, after checking a known shape against the limits of section 2.
    fn mk(&mut self, elem: Option<&str>, shape: Option<Vec<usize>>, sp: Sp) -> At {
        if let Some(s) = &shape {
            if s.iter().any(|&d| d == 0) {
                self.diag("array-shape", sp, "the result would have a dimension of 0: every dimension is a positive integer".to_string(), None);
                return At::any();
            }
            if s.len() > MAX_RANK {
                self.diag("array-shape", sp, format!("the result would have rank {}; the rank is 1 to {}", s.len(), MAX_RANK), None);
                return At::any();
            }
        }
        At::num(elem, shape, Const::Nc)
    }

    // -- assignment ----------------------------------------------------------------------
    fn assign(&mut self, n: &Rc<Node>, left: &Rc<Node>, right: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let lsp = n.arg_span(0, sp);
        let mut v;
        if right.is_fn_literal() {
            self.diag("fn-value", n.arg_span(1, sp), "a function is assigned to a variable inside a function".to_string(), Some("define the function at top level"));
            v = At::any();
        } else {
            v = self.ev(right, n.arg_span(1, sp), fr);
        }
        if v.kind == AKind::Void {
            self.diag("void-value", n.arg_span(1, sp), "the result of a void extern is assigned".to_string(), None);
            v = At::any();
        }
        let v = v.plain();
        if let Kind::Destructure(names) = &left.k {
            let mut item = At::any();
            if v.kind == AKind::Num {
                match &v.shape {
                    None => item = At::num(v.elem.as_deref(), None, Const::Nc),
                    Some(s) if s.is_empty() => {
                        self.diag("shape-mismatch", sp, format!("cannot destructure {} into {} names: it is not an array", v, names.len()), None);
                    }
                    Some(s) if s[0] != names.len() => {
                        self.diag("shape-mismatch", sp, format!("cannot destructure {} into {}: it has {}", v, plural(names.len(), "name", None), plural(s[0], "item", None)), None);
                    }
                    Some(s) => item = At::num(v.elem.as_deref(), Some(s[1..].to_vec()), Const::Nc),
                }
            }
            for q in names {
                self.store(q, &item, lsp, fr);
            }
            return v;
        }
        match &left.k {
            Kind::Sym(name) => {
                self.store(name, &v, lsp, fr);
            }
            _ => {
                self.diag("strict-syntax", lsp, "the left side of :: must be a name".to_string(), None);
            }
        }
        v
    }

    fn store(&mut self, name: &str, v: &At, sp: Sp, fr: usize) {
        if self.frames[fr].vars.contains_key(name) {
            let (old, hint) = {
                let var = &self.frames[fr].vars[name];
                (var.ty.clone(), var.hint.clone())
            };
            let new_ty: At;
            if v.kind != AKind::Num || old.as_ref().map(|o| o.kind != AKind::Num).unwrap_or(false) {
                new_ty = At::any();
            } else if old.is_none() {
                let elem = if hint.as_deref() == Some("float") && v.elem_is("int") { Some("float".to_string()) } else { v.elem.clone() };
                new_ty = At::num(elem.as_deref(), v.shape.clone(), Const::Nc);
            } else {
                let old = old.unwrap();
                let mut shape = old.shape.clone();
                if old.shape.is_some() && v.shape.is_some() && old.shape != v.shape {
                    self.diag("shape-change", sp, format!("{} was {} and is assigned {}: the shape of a variable is fixed", name, old, v), None);
                } else if v.shape.is_none() {
                    shape = None;
                }
                if (old.fixed() || v.fixed()) && old.elem.is_some() && v.elem.is_some() && old.elem != v.elem {
                    let which = if old.fixed() { old.elem.clone().unwrap() } else { v.elem.clone().unwrap() };
                    self.diag("fixed", sp, format!("{} was {} and is assigned {}: a variable does not change its type", name, old, v), Some(&format!("convert the value: .fixed(\"{}\";value)", which)));
                    let var = self.frames[fr].vars.get_mut(name).unwrap();
                    var.ty = Some(At::any());
                    var.set = true;
                    return;
                }
                let elem = if old.elem_is("float") || v.elem_is("float") { Some("float".to_string()) } else if old.elem == v.elem { old.elem.clone() } else { None };
                new_ty = At::num(elem.as_deref(), shape, Const::Nc);
            }
            let var = self.frames[fr].vars.get_mut(name).unwrap();
            var.ty = Some(new_ty);
            var.set = true;
            return;
        }
        let mut f = self.frames[fr].parent;
        while let Some(i) = f {
            if self.frames[i].vars.contains_key(name) {
                self.diag("closure-assign", sp, format!("the nested function assigns {}, a variable of the function it is written in", name), None);
                let var = self.frames[i].vars.get_mut(name).unwrap();
                var.set = true; // reported once, here; not again where it is read
                var.ty = Some(At::any());
                return;
            }
            f = self.frames[i].parent;
        }
        if let Some(g) = self.r.globals.get(name) {
            let have = g.at.clone();
            let is_param = g.param;
            if is_param {
                self.diag("param", sp, format!("{} is a parameter: a compiled function reads it and cannot assign it", name), Some("a parameter is written by the host between runs; keep the running value in another global"));
                return;
            }
            if v.kind == AKind::Num && have.kind == AKind::Num {
                if have.shape.is_some() && v.shape.is_some() && have.shape != v.shape {
                    self.diag("shape-change", sp, format!("the global {} is {} and is assigned {}: the shape of a global is fixed", name, have.plain(), v), None);
                } else if have.elem.is_some() && v.elem.is_some() && have.elem != v.elem {
                    let hint = if have.elem_is("int") {
                        "state must not change type: floor the value with _, or define the global as a float".to_string()
                    } else if have.elem_is("float") {
                        "state must not change type: write the value as a float (0.0, or multiply by 1.0)".to_string()
                    } else {
                        format!("state must not change type: convert the value with .fixed(\"{}\";value)", have.elem.clone().unwrap())
                    };
                    self.diag("shape-change", sp, format!("the global {} is {} and is assigned {}: the type of a global is fixed", name, have.plain(), v), Some(&hint));
                }
            }
            return;
        }
        if self.r.functions.contains(name) || self.r.externs.contains(name) {
            let what = if self.r.functions.contains(name) { "function" } else { "extern" };
            self.diag("redefinition", sp, format!("{} is a {}; it cannot be assigned", name, what), None);
            return;
        }
        if name.starts_with('.') {
            self.diag("sys-fn", sp, format!("{} is a system name; it cannot be assigned in a compiled program", name), None);
            return;
        }
        self.diag("undefined-name", sp, format!("{} is assigned but is neither a local nor a global", name), Some(&format!("declare it as a local ({{[{}];...}}) or define the global at top level first", name)));
    }

    // -- calls -----------------------------------------------------------------------------
    fn call(&mut self, n: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let Kind::Call { callee, args, .. } = &n.k else { return At::any() };
        let csp = n.op_span.unwrap_or(sp);
        if args.iter().any(|q| q.is_missing()) {
            for (i, q) in args.iter().enumerate() {
                if !q.is_missing() {
                    self.ev(q, n.arg_span(i, sp), fr);
                }
            }
            let who = match &callee.k {
                Kind::Sym(s) | Kind::QSym(s) => s.clone(),
                _ => "the function".to_string(),
            };
            self.diag("projection", sp, format!("the call of {} leaves an argument out: that is a projection, a function value", who), None);
            return At::any();
        }
        let name = match &callee.k {
            Kind::Sym(s) => s.clone(),
            _ => {
                for (i, q) in args.iter().enumerate() {
                    self.ev(q, n.arg_span(i, sp), fr);
                }
                self.diag("fn-value", csp, "a function literal is called in place".to_string(), Some("give it a name at top level and call it by name"));
                return At::any();
            }
        };
        let mut f = Some(fr);
        while let Some(i) = f {
            if self.frames[i].vars.contains_key(&name) {
                for (k, q) in args.iter().enumerate() {
                    self.ev(q, n.arg_span(k, sp), fr);
                }
                self.diag("fn-value", csp, format!("{} is a variable called as a function: functions cannot be passed or stored", name), None);
                return At::any();
            }
            f = self.frames[i].parent;
        }
        if self.r.functions.contains(&name) {
            let ats: Vec<At> = args.iter().enumerate().map(|(i, q)| self.arg(q, n.arg_span(i, sp), fr)).collect();
            return self.call_function(&name, &ats, sp);
        }
        if self.r.externs.contains(&name) {
            return self.call_extern(&name, n, args, sp, fr);
        }
        if name == ".f" {
            self.diag("recursion", csp, ".f is the running function: calling it is recursion".to_string(), None);
            return At::any();
        }
        if name == ".fixed" || name == ".float" {
            return self.conversion(&name, n, args, sp, csp, fr);
        }
        if name.starts_with('.') {
            self.diag("sys-fn", csp, format!("{} is a system function; it does not exist in a compiled program", name), None);
            return At::any();
        }
        for (i, q) in args.iter().enumerate() {
            self.ev(q, n.arg_span(i, sp), fr);
        }
        if self.r.globals.contains(&name) {
            self.diag("fn-value", csp, format!("{} is a global, not a function", name), None);
        } else if !self.broken.contains(&name) {
            self.diag("undefined-name", csp, format!("the function {} is not defined", name), Some("a function provided by the target is declared with .extern"));
        }
        At::any()
    }

    /// `.fixed(type;x)` and `.float(x)`: the conversions between fixed-point numbers and the other types.
    fn conversion(&mut self, name: &str, n: &Rc<Node>, args: &[Rc<Node>], sp: Sp, csp: Sp, fr: usize) -> At {
        let want = if name == ".fixed" { 2 } else { 1 };
        if args.len() != want {
            for (i, q) in args.iter().enumerate() {
                if is_plain_str(q).is_none() {
                    self.ev(q, n.arg_span(i, sp), fr);
                }
            }
            self.diag("fixed", csp, format!("{} takes {}, got {}", name, plural(want, "argument", None), args.len()), None);
            return At::any();
        }
        let mut ty: Option<Ty> = None;
        if name == ".fixed" {
            let tsp = n.arg_span(0, sp);
            match is_plain_str(&args[0]) {
                None => self.diag("fixed", tsp, "the type of .fixed must be a string literal such as \"fixed<32,16>\"".to_string(), None),
                Some(text) => match sig::parse_type(text) {
                    Err(e) => {
                        self.diag("signature", tsp, format!(".fixed type: {}", e.message), None);
                    }
                    Ok(t) => {
                        if !is_fixed(&t.elem) || !t.shape.is_empty() {
                            self.diag("fixed", tsp, format!(".fixed converts to fixed<W,I>, not {}", t), None);
                        } else {
                            ty = Some(t);
                        }
                    }
                },
            }
        }
        let v = self.arg(&args[args.len() - 1], n.arg_span(args.len() - 1, sp), fr);
        if v.kind == AKind::Void {
            self.diag("void-value", sp, format!("the argument of {} is the result of a void extern", name), None);
            return At::any();
        }
        if v.kind != AKind::Num || (name == ".fixed" && ty.is_none()) {
            return At::any();
        }
        if name == ".fixed" {
            return At::num(Some(&ty.unwrap().elem), v.shape.clone(), Const::Nc);
        }
        At::num(Some("float"), v.shape.clone(), Const::Nc)
    }

    fn arg(&mut self, q: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let is_fn = q.is_fn_literal() || matches!(&q.k, Kind::Sym(s) if self.is_function(s, fr));
        if is_fn {
            self.diag("fn-value", sp, "a function is passed as an argument".to_string(), Some("a function may only stand directly as the verb of an adverb"));
            return At::any();
        }
        self.ev(q, sp, fr)
    }

    fn call_extern(&mut self, name: &str, n: &Rc<Node>, args: &[Rc<Node>], sp: Sp, fr: usize) -> At {
        let decl = self.r.externs.get(name).unwrap().clone();
        let sig = &decl.signature;
        if args.len() != sig.arity() {
            for (i, q) in args.iter().enumerate() {
                if is_plain_str(q).is_none() {
                    self.ev(q, n.arg_span(i, sp), fr);
                }
            }
            self.diag("extern-call", sp, format!("{} is declared \"{}\" and takes {}, got {}", decl.name, sig, plural(sig.arity(), "argument", None), args.len()), None);
            return from_ty(&sig.result);
        }
        for (i, (q, ty)) in args.iter().zip(sig.params.iter()).enumerate() {
            let asp = n.arg_span(i, sp);
            if ty.elem == "str" {
                if is_plain_str(q).is_none() {
                    self.ev(q, asp, fr);
                    self.diag("extern-call", asp, format!("argument {} of {} is declared str: it must be a string literal", i + 1, decl.name), None);
                }
                continue;
            }
            let a = self.arg(q, asp, fr);
            self.extern_arg(&decl, i, ty, &a, asp);
        }
        from_ty(&sig.result)
    }

    fn extern_arg(&mut self, decl: &ExternDecl, i: usize, ty: &Ty, a: &At, asp: Sp) {
        let what = format!("argument {} of {}", i + 1, decl.name);
        if ty.elem == "str" {
            self.diag("extern-call", asp, format!("{} is declared str: it must be a string literal", what), None);
            return;
        }
        if a.kind == AKind::Void {
            self.diag("void-value", asp, format!("{} is the result of a void extern", what), None);
            return;
        }
        if a.kind != AKind::Num {
            return;
        }
        if a.shape.is_some() && a.shape.as_ref().unwrap() != &ty.shape {
            self.diag("extern-call", asp, format!("{} is declared {}, got {}", what, ty, a), None);
        } else if a.elem_is("float") && ty.elem == "int" {
            self.diag("extern-call", asp, format!("{} is declared {}, got {}", what, ty, a), Some("floor the value with _ to pass an int"));
        }
    }

    // -- adverbs ------------------------------------------------------------------------------
    /// The verb of an adverb as a function from argument values to a result value, or None (after a diagnostic) if it cannot be one. `need`
    /// is the number of arguments the adverb gives it.
    fn verb(&mut self, v: &Rc<Node>, need: usize, sp: Sp, fr: usize, role: &str) -> Option<Verb> {
        match &v.k {
            Kind::Op(glyph) => {
                if glyph == "::" || glyph == ":=" {
                    self.diag("unsupported-op", sp, format!("{} cannot be the verb of an adverb", glyph), Some("wrap it in a function literal"));
                    return None;
                }
                if need == 1 {
                    Some(Verb::Monad(glyph.clone()))
                } else {
                    Some(Verb::Dyad(glyph.clone()))
                }
            }
            Kind::Sym(name) => {
                let mut f = Some(fr);
                while let Some(i) = f {
                    if self.frames[i].vars.contains_key(name) {
                        self.diag("fn-value", sp, format!("{} is a variable used as the {} of an adverb: functions cannot be passed or stored", name, role), None);
                        return None;
                    }
                    f = self.frames[i].parent;
                }
                if let Some(fnc) = self.r.functions.get(name) {
                    let arity = fnc.params.len();
                    if arity != need {
                        self.diag("arity", sp, format!("{} takes {}; here it is given {}", name, plural(arity, "argument", None), need), None);
                        return None;
                    }
                    return Some(Verb::Func(name.clone()));
                }
                if let Some(decl) = self.r.externs.get(name) {
                    if decl.signature.arity() != need {
                        let s = decl.signature.to_string();
                        self.diag("extern-call", sp, format!("{} is declared \"{}\"; here it is given {}", name, s, plural(need, "argument", None)), None);
                        return None;
                    }
                    return Some(Verb::Extern(name.clone()));
                }
                self.read(name, sp, fr); // reports .f, a system name, an undefined name
                None
            }
            Kind::FnLit { params, .. } => {
                let have = params.len();
                if have != need {
                    self.diag("arity", sp, format!("the function literal takes {}; here it is given {}", plural(have, "argument", None), need), None);
                    return None;
                }
                Some(Verb::Lit(v.clone(), fr))
            }
            Kind::Monad(..) | Kind::Dyad(..) | Kind::OpN(..) | Kind::Call { .. } | Kind::Chain { .. } | Kind::DictLit(_) => {
                self.diag("fn-value", sp, format!("the {} of the adverb is computed: it must be an operator, a function name or a function literal", role), None);
                None
            }
            _ => {
                self.diag("fn-value", sp, format!("the {} of the adverb must be an operator, a function name or a function literal", role), None);
                None
            }
        }
    }

    fn apply(&mut self, v: &Verb, args: &[At], at: Sp) -> At {
        match v {
            Verb::Monad(g) => self.ty_monad(g, &args[0], at),
            Verb::Dyad(g) => self.ty_dyad(g, &args[0], &args[1], at),
            Verb::Func(name) => self.call_function(name, args, at),
            Verb::Extern(name) => {
                let decl = self.r.externs.get(name).unwrap().clone();
                for (i, (ty, a)) in decl.signature.params.iter().zip(args.iter()).enumerate() {
                    self.extern_arg(&decl, i, ty, a, at);
                }
                from_ty(&decl.signature.result)
            }
            Verb::Lit(node, fr) => {
                let plain: Vec<At> = args.iter().map(|a| a.plain()).collect();
                self.run_literal(node, &plain, Some(*fr)).plain()
            }
        }
    }

    fn adverb(&mut self, n: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let Kind::Chain { verb, advs, dyadic, l, r } = &n.k else { return At::any() };
        let dyadic = *dyadic;
        let vsp = n.op_span.unwrap_or(sp);
        if advs.len() != 1 {
            self.operands(n, l, r, dyadic, sp, fr);
            self.diag("unsupported-op", vsp, format!("the adverb chain {} is not in the profile", advs.join("")), Some("apply one adverb at a time: wrap the inner one in a function literal, {+/x}'a"));
            return At::any();
        }
        let glyph = advs[0].clone();
        let row = lookup(&glyph, if dyadic { "adverb2" } else { "adverb1" });
        let row = match row {
            Some(r0) if r0.allowed => r0,
            other => {
                self.operands(n, l, r, dyadic, sp, fr);
                match other {
                    None => self.diag("unsupported-op", vsp, format!("the adverb {} is not in the profile", glyph), None),
                    Some(r0) => {
                        let rule = tables::rule(r0.rule.unwrap_or("unsupported-op")).map(|x| x.id).unwrap_or("unsupported-op");
                        self.diag(rule, vsp, format!("{} ({}): {}", glyph, r0.name, r0.note), None);
                    }
                }
                return At::any();
            }
        };
        let kind = row.id;
        if kind == "while" {
            let b = self.ev(r, n.arg_span(1, sp), fr);
            let pred = self.verb(l, 1, n.arg_span(0, sp), fr, "predicate");
            let f = self.verb(verb, 1, vsp, fr, "verb");
            return self.loop_(kind, f, pred, &b, sp, Some(n.arg_span(0, sp)));
        }
        if kind == "iterate" {
            let b = self.ev(r, n.arg_span(1, sp), fr);
            let count = self.ev(l, n.arg_span(0, sp), fr);
            if count.kind == AKind::Num && (count.shape.as_ref().map(|s| !s.is_empty()).unwrap_or(false) || count.elem_is("float")) {
                self.diag("operand", n.arg_span(0, sp), format!("the count of Iterate is {}; it must be an int scalar", count), None);
            }
            let mut refs = Refs::default();
            self.scan(l, sp, &HashSet::new(), &mut refs);
            let hit: Vec<String> = refs.refs.iter().map(|(k, _)| k.clone()).filter(|k| self.r.params.contains(k) && !self.frame_chain(fr).any(|f| self.frames[f].vars.contains_key(k))).collect();
            if let Some(h) = hit.first() {
                self.diag("param", n.arg_span(0, sp), format!("the count of Iterate is the parameter {}: a loop bound cannot depend on one, since the hardware is built for a fixed number of iterations", h), Some("iterate a fixed maximum and keep the value once the parameter's count is reached"));
            }
            let f = self.verb(verb, 1, vsp, fr, "verb");
            return self.loop_(kind, f, None, &b, sp, None);
        }
        let need = if kind == "each" { 1 } else { 2 };
        let (a, b);
        if dyadic {
            b = self.ev(r, n.arg_span(1, sp), fr);
            a = Some(self.ev(l, n.arg_span(0, sp), fr));
        } else {
            b = self.ev(r, n.arg_span(0, sp), fr);
            a = None;
        }
        let f = self.verb(verb, need, vsp, fr, "verb");
        for v in [a.as_ref(), Some(&b)].into_iter().flatten() {
            if v.kind == AKind::Void {
                self.diag("void-value", sp, format!("an operand of {} ({}) is the result of a void extern", glyph, row.name), None);
                return At::any();
            }
        }
        let Some(f) = f else { return At::any() };
        if b.kind != AKind::Num || a.as_ref().map(|x| x.kind != AKind::Num).unwrap_or(false) {
            return At::any();
        }
        if b.shape.as_ref().map(|s| s.is_empty()).unwrap_or(false) {
            self.diag("operand", sp, format!("{} ({}) needs an array; the operand is {}", glyph, row.name, b), None);
            return At::any();
        }
        let item = At::num(b.elem.as_deref(), b.shape.as_ref().map(|s| s[1..].to_vec()), Const::Nc);
        let mut n_items = b.shape.as_ref().map(|s| s[0]);
        if kind == "each" {
            let r0 = self.apply(&f, &[item], sp);
            return self.stacked(&r0, n_items, sp);
        }
        if kind == "each2" {
            let a = a.unwrap();
            if a.shape.as_ref().map(|s| s.is_empty()).unwrap_or(false) {
                self.diag("operand", sp, format!("{} ({}) needs two arrays; the left operand is {}", glyph, row.name, a), None);
                return At::any();
            }
            if let (Some(sa), Some(ni)) = (&a.shape, n_items) {
                if sa[0] != ni {
                    self.diag("shape-mismatch", sp, format!("{} ({}) pairs the items of {} and {}: the lengths differ", glyph, row.name, a, b), Some("KlongPy stops at the shorter operand; compiled code needs equal lengths"));
                    return At::any();
                }
            }
            if n_items.is_none() {
                if let Some(sa) = &a.shape {
                    n_items = Some(sa[0]);
                }
            }
            let left = At::num(a.elem.as_deref(), a.shape.as_ref().map(|s| s[1..].to_vec()), Const::Nc);
            let r0 = self.apply(&f, &[left, item], sp);
            return self.stacked(&r0, n_items, sp);
        }
        if kind == "each-pair" {
            let r0 = self.apply(&f, &[item.clone(), item], sp);
            if let Some(ni) = n_items {
                if ni < 2 {
                    self.diag("array-shape", sp, format!("{} ({}) of {} has no pairs: the result would be empty", glyph, row.name, b), None);
                    return At::any();
                }
            }
            return self.stacked(&r0, n_items.map(|x| x - 1), sp);
        }
        // over and scan, with or without a seed
        let mut seed = match &a {
            Some(x) => x.plain(),
            None => item.clone(),
        };
        if let Some(x) = &a {
            if item.fixed() && x.elem_is("int") && fixed_holds(item.elem.as_deref().unwrap(), &x.c) == Some(true) {
                seed = At::num(item.elem.as_deref(), x.shape.clone(), Const::Nc); // a constant int seed converts exactly
            }
        }
        let acc = self.accumulate(&f, seed, &item, sp, row);
        if acc.kind != AKind::Num {
            return acc;
        }
        if kind == "over" || kind == "over-seed" {
            return acc;
        }
        if n_items.is_none() || acc.shape.is_none() {
            return At::num(acc.elem.as_deref(), None, Const::Nc);
        }
        let mut full = vec![n_items.unwrap() + if kind == "scan-seed" { 1 } else { 0 }];
        full.extend(acc.shape.clone().unwrap());
        self.mk(acc.elem.as_deref(), Some(full), sp)
    }

    fn frame_chain(&self, fr: usize) -> impl Iterator<Item = usize> + '_ {
        let mut cur = Some(fr);
        std::iter::from_fn(move || {
            let c = cur?;
            cur = self.frames[c].parent;
            Some(c)
        })
    }

    fn operands(&mut self, n: &Rc<Node>, l: &Rc<Node>, r: &Rc<Node>, dyadic: bool, sp: Sp, fr: usize) {
        if dyadic {
            if !r.is_fn_literal() {
                self.ev(r, n.arg_span(1, sp), fr);
            }
            if !l.is_fn_literal() {
                self.ev(l, n.arg_span(0, sp), fr);
            }
        } else {
            self.ev(r, n.arg_span(0, sp), fr);
        }
    }

    fn stacked(&mut self, r: &At, n_items: Option<usize>, sp: Sp) -> At {
        if r.kind == AKind::Void {
            return At::void();
        }
        if r.kind != AKind::Num {
            return At::any();
        }
        match (n_items, &r.shape) {
            (Some(n), Some(s)) => {
                let mut full = vec![n];
                full.extend(s.iter().cloned());
                self.mk(r.elem.as_deref(), Some(full), sp)
            }
            _ => At::num(r.elem.as_deref(), None, Const::Nc),
        }
    }

    /// The fixed point of acc = f(acc, item), allowing one promotion from int to float.
    fn accumulate(&mut self, f: &Verb, mut acc: At, item: &At, sp: Sp, row: &Op) -> At {
        for _ in 0..2 {
            self.mute += 1;
            let r = self.apply(f, &[acc.clone(), item.clone()], sp);
            self.mute -= 1;
            if r.kind != AKind::Num || acc.kind != AKind::Num {
                break;
            }
            if r.shape.is_some() && acc.shape.is_some() && r.shape != acc.shape {
                self.diag("loop-state", sp, format!("{} ({}): the accumulator is {} and the verb returns {}; its shape must not change", row.glyph, row.name, acc, r), None);
                return At::any();
            }
            if r.elem_is("float") && acc.elem_is("int") {
                acc = At::num(Some("float"), acc.shape.clone(), Const::Nc);
                continue;
            }
            break; // (an int result of a float accumulator is converted)
        }
        let r = self.apply(f, &[acc.clone(), item.clone()], sp);
        if r.kind == AKind::Void {
            self.diag("void-value", sp, format!("the verb of {} ({}) returns nothing", row.glyph, row.name), None);
            return At::any();
        }
        if r.kind != AKind::Num || acc.kind != AKind::Num {
            return At::any();
        }
        let elem = match (&acc.elem, &r.elem) {
            (Some(a), Some(b)) => promote(a, b),
            _ => None,
        };
        At::num(elem.as_deref(), if acc.shape.is_some() { acc.shape.clone() } else { r.shape.clone() }, Const::Nc)
    }

    fn loop_(&mut self, kind: &str, f: Option<Verb>, pred: Option<Verb>, state: &At, sp: Sp, psp: Option<Sp>) -> At {
        let name = if kind == "iterate" { "Iterate" } else { "While" };
        if state.kind == AKind::Void {
            self.diag("void-value", sp, format!("the starting value of {} is the result of a void extern", name), None);
            return At::any();
        }
        let state = state.plain();
        if let Some(p) = &pred {
            let psp = psp.unwrap_or(sp);
            let c = self.apply(p, &[state.clone()], psp);
            if c.kind == AKind::Num && c.shape.as_ref().map(|s| !s.is_empty()).unwrap_or(false) {
                self.diag("cond-scalar", psp, format!("the predicate of While returns {}; it must be a scalar", c), None);
            } else if c.kind == AKind::Void {
                self.diag("void-value", psp, "the predicate of While returns nothing".to_string(), None);
            }
        }
        let Some(f) = f else { return At::any() };
        let r = self.apply(&f, &[state.clone()], sp);
        if r.kind == AKind::Void {
            self.diag("void-value", sp, format!("the verb of {} returns nothing", name), None);
            return At::any();
        }
        if r.kind != AKind::Num || state.kind != AKind::Num {
            return At::any();
        }
        if (r.shape.is_some() && state.shape.is_some() && r.shape != state.shape) || (r.elem.is_some() && state.elem.is_some() && r.elem != state.elem) {
            self.diag("loop-state", sp, format!("{}: the state starts as {} and the verb returns {}; its type and shape must not change", name, state, r), None);
            return At::any();
        }
        At::num(state.elem.as_deref(), state.shape.clone(), Const::Nc)
    }

    // -- operators -------------------------------------------------------------------------------
    /// The row if the operator is allowed, else None after the diagnostic.
    fn rejected(&mut self, glyph: &str, valence: &str, sp: Sp) -> Option<&'static Op> {
        let Some(row) = lookup(glyph, valence) else {
            self.diag("unsupported-op", sp, format!("the operator {} is not in the profile", glyph), None);
            return None;
        };
        if !row.allowed {
            let shown = if valence == "monad" { format!("{}a", glyph) } else { format!("a{}b", glyph) };
            let rule = tables::rule(row.rule.unwrap_or("unsupported-op")).map(|r| r.id).unwrap_or("unsupported-op");
            self.diag(rule, sp, format!("{} ({}): {}", shown, row.name, row.note), None);
            return None;
        }
        Some(row)
    }

    /// The element type of values that meet in one place (operands of an operator, branches of a conditional, items of a list).
    /// Fixed-point numbers do not mix with floats or with fixed-point numbers of another type, and an int joins a fixed-point number only
    /// as a constant that fits. Err after a diagnostic.
    fn join_elem(&mut self, ats: &[&At], sp: Sp, what: &str) -> Result<Option<String>, ()> {
        let elems: Vec<&Option<String>> = ats.iter().map(|x| &x.elem).collect();
        let fx: Vec<&String> = elems.iter().filter_map(|e| e.as_ref()).filter(|e| is_fixed(e)).collect();
        if fx.is_empty() {
            let mut out: Option<String> = Some("int".to_string());
            for e in &elems {
                out = if out.is_some() { promote_o(&out, e) } else { None };
            }
            return Ok(out);
        }
        if elems.iter().any(|e| e.is_none()) {
            return Ok(None);
        }
        let target = fx[0].clone();
        for x in ats {
            let xe = x.elem.clone().unwrap();
            if xe == target {
                continue;
            }
            if is_fixed(&xe) {
                self.diag("fixed", sp, format!("{} mixes {} and {}", what, target, xe), Some(&format!("convert one of them: .fixed(\"{}\";value) or .fixed(\"{}\";value)", xe, target)));
                return Err(());
            }
            if xe == "float" {
                self.diag("fixed", sp, format!("{} mixes {} with a float: they do not convert implicitly", what, target), Some(&format!("write the float as a fixed-point constant .fixed(\"{}\";value), or convert the {} with .float(value)", target, target)));
                return Err(());
            }
            match fixed_holds(&target, &x.c) {
                None => {
                    self.diag("fixed", sp, format!("{} mixes {} with the int {}, which is not a constant: an int converts only where it is exact", what, target, x), Some(&format!("convert it explicitly: .fixed(\"{}\";value)", target)));
                    return Err(());
                }
                Some(false) => {
                    self.diag("fixed", sp, format!("{}: the int constant does not fit {} (its integer part has {} bits)", what, target, fixed_spec(&target).unwrap().1), Some(&format!("convert it explicitly: .fixed(\"{}\";value) wraps", target)));
                    return Err(());
                }
                Some(true) => {}
            }
        }
        Ok(Some(target))
    }

    fn fixed_dyad(&mut self, row: &Op, glyph: &str, a: &At, b: &At, shape: Option<Vec<usize>>, sp: Sp) -> At {
        let k = row.id;
        let elem = match self.join_elem(&[a, b], sp, &format!("{} ({})", glyph, row.name)) {
            Err(()) => return At::any(),
            Ok(e) => e,
        };
        if !["add", "sub", "mul", "div", "max", "min", "lt", "gt", "eq"].contains(&k) {
            self.diag("fixed", sp, format!("{} ({}) is not defined on fixed-point numbers", glyph, row.name), Some("the fixed-point operators are + - * % | & < > = (and _ to floor to an int); use .float(x) to leave fixed-point for the others"));
            return At::any();
        }
        if let Some(e) = &elem {
            if k == "div" {
                if fixed_spec(e).unwrap().0 > 32 {
                    self.diag("fixed", sp, format!("% (Divide) of {}: fixed-point division is defined for 32 bits or fewer", e), Some("a 64-bit quotient needs a 128-bit divider: divide narrower values and widen the result with .fixed"));
                    return At::any();
                }
                let zero = match &b.c {
                    Const::Int(v) => *v == 0,
                    Const::Flt(f) => *f == 0.0,
                    Const::IArr(v, _) => v.iter().any(|x| *x == 0),
                    Const::FArr(v, _) => v.iter().any(|x| *x == 0.0),
                    _ => false,
                };
                if zero {
                    self.diag("div-zero", sp, format!("{} ({}) divides a fixed-point number by the constant 0", glyph, row.name), Some("compiled code traps on a fixed-point division by zero"));
                    return At::any();
                }
            }
        }
        if ["lt", "gt", "eq"].contains(&k) {
            return At::num(Some("int"), shape, Const::Nc);
        }
        At::num(elem.as_deref(), shape, Const::Nc)
    }

    fn ty_monad(&mut self, glyph: &str, a: &At, sp: Sp) -> At {
        let Some(row) = self.rejected(glyph, "monad", sp) else { return At::any() };
        if a.kind == AKind::Void {
            self.diag("void-value", sp, format!("the operand of {} ({}) is the result of a void extern", glyph, row.name), None);
            return At::any();
        }
        if a.kind != AKind::Num {
            return At::any();
        }
        let k = row.id;
        if a.fixed() && k == "recip" {
            self.diag("fixed", sp, format!("{} ({}) of {}: a reciprocal is a float", glyph, row.name, a.elem.clone().unwrap()), Some(&format!("divide instead: .fixed(\"{}\";1)%x, or 1%x", a.elem.clone().unwrap())));
            return At::any();
        }
        match k {
            "neg" => {
                let c = match &a.c {
                    Const::Int(v) => Const::Int(-*v),
                    Const::Flt(f) => Const::Flt(-*f),
                    Const::IArr(v, nd) => Const::IArr(v.iter().map(|x| -*x).collect(), *nd),
                    Const::FArr(v, nd) => Const::FArr(v.iter().map(|x| -*x).collect(), *nd),
                    other => other.clone(),
                };
                At::num(a.elem.as_deref(), a.shape.clone(), c)
            }
            "floor" => At::num(Some("int"), a.shape.clone(), if a.is_int_const() { a.c.clone() } else { Const::Nc }),
            "recip" => At::num(Some("float"), a.shape.clone(), Const::Nc),
            "not" => At::num(Some("int"), a.shape.clone(), Const::Nc),
            "size" | "shape" => {
                if a.shape.as_ref().map(|s| s.is_empty()).unwrap_or(false) {
                    self.diag("size-of-atom", sp, format!("{} ({}) of {}: the operand is an atom", glyph, row.name, a), None);
                    return At::any();
                }
                let Some(s) = &a.shape else {
                    return At::num(Some("int"), if k == "size" { Some(Vec::new()) } else { None }, Const::Cu);
                };
                if k == "size" {
                    return At::num(Some("int"), Some(Vec::new()), Const::Int(s[0] as i128));
                }
                At::num(Some("int"), Some(vec![s.len()]), Const::IArr(s.iter().map(|d| *d as i128).collect(), 1))
            }
            "first" => match &a.shape {
                None => At::num(a.elem.as_deref(), None, Const::Nc),
                Some(s) => {
                    if !s.is_empty() {
                        At::num(a.elem.as_deref(), Some(s[1..].to_vec()), Const::Nc)
                    } else {
                        a.plain()
                    }
                }
            },
            "reverse" => At::num(a.elem.as_deref(), a.shape.clone(), Const::Nc),
            "transpose" => match &a.shape {
                None => At::num(a.elem.as_deref(), None, Const::Nc),
                Some(s) => {
                    if s.len() != 2 {
                        self.diag("operand", sp, format!("+ (Transpose) needs a rank-2 array; the operand is {}", a), None);
                        return At::any();
                    }
                    At::num(a.elem.as_deref(), Some(vec![s[1], s[0]]), Const::Nc)
                }
            },
            "enumerate" => {
                let Some(c) = self.count(a, sp, "!n (Enumerate)", "n") else { return At::any() };
                match c {
                    Count::Cu => At::num(Some("int"), None, Const::Nc),
                    Count::Val(c) => {
                        if c <= 0 {
                            self.diag("array-shape", sp, format!("!{} would be empty: n must be positive", c), None);
                            return At::any();
                        }
                        let k = if c <= 4096 { Const::IArr((0..c).collect(), 1) } else { Const::Nc };
                        At::num(Some("int"), Some(vec![c as usize]), k)
                    }
                }
            }
            "list" => {
                let shape = a.shape.as_ref().map(|s| {
                    let mut v = vec![1];
                    v.extend(s.iter().cloned());
                    v
                });
                self.mk(a.elem.as_deref(), shape, sp)
            }
            _ => At::any(),
        }
    }

    /// The constant count of take, drop or enumerate: an int, unknown, or None after a diagnostic.
    fn count(&mut self, a: &At, sp: Sp, what: &str, word: &str) -> Option<Count> {
        if a.shape.as_ref().map(|s| !s.is_empty()).unwrap_or(false) {
            self.diag("operand", sp, format!("the {} of {} is {}; it must be an int scalar", word, what, a), None);
            return None;
        }
        if a.elem_is("float") {
            self.diag("operand", sp, format!("the {} of {} is a float; it must be an int", word, what), None);
            return None;
        }
        match &a.c {
            Const::Nc => {
                self.diag("const-count", sp, format!("the {} of {} is not a compile-time constant: the result length would depend on run-time data", word, what), None);
                None
            }
            Const::Cu => Some(Count::Cu),
            Const::Int(v) => Some(Count::Val(*v)),
            Const::Flt(f) => Some(Count::Val(f.trunc() as i128)),
            Const::IArr(v, _) => Some(Count::Val(v.first().copied().unwrap_or(0))),
            Const::FArr(v, _) => Some(Count::Val(v.first().map(|f| f.trunc() as i128).unwrap_or(0))),
        }
    }

    /// The result shape of an atomic dyad: Some(shape) (None inside: not known), or Err after a diagnostic.
    fn prefix(&mut self, a: &At, b: &At, sp: Sp, row: &Op) -> Result<Option<Vec<usize>>, ()> {
        let (Some(sa), Some(sb)) = (&a.shape, &b.shape) else { return Ok(None) };
        if sa.is_empty() {
            return Ok(Some(sb.clone()));
        }
        if sb.is_empty() {
            return Ok(Some(sa.clone()));
        }
        let (short, long) = if sa.len() <= sb.len() { (sa, sb) } else { (sb, sa) };
        if long[..short.len()] != short[..] {
            self.diag("shape-mismatch", sp, format!("cannot apply {} ({}) to {} and {}", row.glyph, row.name, a, b), None);
            return Err(());
        }
        Ok(Some(long.clone()))
    }

    fn ty_dyad(&mut self, glyph: &str, a: &At, b: &At, sp: Sp) -> At {
        let Some(row) = self.rejected(glyph, "dyad", sp) else { return At::any() };
        if a.kind == AKind::Void || b.kind == AKind::Void {
            self.diag("void-value", sp, format!("an operand of {} ({}) is the result of a void extern", glyph, row.name), None);
            return At::any();
        }
        if a.kind != AKind::Num || b.kind != AKind::Num {
            return At::any();
        }
        let k = row.id;
        if row.shape_rule == Some("prefix") {
            let shape = match self.prefix(a, b, sp, row) {
                Err(()) => return At::any(),
                Ok(s) => s,
            };
            let zero = matches!(b.c, Const::Int(0)) && b.elem_is("int");
            if zero && a.elem_is("int") && (k == "idiv" || k == "rem" || (k == "div" && a.c != Const::Nc)) {
                self.diag("div-zero", sp, format!("{} ({}) divides an int by the constant 0", glyph, row.name), Some("compiled code traps on an integer division by zero"));
                return At::any();
            }
            if a.fixed() || b.fixed() {
                return self.fixed_dyad(row, glyph, a, b, shape, sp);
            }
            let elem: Option<String> = match row.type_rule {
                Some("promote") => promote_o(&a.elem, &b.elem),
                Some("float") => Some("float".to_string()),
                Some("int") => Some("int".to_string()),
                Some("int-only") => {
                    for (side, v) in [("left", a), ("right", b)] {
                        if v.elem_is("float") {
                            self.diag("operand", sp, format!(":% (Integer-Divide) needs int operands; the {} operand is {}", side, v), Some("floor the operand with _ first, or divide with %"));
                            return At::any();
                        }
                    }
                    Some("int".to_string())
                }
                _ => {
                    // power
                    if a.elem_is("int") && b.elem_is("int") && matches!(b.c, Const::Int(v) if v >= 0) {
                        Some("int".to_string())
                    } else if a.elem.is_none() || b.elem.is_none() || (a.elem_is("int") && b.elem_is("int") && b.c == Const::Cu) {
                        None
                    } else {
                        Some("float".to_string())
                    }
                }
            };
            let mut c = Const::Nc;
            if a.c != Const::Nc && b.c != Const::Nc {
                c = Const::Cu;
                if let (Const::Int(x), Const::Int(y)) = (&a.c, &b.c) {
                    let (x, y) = (*x, *y);
                    match k {
                        "add" => c = Const::Int(x + y),
                        "sub" => c = Const::Int(x - y),
                        "mul" => c = Const::Int(x.wrapping_mul(y)),
                        "max" => c = Const::Int(x.max(y)),
                        "min" => c = Const::Int(x.min(y)),
                        "idiv" | "rem" if y != 0 => {
                            let q = x.abs() / y.abs();
                            let q = if (x < 0) == (y < 0) { q } else { -q }; // truncates toward zero, as C does
                            c = Const::Int(if k == "idiv" { q } else { x - q * y });
                        }
                        _ => {}
                    }
                }
            }
            if let Some(s) = &shape {
                if s.len() > MAX_RANK {
                    return self.mk(elem.as_deref(), shape, sp);
                }
            }
            let scalar = shape.as_ref().map(|s| s.is_empty()).unwrap_or(false);
            return At::num(elem.as_deref(), shape, if scalar { c } else { Const::Nc });
        }
        match k {
            "match" => {
                if (a.fixed() || b.fixed()) && self.join_elem(&[a, b], sp, "~ (Match)").is_err() {
                    return At::any();
                }
                if a.shape.is_some() && b.shape.is_some() && a.shape != b.shape {
                    self.diag("shape-mismatch", sp, format!("~ (Match) compares {} with {}: the shapes differ, so the answer is always 0", a, b), Some("the profile matches operands of the same shape only"));
                    return At::any();
                }
                At::num(Some("int"), Some(Vec::new()), Const::Nc)
            }
            "index" => {
                if a.shape.as_ref().map(|s| s.is_empty()).unwrap_or(false) {
                    self.diag("operand", sp, format!("cannot index {}: @ (Index) needs an array on the left", a), None);
                    return At::any();
                }
                if b.elem_is("float") {
                    self.diag("operand", sp, format!("the index is {}; an index must be an int", b), Some("floor it with _"));
                    return At::any();
                }
                if let Some(sa) = &a.shape {
                    if !matches!(b.c, Const::Nc | Const::Cu) {
                        let n = sa[0] as i128;
                        if let Some((mn, mx)) = const_minmax(&b.c) {
                            if mn < -n || mx >= n {
                                let worst = if mx >= n { mx } else { mn };
                                self.diag("index-range", sp, format!("index {} is out of range for {}", worst, a), Some(&format!("valid indices are 0 to {}, or -{} to -1 counting from the end", n - 1, n)));
                                return At::any();
                            }
                        }
                    }
                }
                match (&a.shape, &b.shape) {
                    (Some(sa), Some(sb)) => {
                        let mut full = sb.clone();
                        full.extend(sa[1..].iter().cloned());
                        self.mk(a.elem.as_deref(), Some(full), sp)
                    }
                    _ => At::num(a.elem.as_deref(), None, Const::Nc),
                }
            }
            "take" => {
                let c = self.count(a, sp, "n#a (Take)", "count");
                if let Some(sb) = &b.shape {
                    if sb.len() != 1 {
                        self.diag("take-rank", sp, format!("n#a (Take) from {}: take works on rank-1 arrays only", b), None);
                        return At::any();
                    }
                }
                match c {
                    None => At::any(),
                    Some(Count::Cu) => At::num(b.elem.as_deref(), None, Const::Nc),
                    Some(Count::Val(0)) => {
                        self.diag("array-shape", sp, "0#a would be empty: every dimension is a positive integer".to_string(), None);
                        At::any()
                    }
                    Some(Count::Val(c)) => At::num(b.elem.as_deref(), Some(vec![c.unsigned_abs() as usize]), Const::Nc),
                }
            }
            "drop" => {
                let c = self.count(a, sp, "n_a (Drop)", "count");
                if b.shape.as_ref().map(|s| s.is_empty()).unwrap_or(false) {
                    self.diag("operand", sp, format!("n_a (Drop) from {}: the right operand must be an array", b), None);
                    return At::any();
                }
                match c {
                    None => At::any(),
                    Some(Count::Cu) => At::num(b.elem.as_deref(), None, Const::Nc),
                    Some(Count::Val(c)) => {
                        let Some(sb) = &b.shape else { return At::num(b.elem.as_deref(), None, Const::Nc) };
                        let left = sb[0] as i128 - c.abs();
                        if left <= 0 {
                            self.diag("array-shape", sp, format!("dropping {} from {} leaves nothing: every dimension is a positive integer", c.abs(), b), None);
                            return At::any();
                        }
                        let mut full = vec![left as usize];
                        full.extend(sb[1..].iter().cloned());
                        At::num(b.elem.as_deref(), Some(full), Const::Nc)
                    }
                }
            }
            "join" => {
                let elem = match self.join_elem(&[a, b], sp, ", (Join)") {
                    Err(()) => return At::any(),
                    Ok(e) => e,
                };
                let (Some(sa), Some(sb)) = (&a.shape, &b.shape) else { return At::num(elem.as_deref(), None, Const::Nc) };
                if sa.is_empty() && sb.is_empty() {
                    return At::num(elem.as_deref(), Some(vec![2]), Const::Nc);
                }
                if sa.is_empty() || sb.is_empty() {
                    let arr = if sa.is_empty() { sb } else { sa };
                    if arr.len() != 1 {
                        self.diag("shape-mismatch", sp, format!("joining {} with {} would give a ragged list", a, b), None);
                        return At::any();
                    }
                    return At::num(elem.as_deref(), Some(vec![arr[0] + 1]), Const::Nc);
                }
                if sa.len() != sb.len() || sa[1..] != sb[1..] {
                    let hint = if (sa.len() as i64 - sb.len() as i64).abs() == 1 { Some("to append a row to a matrix write m,,row") } else { None };
                    self.diag("shape-mismatch", sp, format!("joining {} with {} would give a ragged list", a, b), hint);
                    return At::any();
                }
                let mut full = vec![sa[0] + sb[0]];
                full.extend(sa[1..].iter().cloned());
                At::num(elem.as_deref(), Some(full), Const::Nc)
            }
            "rotate" => {
                if a.shape.as_ref().map(|s| !s.is_empty()).unwrap_or(false) || a.elem_is("float") {
                    self.diag("operand", sp, format!("the count of n:+a (Rotate) is {}; it must be an int scalar", a), None);
                    return At::any();
                }
                if let Some(sb) = &b.shape {
                    if sb.len() != 1 {
                        self.diag("operand", sp, format!("n:+a (Rotate) of {}: the profile rotates rank-1 arrays only", b), Some("KlongPy rotates the flattened data of a matrix"));
                        return At::any();
                    }
                }
                At::num(b.elem.as_deref(), b.shape.clone(), Const::Nc)
            }
            "reshape" => {
                if a.elem_is("float") {
                    self.diag("operand", sp, format!("the shape of s:^a (Reshape) is {}; it must hold ints", a), None);
                    return At::any();
                }
                if a.c == Const::Nc {
                    self.diag("const-count", sp, "the shape of s:^a (Reshape) is not a compile-time constant: the result shape would depend on run-time data".to_string(), None);
                    return At::any();
                }
                if a.c == Const::Cu {
                    return At::num(b.elem.as_deref(), None, Const::Nc);
                }
                let (dims, ndim): (Vec<i128>, usize) = match &a.c {
                    Const::Int(v) => (vec![*v], 0),
                    Const::Flt(f) => (vec![f.trunc() as i128], 0),
                    Const::IArr(v, nd) => (v.clone(), *nd),
                    Const::FArr(v, nd) => (v.iter().map(|f| f.trunc() as i128).collect(), *nd),
                    _ => (Vec::new(), 0),
                };
                if ndim > 1 {
                    self.diag("operand", sp, format!("the shape of s:^a (Reshape) is {}; it must be an int or a list of ints", a), None);
                    return At::any();
                }
                if ndim == 0 && b.shape.as_ref().map(|s| s.len() > 1).unwrap_or(false) {
                    self.diag("operand", sp, format!("n:^a (Reshape) of {} with a scalar n: the profile reshapes a matrix by a list of dimensions only", b), Some("write the shape as a list: [n]:^a"));
                    return At::any();
                }
                let shape: Vec<usize> = dims.iter().map(|d| if *d < 0 { 0 } else { *d as usize }).collect();
                // (a negative dimension is not a positive one: `mk` reports it as an empty dimension)
                self.mk(b.elem.as_deref(), Some(shape), sp)
            }
            "amend" => {
                self.diag("amend-form", sp, ":= (Amend) cannot be the verb of an adverb".to_string(), None);
                At::any()
            }
            _ => {
                self.diag("unsupported-op", sp, format!("{} ({}) cannot be used here", glyph, row.name), None);
                At::any()
            }
        }
    }

    fn amend(&mut self, n: &Rc<Node>, target: &Rc<Node>, pair: &Rc<Node>, sp: Sp, fr: usize) -> At {
        let mut psp = n.arg_span(1, sp);
        let (v_node, i_node) = match &pair.k {
            Kind::Dyad(op, v, i) if op == "," => (v.clone(), i.clone()),
            _ => {
                self.ev(pair, psp, fr);
                self.ev(target, n.arg_span(0, sp), fr);
                self.diag("amend-form", psp, "the right side of := (Amend) must be written v,i: one value, then the index or indices".to_string(), None);
                return At::any();
            }
        };
        if let Some(s) = pair.span {
            psp = s;
        }
        let idx = self.ev(&i_node, pair.arg_span(1, psp), fr);
        let val = self.ev(&v_node, pair.arg_span(0, psp), fr);
        let arr = self.ev(target, n.arg_span(0, sp), fr);
        for v in [&idx, &val, &arr] {
            if v.kind == AKind::Void {
                self.diag("void-value", sp, "an operand of := (Amend) is the result of a void extern".to_string(), None);
                return At::any();
            }
        }
        if arr.kind != AKind::Num {
            return At::any();
        }
        let mut ok = true;
        if let Some(s) = &arr.shape {
            if s.len() != 1 {
                self.diag("amend-form", sp, format!(":= (Amend) of {}: the profile amends rank-1 arrays only", arr), None);
                ok = false;
            }
        }
        if val.kind == AKind::Num && val.shape.as_ref().map(|s| !s.is_empty()).unwrap_or(false) {
            self.diag("amend-form", pair.arg_span(0, psp), format!("the value of := (Amend) is {}; it must be one scalar", val), None);
            ok = false;
        }
        if idx.kind == AKind::Num {
            if idx.elem_is("float") {
                self.diag("operand", pair.arg_span(1, psp), format!("the index of := (Amend) is {}; an index must be an int", idx), None);
                ok = false;
            } else if idx.shape.as_ref().map(|s| s.len() > 1).unwrap_or(false) {
                self.diag("amend-form", pair.arg_span(1, psp), format!("the index of := (Amend) is {}; it must be a scalar or a rank-1 list", idx), None);
                ok = false;
            } else if ok && arr.shape.is_some() && !matches!(idx.c, Const::Nc | Const::Cu) {
                let size = arr.shape.as_ref().unwrap()[0] as i128;
                if let Some((mn, mx)) = const_minmax(&idx.c) {
                    if mn < -size || mx >= size {
                        let worst = if mx >= size { mx } else { mn };
                        self.diag("index-range", pair.arg_span(1, psp), format!("index {} is out of range for {}", worst, arr), None);
                        ok = false;
                    }
                }
            }
        }
        if !ok {
            return At::any();
        }
        if val.kind != AKind::Num {
            return At::num(None, arr.shape.clone(), Const::Nc);
        }
        let elem = match self.join_elem(&[&arr, &val], sp, ":= (Amend)") {
            Err(()) => return At::any(),
            Ok(e) => e,
        };
        if let Some(e) = &elem {
            if is_fixed(e) && arr.elem.as_deref() != Some(e.as_str()) {
                self.diag("fixed", sp, format!(":= (Amend) of {} with {}: the array would change its type", arr, val), Some(&format!("convert the value: .fixed(\"{}\";value)", arr.elem.clone().unwrap_or_default())));
                return At::any();
            }
        }
        At::num(elem.as_deref(), arr.shape.clone(), Const::Nc)
    }
}

/// Row `i` of a list literal that holds `n` rows, as the value that is a global when the definition names several (`[a b]::[1 2]`).
fn row_of(l: &Lit, n: usize, i: usize) -> Option<LitVal> {
    match l {
        Lit::Num { shape, float, ints, flts } => {
            if shape.is_empty() || shape[0] != n {
                return None;
            }
            let per: usize = shape[1..].iter().product();
            if shape.len() == 1 {
                return Some(if *float { LitVal::Flt(flts[i]) } else { LitVal::Int(ints[i]) });
            }
            Some(LitVal::Arr(Lit::Num {
                shape: shape[1..].to_vec(),
                float: *float,
                ints: if *float { Vec::new() } else { ints[i * per..(i + 1) * per].to_vec() },
                flts: if *float { flts[i * per..(i + 1) * per].to_vec() } else { Vec::new() },
            }))
        }
        Lit::Bad(_, rows) => {
            if rows.len() != n {
                return None;
            }
            Some(match &rows[i] {
                Lit::Scalar(s) => match s {
                    Scalar::Int(v) => LitVal::Int(*v),
                    Scalar::Flt(f) => LitVal::Flt(*f),
                    Scalar::Sym(n) => LitVal::QSym(n.clone()),
                    Scalar::Char => LitVal::Char,
                    Scalar::Str(_) => LitVal::Str,
                    Scalar::Op(_) => LitVal::Func,
                    Scalar::Dict => LitVal::Dict,
                },
                other => LitVal::Arr(other.clone()),
            })
        }
        Lit::Scalar(_) => None,
    }
}

/// Is a compile-time value acceptable as an initialised global? Its type, or the rule and a one-line reason.
pub fn validate_global(value: &LitVal, int_bits: u32) -> Result<Ty, (&'static str, String)> {
    let (lo, hi) = int_range(int_bits);
    match value {
        LitVal::Fixed(t) => {
            let size: usize = t.shape.iter().product();
            if !t.shape.is_empty() && size == 0 {
                return Err(("global-array", "an empty list cannot be a global value: every dimension is a positive integer".to_string()));
            }
            if t.shape.len() > MAX_RANK {
                return Err(("global-array", format!("an array of rank {} cannot be a global value: the rank is 1 to {}", t.shape.len(), MAX_RANK)));
            }
            Ok(t.clone())
        }
        LitVal::Undefined => Err(("global-value", ":undefined cannot be a global".to_string())),
        LitVal::Int(v) => {
            if *v < lo || *v > hi {
                return Err(("int-range", format!("{} is outside the range of a {}-bit kg_int", v, int_bits)));
            }
            Ok(Ty::new("int", Vec::new()))
        }
        LitVal::Flt(_) => Ok(Ty::new("float", Vec::new())),
        LitVal::Char => Err(("global-value", "a character cannot be a global".to_string())),
        LitVal::QSym(s) => Err(("global-value", format!("the symbol :{} cannot be a global", s))),
        LitVal::Str => Err(("global-value", "a string cannot be a global".to_string())),
        LitVal::Dict => Err(("global-value", "a dictionary cannot be a global".to_string())),
        LitVal::Func => Err(("global-value", "a function cannot be a global".to_string())),
        LitVal::At(_) => Ok(Ty::new("int", Vec::new())),
        LitVal::Arr(Lit::Bad(b, _)) => {
            let what = match b {
                Bad::Sym(s) => Some(format!("the symbol :{}", s)),
                Bad::Char => Some("a character".to_string()),
                Bad::Str(_) => Some("a string".to_string()),
                Bad::Dict => Some("a dictionary".to_string()),
                Bad::Op(_) => Some("a function".to_string()),
                Bad::Ragged => None,
            };
            match what {
                Some(w) => Err(("global-array", format!("a list that holds {} cannot be a global", w))),
                None => Err(("global-array", "a ragged list (rows of different lengths, or numbers mixed with lists) cannot be a global".to_string())),
            }
        }
        LitVal::Arr(Lit::Scalar(s)) => match s {
            Scalar::Int(v) => validate_global(&LitVal::Int(*v), int_bits),
            Scalar::Flt(f) => validate_global(&LitVal::Flt(*f), int_bits),
            Scalar::Sym(n) => validate_global(&LitVal::QSym(n.clone()), int_bits),
            Scalar::Char => validate_global(&LitVal::Char, int_bits),
            Scalar::Str(_) => validate_global(&LitVal::Str, int_bits),
            Scalar::Op(_) => validate_global(&LitVal::Func, int_bits),
            Scalar::Dict => validate_global(&LitVal::Dict, int_bits),
        },
        LitVal::Arr(Lit::Num { shape, float, ints, .. }) => {
            let size: usize = shape.iter().product();
            if size == 0 {
                return Err(("global-array", "an empty list cannot be a global value: every dimension is a positive integer".to_string()));
            }
            if shape.len() > MAX_RANK {
                return Err(("global-array", format!("an array of rank {} cannot be a global value: the rank is 1 to {}", shape.len(), MAX_RANK)));
            }
            if !*float {
                let (mn, mx) = (*ints.iter().min().unwrap(), *ints.iter().max().unwrap());
                if mn < lo || mx > hi {
                    let worst = if mx > hi { mx } else { mn };
                    return Err(("int-range", format!("{} is outside the range of a {}-bit kg_int", worst, int_bits)));
                }
                return Ok(Ty::new("int", shape.clone()));
            }
            Ok(Ty::new("float", shape.clone()))
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Check a program text against the static profile.
pub fn check_source(text: &str, name: &str, opts: &Options) -> Report {
    let base_dir = match &opts.base_dir {
        Some(b) => b.clone(),
        None => {
            if std::fs::metadata(name).map(|m| m.is_file()).unwrap_or(false) {
                klong_lint::recover::py_dirname(&abspath(name))
            } else {
                std::env::current_dir().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default()
            }
        }
    };
    let lib_dirs: Vec<String> = if opts.lib_dirs.is_empty() {
        std::env::var("KLONG_LIB_DIRS").map(|v| v.split(':').filter(|d| !d.is_empty()).map(String::from).collect()).unwrap_or_default()
    } else {
        opts.lib_dirs.clone()
    };
    let mut c = Checker {
        mods: vec![ModCtx { prefix: String::new(), names: HashSet::new(), aliases: HashMap::new(), bad_aliases: HashSet::new(), export: None }],
        modules: HashMap::new(),
        prefixes: HashSet::new(),
        r: Report { statements: Vec::new(), sources: Vec::new(), diagnostics: Vec::new(), functions: Ordered::new(), globals: Ordered::new(), externs: Ordered::new(), entries: Vec::new(), params: Ordered::new() },
        base_dir,
        int_bits: opts.int_bits,
        resolve: opts.resolve,
        all_functions: opts.all_functions,
        src: 0,
        mute: 0,
        seen: HashSet::new(),
        stack: Vec::new(),
        memo: HashMap::new(),
        reported: HashSet::new(),
        loaded: HashSet::new(),
        kinds: HashMap::new(),
        refs: HashMap::new(),
        broken: BTreeSet::new(),
        frames: Vec::new(),
        sess: Session::new(lib_dirs),
        given: opts.globals.clone(),
    };
    c.run(name, text);
    c.r
}

fn abspath(p: &str) -> String {
    if p.starts_with('/') {
        return p.to_string();
    }
    let cwd = std::env::current_dir().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
    klong_lint::recover::py_join(&cwd, p)
}
