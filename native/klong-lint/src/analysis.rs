//! The analysis of one source: tokens, frames, the parse (for function metadata and E001) and the definition table that
//! W002, W012 and E002 consult.
use crate::load::{find, Session};
use crate::scan::{scan, Frame, Meta, FK};
use crate::tok::{tokenize, Source, Tok};
use klong_core::error::Source as CoreSource;
use klong_core::lex::K;
use klong_core::parse::parse_next;
use klong_core::sym::{Interner, Warning};
use klong_core::tree::{Node, Tree};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

pub const ARGS: [&str; 3] = ["x", "y", "z"];

/// calls that bring in names the linter cannot see
const UNKNOWN_DEFINERS: [&str; 5] = [".py", ".pyf", ".pya", ".E", ".fpga"];

pub const MAX_DEPTH: usize = 8;

/// An E001 candidate: what the parser said, and where (byte offsets).
#[derive(Clone, Debug)]
pub struct Problem {
    pub span: (usize, usize),
    pub message: String,
    pub hint: Option<String>,
}

pub struct Def {
    pub idx: usize,
    /// index of the first token of the defining statement
    pub start: usize,
    /// the function frame of the right-hand side, if it is a literal
    pub fnf: Option<usize>,
}

/// A module that `.l` bound to a name (`.l("file")` to the file's stem, `.l("file";:alias)` to the alias): what the linter knows of it.
pub struct ModImport {
    pub alias: String,
    /// The token of the `.l` that binds it.
    pub idx: usize,
    /// At the top level of the file (not in a function).
    pub top: bool,
    /// The keys of the module's `export` when the file was found and its export is a literal dictionary.
    pub exports: Option<BTreeSet<String>>,
}

pub struct Analysis {
    pub src: Source,
    pub toks: Vec<Tok>,
    pub trivia: Vec<(usize, usize)>,
    pub frames: Vec<Frame>,
    pub ctx: Vec<usize>,
    pub open_frame: HashMap<usize, usize>,
    pub problems: Vec<Problem>,
    pub base_dir: String,
    pub defs: BTreeMap<String, Vec<Def>>,
    pub fn_assigned: HashSet<String>,
    pub local_names: HashSet<String>,
    /// (index of the token that ends the `.l` statement, the names it brings)
    pub imports: Vec<(usize, Vec<String>)>,
    pub import_anywhere: HashSet<String>,
    pub unknown: bool,
    pub import_limits: HashMap<String, Option<u32>>,
    /// The modules `.l` bound to a name, in order.
    pub mod_imports: Vec<ModImport>,
    /// The keys of this file's own `export`, when it assigns one that is a literal dictionary.
    pub export_keys: Option<BTreeSet<String>>,
    /// Whether the file assigns `export` at its top level: it is a module (`klong_core::module`, the decision the interpreter makes).
    pub has_export: bool,
    /// The statements that call `.extern` and discard the value (span of the statement).
    pub discarded_externs: Vec<(usize, usize)>,
    /// `.l` of a module by `.l(path;[names])` for entries the module does not have: (token of the `.l`, the module's spec, the name, the keys it has).
    pub missing_entries: Vec<(usize, String, String, Vec<String>)>,
    /// `.l("name")` that finds a file beside the linted file (or in KLONGPATH) with the name of a bundled library file: (token of the string, name, path).
    pub shadows: Vec<(usize, String, String)>,
}

fn merge_limit(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        _ => None,
    }
}

impl Analysis {
    pub fn text(&self) -> &str {
        &self.src.text
    }

    /// The Python value of token `i`: its text, a name without its colon, a string's body.
    pub fn val(&self, i: usize) -> &str {
        self.toks[i].val(&self.src.text)
    }

    pub fn is_cop(&self, i: usize, glyph: &str) -> bool {
        i < self.toks.len() && self.toks[i].k == K::Cop && self.val(i) == glyph
    }

    pub fn is_op(&self, i: usize, glyph: &str) -> bool {
        i < self.toks.len() && self.toks[i].k == K::Op && self.val(i) == glyph
    }

    pub fn new(src: Source, base_dir: &str, resolve: bool, seen: &[String], depth: usize, full: bool, sess: &mut Session) -> Analysis {
        let (toks, trivia) = tokenize(&src.text);
        let sc = scan(&toks, &src.text);
        let mut an = Analysis {
            src,
            toks,
            trivia,
            frames: sc.frames,
            ctx: sc.ctx,
            open_frame: sc.open_frame,
            problems: Vec::new(),
            base_dir: base_dir.to_string(),
            defs: BTreeMap::new(),
            fn_assigned: HashSet::new(),
            local_names: HashSet::new(),
            imports: Vec::new(),
            import_anywhere: HashSet::new(),
            unknown: false,
            import_limits: HashMap::new(),
            mod_imports: Vec::new(),
            export_keys: None,
            has_export: false,
            discarded_externs: Vec::new(),
            missing_entries: Vec::new(),
            shadows: Vec::new(),
        };
        an.read_module_facts();
        an.parse(full);
        an.definitions(resolve, seen, depth, sess);
        an
    }

    // -- parse ----------------------------------------------------------------------------------
    fn parse(&mut self, full: bool) {
        let (metas, problems0) = parse_level(&self.src, 0, &self.toks, &self.ctx);
        for f in self.frames.iter_mut() {
            if f.kind == FK::Fn {
                f.meta = metas.get(&self.toks[f.o()].s).copied();
            }
        }
        if full {
            self.problems = problems0;
            let (_, problems1) = parse_level(&self.src, 1, &self.toks, &self.ctx);
            let mut seen: HashSet<usize> = self.problems.iter().map(|p| p.span.0).collect();
            for p in problems1 {
                if seen.insert(p.span.0) {
                    self.problems.push(p);
                }
            }
        }
    }

    // -- statements -------------------------------------------------------------------------------
    /// Index of the token that ends the statement containing token `i`.
    pub fn stmt_end(&self, i: usize) -> usize {
        let f = self.ctx[i];
        let fr = &self.frames[f];
        let n = self.toks.len();
        let top = fr.kind == FK::Top;
        let mut j = i + 1;
        while j < n {
            if self.ctx[j] == f {
                let k = self.toks[j].k;
                if k == K::Semi || (top && k == K::Newline) || k == K::Eof || Some(j) == fr.close {
                    return j;
                }
            } else if fr.close == Some(j) {
                return j;
            }
            j += 1;
        }
        n - 1
    }

    /// Index of the first token of the statement containing token `i`.
    pub fn stmt_start(&self, i: usize) -> usize {
        let f = self.ctx[i];
        let fr = &self.frames[f];
        let top = fr.kind == FK::Top;
        let mut j = i as isize - 1;
        while j >= 0 {
            let ju = j as usize;
            if fr.open == Some(ju) {
                return ju + 1;
            }
            if self.ctx[ju] == f {
                let k = self.toks[ju].k;
                if k == K::Semi || (top && k == K::Newline) {
                    return ju + 1;
                }
            }
            j -= 1;
        }
        0
    }

    // -- definitions --------------------------------------------------------------------------------
    fn add_def(&mut self, nm: &str, idx: usize, start: usize, fnf: Option<usize>) {
        self.defs.entry(nm.to_string()).or_default().push(Def { idx, start, fnf });
    }

    fn definitions(&mut self, resolve: bool, seen: &[String], depth: usize, sess: &mut Session) {
        let n = self.toks.len();
        for f in &self.frames {
            if f.kind == FK::Fn {
                self.local_names.extend(f.locals.iter().cloned());
                if let Some(p) = &f.params {
                    self.local_names.extend(p.iter().cloned());
                }
            }
        }
        for i in 0..n {
            let t = &self.toks[i];
            let f = self.ctx[i];
            if self.frames[f].kind.is_data() {
                continue;
            }
            if t.k != K::Name {
                continue;
            }
            // (a qualified name defines nothing: a.b::v sets an entry)
            if t.is_qualified(&self.src.text) {
                continue;
            }
            let nm = self.val(i).to_string();
            let nxt = if i + 1 < n { Some(&self.toks[i + 1]) } else { None };
            let mut fn_rhs: Option<usize> = None;
            let mut is_def = false;
            if self.is_cop(i + 1, "::") {
                is_def = true;
                if i + 2 < n && self.toks[i + 2].k == K::LBrace {
                    fn_rhs = self.open_frame.get(&(i + 2)).copied();
                }
            } else if let Some(nx) = nxt {
                if nx.k == K::CallP || (nx.k == K::LParen && !nx.ws) {
                    if let Some(&cf) = self.open_frame.get(&(i + 1)) {
                        if self.frames[cf].is_header {
                            is_def = true;
                            if let Some(c) = self.frames[cf].close {
                                fn_rhs = self.open_frame.get(&(c + 2)).copied();
                            }
                        }
                    }
                }
            }
            let call_next = matches!(nxt, Some(nx) if matches!(nx.k, K::LParen | K::CallP) && !nx.ws);
            if is_def {
                if self.frames[f].fnf.is_none() {
                    let (s, _) = (self.stmt_start(i), ());
                    self.add_def(&nm, i, s, fn_rhs);
                } else {
                    self.fn_assigned.insert(nm);
                }
            } else if UNKNOWN_DEFINERS.contains(&nm.as_str()) && call_next {
                self.unknown = true;
            } else if nm == ".bkf" && call_next {
                self.backend_functions(i, f);
            } else if nm == ".l" && call_next {
                self.load(i, f, resolve, seen, depth, sess);
            } else if nm == ".param" && call_next {
                self.param_(i, f);
            }
        }
        // destructuring [a b]::... defines its names
        for fi in 0..self.frames.len() {
            let (kind, is_target, close, fnf, open) = {
                let f = &self.frames[fi];
                (f.kind, f.is_target, f.close, f.fnf, f.o())
            };
            if kind == FK::Data && is_target && close.is_some() {
                let close = close.unwrap();
                if fnf.is_none() {
                    let start = self.stmt_start(open);
                    for j in open + 1..close {
                        if self.toks[j].k == K::Name && self.ctx[j] == fi {
                            let nm = self.val(j).to_string();
                            self.add_def(&nm, j, start, None);
                        }
                    }
                } else {
                    for j in open + 1..close {
                        if self.toks[j].k == K::Name && self.ctx[j] == fi {
                            let nm = self.val(j).to_string();
                            self.fn_assigned.insert(nm);
                        }
                    }
                }
            }
        }
    }

    /// What the parse tree says of this file as a module, by the shared decision (`klong_core::module`): whether it assigns `export`, the
    /// keys when that is a literal dictionary, and the top-level statements that call `.extern` and discard the value.
    fn read_module_facts(&mut self) {
        let text = self.src.text.as_str();
        if !text.contains(klong_core::module::EXPORT) && !text.contains(".extern") {
            return;
        }
        let rec = crate::recover::parse_recovering(&self.src.name, text, 0);
        for st in &rec.statements {
            if self.export_keys.is_none() && !self.has_export {
                if let Some(i) = klong_core::module::interface(&st.tree, &rec.syms, st.node) {
                    self.has_export = true;
                    self.export_keys = i.entries.map(|e| e.into_iter().map(|(k, _)| k).collect());
                }
            }
            if klong_core::module::discarded_extern(&st.tree, &rec.syms, st.node) {
                self.discarded_externs.push(st.span);
            }
        }
    }

    fn backend_functions(&mut self, i: usize, f: usize) {
        // .bkf("exp") and .bkf(["exp";"sin"]) bind those names; any other argument binds unknown names.
        let cf = self.open_frame.get(&(i + 1)).copied();
        let mut names: Vec<String> = Vec::new();
        let mut ok = matches!(cf, Some(c) if self.frames[c].close.is_some());
        if ok {
            let close = self.frames[cf.unwrap()].close.unwrap();
            let mut j = i + 2;
            while j < close {
                let t = &self.toks[j];
                match t.k {
                    K::Str if !t.unt => names.push(self.val(j).to_string()),
                    K::Newline | K::LBrack | K::RBrack | K::Semi => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
                j += 1;
            }
        }
        if !ok {
            self.unknown = true;
            return;
        }
        if self.frames[f].fnf.is_none() {
            let start = self.stmt_start(i);
            for nm in names {
                self.add_def(&nm, i, start, None);
            }
        } else {
            self.import_anywhere.extend(names);
        }
    }

    /// `.param("name";"type")` binds the name (a static-profile program). A first argument that is not a string literal binds an unknown name.
    fn param_(&mut self, i: usize, f: usize) {
        let n = self.toks.len();
        let arg_ok = i + 3 < n && {
            let a = &self.toks[i + 2];
            a.k == K::Str && !a.unt && matches!(self.toks[i + 3].k, K::Semi | K::RParen)
        };
        if !arg_ok {
            self.unknown = true;
            return;
        }
        let nm = self.val(i + 2).to_string();
        if self.frames[f].fnf.is_none() {
            let s = self.stmt_start(i);
            self.add_def(&nm, i, s, None);
        } else {
            self.import_anywhere.insert(nm);
        }
    }

    /// `.l("path")`, `.l("path";:alias)`, `.l("path";["a" "b"])`: what the file brings. A module binds one name (the file's stem, the alias)
    /// or the entries named; a classic file brings its names.
    fn load(&mut self, i: usize, f: usize, resolve: bool, seen: &[String], depth: usize, sess: &mut Session) {
        let n = self.toks.len();
        let arg_ok = i + 3 < n && {
            let a = &self.toks[i + 2];
            a.k == K::Str && !a.unt && matches!(self.toks[i + 3].k, K::RParen | K::Semi)
        };
        if !arg_ok {
            self.unknown = true;
            return;
        }
        let spec = self.val(i + 2).to_string();
        // the second argument, when it is a literal: a symbol, or a list of strings
        enum Second {
            None,
            Alias(String),
            Names(Vec<String>),
        }
        let second = if self.toks[i + 3].k == K::RParen {
            Some(Second::None)
        } else if i + 5 < n && self.toks[i + 4].k == K::QName && self.toks[i + 5].k == K::RParen {
            Some(Second::Alias(self.val(i + 4).to_string()))
        } else if i + 4 < n && self.toks[i + 4].k == K::LBrack {
            let mut j = i + 5;
            let mut names = Vec::new();
            while j < n && self.toks[j].k == K::Str && !self.toks[j].unt {
                names.push(self.val(j).to_string());
                j += 1;
            }
            (j + 1 < n && self.toks[j].k == K::RBrack && self.toks[j + 1].k == K::RParen).then_some(Second::Names(names))
        } else {
            None
        };
        let Some(second) = second else {
            self.unknown = true;
            return;
        };
        let mut info: Option<Rc<Analysis>> = None;
        if resolve {
            if let Some(loc) = find(sess, &spec, &self.base_dir) {
                if loc.shadows {
                    self.shadows.push((i + 2, spec.clone(), loc.path.clone()));
                }
                info = sess.load_info(&loc.path, seen, depth + 1);
            }
        }
        let info = match info {
            Some(a) => a,
            None => {
                self.unknown = true;
                return;
            }
        };
        let top = self.frames[f].fnf.is_none();
        if info.has_export {
            // a module: it brings the names it binds, and nothing else
            let bound: Vec<String> = match &second {
                Second::None => vec![klong_core::module::stem(&spec).to_string()],
                Second::Alias(a) => vec![a.clone()],
                Second::Names(ns) => ns.clone(),
            };
            if let (Second::Names(ns), Some(keys)) = (&second, &info.export_keys) {
                for nm in ns.iter().filter(|nm| !keys.contains(*nm)) {
                    self.missing_entries.push((i, spec.clone(), nm.clone(), keys.iter().cloned().collect()));
                }
            }
            for nm in &bound {
                if !klong_core::module::is_binding_name(nm) {
                    continue;
                }
                if top {
                    let s = self.stmt_start(i);
                    self.add_def(nm, i, s, None);
                } else {
                    self.import_anywhere.insert(nm.clone());
                }
            }
            if !matches!(second, Second::Names(_)) {
                if let Some(a) = bound.first() {
                    self.mod_imports.push(ModImport { alias: a.clone(), idx: i, top, exports: info.export_keys.clone() });
                }
            }
            return;
        }
        if info.unknown {
            self.unknown = true;
        }
        for (nm, lim) in info.limits() {
            let merged = match self.import_limits.get(&nm) {
                Some(old) => merge_limit(*old, lim),
                None => lim,
            };
            self.import_limits.insert(nm, merged);
        }
        let names = info.exported();
        if top {
            let e = self.stmt_end(i);
            self.imports.push((e, names.into_iter().collect()));
        } else {
            self.import_anywhere.extend(names);
        }
    }

    // -- exported view ---------------------------------------------------------------------------------
    /// Names this source defines at top level, with what it loads.
    pub fn exported(&self) -> BTreeSet<String> {
        let mut names: BTreeSet<String> = self.defs.keys().cloned().collect();
        for (_, ns) in &self.imports {
            names.extend(ns.iter().cloned());
        }
        names.extend(self.import_anywhere.iter().cloned());
        names
    }

    /// name -> how many arguments the function bound to it takes (None: not known to be a function).
    pub fn limits(&self) -> BTreeMap<String, Option<u32>> {
        let mut out: BTreeMap<String, Option<u32>> = self.import_limits.iter().map(|(k, v)| (k.clone(), *v)).collect();
        for (nm, ds) in &self.defs {
            let mut lim = Some(0u32);
            for d in ds {
                let x = match d.fnf {
                    Some(fnf) => self.fn_limit(fnf),
                    None => None,
                };
                lim = merge_limit(lim, x);
            }
            if let Some(o) = out.get(nm) {
                lim = merge_limit(lim, *o);
            }
            out.insert(nm.clone(), lim);
        }
        out
    }

    /// The most arguments a function literal takes, or None when the tree did not give its metadata.
    pub fn fn_limit(&self, fnf: usize) -> Option<u32> {
        let f = &self.frames[fnf];
        if let Some(p) = &f.params {
            return Some(p.len() as u32);
        }
        let m = f.meta?;
        match m.params {
            Some(p) => Some(p as u32),
            None => Some(m.arity),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------------------------------------

/// The function literals of a tree: start offset -> metadata.
fn collect_meta(tree: &Tree, out: &mut HashMap<usize, Meta>) {
    for (id, node) in tree.nodes.iter().enumerate() {
        if let Node::Fn { a, has_args, arity, literal, .. } = node {
            if *has_args && !*literal {
                continue;
            }
            let sp = match tree.spans.get(id) {
                Some(&(0, 0)) | None => continue,
                Some(&s) => s,
            };
            let params = match tree.nodes.get(a.0 as usize) {
                Some(Node::Body { params, has_params: true, .. }) => Some(params.len as usize),
                _ => None,
            };
            out.entry(sp.0 as usize).or_insert(Meta { arity: *arity as u32, params });
        }
    }
}

enum Seg {
    Ok { end: usize, warnings: Vec<Warning> },
    Err(Problem),
}

/// What Python's `prog(source, pos)` returns for the program from `pos`: it reads statements until something that is
/// not a separator follows one. Its warnings are sorted, each (span, code) once, and a list or dictionary left open
/// inside another that is left open is one.
fn prog_from(source: &Rc<CoreSource>, pos: usize, strict: u8, syms: &mut Interner, module: &mut Option<String>, metas: &mut HashMap<usize, Meta>) -> Seg {
    let n = source.text.len();
    let w0 = syms.warnings.len();
    // every warning of the segment is kept: a diagnostic lost to the interpreter's cap is a finding the user never sees
    syms.max_warnings = usize::MAX;
    let mut cur = pos;
    let mut end_so_far = pos;
    let seg_end;
    // (a segment that fails gives no statements at all, as in Python: its function literals have no metadata)
    let mut seg_metas: HashMap<usize, Meta> = HashMap::new();
    loop {
        let before = syms.warnings.len();
        let saved = module.clone();
        let mut window = 256usize;
        let mut r = parse_next(source, cur, n, &mut window, None, syms, module, strict, 512, None);
        if syms.warnings.len() > before {
            // a window that was too short may have warned about its own end: read again with the whole text
            syms.warnings.truncate(before);
            *module = saved;
            let mut window = n.max(1);
            r = parse_next(source, cur, n, &mut window, None, syms, module, strict, 512, None);
        }
        match r {
            Err(e) => {
                syms.warnings.truncate(w0);
                let span = e.span.map(|s| (s.0 as usize, s.1 as usize)).unwrap_or((pos, pos));
                let message = e.message.to_string();
                let message = if message.is_empty() { "KlongSyntaxError".to_string() } else { message };
                return Seg::Err(Problem { span, message, hint: e.hint.as_ref().map(|h| h.to_string()) });
            }
            Ok(next) => {
                if next.skipped {
                    cur = next.end;
                    end_so_far = cur;
                    continue;
                }
                match next.parsed {
                    None => {
                        let junk = syms.warnings[before..].iter().any(|w| w.code == "trailing-input" || w.code == "stray-closer");
                        seg_end = if junk { end_so_far } else { n };
                        break;
                    }
                    Some(p) => {
                        collect_meta(&p.tree, &mut seg_metas);
                        if next.last {
                            seg_end = next.stmt_end;
                            break;
                        }
                        cur = next.end;
                        end_so_far = cur;
                    }
                }
            }
        }
    }
    for (k, v) in seg_metas {
        metas.entry(k).or_insert(v);
    }
    let mut ws: Vec<Warning> = syms.warnings.drain(w0..).collect();
    ws.sort_by_key(|w| w.span);
    let mut out: Vec<Warning> = Vec::new();
    let mut open_seen = false;
    for w in ws {
        if out.iter().any(|o| o.span == w.span && o.code == w.code) {
            continue;
        }
        if w.code == "unterminated-list" || w.code == "unterminated-dict" {
            if open_seen {
                continue;
            }
            open_seen = true;
        }
        out.push(w);
    }
    Seg::Ok { end: seg_end, warnings: out }
}

/// Offset just after the first top-level `;` or newline at or after `offset`.
pub fn next_boundary(toks: &[Tok], ctx: &[usize], offset: usize) -> Option<usize> {
    let mut i = match toks.binary_search_by(|t| t.s.cmp(&offset)) {
        Ok(mut k) => {
            // the last token that starts at or before offset (bisect_right - 1)
            while k + 1 < toks.len() && toks[k + 1].s == offset {
                k += 1;
            }
            k
        }
        Err(k) => k.saturating_sub(1),
    };
    if i >= toks.len() {
        i = toks.len() - 1;
    }
    while i < toks.len() {
        let t = &toks[i];
        if t.s >= offset && ctx[i] == 0 && matches!(t.k, K::Semi | K::Newline) {
            return Some(t.e);
        }
        i += 1;
    }
    None
}

/// Parse a source at a strict level, resuming after each problem at the next top-level statement. Returns the
/// function metadata and the problems.
fn parse_level(source: &Source, strict: u8, toks: &[Tok], ctx: &[usize]) -> (HashMap<usize, Meta>, Vec<Problem>) {
    let text = &source.text;
    let n = text.len();
    let core_src = Rc::new(CoreSource { name: source.name.clone(), text: std::borrow::Cow::Owned(text.clone()) });
    let mut syms = Interner::new();
    let mut module: Option<String> = None;
    let mut metas: HashMap<usize, Meta> = HashMap::new();
    let mut problems: Vec<Problem> = Vec::new();
    let mut pos = 0usize;
    let mut guard = 0;
    while pos < n && guard < 10000 {
        guard += 1;
        let stop: Option<usize>;
        match prog_from(&core_src, pos, strict, &mut syms, &mut module, &mut metas) {
            Seg::Err(p) => {
                stop = Some(p.span.0);
                problems.push(p);
            }
            Seg::Ok { end, warnings } => {
                let had = !warnings.is_empty();
                let first = warnings.iter().map(|w| w.span.0 as usize).min();
                for w in warnings {
                    let message = match &w.note {
                        Some(note) => format!("{}; {}", w.message, note),
                        None => w.message.clone(),
                    };
                    problems.push(Problem { span: (w.span.0 as usize, w.span.1 as usize), message, hint: w.hint.clone() });
                }
                if had {
                    if end >= n {
                        break;
                    }
                    stop = Some(first.unwrap_or(end).min(end));
                } else if end >= n {
                    break;
                } else {
                    stop = Some(end);
                }
            }
        }
        let stop = match stop {
            Some(s) => s,
            None => break,
        };
        match next_boundary(toks, ctx, stop) {
            Some(nxt) if nxt > pos => pos = nxt,
            _ => break,
        }
    }
    (metas, problems)
}
