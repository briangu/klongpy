//! The rules. Each works on the tokens and frames of an [`Analysis`] (so it also runs when the parse fails); only E001
//! (the parser's own message) and the arity of function literals (E002) come from the parse.
use crate::analysis::{Analysis, ARGS};
use crate::builtins::builtin_names;
use crate::scan::FK;
use klong_core::lex::K;
use std::collections::{BTreeSet, HashMap, HashSet};

pub struct Found {
    pub code: &'static str,
    pub span: (usize, usize),
    pub message: String,
    pub hint: Option<String>,
}

const ADVERB_OPS: [&str; 5] = ["'", "/", "\\", "\\~", "\\*"];
const ADVERB_COPS: [&str; 5] = [":\\", ":'", ":/", ":~", ":*"];

fn operand_end(k: K) -> bool {
    matches!(k, K::Num | K::Char | K::Str | K::Name | K::QName | K::RParen | K::RBrack | K::RBrace)
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("{} {}", n, word)
    } else {
        format!("{} {}s", n, word)
    }
}

fn is_arg(s: &str) -> bool {
    ARGS.contains(&s)
}

pub struct Linter<'a> {
    an: &'a Analysis,
    define: &'a BTreeSet<String>,
    pub found: Vec<Found>,
    underscored: HashSet<usize>,
}

impl<'a> Linter<'a> {
    pub fn new(an: &'a Analysis, define: &'a BTreeSet<String>) -> Linter<'a> {
        Linter { an, define, found: Vec::new(), underscored: HashSet::new() }
    }

    fn add(&mut self, code: &'static str, span: (usize, usize), message: String, hint: Option<&str>) {
        self.found.push(Found { code, span, message, hint: hint.map(String::from) });
    }

    fn span_of(&self, i: usize, j: Option<usize>) -> (usize, usize) {
        (self.an.toks[i].s, self.an.toks[j.unwrap_or(i)].e)
    }

    fn kind_of(&self, i: usize) -> FK {
        self.an.frames[self.an.ctx[i]].kind
    }

    /// Index of the token before `i` in the same expression: newlines are skipped except at top level.
    fn prev(&self, i: usize) -> isize {
        let mut j = i as isize - 1;
        if self.kind_of(i) != FK::Top {
            while j >= 0 && self.an.toks[j as usize].k == K::Newline {
                j -= 1;
            }
        }
        j
    }

    /// Every rule, in the order of `lint.py`. A rule must never take the linter down: one that panics is reported on
    /// standard error and the others still run.
    pub fn run(&mut self) {
        let rules: [(&str, fn(&mut Linter<'a>)); 17] = [
            ("e001", Linter::e001),
            ("w001", Linter::w001),
            ("w010", Linter::w010),
            ("w011", Linter::w011),
            ("w012_w017", Linter::w012_w017),
            ("w013", Linter::w013),
            ("w014", Linter::w014),
            ("w015", Linter::w015),
            ("w016", Linter::w016),
            ("w018", Linter::w018),
            ("w019", Linter::w019),
            ("w020", Linter::w020),
            ("w024", Linter::w024),
            ("w003", Linter::w003),
            ("w002", Linter::w002),
            ("e002", Linter::e002),
            ("modules", Linter::modules),
        ];
        for (name, rule) in rules {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rule(self)));
            if r.is_err() {
                eprintln!("klong-lint: {} failed on {}", name, self.an.src.name);
            }
        }
    }

    // -- W024: a comparison immediately followed by Group of a scalar threshold -------------
    fn w024(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        for i in 1..toks.len().saturating_sub(3) {
            if !an.is_op(i, ">") && !an.is_op(i, "<") {
                continue;
            }
            if !an.is_op(i + 1, "=") || toks[i + 1].s != toks[i].e || self.kind_of(i).is_data() || !operand_end(toks[i - 1].k) {
                continue;
            }
            let start = i + 2;
            let number = if an.is_op(start, "-") { start + 1 } else { start };
            if toks[number].k != K::Num {
                continue;
            }
            // Group has the whole expression to its right. A list-valued expression such as
            // 1+items is a valid operand even when it starts with a scalar literal.
            let mut end = number + 1;
            while toks[end].k == K::Newline && self.kind_of(i) != FK::Top {
                end += 1;
            }
            if !matches!(toks[end].k, K::Eof | K::Semi | K::Newline | K::RParen | K::RBrack | K::RBrace | K::Elif) {
                continue;
            }
            let comparison = an.val(i);
            let opposite = if comparison == ">" { "<" } else { ">" };
            let threshold = &an.text()[toks[start].s..toks[number].e];
            let left = match toks[i - 1].k {
                K::Name | K::Num | K::Char => &an.text()[toks[i - 1].s..toks[i - 1].e],
                _ => "(left)",
            };
            self.add("W024", self.span_of(i, Some(i + 1)), format!("{}= applies Group (=) to a scalar threshold", comparison),
                     Some(&format!("use ~{}{}{} for an inclusive comparison", left, opposite, threshold)));
        }
    }

    // -- E001 ------------------------------------------------------------------------------------------
    fn e001(&mut self) {
        let ps: Vec<_> = self.an.problems.clone();
        for p in ps {
            self.found.push(Found { code: "E001", span: p.span, message: p.message, hint: p.hint });
        }
    }

    // -- W001 ------------------------------------------------------------------------------------------
    fn w001(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let n = toks.len();
        let mut i = 0;
        while i + 2 < n {
            let (a, b, c) = (&toks[i], &toks[i + 1], &toks[i + 2]);
            if a.k == K::Name && b.k == K::Op && an.val(i + 1) == "_" && !b.ws && c.k == K::Name && !c.ws && !is_arg(an.val(i)) && !is_arg(an.val(i + 2)) {
                let mut j = i + 2;
                while j + 2 < n && toks[j + 1].k == K::Op && an.val(j + 1) == "_" && !toks[j + 1].ws && toks[j + 2].k == K::Name && !toks[j + 2].ws && !is_arg(an.val(j + 2)) {
                    j += 2;
                }
                for q in i..=j {
                    self.underscored.insert(q);
                }
                let name = &an.text()[a.s..toks[j].e];
                let parts = name.split('_').collect::<Vec<_>>().join(" _ ");
                let sp = self.span_of(i, Some(j));
                self.add(
                    "W001",
                    sp,
                    format!("'{}' reads as {}: _ is the Drop operator, names cannot contain underscores", name, parts),
                    Some("rename it (myVar; a dot is not part of a name, it reads an entry of a dictionary), or put spaces around _ if you mean Drop"),
                );
                i = j + 1;
            } else {
                i += 1;
            }
        }
    }

    // -- W010 ------------------------------------------------------------------------------------------
    fn w010(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let n = toks.len();
        if n < 3 {
            return;
        }
        for i in 0..n - 2 {
            if !an.is_op(i, "-") {
                continue;
            }
            if self.kind_of(i).is_data() {
                continue;
            }
            let (num, op) = (&toks[i + 1], &toks[i + 2]);
            let opv = an.val(i + 2);
            if num.k != K::Num || op.k != K::Op || opv == "*" || opv == "%" || ADVERB_OPS.contains(&opv) {
                continue;
            }
            let j = self.prev(i);
            if j >= 0 && operand_end(toks[j as usize].k) {
                continue; // a dyadic minus
            }
            let lit = &an.text()[toks[i].s..num.e];
            let inner = lit[1..].trim();
            self.add(
                "W010",
                (toks[i].s, op.e),
                format!("a leading - negates everything to its right: {}{}... is -({}{}...)", lit, opv, inner, opv),
                Some(&format!("write ({}){}... to negate only the number", lit, opv)),
            );
        }
    }

    // -- W011 / W003 helpers -----------------------------------------------------------------------------
    /// (fn frame, set of 1..3) of the arguments each function mentions outside nested literals.
    fn fn_mentions(&self) -> Vec<(usize, BTreeSet<usize>)> {
        let an = self.an;
        let mut out: Vec<(usize, BTreeSet<usize>)> = Vec::new();
        for (i, t) in an.toks.iter().enumerate() {
            if matches!(t.k, K::Name | K::QName) && is_arg(an.val(i)) {
                let f = &an.frames[an.ctx[i]];
                let fnf = match f.fnf {
                    Some(x) => x,
                    None => continue,
                };
                if f.kind.is_data() || an.frames[fnf].locals.iter().any(|l| l == an.val(i)) {
                    continue;
                }
                let a = ARGS.iter().position(|x| *x == an.val(i)).unwrap() + 1;
                match out.iter_mut().find(|(g, _)| *g == fnf) {
                    Some((_, s)) => {
                        s.insert(a);
                    }
                    None => out.push((fnf, BTreeSet::from([a]))),
                }
            }
        }
        out
    }

    /// name -> [(call frame, argument count)] for every call written name(...).
    fn calls(&self) -> HashMap<&'a str, Vec<(usize, usize)>> {
        let an = self.an;
        let mut out: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
        for (fi, f) in an.frames.iter().enumerate() {
            if f.kind == FK::Call && !f.is_header && f.close.is_some() && f.o() > 0 && an.toks[f.o() - 1].k == K::Name {
                out.entry(an.val(f.o() - 1)).or_default().push((fi, self.count_args(fi)));
            }
        }
        out
    }

    fn w011(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let mut calls: Option<HashMap<&str, Vec<(usize, usize)>>> = None;
        for (fnf, used) in self.fn_mentions() {
            let fr = &an.frames[fnf];
            if matches!(&fr.params, Some(p) if !p.is_empty()) {
                continue;
            }
            let hi = *used.iter().max().unwrap();
            let missing: Vec<usize> = (1..hi).filter(|a| !used.contains(a)).collect();
            if missing.is_empty() {
                continue;
            }
            let mut short: Option<(usize, usize)> = None;
            let fo = fr.o();
            if let Some(close) = fr.close.filter(|&c| c + 1 < toks.len() && matches!(toks[c + 1].k, K::LParen | K::CallP) && !toks[c + 1].ws) {
                if let Some(&cf) = an.open_frame.get(&(close + 1)) {
                    if an.frames[cf].close.is_some() {
                        let c = self.count_args(cf);
                        if c < hi {
                            short = Some((cf, c));
                        }
                    }
                }
            } else if fo >= 2 && an.is_cop(fo - 1, "::") && toks[fo - 2].k == K::Name && an.frames[an.ctx[fo - 2]].fnf.is_none() {
                let name = an.val(fo - 2);
                let ndefs = an.defs.get(name).map(|d| d.len()).unwrap_or(0);
                if ndefs == 1 && !an.local_names.contains(name) && !an.fn_assigned.contains(name) {
                    if calls.is_none() {
                        calls = Some(self.calls());
                    }
                    for &(cf, c) in calls.as_ref().unwrap().get(name).map(|v| v.as_slice()).unwrap_or(&[]) {
                        if c < hi {
                            short = Some((cf, c));
                            break;
                        }
                    }
                }
            }
            let (cf, c) = match short {
                Some(s) => s,
                None => continue,
            };
            let line = an.src.line_col(toks[an.frames[cf].o()].s).0;
            let miss = missing.iter().map(|a| ARGS[a - 1]).collect::<Vec<_>>().join(" or ");
            let sp = self.span_of(fo, None);
            self.add(
                "W011",
                sp,
                format!("function uses {} but not {}: the call at line {} passes {} and gets a projection, not a value", ARGS[hi - 1], miss, line, plural(c, "argument")),
                Some(&format!("its arity is {}; pass {} too, or use the arguments you pass", hi, miss)),
            );
        }
    }

    // -- W003 -----------------------------------------------------------------------------------------
    /// A declared local that is never mentioned. Klong scopes dynamically, so a function can also read its caller's
    /// locals: a local whose name is mentioned as a free name anywhere outside the function is not reported.
    fn w003(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let mut free: Option<HashMap<&str, Vec<usize>>> = None;
        for fi in 0..an.frames.len() {
            let fr = &an.frames[fi];
            if fr.kind != FK::Fn || fr.locals.is_empty() {
                continue;
            }
            if free.is_none() {
                free = Some(self.free_mentions());
            }
            let free = free.as_ref().unwrap();
            let end = fr.close.unwrap_or(toks.len());
            let (dstart, dend) = fr.decl.unwrap();
            let mut used: HashSet<&str> = HashSet::new();
            let mut first: HashMap<&str, usize> = HashMap::new();
            for j in fr.o() + 1..end {
                let t = &toks[j];
                if matches!(t.k, K::Name | K::QName) && !(dstart <= t.s && t.s < dend) {
                    used.insert(an.val(j));
                }
            }
            for j in fr.o() + 1..end {
                let t = &toks[j];
                if t.k == K::Name && dstart <= t.s && t.s < dend {
                    first.entry(an.val(j)).or_insert(j);
                }
            }
            for nm in &fr.locals {
                let nm = nm.as_str();
                if used.contains(nm) || !first.contains_key(nm) {
                    continue;
                }
                if free.get(nm).map(|v| v.iter().any(|&j| !(fr.o() <= j && j < end))).unwrap_or(false) {
                    continue;
                }
                let sp = self.span_of(first[nm], None);
                self.add("W003", sp, format!("local '{}' is declared but never used", nm), Some("remove it from the locals list"));
            }
        }
    }

    /// name -> [token index] of every mention of a name that no enclosing function declares.
    fn free_mentions(&self) -> HashMap<&'a str, Vec<usize>> {
        let an = self.an;
        let mut out: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, t) in an.toks.iter().enumerate() {
            if t.k != K::Name {
                continue;
            }
            let f = &an.frames[an.ctx[i]];
            if f.kind.is_data() {
                continue;
            }
            let nm = an.val(i);
            let mut fnf = f.fnf;
            let mut declared = false;
            while let Some(g) = fnf {
                let fr = &an.frames[g];
                if fr.locals.iter().any(|l| l == nm) || fr.params.as_ref().map(|p| p.iter().any(|l| l == nm)).unwrap_or(false) {
                    declared = true;
                    break;
                }
                fnf = fr.parent.and_then(|p| an.frames[p].fnf);
            }
            if !declared {
                out.entry(nm).or_default().push(i);
            }
        }
        out
    }

    // -- names visible at a token -------------------------------------------------------------------------
    /// Locals and named parameters of the functions around frame `f`.
    fn declared_at(&self, f: usize) -> HashSet<&'a str> {
        let an = self.an;
        let mut names: HashSet<&str> = HashSet::new();
        let mut fnf = an.frames[f].fnf;
        while let Some(g) = fnf {
            let fr = &an.frames[g];
            names.extend(fr.locals.iter().map(|s| s.as_str()));
            if let Some(p) = &fr.params {
                names.extend(p.iter().map(|s| s.as_str()));
            }
            fnf = fr.parent.and_then(|p| an.frames[p].fnf);
        }
        names
    }

    /// name -> index of the first token from which a top-level reference is defined.
    fn first_available(&self) -> HashMap<&'a str, usize> {
        let an = self.an;
        let mut first: HashMap<&str, usize> = HashMap::new();
        for (nm, ds) in &an.defs {
            first.insert(nm.as_str(), ds.iter().map(|d| d.start).min().unwrap());
        }
        for (end, names) in &an.imports {
            for nm in names {
                match first.get(nm.as_str()) {
                    Some(&f) if f <= *end => {}
                    _ => {
                        first.insert(nm.as_str(), *end);
                    }
                }
            }
        }
        first
    }

    // -- W012 and W017 ---------------------------------------------------------------------------------------
    fn w012_w017(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let first = self.first_available();
        let mut anywhere: HashSet<&str> = an.defs.keys().map(|s| s.as_str()).collect();
        anywhere.extend(an.import_anywhere.iter().map(|s| s.as_str()));
        for (_, names) in &an.imports {
            anywhere.extend(names.iter().map(|s| s.as_str()));
        }
        for fi in 0..an.frames.len() {
            let f = &an.frames[fi];
            if f.kind != FK::Data || f.is_decl || f.is_target {
                continue;
            }
            let end = f.close.unwrap_or(toks.len());
            // an entry of a dictionary literal: bare names are keys
            let takes_symbols = matches!(f.parent, Some(p) if an.frames[p].kind == FK::Dict) || self.takes_symbols(fi);
            let mut bad: Vec<usize> = Vec::new();
            let mut declared: Option<HashSet<&str>> = None;
            for j in f.o() + 1..end {
                if an.ctx[j] != fi {
                    continue;
                }
                let t = &toks[j];
                match t.k {
                    K::Name => {
                        let nm = an.val(j);
                        if declared.is_none() {
                            declared = Some(self.declared_at(f.parent.unwrap_or(0)));
                        }
                        if is_arg(nm) || nm == ".f" || takes_symbols {
                            continue;
                        }
                        let ok = declared.as_ref().unwrap().contains(nm)
                            || (if f.fnf.is_none() { first.get(nm).map(|&e| e <= j).unwrap_or(false) } else { anywhere.contains(nm) });
                        if ok {
                            let sp = self.span_of(j, None);
                            self.add("W012", sp, format!("'{}' in a list is the symbol :{}, not the value of {}", nm, nm, nm), Some("use [;a;b] to list values"));
                        }
                    }
                    K::Semi | K::LParen | K::RParen | K::LBrace | K::RBrace | K::Cond | K::Elif | K::CallP | K::Cop => bad.push(j),
                    K::Op => {
                        if an.val(j) == "-" && j + 1 < toks.len() && toks[j + 1].k == K::Num && !toks[j + 1].ws {
                            continue;
                        }
                        if an.val(j) == "_" && j > 0 && toks[j - 1].k == K::Name && !t.ws {
                            continue; // W001 reports my_var
                        }
                        bad.push(j);
                    }
                    _ => {}
                }
            }
            if !bad.is_empty() {
                let j = bad[0];
                let t = &toks[j];
                let glyph = an.val(j);
                let more = if bad.len() > 1 { format!(" (and {})", plural(bad.len() - 1, "more item")) } else { String::new() };
                let msg = if t.k == K::Semi {
                    format!("';' inside a list is the string \";\"{}", more)
                } else if matches!(t.k, K::Op | K::Cop) {
                    format!("operator {} inside a list is an operator object, not a computation{}", glyph, more)
                } else {
                    format!("'{}' inside a list is the string \"{}\"{}", glyph, glyph, more)
                };
                let sp = self.span_of(j, None);
                self.add("W017", sp, msg, Some("use [;a;b] for a list of values; separate data with spaces"));
            }
        }
    }

    /// Whether the data list is where the language wants symbols: the object of `.pyc([obj "method"];...)`, or the
    /// variables of the autograd forms `f:>[a c]`, `[a c]∇f`, `[a c]∂f`.
    fn takes_symbols(&self, fi: usize) -> bool {
        let an = self.an;
        let toks = &an.toks;
        let f = &an.frames[fi];
        if f.o() > 0 {
            let p = f.o() - 1;
            if (toks[p].k == K::Cop && an.val(p) == ":>") || (toks[p].k == K::Op && matches!(an.val(p), "∇" | "∂")) {
                return true;
            }
        }
        if let Some(c) = f.close {
            if c + 1 < toks.len() && toks[c + 1].k == K::Op && matches!(an.val(c + 1), "∇" | "∂") {
                return true;
            }
        }
        if let Some(p) = f.parent {
            let c = &an.frames[p];
            if c.kind == FK::Call && c.o() > 0 && an.val(c.o() - 1) == ".pyc" {
                let mut j = c.o() + 1;
                while toks[j].k == K::Newline {
                    j += 1;
                }
                return j == f.o();
            }
        }
        false
    }

    // -- W013 ------------------------------------------------------------------------------------------------
    fn w013(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        for fi in 0..an.frames.len() {
            let f = &an.frames[fi];
            if f.kind != FK::Cond || f.close.is_none() {
                continue;
            }
            let close = f.close.unwrap();
            let semis: Vec<usize> = (f.o() + 1..close).filter(|&j| an.ctx[j] == fi && toks[j].k == K::Semi).collect();
            if semis.len() < 2 {
                continue;
            }
            let mut a = semis[semis.len() - 1] + 1;
            while toks[a].k == K::Newline {
                a += 1;
            }
            let nested = an.open_frame.get(&a).copied();
            if toks[a].k != K::Cond {
                continue;
            }
            let nested = match nested {
                Some(n) if an.frames[n].close.is_some() => n,
                _ => continue,
            };
            let mut b = an.frames[nested].close.unwrap() + 1;
            while toks[b].k == K::Newline {
                b += 1;
            }
            if b == close {
                let sp = self.span_of(a, None);
                self.add("W013", sp, ":[ in the else branch of :[ can be written as the :| chain".to_string(), Some(":[c1;t1:|c2;t2;e] replaces :[c1;t1;:[c2;t2;e]]"));
            }
        }
    }

    // -- W014 ------------------------------------------------------------------------------------------------
    fn w014(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        for i in 1..toks.len() {
            let t = &toks[i];
            if !t.ws {
                continue;
            }
            let k = t.k;
            let adv = (k == K::Op && ADVERB_OPS.contains(&an.val(i))) || (k == K::Cop && ADVERB_COPS.contains(&an.val(i)));
            if !(adv || matches!(k, K::LParen | K::CallP)) {
                continue;
            }
            if self.kind_of(i).is_data() {
                continue;
            }
            let p = &toks[i - 1];
            if !matches!(p.k, K::Name | K::QName | K::RBrace) {
                continue;
            }
            let call = matches!(k, K::LParen | K::CallP);
            if call && self.is_infix(i - 1) {
                continue; // 1 f (2): f is the infix verb and (2) its right operand
            }
            let what = if p.k != K::RBrace { &an.text()[p.s..p.e] } else { "}" };
            let sp = self.span_of(i, None);
            if call {
                self.add(
                    "W014",
                    sp,
                    format!("whitespace between {} and {}: this is not a call, the parenthesised part is dropped", what, an.val(i)),
                    Some(&format!("write {}(...) with no space", what)),
                );
            } else {
                self.add(
                    "W014",
                    sp,
                    format!("whitespace between {} and the adverb {}: an adverb must follow its verb directly", what, an.val(i)),
                    Some(&format!("write {}{} with no space", what, an.val(i))),
                );
            }
        }
    }

    /// Whether the name or `}` at token `j` stands in the verb position of an infix call (1 f 2).
    fn is_infix(&self, j: usize) -> bool {
        let an = self.an;
        let mut j = j;
        if an.toks[j].k == K::RBrace {
            let f = &an.frames[an.ctx[j]];
            if f.kind != FK::Fn || f.o() < 1 {
                return false;
            }
            j = f.o();
        }
        let q = if self.kind_of(j) != FK::Top { self.prev(j) } else { j as isize - 1 };
        if q < 0 {
            return false;
        }
        operand_end(an.toks[q as usize].k)
    }

    // -- W015 ------------------------------------------------------------------------------------------------
    fn w015(&mut self) {
        let an = self.an;
        let tr = &an.trivia;
        for (i, t) in an.toks.iter().enumerate() {
            if t.k == K::Newline && t.nlc {
                let kind = self.kind_of(i);
                let where_ = match kind {
                    FK::Fn => "a function body",
                    FK::Cond => "a conditional",
                    FK::Call => "an argument list",
                    FK::EList => "an evaluated list",
                    _ => continue,
                };
                // bisect_right(starts, t.s) - 1
                let k = tr.partition_point(|c| c.0 <= t.s) as isize - 1;
                let span = if k >= 0 { tr[k as usize] } else { (t.s, t.e) };
                self.add(
                    "W015",
                    span,
                    format!("newline after a comment acts as ';' inside {}", where_),
                    Some("write the ; yourself (at strict level 1 this newline is an ordinary newline)"),
                );
            }
        }
    }

    // -- W016 / W018 -------------------------------------------------------------------------------------------
    fn w016(&mut self) {
        let an = self.an;
        let text = an.text();
        for t in an.toks.iter() {
            if t.k == K::Num && text.as_bytes()[t.s] == b':' {
                self.add("W016", (t.s, t.e), format!("'{}' is the number {}: the colon is ignored", &text[t.s..t.e], &text[t.s + 1..t.e]), Some("remove the colon"));
            }
        }
    }

    fn w018(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let text = an.text();
        for i in 0..toks.len().saturating_sub(1) {
            let (a, b) = (&toks[i], &toks[i + 1]);
            if a.k == K::Num && b.k == K::Name && !b.ws {
                let s = &text[a.s..b.e];
                self.add(
                    "W018",
                    (a.s, b.e),
                    format!("'{}' is the number {} followed by the name {}", s, &text[a.s..a.e], an.val(i + 1)),
                    Some("add a space if the name is an infix function, or fix the number"),
                );
            }
        }
    }

    // -- W019 ------------------------------------------------------------------------------------------------
    fn w019(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        for i in 0..toks.len().saturating_sub(1) {
            let t = &toks[i];
            if t.k == K::Name && is_arg(an.val(i)) && an.is_cop(i + 1, "::") {
                let f = &an.frames[an.ctx[i]];
                if !f.kind.is_data() && !f.is_header {
                    if let Some(g) = f.fnf {
                        if !an.frames[g].locals.iter().any(|l| l == an.val(i)) {
                            self.add(
                                "W019",
                                (t.s, toks[i + 1].e),
                                format!("assignment to {}, which is an argument of the function", an.val(i)),
                                Some("assign to a declared local instead: {[a];a::...}"),
                            );
                        }
                    }
                }
            }
        }
    }

    // -- W020 ------------------------------------------------------------------------------------------------
    fn w020(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        if toks.len() < 2 {
            return;
        }
        for i in 1..toks.len() - 1 {
            if !(an.is_op(i, ",") && toks[i + 1].k == K::Dict) {
                continue;
            }
            if self.kind_of(i).is_data() {
                continue;
            }
            let j = self.prev(i);
            if j < 0 {
                continue;
            }
            let j = j as usize;
            let p = &toks[j];
            let mut two = false;
            let mut what = String::new();
            if p.k == K::RBrack {
                let lf = &an.frames[an.ctx[j]];
                if lf.kind == FK::Data && lf.close == Some(j) {
                    two = self.count_items(an.ctx[j]) == 2;
                    what = "a two-element list".to_string();
                }
            } else if p.k == K::Name {
                let nm = an.val(j);
                if let Some(ds) = an.defs.get(nm) {
                    if ds.iter().all(|d| self.is_two_list(d.idx)) && !an.fn_assigned.contains(nm) {
                        two = true;
                        what = format!("'{}', a two-element list,", nm);
                    }
                }
            }
            if two {
                self.add(
                    "W020",
                    (toks[i].s, toks[i + 1].e),
                    format!("{} joined to a dictionary with , becomes a dictionary entry", what),
                    Some("to append the dictionary to the list use ,, "),
                );
            }
        }
    }

    fn is_two_list(&self, idx: usize) -> bool {
        let an = self.an;
        // name::[a b]
        let j = idx + 2;
        if j >= an.toks.len() || an.toks[j].k != K::LBrack {
            return false;
        }
        match an.open_frame.get(&j) {
            Some(&f) => an.frames[f].kind == FK::Data && an.frames[f].close.is_some() && self.count_items(f) == 2,
            None => false,
        }
    }

    /// Number of elements of a data list (a nested list is one element).
    fn count_items(&self, fi: usize) -> usize {
        let an = self.an;
        let toks = &an.toks;
        let f = &an.frames[fi];
        let mut n = 0;
        let mut j = f.o() + 1;
        let end = f.close.unwrap_or(0);
        while j < end {
            if an.ctx[j] == fi && toks[j].k != K::Newline {
                n += 1;
                if an.is_op(j, "-") && toks[j + 1].k == K::Num && !toks[j + 1].ws {
                    j += 1; // a negative number is one element
                }
            }
            j += 1;
        }
        n
    }

    // -- W002 ------------------------------------------------------------------------------------------------
    /// (names, params) of the definitions `f(a;b)::...` that E001 rejects: the names defined, and {(token index,
    /// name)} for each use of a parameter in the statement.
    fn failed_named_defs(&self) -> (HashSet<&'a str>, HashSet<(usize, &'a str)>) {
        let an = self.an;
        let toks = &an.toks;
        let starts: HashMap<usize, usize> = toks.iter().enumerate().map(|(i, t)| (t.s, i)).collect();
        let mut names = HashSet::new();
        let mut params = HashSet::new();
        for p in &an.problems {
            if !p.message.starts_with("named-parameter definition needs") {
                continue;
            }
            let mut found = None;
            for d in 0..3usize {
                if p.span.0 >= d {
                    if let Some(&i) = starts.get(&(p.span.0 - d)) {
                        found = Some(i);
                        break;
                    }
                }
            }
            let i = match found {
                Some(i) => i,
                None => continue,
            };
            let j = an.stmt_start(i);
            if !(j + 1 < toks.len() && toks[j].k == K::Name && matches!(toks[j + 1].k, K::LParen | K::CallP)) {
                continue;
            }
            names.insert(an.val(j));
            let close = match matching_close(an, j + 1) {
                Some(c) => c,
                None => continue,
            };
            let ps: HashSet<&str> = (j + 2..close).filter(|&k| toks[k].k == K::Name).map(|k| an.val(k)).collect();
            for k in j..=an.stmt_end(i) {
                if toks[k].k == K::Name && ps.contains(an.val(k)) {
                    params.insert((k, an.val(k)));
                }
            }
        }
        (names, params)
    }

    fn w002(&mut self) {
        let an = self.an;
        if an.unknown {
            return;
        }
        let toks = &an.toks;
        let first = self.first_available();
        let mut anywhere: HashSet<&str> = first.keys().copied().collect();
        anywhere.extend(an.fn_assigned.iter().map(|s| s.as_str()));
        anywhere.extend(an.local_names.iter().map(|s| s.as_str()));
        anywhere.extend(an.import_anywhere.iter().map(|s| s.as_str()));
        let builtins: HashSet<&str> = builtin_names().iter().map(|s| s.as_str()).collect();
        let n = toks.len();
        let (failed_names, failed_params) = self.failed_named_defs();
        for i in 0..n {
            if toks[i].k != K::Name {
                continue;
            }
            let fi = an.ctx[i];
            let f = &an.frames[fi];
            if f.kind.is_data() || (f.kind == FK::Call && f.is_header) {
                continue;
            }
            let nm = an.val(i);
            if is_arg(nm) || builtins.contains(nm) || self.define.contains(nm) || self.underscored.contains(&i) {
                continue;
            }
            // E001 already reports a named-parameter definition whose right side is not a function: its name counts as
            // defined, and its parameters are not reported
            if failed_names.contains(nm) || failed_params.contains(&(i, nm)) {
                continue;
            }
            if an.is_cop(i + 1, "::") && !toks[i].is_qualified(&an.src.text) {
                continue;
            }
            if i + 1 < n && matches!(toks[i + 1].k, K::LParen | K::CallP) && !toks[i + 1].ws {
                if let Some(&cf) = an.open_frame.get(&(i + 1)) {
                    if an.frames[cf].is_header {
                        continue;
                    }
                }
            }
            if f.fnf.is_none() {
                let e = first.get(nm).copied();
                if matches!(e, Some(e) if e <= i) {
                    continue;
                }
                let sp = self.span_of(i, None);
                if let Some(e) = e {
                    let line = an.src.line_col(toks[e.min(n - 1)].s).0;
                    self.add("W002", sp, format!("'{}' is used before its definition (defined at line {})", nm, line), Some(&format!("an undefined name evaluates to its own symbol, :{}", nm)));
                } else {
                    self.add("W002", sp, format!("'{}' is never defined in this file", nm), Some(&format!("an undefined name evaluates to its own symbol, :{}", nm)));
                }
            } else {
                if anywhere.contains(nm) || self.declared_at(fi).contains(nm) {
                    continue;
                }
                let sp = self.span_of(i, None);
                self.add("W002", sp, format!("'{}' is never defined in this file", nm), Some(&format!("an undefined name evaluates to its own symbol, :{}", nm)));
            }
        }
    }

    // -- E003, E004, W021, W022: qualified names and imports ---------------------------------------------------
    /// `a.b` is `a@"b"`, and `.l` binds a module (docs/design-principles.md, principles 4 and 5):
    /// E003 writes an entry of a module (it is read-only), E004 reads or binds one the module does not export (when the file and its
    /// literal `export` are found), W021 `a.b::v` where `a` is never a dictionary (in version 1 it defined a variable named `a.b`),
    /// W022 a module that `.l` bound and the program never uses, W023 a local file that shadows a bundled library file of the same
    /// name, E005 a statement that calls `.extern` and throws the function away.
    fn modules(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let text = &an.src.text;
        let mut aliases: HashMap<&str, &crate::analysis::ModImport> = HashMap::new();
        for m in &an.mod_imports {
            let once = an.defs.get(&m.alias).map_or(0, |d| d.len()) == 1;
            if m.top && once && !an.local_names.contains(&m.alias) && !an.fn_assigned.contains(&m.alias) {
                aliases.insert(m.alias.as_str(), m);
            }
        }
        let mut used: HashSet<&str> = HashSet::new();
        for i in 0..toks.len() {
            let t = &toks[i];
            if t.k != K::Name || an.frames[an.ctx[i]].kind.is_data() {
                continue;
            }
            let full = &text[t.s..t.e];
            if full.starts_with('.') {
                continue;
            }
            let base = an.val(i);
            let qualified = t.is_qualified(text);
            let is_alias = aliases.get(base).copied();
            if let Some(m) = is_alias {
                if m.idx != i {
                    used.insert(base);
                }
            }
            if !qualified {
                continue;
            }
            let key = full.split('.').nth(1).unwrap_or("");
            let write = an.is_cop(i + 1, "::");
            if let Some(m) = is_alias {
                if write {
                    self.add("E003", (t.s, t.e), format!("cannot assign {}: {} holds a module, which is read-only", full, base), Some("copy it with  e:::{};e::e,m  or change the module's file"));
                } else if let Some(ex) = &m.exports {
                    if !ex.contains(key) {
                        let list: Vec<&str> = ex.iter().map(|s| s.as_str()).collect();
                        self.add("E004", (t.s, t.e), format!("{} has no entry \"{}\"", base, key), Some(&format!("the module exports: {}", list.join(" "))));
                    }
                }
            } else if write {
                let a_param = an.local_names.contains(base) || an.fn_assigned.contains(base) || is_arg(base);
                let defs = an.defs.get(base);
                let never_a_dict = match defs {
                    None => !a_param,
                    Some(ds) => !a_param && ds.iter().all(|d| d.idx + 2 < toks.len() && matches!(toks[d.idx + 2].k, K::Num | K::Str | K::Char | K::LBrack | K::LBrace)),
                };
                if never_a_dict {
                    self.add(
                        "W021",
                        (t.s, t.e),
                        format!("{}::... sets the entry \"{}\" of {}, which is not a dictionary here", full, key, base),
                        Some("in Klong 1 this defined a variable named with a dot; write the name without the dot, or make the base a dictionary"),
                    );
                }
            }
        }
        for m in &an.mod_imports {
            if m.top && !used.contains(m.alias.as_str()) && aliases.contains_key(m.alias.as_str()) {
                let t = &toks[m.idx];
                self.add("W022", (t.s, t.e), format!("the module {} is never used", m.alias), Some("remove the line, or use a qualified name such as module.name"));
            }
        }
        for (idx, spec, name, have) in &an.missing_entries {
            let t = &toks[*idx];
            self.add("E004", (t.s, t.e), format!("{} has no entry \"{}\"", spec, name), Some(&format!("the module exports: {}", have.join(" "))));
        }
        for (idx, name, path) in &an.shadows {
            let t = &toks[*idx];
            self.add("W023", (t.s, t.e), format!("\"{}\" finds {}, which shadows the bundled library file of the same name", name, path), Some(&format!("rename the file, or write \"./{}\" to say that the local file is meant", name)));
        }
        for &sp in &an.discarded_externs {
            self.add("E005", sp, ".extern returns the host function as a value, and this statement throws it away".to_string(), Some("assign it, in a module file that exports it:  adc::.extern(\"read_adc\";\"int -> float\")  and  export:::{;\"adc\";adc}"));
        }
    }

    // -- E002 ------------------------------------------------------------------------------------------------
    fn e002(&mut self) {
        let an = self.an;
        let toks = &an.toks;
        let limits = an.limits();
        for fi in 0..an.frames.len() {
            let f = &an.frames[fi];
            if f.kind != FK::Call || f.is_header || f.close.is_none() || f.o() < 1 {
                continue;
            }
            let ci = f.o() - 1;
            let c = &toks[ci];
            let mut limit: Option<u32> = None;
            let name: String;
            if c.k == K::Name {
                let nm = an.val(ci);
                name = nm.to_string();
                if an.local_names.contains(nm) || an.fn_assigned.contains(nm) || is_arg(nm) {
                    continue;
                }
                if an.defs.contains_key(nm) || an.import_limits.contains_key(nm) {
                    limit = limits.get(nm).copied().flatten();
                }
            } else if c.k == K::RBrace {
                let g = an.ctx[ci];
                if an.frames[g].kind == FK::Fn && an.frames[g].close == Some(ci) {
                    limit = an.fn_limit(g);
                }
                name = "the function".to_string();
            } else {
                continue;
            }
            let limit = match limit {
                Some(l) => l as usize,
                None => continue,
            };
            let count = self.count_args(fi);
            if count > limit {
                let a = if c.k == K::Name { ci } else { an.frames[an.ctx[ci]].o() };
                self.add(
                    "E002",
                    (toks[a].s, toks[f.close.unwrap()].e),
                    format!("{} takes {} but is called with {}: the extra arguments are dropped", name, plural(limit, "argument"), count),
                    Some("an arity error at strict level 2"),
                );
            }
        }
    }

    fn count_args(&self, fi: usize) -> usize {
        let an = self.an;
        let toks = &an.toks;
        let f = &an.frames[fi];
        let mut semis = 0;
        let mut other = false;
        for j in f.o() + 1..f.close.unwrap_or(0) {
            if an.ctx[j] == fi {
                if toks[j].k == K::Semi {
                    semis += 1;
                } else if toks[j].k != K::Newline {
                    other = true;
                }
            } else {
                other = true;
            }
        }
        if semis == 0 && !other {
            0
        } else {
            semis + 1
        }
    }
}

/// Index of the token that closes the bracket opened at index `j`.
fn matching_close(an: &Analysis, j: usize) -> Option<usize> {
    let mut depth = 0i32;
    for i in j..an.toks.len() {
        match an.toks[i].k {
            K::LParen | K::CallP | K::LBrack | K::LBrace | K::Cond | K::Dict => depth += 1,
            K::RParen | K::RBrack | K::RBrace => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}
