//! `klong check --streamable` (`klongpy/static/streamable.py`, docs/hardware.md): which constructs of a program inside the static
//! profile keep a hardware kernel from taking one record per clock cycle.
use crate::check::{Diag, Report};
use crate::emit_c::{hls_work, match_window, HLS_WORK_LIMIT};
use crate::ir::*;
use crate::tables;
use std::collections::{HashMap, HashSet};

pub const TABLE_MAX: usize = 1024;
pub const BLOCK_MAX: usize = HLS_WORK_LIMIT;

fn reachable(entry: &Entry) -> Vec<F> {
    let mut out: Vec<F> = Vec::new();
    let mut todo: Vec<F> = vec![entry.f.clone()];
    let mut seen: HashSet<usize> = HashSet::new();
    while !todo.is_empty() {
        let f = todo.remove(0);
        if !seen.insert(fn_key(&f)) {
            continue;
        }
        out.push(f.clone());
        for n in f.body.walked() {
            for g in n.fnspecs() {
                todo.push(g.clone());
            }
        }
    }
    out
}

fn reads_state(node: &Node, name: &str, memo: &mut HashMap<(usize, String), bool>) -> bool {
    for n in node.walked() {
        if matches!(&n.k, K::GlobalRef { name: g, state: true, .. } if g == name) {
            return true;
        }
        for g in n.fnspecs() {
            let key = (fn_key(g), name.to_string());
            if !memo.contains_key(&key) {
                memo.insert(key.clone(), false);
                let r = reads_state(&g.body, name, memo);
                memo.insert(key.clone(), r);
            }
            if memo[&key] {
                return true;
            }
        }
    }
    false
}

fn slow_operation(node: &Node, memo: &mut HashMap<usize, Option<(String, String)>>) -> Option<(String, String)> {
    for n in node.walked() {
        match &n.k {
            K::Binary { op, a, .. } if matches!(op.as_str(), "add" | "sub" | "mul" | "div" | "rem" | "pow") && a.ty.elem == "float" => {
                return Some((n.class().to_string(), format!("float {}", op)));
            }
            K::Unary { op, .. } if op == "recip" => return Some((n.class().to_string(), "float reciprocal".to_string())),
            K::Binary { op, a, b } if matches!(op.as_str(), "idiv" | "rem" | "div") && a.ty.elem != "float" && b.cval().is_none() => {
                return Some((n.class().to_string(), format!("{} {} by a value that is not a constant", if a.ty.elem != "int" { "fixed-point" } else { "integer" }, op)));
            }
            _ => {}
        }
        for g in n.fnspecs() {
            let k = fn_key(g);
            if !memo.contains_key(&k) {
                memo.insert(k, None);
                let r = slow_operation(&g.body, memo);
                memo.insert(k, r);
            }
            if let Some(r) = memo[&k].clone() {
                return Some(r);
            }
        }
    }
    None
}

/// The diagnostics of the streamable tier, sorted by position.
pub fn check_streamable(program: &Program, report: &Report) -> Vec<Diag> {
    let mut out: Vec<Diag> = Vec::new();
    let mut seen: HashSet<(String, (usize, usize), String)> = HashSet::new();
    let mut add = |rule: &'static str, node: &Node, f: &FnSpec, message: String| {
        let sp = node.span.or(f.span).unwrap_or((0, 0));
        if !seen.insert((rule.to_string(), sp, message.clone())) {
            return;
        }
        let r = tables::rule(rule).unwrap();
        out.push(Diag { rule: r.id, span: sp, src: f.source, message, hint: r.hint.map(String::from), function: None });
    };
    for e in &program.entries {
        let fns = reachable(e);
        let window = match_window(program, e);
        for f in &fns {
            for n in f.body.walked() {
                match &n.k {
                    K::Loop { kind, count, .. } => {
                        if kind == "while" {
                            add("stream-loop", n, f, "a While loop: its number of iterations depends on the data, so the call has no fixed number of cycles".to_string());
                        } else {
                            let c = count.as_ref().unwrap();
                            if c.cval().is_none() || !c.ty.shape.is_empty() || c.effects() {
                                add("stream-loop", n, f, "an Iterate whose count is not a compile-time constant: the number of iterations depends on the data".to_string());
                            }
                        }
                    }
                    K::Index { a, i: idx, .. } | K::Gather { a, idx, .. } | K::Amend { a, idx, .. } => {
                        let size: usize = a.ty.shape.iter().product();
                        if !a.ty.shape.is_empty() && size > TABLE_MAX && idx.cval().is_none() {
                            add("stream-table", n, f, format!("a data-dependent index into an array of {} items ({}): more than {}", size, a.ty, TABLE_MAX));
                        }
                    }
                    K::Assign { target, value } => {
                        if let K::GlobalRef { name, state: true, .. } = &target.k {
                            if reads_state(value, name, &mut HashMap::new()) {
                                if let Some(slow) = slow_operation(value, &mut HashMap::new()) {
                                    add("stream-state", n, f, format!("the state {} is read and rewritten through a {}: a multi-cycle operation inside the loop that carries it from one record to the next", name, slow.1));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let work = hls_work(program, e);
        if work > BLOCK_MAX && window.is_none() {
            add("stream-block", &e.f.body, &e.f, format!("entry {} does about {} element operations per call (more than {}) and is not a sliding window", e.decl.name, work, BLOCK_MAX));
        }
    }
    let _ = report;
    out.sort_by(|a, b| (a.span.0, a.span.1, a.rule).cmp(&(b.span.0, b.span.1, b.rule)));
    out
}
