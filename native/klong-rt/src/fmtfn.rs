//! `.fmt(template; args)` and `.pf` (KlongPy `_fmt_parse`, `_fmt_render`).
use alloc::string::String;
use alloc::vec::Vec;

use klong_core::error::Result;
use klong_core::fmt::format_value;
use klong_core::value::{try_push_char, try_push_str, Value};
use klong_core::{Error, Flt, Int, Interp};

enum Part<'a> {
    Text(&'a str),
    /// One literal brace (`{{` or `}}`).
    Brace(char),
    Hole(Option<usize>, Option<&'a str>),
}

/// Python's `repr` of a string: single quotes, unless the text has a single quote and no double quote.
fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = core::fmt::Write::write_fmt(&mut out, format_args!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

fn bad(template: &str, what: String) -> Error {
    Error::range(klong_core::msg!(".fmt: {} in template {}", what, py_repr(template)))
}

/// What Python calls the type of a value (KlongPy's message for a template that is not a string).
fn py_type(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        #[cfg(feature = "bigint")]
        Value::Big(_) => "int",
        Value::Flt(_) => "float",
        Value::Char(_) => "KGChar",
        Value::Sym(_) => "KGSym",
        Value::Str(_) => "str",
        Value::Dict(_) => "KGDict",
        Value::Ints(_) | Value::Flts(_) | Value::List(_) => "ndarray",
        Value::Fn(_) | Value::Prim(_) => "KGFn",
        _ => "object",
    }
}

/// `-?digits(.digits)?`
fn spec_ok(s: &str) -> bool {
    let s = s.strip_prefix('-').unwrap_or(s);
    let (a, b) = match s.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    let digits = |t: &str| !t.is_empty() && t.bytes().all(|c| c.is_ascii_digit());
    digits(a) && b.map(digits).unwrap_or(true)
}

fn parse(t: &str) -> Result<Vec<Part<'_>>> {
    let mut parts = Vec::new();
    fn push<'a>(p: Part<'a>, parts: &mut Vec<Part<'a>>) -> Result<()> {
        parts.try_reserve(1).map_err(|_| Error::memory())?;
        parts.push(p);
        Ok(())
    }
    let b = t.as_bytes();
    let mut i = 0;
    let mut lit = 0;
    while i < b.len() {
        match b[i] {
            b'{' => {
                if lit < i {
                    push(Part::Text(&t[lit..i]), &mut parts)?;
                }
                if b.get(i + 1) == Some(&b'{') {
                    push(Part::Brace('{'), &mut parts)?;
                    i += 2;
                    lit = i;
                    continue;
                }
                let j = match t[i..].find('}') {
                    Some(j) => i + j,
                    None => return Err(bad(t, String::from("unmatched '{'"))),
                };
                let body = &t[i + 1..j];
                let (idx, spec) = match body.split_once(':') {
                    Some((a, s)) => (a, Some(s)),
                    None => (body, None),
                };
                if !idx.is_empty() && !idx.bytes().all(|c| c.is_ascii_digit()) {
                    return Err(bad(t, klong_core::msg!("bad placeholder '{{{}}}'", body)));
                }
                if let Some(s) = spec {
                    let sb = s.strip_prefix('-').unwrap_or(s).as_bytes();
                    if sb.len() >= 2 && sb[0] == b'0' && sb[1].is_ascii_digit() {
                        return Err(bad(t, klong_core::msg!("bad format '{}' in placeholder '{{{}}}': zero padding is not supported, a width pads with spaces", s, body)));
                    }
                    if !spec_ok(s) {
                        return Err(bad(t, klong_core::msg!("bad format '{}' in placeholder '{{{}}}'", s, body)));
                    }
                }
                let idx = if idx.is_empty() { None } else { Some(idx.parse::<usize>().map_err(|_| bad(t, klong_core::msg!("bad placeholder '{{{}}}'", body)))?) };
                push(Part::Hole(idx, spec), &mut parts)?;
                i = j + 1;
                lit = i;
            }
            b'}' => {
                if b.get(i + 1) != Some(&b'}') {
                    return Err(bad(t, String::from("unmatched '}'")));
                }
                if lit < i {
                    push(Part::Text(&t[lit..i]), &mut parts)?;
                }
                push(Part::Brace('}'), &mut parts)?;
                i += 2;
                lit = i;
            }
            _ => i += 1,
        }
    }
    if lit < b.len() {
        push(Part::Text(&t[lit..]), &mut parts)?;
    }
    Ok(parts)
}

pub fn render(ip: &mut Interp, a: &[Value], name: &str) -> Result<String> {
    // the arguments may be left out when the template needs none (KlongPy `_fmt_check_call`, `_fmt_render`)
    if a.is_empty() {
        return Err(Error::typ(klong_core::msg!("{}: needs a template string (and its arguments)", name)));
    }
    if a.len() > 2 {
        return Err(Error::typ(klong_core::msg!("{}: takes a template and one argument (a list for several), got {} arguments", name, a.len())));
    }
    let template = match &a[0] {
        Value::Str(t) => t.clone(),
        t => return Err(Error::typ(klong_core::msg!(".fmt: template must be a string, got {}", py_type(t)))),
    };
    let template = template.as_str();
    let hosted = match a.get(1).and_then(|y| y.host_items()) {
        Some(r) => Some(r?),
        None => None,
    };
    let args: Vec<Value> = match hosted.as_ref().or(a.get(1)) {
        Some(y) if y.is_list() => y.items()?,
        Some(y) => alloc::vec![y.clone()],
        None => {
            let holes: Vec<Option<usize>> = parse(template)?.iter().filter_map(|p| if let Part::Hole(i, _) = p { Some(*i) } else { None }).collect();
            if !holes.is_empty() {
                let mut names = String::new();
                for (k, h) in holes.iter().enumerate() {
                    if k > 0 {
                        names.push_str(", ");
                    }
                    match h {
                        None => names.push_str("{}"),
                        Some(i) => names.push_str(&klong_core::msg!("{{{}}}", i)),
                    }
                }
                return Err(bad(template, klong_core::msg!("{} placeholder(s) ({}) but no arguments; call .fmt(template;arguments)", holes.len(), names)));
            }
            Vec::new()
        }
    };
    let parts = parse(template)?;
    let holes = parts.iter().filter(|p| matches!(p, Part::Hole(..))).count();
    let auto = parts.iter().filter(|p| matches!(p, Part::Hole(None, _))).count();
    if auto != 0 && auto != holes {
        return Err(bad(template, String::from("cannot mix {} and {N} placeholders")));
    }
    if auto != 0 && auto != args.len() {
        return Err(bad(template, klong_core::msg!("{} placeholder(s) but {} argument(s)", auto, args.len())));
    }
    let mut out = String::new();
    let mut next = 0;
    for p in &parts {
        match p {
            Part::Text(s) => try_push_str(&mut out, s)?,
            Part::Brace(c) => try_push_char(&mut out, *c)?,
            Part::Hole(idx, spec) => {
                let idx = match idx {
                    Some(i) => *i,
                    None => {
                        next += 1;
                        next - 1
                    }
                };
                let v = match args.get(idx) {
                    Some(v) => v,
                    None => return Err(bad(template, klong_core::msg!("index {} out of range for {} argument(s)", idx, args.len()))),
                };
                let text = match spec {
                    None => format_value(&ip.syms, v, true)?,
                    Some(s) if s.contains('.') => {
                        let u = v.unhost(".fmt")?;
                        let x = match &u {
                            Value::Int(i) => *i as Flt,
                            Value::Flt(f) => *f,
                            _ => return Err(Error::typ(klong_core::msg!(".fmt: format '{}' needs a number, got {} in template {}", s, v.py_repr(Some(&ip.syms)), py_repr(template)))),
                        };
                        let w = s.parse::<Flt>().map_err(|_| bad(template, klong_core::msg!("bad format '{}'", s)))?;
                        match klong_core::ops::format2(ip, &Value::Flt(w), &Value::Flt(x))? {
                            Value::Str(t) => klong_core::value::try_str(t.as_str())?,
                            _ => String::new(),
                        }
                    }
                    Some(s) => {
                        let w = s.parse::<Int>().map_err(|_| bad(template, klong_core::msg!("bad format '{}'", s)))?;
                        let shown = Value::str(format_value(&ip.syms, v, true)?);
                        match klong_core::ops::format2(ip, &Value::Int(w), &shown)? {
                            Value::Str(t) => klong_core::value::try_str(t.as_str())?,
                            _ => String::new(),
                        }
                    }
                };
                try_push_str(&mut out, &text)?;
            }
        }
    }
    Ok(out)
}
