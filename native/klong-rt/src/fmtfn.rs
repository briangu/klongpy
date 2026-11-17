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

fn bad(template: &str, what: String) -> Error {
    Error::range(klong_core::msg!(".fmt: {} in template \"{}\"", what, template))
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
    let (template, y) = match a {
        [Value::Str(t), y] => (t.clone(), y),
        [t, _] => return Err(Error::typ(klong_core::msg!("{}: template must be a string, got {}", name, t.describe()))),
        _ => return Err(Error::arity(klong_core::msg!("{} takes 2 arguments, got {}", name, a.len()))),
    };
    let template = template.as_str();
    let args: Vec<Value> = if y.is_list() { y.items()? } else { alloc::vec![y.clone()] };
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
                        let x = match v {
                            Value::Int(i) => *i as Flt,
                            Value::Flt(f) => *f,
                            _ => return Err(Error::typ(klong_core::msg!(".fmt: format '{}' needs a number, got {} in template \"{}\"", s, v.describe(), template))),
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
