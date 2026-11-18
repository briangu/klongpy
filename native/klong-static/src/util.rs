//! Small helpers: Python's `repr` of a string, and the `-v` and error layout shared with `kgpy check`.

/// Python's `repr` of a str.
pub fn py_repr(s: &str) -> String {
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
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

pub fn plural(n: usize, word: &str, many: Option<&str>) -> String {
    if n == 1 {
        format!("{} {}", n, word)
    } else {
        match many {
            Some(m) => format!("{} {}", n, m),
            None => format!("{} {}s", n, word),
        }
    }
}
