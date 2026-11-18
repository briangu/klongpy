//! `klong lint FILE...`: the options of `kgpy lint`, the output and the exit status.
use crate::{lint_files, parse_codes, parse_names, Options};
use std::io::Write;

/// Python's `repr` of a str, for the error text of an unknown code.
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

fn usage(prog: &str) -> String {
    let head = format!("usage: {} ", prog);
    let line = "[-h] [--select CODES] [--ignore CODES] [--define NAMES] [-v]";
    let tail = "FILE [FILE ...]";
    if head.len() + line.len() + 1 + tail.len() > 78 {
        format!("{}{}\n{}{}\n", head, line, " ".repeat(head.len()), tail)
    } else {
        format!("{}{} {}\n", head, line, tail)
    }
}

fn help(prog: &str) -> String {
    format!(
        "{}\nCheck FILEs for the classic Klong traps without running them, then exit (1 if\nany error was found).\n\n\
positional arguments:\n  FILE            the programs to check (- is stdin)\n\n\
options:\n  -h, --help      show this help message and exit\n  --select CODES  report only these codes (comma-separated)\n  \
--ignore CODES  leave these codes out (comma-separated)\n  --define NAMES  names the host defines before the program runs (comma-\n                  separated)\n  \
-v, --verbose   enable verbose output\n",
        usage(prog)
    )
}

enum Parsed {
    Run { files: Vec<String>, select: Option<String>, ignore: Option<String>, define: Option<String>, verbose: bool },
    Exit(i32),
}

fn parse_args(prog: &str, args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Parsed {
    let fail = |err: &mut dyn Write, msg: String| {
        let _ = write!(err, "{}{}: error: {}\n", usage(prog), prog, msg);
        Parsed::Exit(2)
    };
    let mut files = Vec::new();
    let mut unknown = Vec::new();
    let (mut select, mut ignore, mut define) = (None, None, None);
    let mut verbose = false;
    let longs = ["--help", "--select", "--ignore", "--define", "--verbose"];
    let mut i = 0;
    let mut only_files = false;
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if only_files || a == "-" || !a.starts_with('-') || (a.len() > 1 && a[1..].chars().all(|c| c.is_ascii_digit() || c == '.') && !a.starts_with("--")) {
            files.push(a.clone());
            continue;
        }
        if a == "--" {
            only_files = true;
            continue;
        }
        let (name, inline): (String, Option<String>) = if a.starts_with("--") {
            match a.find('=') {
                Some(p) => (a[..p].to_string(), Some(a[p + 1..].to_string())),
                None => (a.clone(), None),
            }
        } else {
            // a short option, possibly followed by its value or by more flags
            let c = &a[..2.min(a.len())];
            if a.len() > 2 && (c == "-v" || c == "-h") {
                // -vh style: only -v and -h exist; treat the rest as more flags
                let mut bad = false;
                for ch in a[1..].chars() {
                    match ch {
                        'v' => verbose = true,
                        'h' => {
                            let _ = out.write_all(help(prog).as_bytes());
                            return Parsed::Exit(0);
                        }
                        _ => bad = true,
                    }
                }
                if bad {
                    unknown.push(a.clone());
                }
                continue;
            }
            (c.to_string(), None)
        };
        let full: Option<&str> = if name.starts_with("--") {
            let m: Vec<&&str> = longs.iter().filter(|l| **l == name || l.starts_with(name.as_str())).collect();
            if let Some(e) = longs.iter().find(|l| **l == name) {
                Some(*e)
            } else if m.len() == 1 {
                Some(**m.first().unwrap())
            } else if m.len() > 1 {
                let opts: Vec<&str> = m.iter().map(|s| **s).collect();
                return fail(err, format!("ambiguous option: {} could match {}", name, opts.join(", ")));
            } else {
                None
            }
        } else {
            match name.as_str() {
                "-h" => Some("--help"),
                "-v" => Some("--verbose"),
                _ => None,
            }
        };
        match full {
            Some("--help") => {
                let _ = out.write_all(help(prog).as_bytes());
                return Parsed::Exit(0);
            }
            Some("--verbose") => verbose = true,
            Some(opt @ ("--select" | "--ignore" | "--define")) => {
                let v = match inline {
                    Some(v) => v,
                    None => {
                        if i < args.len() && (!args[i].starts_with('-') || args[i] == "-" || args[i].len() == 1) {
                            i += 1;
                            args[i - 1].clone()
                        } else {
                            return fail(err, format!("argument {}: expected one argument", opt));
                        }
                    }
                };
                match opt {
                    "--select" => select = Some(v),
                    "--ignore" => ignore = Some(v),
                    _ => define = Some(v),
                }
            }
            _ => unknown.push(a.clone()),
        }
    }
    if files.is_empty() && unknown.is_empty() {
        return fail(err, "the following arguments are required: FILE".to_string());
    }
    if !unknown.is_empty() {
        if files.is_empty() {
            return fail(err, "the following arguments are required: FILE".to_string());
        }
        return fail(err, format!("unrecognized arguments: {}", unknown.join(" ")));
    }
    Parsed::Run { files, select, ignore, define, verbose }
}

/// The body of `klong lint FILE...` (`args` follow the word `lint`). Prints the findings and a summary and returns the
/// exit status: 1 if any E finding was reported, 2 if a file could not be read (and no E finding), else 0.
pub fn lint_main(prog: &str, args: &[String], lib_dirs: Vec<String>, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let (files, select, ignore, define, verbose) = match parse_args(prog, args, out, err) {
        Parsed::Run { files, select, ignore, define, verbose } => (files, select, ignore, define, verbose),
        Parsed::Exit(c) => return c,
    };
    let mut opts = Options { lib_dirs, ..Options::default() };
    match select.as_deref().map(parse_codes).transpose() {
        Ok(s) => opts.select = s,
        Err(e) => {
            let _ = writeln!(err, "{}: {}", prog, e);
            return 2;
        }
    }
    match ignore.as_deref().map(parse_codes).transpose() {
        Ok(s) => opts.ignore = s.unwrap_or_default(),
        Err(e) => {
            let _ = writeln!(err, "{}: {}", prog, e);
            return 2;
        }
    }
    opts.define = define.as_deref().map(parse_names).unwrap_or_default();
    let mut errors = 0usize;
    let mut warnings = 0usize;
    let mut unreadable = 0usize;
    for r in lint_files(&files, &opts) {
        match r.result {
            Err(why) => {
                let _ = writeln!(err, "{}: cannot read: {}", r.path, why);
                unreadable += 1;
            }
            Ok(found) => {
                for d in found {
                    let _ = writeln!(out, "{}", d.format(verbose));
                    if d.severity == "error" {
                        errors += 1;
                    } else {
                        warnings += 1;
                    }
                }
            }
        }
    }
    let n = files.len() - unreadable;
    let pl = |n: usize, w: &str| if n == 1 { format!("{} {}", n, w) } else { format!("{} {}s", n, w) };
    let _ = writeln!(out, "{}, {} in {}", pl(errors, "error"), pl(warnings, "warning"), pl(n, "file"));
    if errors > 0 {
        1
    } else if unreadable > 0 {
        2
    } else {
        0
    }
}
