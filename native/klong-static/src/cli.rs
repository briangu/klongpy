//! `kg check FILE...`: the options, the output and the exit status of `kgpy check`.
use crate::check::{check_source, Diag, Options, Report};
use crate::util::plural;
use std::io::Write;

const CLIP: usize = 120;

/// The error laid out as `KlongError.render` lays it out: `name:line:col: static error: message [rule]`, the source line, carets, the hint.
pub fn render(d: &Diag, r: &Report) -> String {
    let src = &r.sources[d.src];
    let text = &src.source.text;
    let (line, col) = src.source.line_col(d.span.0);
    let name = &src.name;
    let head = format!("static error: {} [{}]", d.message, d.rule);
    let where_ = if name.is_empty() || name.starts_with("<repl") { format!("{}:{}", line, col) } else { format!("{}:{}:{}", name, line, col) };
    let mut out = format!("{}: {}", where_, head);
    // the text of the line
    let mut starts: Vec<usize> = vec![0];
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    let s = starts[line - 1];
    let e = if line < starts.len() { starts[line] - 1 } else { text.len() };
    let mut chars: Vec<char> = text[s..e].chars().map(|c| if c == '\t' { ' ' } else { c }).collect();
    let mut start = col - 1;
    let span_chars = text.get(d.span.0..d.span.1.max(d.span.0)).map(|t| t.chars().count()).unwrap_or(d.span.1.saturating_sub(d.span.0));
    let mut width = span_chars.min(chars.len().saturating_sub(start)).max(1);
    if chars.len() > CLIP {
        let lo = start.saturating_sub(CLIP / 2).min(chars.len() - CLIP);
        chars = chars[lo..lo + CLIP].to_vec();
        start -= lo;
        width = width.min(CLIP - start).max(1);
    }
    let shown: String = chars.iter().collect();
    out.push_str(&format!("\n  {}\n  {}{}", shown, " ".repeat(start), "^".repeat(width)));
    if let Some(h) = &d.hint {
        if !h.is_empty() {
            out.push_str(&format!("\n  hint: {}", h));
        }
    }
    out
}

fn describe(r: &Report, out: &mut dyn Write) {
    for e in &r.entries {
        let _ = writeln!(out, "  entry   {}: {}", e.name, e.signature);
    }
    for x in r.externs.values() {
        let _ = writeln!(out, "  extern  {}: {} (C {})", x.name, x.signature, x.c_symbol);
    }
    for g in r.globals.values() {
        let kind = if g.assigned { "state" } else { "const" };
        let ty = match g.ty() {
            Some(t) => t.to_string(),
            None => "computed on the host; type not known without evaluating".to_string(),
        };
        let _ = writeln!(out, "  global  {}: {} ({})", g.name, ty, kind);
    }
    for f in r.functions.values() {
        if !f.reachable {
            let _ = writeln!(out, "  fn      {}: host only (no entry reaches it)", f.name);
            continue;
        }
        for (args, result) in &f.specializations {
            let shown: Vec<String> = args.iter().map(|a| a.as_ref().map(|t| t.to_string()).unwrap_or_else(|| "?".to_string())).collect();
            let res = result.as_ref().map(|t| t.to_string()).unwrap_or_else(|| "?".to_string());
            let _ = writeln!(out, "  fn      {}({}) -> {}", f.name, shown.join("; "), res);
        }
    }
}

fn usage(prog: &str) -> String {
    let head = format!("usage: {} ", prog);
    let line = "[-h] [--int64] [--streamable | --dump-ir] [-v]";
    let tail = "FILE [FILE ...]";
    if head.len() + line.len() + 1 + tail.len() > 78 {
        format!("{}{}\n{}{}\n", head, line, " ".repeat(head.len()), tail)
    } else {
        format!("{}{} {}\n", head, line, tail)
    }
}

fn help(prog: &str) -> String {
    format!(
        "{}\nCheck that FILEs are inside the static profile (the subset that compiles ahead\nof time) without running them, then exit (1 if any is outside).\n\n\
positional arguments:\n  FILE           the programs to check (- is stdin)\n\n\
options:\n  -h, --help     show this help message and exit\n  --int64        kg_int is 64 bits wide (the default is 32)\n  \
--streamable   also say what keeps a kernel from taking one record per clock\n                 cycle\n  \
--dump-ir      evaluate the compile-time definitions and print the typed IR\n                 instead of checking\n  \
-v, --verbose  list the functions, globals and declarations of each file\n",
        usage(prog)
    )
}

struct Args {
    files: Vec<String>,
    int64: bool,
    streamable: bool,
    dump_ir: bool,
    verbose: bool,
}

enum Parsed {
    Run(Args),
    Exit(i32),
}

fn parse_args(prog: &str, args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> Parsed {
    let fail = |err: &mut dyn Write, msg: String| {
        let _ = write!(err, "{}{}: error: {}\n", usage(prog), prog, msg);
        Parsed::Exit(2)
    };
    let mut a = Args { files: Vec::new(), int64: false, streamable: false, dump_ir: false, verbose: false };
    let mut unknown = Vec::new();
    let longs = ["--help", "--int64", "--streamable", "--dump-ir", "--verbose"];
    let mut only_files = false;
    for arg in args {
        if only_files || arg == "-" || !arg.starts_with('-') || (arg.len() > 1 && arg[1..].chars().all(|c| c.is_ascii_digit() || c == '.') && !arg.starts_with("--")) {
            a.files.push(arg.clone());
            continue;
        }
        if arg == "--" {
            only_files = true;
            continue;
        }
        let name = match arg.find('=') {
            Some(p) if arg.starts_with("--") => arg[..p].to_string(),
            _ => arg.clone(),
        };
        let full: Option<&str> = if name.starts_with("--") {
            let m: Vec<&&str> = longs.iter().filter(|l| l.starts_with(name.as_str())).collect();
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
        } else if arg.len() >= 2 && arg[1..].chars().all(|c| c == 'v' || c == 'h') {
            // -v, -h and runs of them
            for c in arg[1..].chars() {
                if c == 'h' {
                    let _ = out.write_all(help(prog).as_bytes());
                    return Parsed::Exit(0);
                }
                a.verbose = true;
            }
            continue;
        } else {
            None
        };
        match full {
            Some("--help") => {
                let _ = out.write_all(help(prog).as_bytes());
                return Parsed::Exit(0);
            }
            Some("--int64") => a.int64 = true,
            Some("--verbose") => a.verbose = true,
            Some("--streamable") => {
                if a.dump_ir {
                    return fail(err, "argument --streamable: not allowed with argument --dump-ir".to_string());
                }
                a.streamable = true;
            }
            Some("--dump-ir") => {
                if a.streamable {
                    return fail(err, "argument --dump-ir: not allowed with argument --streamable".to_string());
                }
                a.dump_ir = true;
            }
            _ => unknown.push(arg.clone()),
        }
    }
    if a.files.is_empty() {
        return fail(err, "the following arguments are required: FILE".to_string());
    }
    if !unknown.is_empty() {
        return fail(err, format!("unrecognized arguments: {}", unknown.join(" ")));
    }
    Parsed::Run(a)
}

/// The body of `kg check FILE...` (`args` follow the word `check`). Every problem of every file is printed to stderr; a file inside
/// the profile gets one line on stdout. Returns the exit status: 1 if any file is outside the profile, 2 if a file could not be read
/// (and none was outside), else 0.
pub fn check_main(prog: &str, args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let a = match parse_args(prog, args, out, err) {
        Parsed::Run(a) => a,
        Parsed::Exit(c) => return c,
    };
    if a.dump_ir {
        return crate::compile::dump_ir_main(&a.files, if a.int64 { 64 } else { 32 }, out, err);
    }
    run_check_ex(prog, &a.files, a.int64, a.verbose, a.streamable, Vec::new(), out, err)
}

/// What `check_main` does once the options are known: check each file, print what `kgpy check` prints, return its status.
pub fn run_check(prog: &str, files: &[String], int64: bool, verbose: bool, lib_dirs: Vec<String>, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    run_check_ex(prog, files, int64, verbose, false, lib_dirs, out, err)
}

/// `run_check` with `--streamable`.
#[allow(clippy::too_many_arguments)]
pub fn run_check_ex(_prog: &str, files: &[String], int64: bool, verbose: bool, streamable: bool, lib_dirs: Vec<String>, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let a = Args { files: files.to_vec(), int64, streamable: false, dump_ir: false, verbose };
    let opts = Options { int_bits: if a.int64 { 64 } else { 32 }, lib_dirs: lib_dirs.clone(), ..Options::default() };
    let (mut bad, mut unreadable, mut total) = (0usize, 0usize, 0usize);
    for path in &a.files {
        let read = if path == "-" {
            use std::io::Read;
            let mut buf = Vec::new();
            match std::io::stdin().read_to_end(&mut buf) {
                Ok(_) => klong_lint::decode(&buf).map(|t| ("<stdin>".to_string(), t)),
                Err(e) => Err(e.to_string()),
            }
        } else {
            klong_lint::read_text(path).map(|t| (path.clone(), t))
        };
        let (name, text) = match read {
            Ok(x) => x,
            Err(why) => {
                let _ = writeln!(err, "{}: cannot read: {}", path, why);
                unreadable += 1;
                continue;
            }
        };
        let report = check_source(&text, &name, &opts);
        if !report.diagnostics.is_empty() {
            bad += 1;
            total += report.diagnostics.len();
            for d in &report.diagnostics {
                let _ = writeln!(err, "{}", render(d, &report));
            }
        } else {
            let mut note = if report.entries.is_empty() { "; no .entry, so every function was checked with arguments of unknown type".to_string() } else { String::new() };
            let un = report.unresolved();
            if !un.is_empty() {
                let n = un.len();
                let shown: Vec<&str> = un.iter().take(4).map(|s| s.as_str()).collect();
                note.push_str(&format!(
                    "; {} ({}{}) {} computed on the host: checked for form only until the program is compiled",
                    plural(n, "global", None),
                    shown.join(", "),
                    if n > 4 { ", ..." } else { "" },
                    if n == 1 { "is" } else { "are" }
                ));
            }
            let _ = writeln!(out, "{}: inside the static profile ({}){}", path, report.summary(), note);
            if streamable && path != "-" {
                match crate::compile::compile_program_in(path, None, if a.int64 { 64 } else { 32 }, None, std::collections::HashMap::new(), &lib_dirs) {
                    Err(f) => {
                        for t in f.render() {
                            let _ = writeln!(err, "{}", t);
                        }
                        bad += 1;
                        total += 1;
                    }
                    Ok(c) => {
                        let found = crate::streamable::check_streamable(&c.program, &c.report);
                        if found.is_empty() {
                            let _ = writeln!(out, "{}: streamable: nothing keeps a kernel from taking one record per clock cycle", path);
                        } else {
                            bad += 1;
                            total += found.len();
                            for d in &found {
                                let _ = writeln!(err, "{}", render(d, &c.report));
                            }
                        }
                    }
                }
            }
        }
        if a.verbose {
            describe(&report, out);
        }
    }
    if total > 0 {
        let _ = writeln!(err, "{} in {}", plural(total, "static error", None), plural(bad, "file", None));
    }
    if bad > 0 {
        1
    } else if unreadable > 0 {
        2
    } else {
        0
    }
}
