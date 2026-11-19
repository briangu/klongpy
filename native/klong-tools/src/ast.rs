//! `kg ast FILE|-e EXPR`: the canonical parse tree of a program (`conformance/README.md`), the text `kgpy ast` printed.
use crate::ToolError;
use klong_core::Config;
use klong_host::{Console, HostPlatform};
use klong_rt::Runtime;
use std::io::{Read, Write};

/// Parse a text on a fresh runtime (nothing is evaluated) and hand the tree to `f`, on a thread with a big stack. A text that does not
/// parse is the syntax error the evaluator would raise.
pub fn with_parse<T: Send + 'static>(
    text: &str,
    name: &str,
    f: impl FnOnce(&klong_core::Interp, &klong_core::parse::Parsed) -> T + Send + 'static,
) -> Result<T, ToolError> {
    let (text, name) = (text.to_string(), name.to_string());
    let r = crate::big_stack(move || {
        let cfg = Config { stack_limit: Some(crate::STACK_BYTES - crate::STACK_BYTES / 4), ..Config::default() };
        let mut rt = Runtime::new(HostPlatform::new(Console::Buffer(Vec::new())), cfg, None);
        rt.interp.parse(&text, &name).map(|p| f(&rt.interp, &p)).map_err(ToolError::from)
    });
    r.unwrap_or_else(|| Err(ToolError::system("cannot start the parser thread")))
}

/// The parse of a text, dumped: one line per statement. Nothing is evaluated.
pub fn ast_text(text: &str, name: &str) -> Result<String, ToolError> {
    with_parse(text, name, |ip, p| klong_core::dump::dump(&p.tree, &ip.syms, &p.stmts))
}

const USAGE: &str = "usage: kg ast [-h] (FILE | -e EXPR)\n";

const HELP: &str = "usage: kg ast [-h] (FILE | -e EXPR)

Print the canonical parse tree of FILE (or of EXPR) and exit. Nothing is run.

positional arguments:
  FILE        the program to parse (- reads standard input)

options:
  -h, --help  show this help message and exit
  -e EXPR, --expr EXPR
              parse this text instead of a file
";

/// The body of `kg ast`: `args` follow the word `ast`. Returns the exit status: 0, 1 when the text does not parse or the file
/// cannot be read, 2 for a wrong command line.
pub fn ast_main(args: &[String], stdin: &mut dyn Read, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let (mut file, mut expr): (Option<String>, Option<String>) = (None, None);
    let mut i = 0;
    let mut only_files = false;
    let bad = |err: &mut dyn Write, m: String| -> i32 {
        let _ = write!(err, "{}kg ast: error: {}\n", USAGE, m);
        2
    };
    while i < args.len() {
        let a = &args[i];
        i += 1;
        if !only_files && a == "--" {
            only_files = true;
        } else if !only_files && (a == "-h" || a == "--help") {
            let _ = out.write_all(HELP.as_bytes());
            return 0;
        } else if !only_files && (a == "-e" || a == "--expr" || a.starts_with("--expr=") || (a.starts_with("-e") && a.len() > 2)) {
            let v = if a == "-e" || a == "--expr" {
                match args.get(i) {
                    Some(v) => {
                        i += 1;
                        v.clone()
                    }
                    None => return bad(err, format!("argument -e/--expr: expected one argument")),
                }
            } else if let Some(v) = a.strip_prefix("--expr=") {
                v.to_string()
            } else {
                a[2..].to_string()
            };
            if expr.is_some() {
                return bad(err, "argument -e/--expr: given twice".to_string());
            }
            expr = Some(v);
        } else if !only_files && a != "-" && a.starts_with('-') {
            return bad(err, format!("unrecognized arguments: {}", a));
        } else if file.is_none() {
            file = Some(a.clone());
        } else {
            return bad(err, format!("unrecognized arguments: {}", a));
        }
    }
    let (text, name) = match (&file, &expr) {
        (Some(_), Some(_)) => return bad(err, "give FILE or -e EXPR, not both".to_string()),
        (None, None) => return bad(err, "the following arguments are required: FILE".to_string()),
        (None, Some(e)) => (e.clone(), "<-e>".to_string()),
        (Some(f), None) => {
            // read as `kgpy ast` reads: UTF-8, and `\r\n` and `\r` read as `\n` (Python's text mode)
            let mut bytes = Vec::new();
            let read = if f == "-" { stdin.read_to_end(&mut bytes).map(|_| ()) } else { std::fs::read(f).map(|b| bytes = b) };
            if let Err(e) = read {
                let why = e.to_string();
                let why = why.split(" (os error").next().unwrap_or(&why).to_string();
                let _ = writeln!(err, "kg ast: {}: {}", f, why);
                return 1;
            }
            match klong_lint::decode(&bytes) {
                Ok(t) => (t, if f == "-" { "<stdin>".to_string() } else { f.clone() }),
                Err(why) => {
                    let _ = writeln!(err, "kg ast: {}: {}", f, why);
                    return 1;
                }
            }
        }
    };
    match ast_text(&text, &name) {
        Ok(s) => {
            let _ = out.write_all(s.as_bytes());
            0
        }
        Err(e) => {
            let _ = writeln!(err, "{}", e.rendered);
            1
        }
    }
}
