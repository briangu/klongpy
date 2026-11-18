//! A small `argparse` look-alike for the subcommands that take options with values (`kgpy compile c`, `kgpy fpga source`): the usage
//! line wrapped as argparse wraps it, `--opt=value`, `-oDIR`, unique prefixes of long options, intermixed positionals, the error
//! texts and the exit status 2.
use std::collections::HashMap;

pub struct Opt {
    pub short: Option<&'static str>,
    pub long: &'static str,
    /// the metavar of an option that takes a value, None for a flag
    pub metavar: Option<&'static str>,
    pub int: bool,
}

pub struct Spec {
    pub prog: String,
    pub opts: Vec<Opt>,
    pub positional: &'static str,
    pub help: String,
}

#[derive(Default)]
pub struct Parsed {
    pub values: HashMap<&'static str, String>,
    pub flags: Vec<&'static str>,
    pub files: Vec<String>,
}

impl Parsed {
    pub fn flag(&self, long: &str) -> bool {
        self.flags.contains(&long)
    }
    pub fn get(&self, long: &str) -> Option<&String> {
        self.values.get(long)
    }
}

pub enum Outcome {
    Run(Parsed),
    Exit(i32),
}

const WIDTH: usize = 78;

fn opt_text(o: &Opt) -> String {
    let name = o.short.unwrap_or(o.long);
    match o.metavar {
        Some(m) => format!("[{} {}]", name, m),
        None => format!("[{}]", name),
    }
}

/// argparse's usage text for the options of the spec and the positional `FILE [FILE ...]`.
pub fn usage(spec: &Spec) -> String {
    let prefix = "usage: ";
    let mut opt_parts = vec!["[-h]".to_string()];
    opt_parts.extend(spec.opts.iter().map(opt_text));
    let pos_parts = vec![spec.positional.to_string()];
    let one: Vec<String> = std::iter::once(spec.prog.clone()).chain(opt_parts.iter().cloned()).chain(pos_parts.iter().cloned()).collect();
    let flat = one.join(" ");
    if prefix.len() + flat.len() <= WIDTH {
        return format!("{}{}\n", prefix, flat);
    }
    let indent = " ".repeat(prefix.len() + spec.prog.len() + 1);
    fn get_lines(parts: &[String], indent: &str, prefix: Option<&str>) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut line: Vec<String> = Vec::new();
        let mut line_len: i64 = match prefix {
            Some(p) => p.len() as i64 - 1,
            None => indent.len() as i64 - 1,
        };
        for part in parts {
            if line_len + 1 + part.len() as i64 > WIDTH as i64 && !line.is_empty() {
                lines.push(format!("{}{}", indent, line.join(" ")));
                line.clear();
                line_len = indent.len() as i64 - 1;
            }
            line.push(part.clone());
            line_len += part.len() as i64 + 1;
        }
        if !line.is_empty() {
            lines.push(format!("{}{}", indent, line.join(" ")));
        }
        if prefix.is_some() {
            lines[0] = lines[0][indent.len()..].to_string();
        }
        lines
    }
    let mut parts = vec![spec.prog.clone()];
    parts.extend(opt_parts);
    let mut lines = get_lines(&parts, &indent, Some(prefix));
    lines.extend(get_lines(&pos_parts, &indent, None));
    format!("{}{}\n", prefix, lines.join("\n"))
}

fn fail(spec: &Spec, err: &mut dyn std::io::Write, msg: String) -> Outcome {
    let _ = write!(err, "{}{}: error: {}\n", usage(spec), spec.prog, msg);
    Outcome::Exit(2)
}

fn looks_negative(s: &str) -> bool {
    s.len() > 1 && s.starts_with('-') && s[1..].chars().all(|c| c.is_ascii_digit() || c == '.') && s[1..].chars().any(|c| c.is_ascii_digit())
}

pub fn parse(spec: &Spec, args: &[String], out: &mut dyn std::io::Write, err: &mut dyn std::io::Write) -> Outcome {
    let mut p = Parsed::default();
    let mut unknown: Vec<String> = Vec::new();
    let mut only_files = false;
    let mut i = 0;
    let longs: Vec<&str> = std::iter::once("--help").chain(spec.opts.iter().map(|o| o.long)).collect();
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if only_files || arg == "-" || !arg.starts_with('-') || looks_negative(arg) {
            p.files.push(arg.clone());
            continue;
        }
        if arg == "--" {
            only_files = true;
            continue;
        }
        // which option is it, and is its value attached?
        let (name, mut attached): (String, Option<String>) = if arg.starts_with("--") {
            match arg.find('=') {
                Some(pos) => (arg[..pos].to_string(), Some(arg[pos + 1..].to_string())),
                None => (arg.clone(), None),
            }
        } else {
            // a short option, maybe with its value attached
            let key = &arg[..2.min(arg.len())];
            if arg.len() > 2 {
                if key == "-h" {
                    (key.to_string(), None)
                } else {
                    (key.to_string(), Some(arg[2..].to_string()))
                }
            } else {
                (key.to_string(), None)
            }
        };
        let long: Option<&'static str> = if name.starts_with("--") {
            if let Some(e) = longs.iter().find(|l| **l == name) {
                Some(*e)
            } else {
                let m: Vec<&&str> = longs.iter().filter(|l| l.starts_with(name.as_str())).collect();
                if m.len() == 1 {
                    Some(**m.first().unwrap())
                } else if m.len() > 1 {
                    let opts: Vec<&str> = m.iter().map(|s| **s).collect();
                    return fail(spec, err, format!("ambiguous option: {} could match {}", name, opts.join(", ")));
                } else {
                    None
                }
            }
        } else if name == "-h" {
            Some("--help")
        } else {
            spec.opts.iter().find(|o| o.short == Some(name.as_str())).map(|o| o.long)
        };
        let Some(long) = long else {
            unknown.push(arg.clone());
            continue;
        };
        if long == "--help" {
            let _ = out.write_all(spec.help.as_bytes());
            return Outcome::Exit(0);
        }
        let o = spec.opts.iter().find(|o| o.long == long).unwrap();
        match o.metavar {
            None => {
                if attached.is_some() && arg.starts_with("--") {
                    return fail(spec, err, format!("argument {}: ignored explicit argument '{}'", flag_names(o), attached.unwrap()));
                }
                // a run of short flags is not needed: no two options are flags with short names
                p.flags.push(o.long);
            }
            Some(_) => {
                if attached.is_none() {
                    match args.get(i) {
                        Some(v) if !v.starts_with('-') || looks_negative(v) || v == "-" => {
                            attached = Some(v.clone());
                            i += 1;
                        }
                        _ => return fail(spec, err, format!("argument {}: expected one argument", flag_names(o))),
                    }
                }
                let v = attached.unwrap();
                if o.int && v.trim().parse::<i64>().is_err() {
                    return fail(spec, err, format!("argument {}: invalid int value: '{}'", flag_names(o), v));
                }
                p.values.insert(o.long, v);
            }
        }
    }
    if p.files.is_empty() {
        return fail(spec, err, "the following arguments are required: FILE".to_string());
    }
    if !unknown.is_empty() {
        return fail(spec, err, format!("unrecognized arguments: {}", unknown.join(" ")));
    }
    Outcome::Run(p)
}

fn flag_names(o: &Opt) -> String {
    match o.short {
        Some(s) => format!("{}/{}", s, o.long),
        None => o.long.to_string(),
    }
}

/// `p.error(msg)` after parsing.
pub fn error(spec: &Spec, err: &mut dyn std::io::Write, msg: &str) -> i32 {
    let _ = write!(err, "{}{}: error: {}\n", usage(spec), spec.prog, msg);
    2
}
