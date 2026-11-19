//! What the Python package needed of the tool and engine API: the report's statements, the typed IR as JSON, a program on a shared runtime,
//! the parameter registry, the tables as data, `globals`, library directories and the streamable option.
use klong_static::check::{LitVal, Options};
use klong_static::compile::{compile_program_in, stage_runtime};
use klong_static::host::{call_entry, params_of, set_param_rt, HostValue, Loaded};
use klong_static::ir_json::program_json;
use klong_static::sig::{check_entry_name, check_extern_name, parse_signature, parse_type};
use klong_static::tables;
use std::collections::HashMap;

const PROG: &str = ".param(\"gain\";\"float\";2.5)\n.extern(\"read.adc\";\"int -> float\")\nacc::0\nf::{acc::acc+x;acc}\n.entry(\"f\";\"int -> int\")\n";

#[test]
fn statements_come_in_the_order_they_run_with_their_spans() {
    let r = klong_static::check_source(PROG, "p.kg", &Options::default());
    let kinds: Vec<(&str, &str)> = r.statements.iter().map(|s| (s.kind, s.name.as_str())).collect();
    assert_eq!(kinds, [("param", "gain"), ("extern", "read.adc"), ("global", "acc"), ("function", "f"), ("entry", "f")]);
    let f = r.statements.iter().find(|s| s.kind == "function").unwrap();
    assert_eq!(&PROG[f.span.0..f.span.1], "f::{acc::acc+x;acc}");
}

#[test]
fn a_load_is_a_statement_and_a_loaded_file_has_its_own_source() {
    let dir = std::env::temp_dir().join(format!("kg_wave3_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib3.kg"), "g::{x+1}\n").unwrap();
    let opts = Options { lib_dirs: vec![dir.to_string_lossy().into_owned()], ..Options::default() };
    let r = klong_static::check_source(".l(\"lib3\")\nf::{g(x)}\n.entry(\"f\";\"int -> int\")\n", "p.kg", &opts);
    assert!(r.ok(), "{:?}", r.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>());
    assert_eq!(r.statements[0].kind, "load");
    assert_eq!(r.sources.len(), 2);
    assert!(r.sources[1].name.ends_with("lib3.kg"), "{}", r.sources[1].name);
    let g = r.statements.iter().find(|s| s.name == "g").unwrap();
    assert_eq!(g.src, 1);
    // compiling reports the same path
    let c = compile_program_in("p.kg", Some(".l(\"lib3\")\nf::{g(x)}\n.entry(\"f\";\"int -> int\")\n"), 32, None, HashMap::new(), &[dir.to_string_lossy().into_owned()]).ok().unwrap();
    assert!(c.report.sources[1].name.ends_with("lib3.kg"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn given_globals_are_validated_as_the_checker_validates_literals() {
    let mut given = HashMap::new();
    given.insert("n".to_string(), LitVal::Int(3));
    given.insert("big".to_string(), LitVal::Int(1 << 40));
    let opts = Options { globals: given, ..Options::default() };
    let r = klong_static::check_source("n::1+2\nbig::2*3\nf::{x+n}\n.entry(\"f\";\"int -> int\")\n", "p.kg", &opts);
    let rules: Vec<&str> = r.diagnostics.iter().map(|d| d.rule).collect();
    assert_eq!(rules, ["int-range"]);
    assert_eq!(r.globals.values().find(|g| g.name == "n").unwrap().ty().unwrap().to_string(), "int");
}

#[test]
fn the_typed_ir_is_json_with_spans_in_characters() {
    let text = "acc::0\nf::{acc::acc+x;acc}\n.entry(\"f\";\"int -> int\")\n";
    let c = compile_program_in("t.kg", Some(text), 32, None, HashMap::new(), &[]).ok().unwrap();
    let j = program_json(&c.program, &c.report);
    for want in ["\"name\": \"t\"", "\"int_bits\": 32", "\"k\": \"Assign\"", "\"k\": \"Binary\"", "\"op\": \"add\"", "\"label\": \"f#0\"", "\"signature\": \"int -> int\"", "\"state\": true", "\"effects\": true"] {
        assert!(j.contains(want), "{}\n{}", want, j);
    }
    // a non-ASCII comment before the code moves nothing: spans are characters
    let text2 = ":\" \u{e9}\u{e9}\u{e9} \"\nf::{x+1}\n.entry(\"f\";\"int -> int\")\n";
    let c2 = compile_program_in("t.kg", Some(text2), 32, None, HashMap::new(), &[]).ok().unwrap();
    let j2 = program_json(&c2.program, &c2.report);
    let start = text2.chars().position(|c| c == 'f').unwrap();
    assert!(j2.contains(&format!("{}", start)), "{}", j2);
}

#[test]
fn a_program_on_a_shared_runtime_shares_its_globals_and_parameters() {
    let mut rt = stage_runtime();
    rt.run(PROG, "p.kg").unwrap();
    let ptr: *mut _ = &mut rt;
    let mut p = unsafe { Loaded::attach(ptr, "p.kg", PROG, HashMap::new()) };
    // a call through the program changes the session's global
    assert_eq!(call_entry(&mut p, "f", &[HostValue::Int(4)], None, false).unwrap(), HostValue::Int(4));
    assert_eq!(rt.interp.global("acc").map(|v| HostValue::from_value(&v)), Some(HostValue::Int(4)));
    // and a global set by the session is what the entry sees
    rt.interp.define_global("acc", klong_core::Value::Int(10)).unwrap();
    assert_eq!(call_entry(&mut p, "f", &[HostValue::Int(1)], None, false).unwrap(), HostValue::Int(11));
}

#[test]
fn the_param_registry_lists_and_sets() {
    let mut rt = stage_runtime();
    rt.run(PROG, "p.kg").unwrap();
    let rows = params_of(&rt);
    assert_eq!(rows.len(), 1);
    assert_eq!((rows[0].name.as_str(), rows[0].ty.to_string()), ("gain", "float".to_string()));
    assert_eq!(rows[0].default, HostValue::Float(2.5));
    assert_eq!(rows[0].value, Some(HostValue::Float(2.5)));
    assert_eq!(set_param_rt(&mut rt, "gain", &HostValue::Int(3)).unwrap(), HostValue::Float(3.0));
    assert_eq!(params_of(&rt)[0].value, Some(HostValue::Float(3.0)));
    assert_eq!(params_of(&rt)[0].default, HostValue::Float(2.5));
    assert!(set_param_rt(&mut rt, "nope", &HostValue::Int(1)).is_err());
}

#[test]
fn the_tables_are_consistent() {
    // every rule an operator row names exists; every kind, type rule and shape rule a row uses is defined
    let kinds: Vec<&str> = tables::OPERAND_KINDS.iter().map(|k| k.0).collect();
    let types: Vec<&str> = tables::TYPE_RULES.iter().map(|k| k.0).collect();
    let shapes: Vec<&str> = tables::SHAPE_RULES.iter().map(|k| k.0).collect();
    let mut seen = std::collections::HashSet::new();
    for o in tables::OPTABLE {
        assert!(seen.insert((o.glyph, o.valence)), "two rows for {} {}", o.glyph, o.valence);
        assert!(o.operands.iter().all(|k| kinds.contains(k)), "{}", o.id);
        if o.allowed {
            assert!(types.contains(&o.type_rule.unwrap()) && shapes.contains(&o.shape_rule.unwrap()), "{}", o.id);
            assert!(o.rule.is_none());
        } else {
            assert!(tables::rule(o.rule.unwrap()).is_some(), "{}", o.id);
            assert!(o.examples.is_empty());
        }
        // an example's signature parses, and its inputs are a JSON array
        for e in o.examples {
            assert!(parse_signature(e.signature, false).is_ok(), "{}: {}", o.id, e.signature);
            if let Some(i) = e.inputs {
                assert!(i.starts_with('[') && i.ends_with(']'), "{}", i);
            }
        }
    }
    for f in tables::FORMS {
        for e in f.examples {
            assert!(parse_signature(e.signature, false).is_ok(), "{}: {}", f.id, e.signature);
        }
    }
    let ids: std::collections::HashSet<&str> = tables::RULES.iter().map(|r| r.id).collect();
    assert_eq!(ids.len(), tables::RULES.len(), "duplicate rule id");
    assert!(tables::STREAM_RULES.iter().all(|r| ids.contains(r.id)));
    assert!(tables::RULES.iter().all(|r| tables::GROUPS.iter().any(|g| g.0 == r.group)));
}

#[test]
fn signature_functions_match_the_documented_messages() {
    assert_eq!(parse_signature("int float[64] -> float", true).unwrap().to_string(), "int float[64] -> float");
    let e = parse_signature("int ->", true).err().unwrap();
    assert_eq!(e.message, "a signature needs a result type after '->' (void for none)");
    let e = parse_signature("str -> int", false).err().unwrap();
    assert_eq!((e.message.as_str(), e.span), ("str is a parameter type of externs only", (0, 3)));
    assert_eq!(parse_type("float[16,241]").unwrap().to_string(), "float[16,241]");
    assert!(check_extern_name("read.adc").is_none());
    assert!(check_entry_name("init").unwrap().contains("compiler generates"));
}

#[test]
fn check_with_streamable_reports_what_keeps_a_kernel_from_streaming() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/static/reject");
    let path = dir.join("stream-loop.kg").to_string_lossy().into_owned();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let plain = klong_static::cli::run_check_ex("t", &[path.clone()], false, false, false, Vec::new(), &mut out, &mut err);
    assert_eq!(plain, 0, "{}", String::from_utf8_lossy(&err));
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let streamed = klong_static::cli::run_check_ex("t", &[path], false, false, true, Vec::new(), &mut out, &mut err);
    assert_eq!(streamed, 1);
    assert!(String::from_utf8_lossy(&err).contains("[stream-loop]"), "{}", String::from_utf8_lossy(&err));
}
